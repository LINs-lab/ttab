# -*- coding: utf-8 -*-
import copy
import functools
from typing import List

import torch
import torch.nn as nn
import ttab.loads.define_dataset as define_dataset
import ttab.model_adaptation.utils as adaptation_utils
from ttab.api import Batch
from ttab.model_adaptation.base_adaptation import BaseAdaptation
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import Metrics
from ttab.utils.auxiliary import fork_rng_with_seed
from ttab.utils.logging import Logger
from ttab.utils.timer import Timer


class T3A(BaseAdaptation):
    """
    T3A: Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization,
    https://openreview.net/forum?id=e_yvNqkJKAW&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2021%2FConference%2FAuthors%23your-submissions),
    https://github.com/matsuolab/T3A

    T3A adjusts a trained linear classifier with the following procedure:
    (1) compute a pseudo-prototype representation for each class.
    (2) classify each sample based on its distance to the pseudo-prototypes.
    """

    def __init__(self, meta_conf, model: nn.Module):
        super(T3A, self).__init__(meta_conf, model)

    def _prior_safety_check(self):

        assert self._meta_conf.top_M > 0, "top_M must be correctly specified"
        assert (
            self._meta_conf.debug is not None
        ), "The state of debug should be specified"
        assert self._meta_conf.n_train_steps > 0, "Adaptation steps requires >= 1."

    def _initialize_model(self, model):
        """Configure model for adaptation."""
        # In T3A, no update on model params.
        model.requires_grad_(False)
        model.eval()

        return model.to(self._meta_conf.device)

    def _initialize_trainable_parameters(self):
        """
        During adaptation, T3A doesn't need to update params in network.
        """
        self._adapt_module_names = []
        self._classifier_layers = []

        # TODO: make this more general
        # Problem description: the naming and structure of classifier layers may change.
        # Be careful: the following list may not cover all cases when using model outside of this library.
        freezed_module_name = ["fc", "classifier", "head"]

        for named_module, module in self._model.named_children():
            if named_module in freezed_module_name:
                assert isinstance(module, nn.Linear)
                self._classifier_layers.append(module)
            else:
                self._adapt_module_names.append(named_module)

        self.warmup_supports = self._classifier_layers[-1].weight.data.to(
            self._meta_conf.device
        )
        self._num_classes = self._classifier_layers[-1].weight.data.size(0)

    def _initialize_optimizer(self, params) -> torch.optim.Optimizer:
        """In T3A, no optimizer is used."""
        pass

    def _post_safety_check(self):
        is_training = self._model.training
        assert not is_training, "T3A does not need train mode: call model.eval()."

        param_grads = [p.requires_grad for p in (self._model.parameters())]
        has_any_params = any(param_grads)
        assert not has_any_params, "adaptation does not need trainable params."

    def initialize(self, seed: int):
        """Initialize the algorithm."""
        self._model = self._initialize_model(model=copy.deepcopy(self._base_model))
        self._initialize_trainable_parameters()
        self._auxiliary_data_cls = define_dataset.ConstructAuxiliaryDataset(
            config=self._meta_conf
        )

        warmup_prob = self.warmup_supports
        for module in self._classifier_layers:
            warmup_prob = module(warmup_prob)

        self.warmup_ent = adaptation_utils.softmax_entropy(warmup_prob)
        self.warmup_labels = nn.functional.one_hot(
            warmup_prob.argmax(1), num_classes=self._num_classes
        ).float()

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent

        self.top_M = self._meta_conf.top_M
        self.softmax = nn.Softmax(-1)

        # only use one-step adaptation.
        if self._meta_conf.n_train_steps > 1:
            self._meta_conf.n_train_steps = 1

    def one_adapt_step(
        self,
        model: torch.nn.Module,
        batch: Batch,
        timer: Timer,
        random_seed: int = None,
    ):
        """adapt the model in one step."""
        with timer("forward"):
            with fork_rng_with_seed(random_seed):
                feas = model.forward_head(
                    model.forward_features(batch._x), pre_logits=True
                )
                feas = feas.view(feas.size(0), -1)

            y_hat = self._model(batch._x)
            label_hat = nn.functional.one_hot(
                y_hat.argmax(1), num_classes=self._num_classes
            ).float()
            ent = adaptation_utils.softmax_entropy(y_hat)

            # prediction.
            assert (
                self.supports.device == feas.device
            ), "Supports and features should be on the same device."
            self.supports = torch.cat([self.supports, feas])
            self.labels = torch.cat([self.labels, label_hat])
            self.ent = torch.cat([self.ent, ent])

            supports, labels = self.select_supports()
            supports = nn.functional.normalize(supports, dim=1)
            weights = supports.T @ (labels)
            adapted_y = feas @ nn.functional.normalize(weights, dim=0)
            loss = adaptation_utils.softmax_entropy(adapted_y).mean(0)

        return {"loss": loss.item(), "yhat": adapted_y, "grads": None}

    def adapt_and_eval(
        self,
        episodic: bool,
        metrics: Metrics,
        model_selection_method: BaseSelection,
        current_batch: Batch,
        previous_batches: List[Batch],
        logger: Logger,
        timer: Timer,
    ):
        """The key entry of test-time adaptation."""
        # some simple initialization.
        log = functools.partial(logger.log, display=self._meta_conf.debug)

        # model_selection method is defined but not used in T3A.
        log(f"\tinitialize selection method={model_selection_method.name}.")
        model_selection_method.initialize()

        # adaptation.
        with timer("test_adaptation"):
            log(f"\tadapt the model for {self._meta_conf.n_train_steps} steps.")
            for _ in range(self._meta_conf.n_train_steps):
                adaptation_result = self.one_adapt_step(
                    self._model, current_batch, timer, random_seed=self._meta_conf.seed
                )

        with timer("evaluate_adaptation_result"):
            metrics.eval(current_batch._y, adaptation_result["yhat"])
            if self._meta_conf.base_data_name in ["waterbirds"]:
                self.tta_loss_computer.loss(
                    adaptation_result["yhat"],
                    current_batch._y,
                    current_batch._g,
                    is_training=False,
                )

    def generate_representation(self, batch: Batch):
        """
        The classifier adaptation needs the feature representations as inputs.
        """

        assert (
            not self._model.training
        ), "The generation process needs model.eval() mode."

        inputs = batch._x
        # targets = batch._y

        feas = self._model.forward_features(inputs)
        feas = feas.view(inputs.size(0), -1)

        target_hat = self._model(inputs)
        ent = adaptation_utils.softmax_entropy(target_hat)
        label_hat = target_hat.argmax(1).float()

        return feas, target_hat, label_hat, ent

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        top_M = self.top_M
        # if top_M == -1:
        #     indices = torch.LongTensor(list(range(len(ent_s))))

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s))))
        for i in range(self._num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat == i][indices2][:top_M])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]

        return self.supports, self.labels

    def reset_warmup(self):
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

    @property
    def name(self):
        return "t3a"
