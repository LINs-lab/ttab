# -*- coding: utf-8 -*-
import copy
from typing import Any, List, Optional, Type

import torch
import torch.nn as nn

import ttab.loads.datasets.loaders as loaders
import ttab.loads.define_dataset as define_dataset
import ttab.model_adaptation.utils as adaptation_utils
from ttab.api import Batch, Quality
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.group_metrics import GroupLossComputer
from ttab.model_selection.metrics import Metrics
from ttab.utils.logging import Logger
from ttab.utils.timer import Timer


class BaseAdaptation(object):
    def __init__(self, meta_conf, base_model: nn.Module):
        self._meta_conf = copy.deepcopy(meta_conf)
        self._base_model = base_model

        self._prior_safety_check()
        self.initialize(seed=self._meta_conf.seed)
        self._post_safety_check()

    def _prior_safety_check(self):

        assert (
            self._meta_conf.debug is not None
        ), "The state of debug should be specified"
        assert self._meta_conf.n_train_steps > 0, "adaptation steps requires >= 1."

    def _post_safety_check(self):
        is_training = self._model.training
        assert is_training, "adaptation needs train mode: call model.train()."

        param_grads = [p.requires_grad for p in (self._model.parameters())]
        has_any_params = any(param_grads)
        has_all_params = all(param_grads)
        assert has_any_params, "adaptation needs some trainable params."
        assert not has_all_params, "not all params are trainable."

    def _initialize_optimizer(self, params) -> torch.optim.Optimizer:
        """Set up optimizer for adaptation process."""
        optimizer = adaptation_utils.define_optimizer(
            self._meta_conf, params, lr=self._meta_conf.lr
        )
        return optimizer

    def initialize(self, seed: int):
        """Initialize the algorithm."""
        self._oracle_model_selection = False
        if self._meta_conf.model_selection_method == "oracle_model_selection":
            self._oracle_model_selection = True
            self.oracle_adaptation_steps = []  # record optimal adaptation steps for each batch
            assert (
                self._meta_conf.data_wise == "batch_wise"
                and self._meta_conf.batch_size > 1
            ), "batch-size should be larger than 1 when working with oracle model selection."

        self._model = self._initialize_model(model=copy.deepcopy(self._base_model))
        self._base_model = copy.deepcopy(self._model)  # update base model
        params, names = self._initialize_trainable_parameters()
        self._optimizer = self._initialize_optimizer(params)
        self._base_optimizer = copy.deepcopy(self._optimizer)
        self._auxiliary_data_cls = define_dataset.ConstructAuxiliaryDataset(
            config=self._meta_conf
        )
        # compute fisher regularizer
        self.fishers = None
        self.ewc_optimizer = torch.optim.SGD(params, 0.001)

        # base model state.
        self.model_state_dict = copy.deepcopy(self._model).state_dict()

    def set_nbstep_ratio(self, nbstep_ratio: float) -> None:
        """
        Use a ratio to help control the number of adaptation steps.
        Only used in experiments for Figure 2c in the paper,
        where we proportionally reduce the number of adaptation steps to fight against batch dependency.
        """
        assert 0 < nbstep_ratio < 1, "invalid ratio number"
        self.nbstep_ratio = nbstep_ratio

    def _get_adaptation_steps(self, index: int = None) -> int:
        """control the number of adaptation steps."""
        if hasattr(self, "nbstep_ratio"):
            # for batch dependency experiment
            if index is None:
                raise ValueError("the given index is empty")
            return max(int(self.optimal_adaptation_steps[index] * self.nbstep_ratio), 1)
        else:
            return self._meta_conf.n_train_steps

    def set_optimal_adaptation_steps(self, optimal_adaptation_steps: list) -> None:
        """Set up the optimal adaptation steps which can be retrieved for each batch."""
        if not isinstance(optimal_adaptation_steps, list):
            raise ValueError("optimal_adaptation_steps should be a list")
        self.optimal_adaptation_steps = optimal_adaptation_steps

    def get_optimal_adaptation_steps(self) -> list:
        if not hasattr(self, "optimal_adaptation_steps"):
            raise ValueError("optimal_adaptation_steps is not set")
        return self.optimal_adaptation_steps

    def construct_group_computer(self, dataset):
        """This function is used to build a new metric tracker for group-wise datasets like waterbirds."""
        criterion = nn.CrossEntropyLoss(reduction="none")
        self.tta_loss_computer = GroupLossComputer(
            criterion=criterion,
            dataset=dataset,
            device=self._meta_conf.device,
        )
        return self.tta_loss_computer

    def reset(self):
        """recover model and optimizer to their initial states."""
        self._model.load_state_dict(self.model_state_dict)
        self._optimizer.load_state_dict(self._base_optimizer.state_dict())

    def get_auxiliary_loader(self, scenario) -> loaders.BaseLoader:
        """setup for auxiliary datasets used in test-time adaptation."""
        return self._auxiliary_data_cls.construct_auxiliary_loader(
            scenario, data_augment=True
        )

    def get_src_data(self, scenario, data_size):
        return self._auxiliary_data_cls.construct_src_dataset(
            scenario, data_size, data_augment=True
        )

    def stochastic_restore(self):
        """Stochastically restorre model parameters to resist catastrophic forgetting."""
        for nm, m in self._model.named_modules():
            for npp, p in m.named_parameters():
                if npp in ["weight", "bias"] and p.requires_grad:
                    mask = (
                        (torch.rand(p.shape) < self._meta_conf.restore_prob)
                        .float()
                        .to(self._meta_conf.device)
                    )
                    with torch.no_grad():
                        p.data = self.model_state_dict[f"{nm}.{npp}"] * mask + p * (
                            1.0 - mask
                        )

    def compute_fishers(self, scenario, data_size):
        """Get fisher regularizer"""
        print("Computing fisher matrices===>")
        self.fisher_dataset, self.fisher_loader = self.get_src_data(scenario, data_size)

        fishers = {}
        train_loss_fn = nn.CrossEntropyLoss().to(self._meta_conf.device)
        for step, _, batch in self.fisher_loader.iterator(
            batch_size=self._meta_conf.batch_size,
            shuffle=True,
            repeat=False,
            ref_num_data=None,
            num_workers=self._meta_conf.num_workers
            if hasattr(self._meta_conf, "num_workers")
            else 2,
            pin_memory=True,
            drop_last=False,
        ):
            outputs = self._model(
                batch._x
            )  # don't need to worry about BN error becasue we use in-distribution data here.
            _, targets = outputs.max(1)
            loss = train_loss_fn(outputs, targets)
            loss.backward()
            for name, param in self._model.named_parameters():
                if param.grad is not None:
                    if step > 1:
                        fisher = (
                            param.grad.data.clone().detach() ** 2 + fishers[name][0]
                        )
                    else:
                        fisher = param.grad.data.clone().detach() ** 2
                    if step == len(self.fisher_dataset):
                        fisher = fisher / step
                    fishers.update({name: [fisher, param.data.clone().detach()]})
            self.ewc_optimizer.zero_grad()
        print("compute fisher matrices finished")
        del self.ewc_optimizer
        self.fishers = fishers

    def adapt_and_eval(
        self,
        metrics: Metrics,
        model_selection_method: Type[BaseSelection],
        current_batch: Batch,
        previous_batches: List[Batch],
        auxiliary_loaders: Optional[List[loaders.BaseLoader]],
        logger: Logger,
        timer: Timer,
    ):
        pass

    def copy_model(self):
        """copy and return the whole model."""
        if self._meta_conf.model_adaptation_method == "ttt":
            return copy.deepcopy(self._base_model)
        return copy.deepcopy(self._model)
