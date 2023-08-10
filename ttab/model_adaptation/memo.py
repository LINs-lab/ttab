# -*- coding: utf-8 -*-

import copy
import functools
from typing import List

import torch
import torch.nn as nn

import ttab.model_adaptation.utils as adaptation_utils
from ttab.api import Batch
from ttab.loads.datasets.cifar.data_aug_cifar import aug_cifar
from ttab.loads.datasets.imagenet.data_aug_imagenet import aug_imagenet
from ttab.loads.datasets.mnist.data_aug_mnist import aug_mnist
from ttab.loads.datasets.yearbook.data_aug_yearbook import aug_yearbook
from ttab.model_adaptation.base_adaptation import BaseAdaptation
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import Metrics
from ttab.utils.auxiliary import fork_rng_with_seed
from ttab.utils.logging import Logger
from ttab.utils.timer import Timer


class MEMO(BaseAdaptation):
    """
    MEMO: Test Time Robustness via Adaptation and Augmentation,
    https://arxiv.org/abs/2110.09506,
    https://github.com/zhangmarvin/memo
    """

    def __init__(self, meta_conf, model: nn.Module):
        super().__init__(meta_conf, model)
        self.transform_helper = self._get_transform_helper()

    def _prior_safety_check(self):

        assert (
            self._meta_conf.aug_size > 0
        ), "The number of augmentation operation requires >= 1."
        assert (
            self._meta_conf.debug is not None
        ), "The state of debug should be specified"
        assert self._meta_conf.n_train_steps > 0, "adaptation steps requires >= 1."

    def _initialize_model(self, model: nn.Module):
        """Configure model for adaptation."""
        # memo works in mode eval(). All params are adapted.
        model.eval()
        model.requires_grad_(True)

        if hasattr(self._meta_conf, "bn_prior_strength") and (
            self._meta_conf.bn_prior_strength > 0
        ):
            print("modifying BN forward pass")
            nn.BatchNorm2d.prior = float(self._meta_conf.bn_prior_strength) / float(
                self._meta_conf.bn_prior_strength + 1
            )
            nn.BatchNorm2d.forward = adaptation_utils.modified_bn_forward

        return model.to(self._meta_conf.device)

    def _initialize_trainable_parameters(self):
        """
        select target parameters for adaptation methods.
        All of params are adapted here.
        """
        self._adapt_module_names = []
        adapt_params = []
        adapt_param_names = []

        for name_module, module in self._model.named_children():
            self._adapt_module_names.append(name_module)
            for name_param, param in module.named_parameters():
                adapt_params.append(param)
                adapt_param_names.append(f"{name_module}.{name_param}")

        assert (
            len(self._adapt_module_names) > 0
        ), "MEMO needs some adaptable model parameters."
        return adapt_params, adapt_param_names

    def _post_safety_check(self):

        # all modules are adapted in memo
        param_grads = [p.requires_grad for p in (self._model.parameters())]
        has_any_params = any(param_grads)
        has_all_params = all(param_grads)
        assert has_any_params, "adaptation needs some trainable params."
        assert has_all_params, "all params are trainable."

    def _get_transform_helper(self):
        """get particular augmentation method for different datasets"""
        if self._meta_conf.base_data_name in ["cifar10", "cifar100"]:
            return aug_cifar
        elif self._meta_conf.base_data_name in [
            "imagenet",
            "officehome",
            "waterbirds",
            "pacs",
        ]:
            return aug_imagenet
        elif self._meta_conf.base_data_name in ["coloredmnist"]:
            return aug_mnist
        elif self._meta_conf.base_data_name in ["yearbook"]:
            return aug_yearbook

    def one_adapt_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        timer: Timer,
        random_seed: int = None,
    ):
        """adapt the model in one step."""
        model.eval()
        # trick from bn_adapt
        if hasattr(self._meta_conf, "bn_prior_strength") and (
            self._meta_conf.bn_prior_strength > 0
        ):
            nn.BatchNorm2d.prior = float(self._meta_conf.bn_prior_strength) / float(
                self._meta_conf.bn_prior_strength + 1
            )
        else:
            nn.BatchNorm2d.prior = 1

        with timer("forward"):
            # Modification for fair comparison: apply gradient accumulation when batch size > 1.
            NUM_ACCUMULATION_STEPS = len(batch._x)
            for i in range(NUM_ACCUMULATION_STEPS):
                inputs = [
                    self.transform_helper(batch._x[i], data_name=self._meta_conf.base_data_name)
                    for _ in range(self._meta_conf.aug_size)
                ]
                inputs = torch.stack(inputs).to(self._meta_conf.device)

                with fork_rng_with_seed(random_seed):
                    y_hat = model(inputs)
                loss, _ = adaptation_utils.marginal_entropy(y_hat)
                loss = loss / NUM_ACCUMULATION_STEPS

                # apply fisher regularization when enabled
                if self.fishers is not None:
                    ewc_loss = 0
                    for name, param in model.named_parameters():
                        if name in self.fishers:
                            ewc_loss += (
                                self._meta_conf.fisher_alpha
                                * (
                                    self.fishers[name][0]
                                    * (param - self.fishers[name][1]) ** 2
                                ).sum()
                            )
                    loss += ewc_loss

                loss.backward()

        with timer("update"):
            grads = dict(
                (name, param.grad.clone().detach())
                for name, param in model.named_parameters()
                if param.grad is not None
            )
            # update parameters using accumulated gradients.
            optimizer.step()
            optimizer.zero_grad()

        # evaluate after adaptation
        with torch.no_grad():
            y_hat = model(batch._x)  # already in eval mode
        nn.BatchNorm2d.prior = 1

        return {
            "optimizer": copy.deepcopy(optimizer).state_dict(),
            "loss": loss.item(),
            "grads": grads,
            "yhat": y_hat,
        }

    def run_multiple_steps(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        model_selection_method: BaseSelection,
        nbsteps: int,
        timer: Timer,
        random_seed: int = None,
    ):
        for step in range(1, nbsteps + 1):
            adaptation_result = self.one_adapt_step(
                model,
                optimizer,
                batch,
                timer,
                random_seed=random_seed,
            )

            model_selection_method.save_state(
                {
                    "model": copy.deepcopy(model).state_dict(),
                    "step": step,
                    "lr": self._meta_conf.lr,
                    **adaptation_result,
                },
                current_batch=batch,
            )

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
        if episodic:
            log("\treset model to initial state during the test time.")
            self.reset()

        log(f"\tinitialize selection method={model_selection_method.name}.")
        model_selection_method.initialize()

        # evaluate the per batch pre-adapted performance. Different with no adaptation.
        if self._meta_conf.record_preadapted_perf:
            with timer("evaluate_preadapted_performance"):
                self._model.eval()
                if hasattr(self._meta_conf, "bn_prior_strength") and (
                    self._meta_conf.bn_prior_strength > 0
                ):
                    nn.BatchNorm2d.prior = float(
                        self._meta_conf.bn_prior_strength
                    ) / float(self._meta_conf.bn_prior_strength + 1)
                else:
                    nn.BatchNorm2d.prior = 1

                with torch.no_grad():
                    yhat = self._model(current_batch._x)
                metrics.eval_auxiliary_metric(
                    current_batch._y, yhat, metric_name="preadapted_accuracy_top1"
                )
                nn.BatchNorm2d.prior = 1

        # adaptation.
        with timer("test_time_adaptation"):
            nbsteps = self._get_adaptation_steps(index=len(previous_batches))
            log(f"\tadapt the model for {nbsteps} steps with lr={self._meta_conf.lr}.")
            self.run_multiple_steps(
                model=self._model,
                optimizer=self._optimizer,
                batch=current_batch,
                model_selection_method=model_selection_method,
                nbsteps=nbsteps,
                timer=timer,
                random_seed=self._meta_conf.seed,
            )

        # select the optimal model, and return the corresponding prediction.
        with timer("select_optimal_model"):
            # build inputs and targets for model selection.
            optimal_state = model_selection_method.select_state()
            log(
                f"\tselect the optimal model ({optimal_state['step']}-th step and lr={optimal_state['lr']}) for the current mini-batch.",
            )

            self._model.load_state_dict(optimal_state["model"])
            model_selection_method.clean_up()

            if self._oracle_model_selection:
                # oracle model selection needs to save steps
                self.oracle_adaptation_steps.append(optimal_state["step"])
                # update optimizer.
                self._optimizer.load_state_dict(optimal_state["optimizer"])

        with timer("evaluate_adaptation_result"):
            metrics.eval(current_batch._y, optimal_state["yhat"])
            if self._meta_conf.base_data_name in ["waterbirds"]:
                self.tta_loss_computer.loss(
                    optimal_state["yhat"],
                    current_batch._y,
                    current_batch._g,
                    is_training=False,
                )

        # stochastic restore part of model parameters if enabled.
        if self._meta_conf.stochastic_restore_model:
            self.stochastic_restore()

    @property
    def name(self):
        return "memo"
