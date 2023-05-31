# -*- coding: utf-8 -*-
import copy
import functools
from typing import List

import numpy as np
import torch
import torch.nn as nn
import ttab.model_adaptation.utils as adaptation_utils
from ttab.api import Batch
from ttab.model_adaptation.base_adaptation import BaseAdaptation
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import Metrics
from ttab.utils.auxiliary import fork_rng_with_seed
from ttab.utils.logging import Logger
from ttab.utils.timer import Timer


class SAR(BaseAdaptation):
    """
    Towards Stable Test-Time Adaptation in Dynamic Wild World,
    https://arxiv.org/abs/2302.12400,
    https://github.com/mr-eggplant/SAR
    """

    def __init__(self, meta_conf, model: nn.Module):
        super(SAR, self).__init__(meta_conf, model)
        self.ema = None  # to record the moving average of model output entropy, as model recovery criteria

    def _initialize_model(self, model: nn.Module):
        """Configure model for adaptation."""
        model.train()
        # disable grad, to (re-)enable only what specified adaptation method updates
        model.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # bn module always uses batch statistics, in both training and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)
        return model.to(self._meta_conf.device)

    def _initialize_trainable_parameters(self):
        """
        Collect the affine scale + shift parameters from norm layers.

        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        self._adapt_module_names = []
        adapt_params = []
        adapt_param_names = []

        for name_module, module in self._model.named_modules():
            # skip top layers for adaptation: layer4 for ResNets
            if "layer4" in name_module:
                continue
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                self._adapt_module_names.append(name_module)
                for name_parameter, parameter in module.named_parameters():
                    if name_parameter in ["weight", "bias"]:
                        adapt_params.append(parameter)
                        adapt_param_names.append(f"{name_module}.{name_parameter}")

        assert (
            len(self._adapt_module_names) > 0
        ), "SAR needs some adaptable model parameters."
        return adapt_params, adapt_param_names

    def _initialize_optimizer(self, params) -> torch.optim.Optimizer:
        """Set up optimizer for adaptation process."""
        # particular setup of optimizer for oracle model selection.
        base_optimizer = torch.optim.SGD
        optimizer = adaptation_utils.SAM(
            params,
            base_optimizer=base_optimizer,
            lr=self._meta_conf.lr,
            momentum=self._meta_conf.momentum
            if hasattr(self._meta_conf, "momentum")
            else 0.9,
        )
        return optimizer

    def reset(self):
        """recover model and optimizer to their initial states."""
        self._model.load_state_dict(self.model_state_dict)
        self._optimizer.load_state_dict(self._base_optimizer.state_dict())
        self.ema = None

    @staticmethod
    def update_ema(ema: float, new_data: float):
        if ema is None:
            return new_data
        else:
            with torch.no_grad():
                return 0.9 * ema + (1 - 0.9) * new_data

    def one_adapt_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        ema: float,
        timer: Timer,
        random_seed: int = None,
    ):
        """adapt the model in one step."""
        optimizer.zero_grad()
        with timer("forward"):
            with fork_rng_with_seed(random_seed):
                y_hat = model(batch._x)
            entropys = adaptation_utils.softmax_entropy(y_hat)
            filter_ids_1 = torch.where(entropys < self._meta_conf.sar_margin_e0)
            entropys = entropys[filter_ids_1]
            loss = entropys.mean(0)

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

        with timer("backward"):
            loss.backward()
            optimizer.first_step(
                zero_grad=True
            )  # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
            entropys2 = adaptation_utils.softmax_entropy(model(batch._x))
            entropys2 = entropys2[filter_ids_1]  # second time forward
            loss_second_value = entropys2.clone().detach().mean(0)
            filter_ids_2 = torch.where(
                entropys2 < self._meta_conf.sar_margin_e0
            )  # here filtering reliable samples again, since model weights have been changed to \Theta+\hat{\epsilon(\Theta)}
            loss_second = entropys2[filter_ids_2].mean(0)
            if not np.isnan(loss_second.item()):
                ema = self.update_ema(
                    ema, loss_second.item()
                )  # record moving average loss values for model recovery

            # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
            loss_second.backward()
            optimizer.second_step(zero_grad=True)

            grads = dict(
                (name, param.grad.clone().detach())
                for name, param in model.named_parameters()
                if param.grad is not None
            )

        # perform model recovery
        reset_flag = False
        if ema is not None:
            if ema < self._meta_conf.reset_constant_em:
                print("ema < 0.2, now reset the model")
                reset_flag = True

        return (
            {
                "optimizer": copy.deepcopy(optimizer).state_dict(),
                "loss": loss.item(),
                "grads": grads,
                "yhat": y_hat,
            },
            reset_flag,
            ema,
        )

    def run_multiple_steps(
        self,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        model_selection_method: BaseSelection,
        nbsteps: int,
        timer: Timer,
        random_seed: int = None,
    ):
        for step in range(1, nbsteps + 1):
            adaptation_result, reset_flag, ema = self.one_adapt_step(
                self._model,
                optimizer,
                batch,
                self.ema,
                timer,
                random_seed=random_seed,
            )

            model_selection_method.save_state(
                {
                    "model": copy.deepcopy(self._model).state_dict(),
                    "step": step,
                    "lr": self._meta_conf.lr,
                    **adaptation_result,
                },
                current_batch=batch,
            )

            # reset model when it begins to collapse.
            if reset_flag:
                self.reset()
            self.ema = ema

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
                with torch.no_grad():
                    yhat = self._model(current_batch._x)
                self._model.train()
                metrics.eval_auxiliary_metric(
                    current_batch._y, yhat, metric_name="preadapted_accuracy_top1"
                )

        # adaptation.
        with timer("test_time_adaptation"):
            nbsteps = self._get_adaptation_steps(index=len(previous_batches))
            log(f"\tadapt the model for {nbsteps} steps with lr={self._meta_conf.lr}.")
            self.run_multiple_steps(
                optimizer=self._optimizer,
                batch=current_batch,
                model_selection_method=model_selection_method,
                nbsteps=nbsteps,
                timer=timer,
                random_seed=self._meta_conf.seed,
            )

        # select the optimal checkpoint, and return the corresponding prediction.
        with timer("select_optimal_checkpoint"):
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
        return "sar"
