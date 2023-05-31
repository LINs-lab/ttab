# -*- coding: utf-8 -*-
import copy
import functools
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import ttab.model_adaptation.utils as adaptation_utils
from ttab.api import Batch
from ttab.model_adaptation.base_adaptation import BaseAdaptation
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import Metrics
from ttab.utils.auxiliary import fork_rng_with_seed
from ttab.utils.logging import Logger
from ttab.utils.timer import Timer


class EATA(BaseAdaptation):
    """
    Efficient Test-Time Model Adaptation without Forgetting,
    https://arxiv.org/abs/2204.02610,
    https://github.com/mr-eggplant/EATA
    """

    def __init__(self, meta_conf, model: nn.Module):
        super(EATA, self).__init__(meta_conf, model)
        self.current_model_probs = None
        # filter samples.
        self.num_samples_update_1 = (
            0  # number of samples after First filtering, exclude unreliable samples
        )
        self.num_samples_update_2 = 0  # number of samples after Second filtering, exclude both unreliable and redundant samples

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
            if isinstance(
                module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)
            ):  # only bn is used in the paper.
                self._adapt_module_names.append(name_module)
                for name_parameter, parameter in module.named_parameters():
                    if name_parameter in ["weight", "bias"]:
                        adapt_params.append(parameter)
                        adapt_param_names.append(f"{name_module}.{name_parameter}")

        assert (
            len(self._adapt_module_names) > 0
        ), "EATA needs some adaptable model parameters."
        return adapt_params, adapt_param_names

    @staticmethod
    def update_model_probs(current_model_probs, new_probs):
        if current_model_probs is None:
            if new_probs.size(0) == 0:
                return None
            else:
                with torch.no_grad():
                    return new_probs.mean(0)
        else:
            if new_probs.size(0) == 0:
                with torch.no_grad():
                    return current_model_probs
            else:
                with torch.no_grad():
                    return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)

    def reset_model_probs(self, probs):
        self.current_model_probs = probs

    def one_adapt_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        current_model_probs,
        timer: Timer,
        random_seed: int = None,
    ):
        """adapt the model in one step."""
        with timer("forward"):
            with fork_rng_with_seed(random_seed):
                y_hat = model(batch._x)  # adapt
            entropys = adaptation_utils.softmax_entropy(y_hat)
            # filter unreliable samples
            filter_ids_1 = torch.where(entropys < self._meta_conf.eata_margin_e0)
            ids1 = filter_ids_1
            ids2 = torch.where(ids1[0] > -0.1)
            entropys = entropys[filter_ids_1]
            # filter redundant samples
            if current_model_probs is not None:
                cosine_similarities = F.cosine_similarity(
                    current_model_probs.unsqueeze(dim=0),
                    y_hat[filter_ids_1].softmax(1),
                    dim=1,
                )
                filter_ids_2 = torch.where(
                    torch.abs(cosine_similarities) < self._meta_conf.eata_margin_d0
                )
                entropys = entropys[filter_ids_2]
                ids2 = filter_ids_2
                updated_probs = self.update_model_probs(
                    current_model_probs, y_hat[filter_ids_1][filter_ids_2].softmax(1)
                )
            else:
                updated_probs = self.update_model_probs(
                    current_model_probs, y_hat[filter_ids_1].softmax(1)
                )
            coeff = 1 / (
                torch.exp(entropys.clone().detach() - self._meta_conf.eata_margin_e0)
            )
            # implementation version 1, compute loss, all samples backward (some unselected are masked)
            entropys = entropys.mul(coeff)  # reweight entropy losses for diff. samples
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
            if batch._x[ids1][ids2].size(0) != 0:
                loss.backward()
                grads = dict(
                    (name, param.grad.clone().detach())
                    for name, param in model.named_parameters()
                    if param.grad is not None
                )
                optimizer.step()
            optimizer.zero_grad()
        return (
            {
                "optimizer": copy.deepcopy(optimizer).state_dict(),
                "loss": loss.item(),
                "grads": grads if batch._x[ids1][ids2].size(0) != 0 else None,
                "yhat": y_hat,
            },
            entropys.size(0),
            filter_ids_1[0].size(0),
            updated_probs,
        )

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
            (
                adaptation_result,
                num_counts_2,
                num_counts_1,
                updated_probs,
            ) = self.one_adapt_step(
                model,
                optimizer,
                batch,
                self.current_model_probs,
                timer,
                random_seed=random_seed,
            )
            self.num_samples_update_2 += num_counts_2
            self.num_samples_update_1 += num_counts_1
            self.reset_model_probs(updated_probs)

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
                model=self._model,
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
        return "eata"
