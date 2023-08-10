# -*- coding: utf-8 -*-
import copy
import functools
from typing import List

import torch
import torch.nn as nn

import ttab.loads.define_dataset as define_dataset
import ttab.model_adaptation.utils as adaptation_utils
from ttab.api import Batch
from ttab.loads.datasets.cifar.data_aug_cifar import tr_transforms_cifar
from ttab.loads.datasets.imagenet.data_aug_imagenet import tr_transforms_imagenet
from ttab.loads.datasets.mnist.data_aug_mnist import tr_transforms_mnist
from ttab.loads.datasets.yearbook.data_aug_yearbook import tr_transforms_yearbook
from ttab.model_adaptation.base_adaptation import BaseAdaptation
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import Metrics
from ttab.utils.auxiliary import fork_rng_with_seed
from ttab.utils.logging import Logger
from ttab.utils.timer import Timer


class TTT(BaseAdaptation):
    """
    Test-Time Training with Self-Supervision for Generalization under Distribution Shifts.
    https://arxiv.org/abs/1909.13231,
    https://github.com/yueatsprograms/ttt_cifar_release
    """

    def __init__(self, meta_conf, model: nn.Module):
        super(TTT, self).__init__(meta_conf, model)

    def _prior_safety_check(self):

        assert hasattr(
            self._meta_conf, "entry_of_shared_layers"
        ), "The shared argument must be specified"
        assert (
            self._meta_conf.aug_size > 0
        ), "the size of augmented batches must be specified >= 1"
        assert (
            self._meta_conf.threshold_ttt > 0
        ), "The threshold_ttt argument must be specified > 0"
        assert (
            self._meta_conf.debug is not None
        ), "The state of debug should be specified"
        assert self._meta_conf.n_train_steps > 0, "adaptation steps requires >= 1."

    def _initialize_model(self, model: nn.Module):
        """Configure model for adaptation."""
        main_model = model.main_model
        ext = model.ext
        head = model.head
        ssh = model.ssh

        return (
            main_model.to(self._meta_conf.device),
            ext.to(self._meta_conf.device),
            head.to(self._meta_conf.device),
            ssh.to(self._meta_conf.device),
        )

    def _initialize_trainable_parameters(self):
        """
        Setup parameters for training and adaptation.
        """
        adapt_params = []
        adapt_param_names = []
        self._model.requires_grad_(False)

        for layer_name, layer_module in self._ext.layers.items():
            layer_module.requires_grad_(True)
            for named_param, param in layer_module.named_parameters():
                adapt_params.append(param)
                adapt_param_names.append(f"ext.{layer_name}.{named_param}")

        for named_param, param in self._head.named_parameters():
            adapt_params.append(param)
            adapt_param_names.append(f"head.{named_param}")

        # head of the main branch should in eval mode, the other modules work in train mode.
        self._model.eval()
        self._ssh.make_train()
        return adapt_params, adapt_param_names

    def _post_safety_check(self):
        is_training = self._ssh.training
        assert (
            is_training
        ), "The feature extractor and self-supervised head need training mode: call model.train()."

        param_grads = [p.requires_grad for p in self._ssh.parameters()]
        has_any_params = any(param_grads)
        has_all_params = all(param_grads)
        assert has_any_params, "adaptation needs some trainable params in ssh."
        assert not has_all_params, "not all params are trainable in ssh."

    def initialize(self, seed: int):
        """Initialize the algorithm."""
        if self._meta_conf.model_selection_method == "oracle_model_selection":
            self._oracle_model_selection = True
            self.oracle_adaptation_steps = []
            assert (
                self._meta_conf.data_wise == "batch_wise"
                and self._meta_conf.batch_size > 1
            ), "batch-size should be larger than 1 when working with oracle model selection."
        else:
            self._oracle_model_selection = False

        self._model, self._ext, self._head, self._ssh = self._initialize_model(
            model=copy.deepcopy(self._base_model)
        )
        params, names = self._initialize_trainable_parameters()
        self._optimizer = self._initialize_optimizer(params)
        self._base_optimizer = copy.deepcopy(self._optimizer)
        self._auxiliary_data_cls = define_dataset.ConstructAuxiliaryDataset(
            config=self._meta_conf
        )
        self.transform_helper = self._get_transform_helper()

        self.model_state_dict = copy.deepcopy(self._model).state_dict()
        self.ssl_head_state_dict = copy.deepcopy(self._head).state_dict()

        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        # fisher regularizer
        self.fishers = None
        self.ewc_optimizer = torch.optim.SGD(params, 0.001)

    def reset(self):
        """recover model and optimizer to their initial states."""
        self._model.load_state_dict(self.model_state_dict)
        self._head.load_state_dict(self.ssl_head_state_dict)
        self._optimizer.load_state_dict(self._base_optimizer.state_dict())

    def _get_transform_helper(self):
        """get particular augmentation method for different datasets"""
        if self._meta_conf.base_data_name in ["cifar10", "cifar100"]:
            return tr_transforms_cifar
        elif self._meta_conf.base_data_name in [
            "imagenet",
            "officehome",
            "waterbirds",
            "pacs",
        ]:
            return tr_transforms_imagenet
        elif self._meta_conf.base_data_name in ["coloredmnist"]:
            return tr_transforms_mnist
        elif self._meta_conf.base_data_name in ["yearbook"]:
            return tr_transforms_yearbook

    def one_adapt_step(
        self,
        model: torch.nn.Module,
        ssh: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        timer: Timer,
        random_seed: int = None,
    ):
        """adapt the model in one step."""
        with timer("forward"):
            # Modification for fair comparison: apply gradient accumulation when batch size > 1.
            inputs, targets = batch._x, batch._y
            NUM_ACCUMULATION_STEPS = len(batch._x)
            for image_idx in range(NUM_ACCUMULATION_STEPS):
                aug_inputs = [
                    self.transform_helper(inputs[image_idx], data_name=self._meta_conf.base_data_name)
                    for _ in range(self._meta_conf.aug_size)
                ]
                aug_inputs = torch.stack(aug_inputs).to(self._meta_conf.device)

                inputs_ssh, targets_ssh = adaptation_utils.rotate_batch(
                    aug_inputs,
                    self._meta_conf.rotation_type,
                    self._meta_conf.device,
                    self.generator,
                )

                with fork_rng_with_seed(random_seed):
                    outputs_ssh = ssh(inputs_ssh)
                loss = (
                    nn.CrossEntropyLoss()(outputs_ssh, targets_ssh)
                    / NUM_ACCUMULATION_STEPS
                )

                # apply fisher regularization when enabled
                if self.fishers is not None:
                    ewc_loss = 0
                    for name, param in ssh.named_parameters():
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
            optimizer.step()
            optimizer.zero_grad()

        # evaluate after adaptation
        model.eval()
        with torch.no_grad():
            y_hat = model(batch._x)

        return {
            "optimizer": copy.deepcopy(optimizer).state_dict(),
            "loss": loss.item(),
            "grads": grads,
            "yhat": y_hat,
        }

    def run_multiple_steps(
        self,
        model: nn.Module,
        ssh: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        model_selection_method: BaseSelection,
        nbsteps: int,
        timer: Timer,
        random_seed: int = None,
    ):
        ssh.make_eval()
        with torch.no_grad():
            y_hat_init = ssh(batch._x)
        confidence = nn.functional.softmax(y_hat_init, dim=1)[:, 0]
        need_adapt = confidence < (
            self._meta_conf.threshold_ttt + 0.001
        )  # use a sample to adapt only if its confidence is below threshold
        if need_adapt.sum() > 0:
            inputs_to_adapt = batch._x[need_adapt]
            targets_to_adapt = batch._y[need_adapt]
            batch_to_adapt = Batch(inputs_to_adapt, targets_to_adapt).to(
                self._meta_conf.device
            )
            for step in range(1, nbsteps + 1):
                ssh.make_train()
                adaptation_result = self.one_adapt_step(
                    model,
                    ssh,
                    optimizer,
                    batch_to_adapt,
                    timer,
                    random_seed=self._meta_conf.seed,
                )
                model_selection_method.save_state(
                    {
                        "main_model": copy.deepcopy(model).state_dict(),
                        "ssh": copy.deepcopy(ssh).state_dict(),
                        "step": step,
                        "lr": self._meta_conf.lr,
                        **adaptation_result,
                    },
                    current_batch=batch,
                )
        else:
            model.eval()
            with torch.no_grad():
                y_hat = model(batch._x)
            model_selection_method.save_state(
                {
                    "main_model": copy.deepcopy(model).state_dict(),
                    "ssh": copy.deepcopy(ssh).state_dict(),
                    "optimizer": copy.deepcopy(optimizer).state_dict(),
                    "step": 0,
                    "lr": 0,
                    "yhat": y_hat,
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
                metrics.eval_auxiliary_metric(
                    current_batch._y, yhat, metric_name="preadapted_accuracy_top1"
                )

        with timer("test_time_training"):
            nbsteps = self._get_adaptation_steps(index=len(previous_batches))
            log(f"\tadapt the model for {nbsteps} steps with lr={self._meta_conf.lr}.")
            self.run_multiple_steps(
                model=self._model,
                ssh=self._ssh,
                optimizer=self._optimizer,
                batch=current_batch,
                model_selection_method=model_selection_method,
                nbsteps=nbsteps,
                timer=timer,
                random_seed=self._meta_conf.seed,
            )
        # select the optimal checkpoint, and return the corresponding prediction.
        with timer("select_optimal_model"):
            optimal_state = model_selection_method.select_state()
            log(
                f"\tselect the optimal model ({optimal_state['step']}-th step and lr={optimal_state['lr']}) for the current mini-batch.",
            )

            self._model.load_state_dict(optimal_state["main_model"])
            self._ssh.load_state_dict(optimal_state["ssh"])
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
        return "ttt"
