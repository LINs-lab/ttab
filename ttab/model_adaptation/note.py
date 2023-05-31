# -*- coding: utf-8 -*-
import copy
import functools
import random
import warnings
from typing import List

import torch
import torch.nn as nn
import ttab.model_adaptation.utils as adaptation_utils
from ttab.api import Batch
from ttab.loads.define_model import load_pretrained_model
from ttab.model_adaptation.base_adaptation import BaseAdaptation
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import Metrics
from ttab.utils.auxiliary import fork_rng_with_seed
from ttab.utils.logging import Logger
from ttab.utils.timer import Timer


class NOTE(BaseAdaptation):
    """
    NOTE: Robust Continual Test-time Adaptation Against Temporal Correlation,
    https://arxiv.org/abs/2208.05117,
    https://github.com/TaesikGong/NOTE
    """

    def __init__(self, meta_conf, model: nn.Module):
        super(NOTE, self).__init__(meta_conf, model)
        self.fifo = FIFO(capacity=self._meta_conf.update_every_x)
        self.memory = self.define_memory()
        self.entropy_loss = adaptation_utils.HLoss(
            temp_factor=self._meta_conf.temperature
        )

    def _prior_safety_check(self):

        assert self._meta_conf.use_learned_stats, "NOTE uses batch-free evaluation."
        assert (
            self._meta_conf.debug is not None
        ), "The state of debug should be specified"
        assert self._meta_conf.n_train_steps > 0, "adaptation steps requires >= 1."

    def convert_iabn(self, module: nn.Module, **kwargs):
        """
        Recursively convert all BatchNorm to InstanceAwareBatchNorm.
        """
        module_output = module
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            IABN = (
                adaptation_utils.InstanceAwareBatchNorm2d
                if isinstance(module, nn.BatchNorm2d)
                else adaptation_utils.InstanceAwareBatchNorm1d
            )
            module_output = IABN(
                num_channels=module.num_features,
                k=self._meta_conf.iabn_k,
                eps=module.eps,
                momentum=module.momentum,
                threshold=self._meta_conf.threshold_note,
                affine=module.affine,
            )

            module_output._bn = copy.deepcopy(module)

        for name, child in module.named_children():
            module_output.add_module(name, self.convert_iabn(child, **kwargs))
        del module
        return module_output

    def _initialize_model(self, model: nn.Module):
        """Configure model for adaptation."""
        # IABN
        if hasattr(self._meta_conf, "iabn") and self._meta_conf.iabn:
            # check BN layers
            bn_flag = False
            for name_module, module in model.named_modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    bn_flag = True
            if not bn_flag:
                warnings.warn(
                    "IABN needs bn layers, while there is no bn in the base model."
                )
            self.convert_iabn(model)
            load_pretrained_model(self._meta_conf, model)

        # disable grad, to (re-)enable only what specified adaptation method updates
        model.requires_grad_(False)
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if self._meta_conf.use_learned_stats:
                    module.track_running_stats = True
                    module.momentum = self._meta_conf.bn_momentum
                else:
                    # with below, this module always uses the test batch statistics (no momentum)
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            if hasattr(self._meta_conf, "iabn") and self._meta_conf.iabn:
                if isinstance(
                    module,
                    (
                        adaptation_utils.InstanceAwareBatchNorm2d,
                        adaptation_utils.InstanceAwareBatchNorm1d,
                    ),
                ):
                    for param in module.parameters():
                        param.requires_grad = True

        return model.to(self._meta_conf.device)

    def _initialize_trainable_parameters(self):
        """select target params for adaptation methods."""
        self._adapt_module_names = []
        adapt_params = []
        adapt_param_names = []

        for name_module, module in self._model.named_children():
            self._adapt_module_names.append(name_module)
            for name_param, param in module.named_parameters():
                adapt_params.append(param)
                adapt_param_names.append(f"{name_module}.{name_param}")

        return adapt_params, adapt_param_names

    def define_memory(self):
        """
        Define memory type.
        A replay memory manages a buffer to replay previous data for future learning to prevent catastrophic forgetting.
        """
        if self._meta_conf.memory_type == "FIFO":
            mem = FIFO(capacity=self._meta_conf.memory_size)
        elif self._meta_conf.memory_type == "Reservoir":
            mem = Reservoir(capacity=self._meta_conf.memory_size)
        elif self._meta_conf.memory_type == "PBRS":
            mem = PBRS(
                capacity=self._meta_conf.memory_size,
                num_class=self._meta_conf.statistics["n_classes"],
            )

        return mem

    def update_memory(self, current_batch: Batch):
        for i in range(len(current_batch)):
            current_sample = current_batch[i]
            self.fifo.add_instance(current_sample)
            with torch.no_grad():
                self._model.eval()
                if self._meta_conf.memory_type in ["FIFO", "Reservoir"]:
                    self.memory.add_instance(current_sample)
                elif self._meta_conf.memory_type in ["PBRS"]:
                    f, c = current_sample[0].to(self._meta_conf.device), current_sample[
                        1
                    ].to(self._meta_conf.device)

                    logit = self._model(f.unsqueeze(0))
                    pseudo_cls = logit.max(1, keepdim=False)[1][0]
                    self.memory.add_instance([f, pseudo_cls, c, 0])

    def one_adapt_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        memory_sampled_feats: torch.Tensor,
        timer: Timer,
        random_seed: int = None,
    ):
        """adapt the model in one step."""
        with timer("forward"):
            with fork_rng_with_seed(random_seed):
                y_hat = model(memory_sampled_feats)
            loss = self.entropy_loss(y_hat)

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
            grads = dict(
                (name, param.grad.clone().detach())
                for name, param in model.named_parameters()
                if param.grad is not None
            )
            optimizer.step()
            optimizer.zero_grad()
        return {
            "optimizer": copy.deepcopy(optimizer).state_dict(),
            "loss": loss.item(),
            "grads": grads,
        }

    def run_multiple_steps(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        memory_sampled_feats: torch.Tensor,
        current_batch: Batch,
        model_selection_method: BaseSelection,
        nbsteps: int,
        timer: Timer,
        random_seed: int = None,
    ):
        for step in range(1, nbsteps + 1):
            adaptation_result = self.one_adapt_step(
                model,
                optimizer,
                memory_sampled_feats,
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
                current_batch=current_batch,
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

        # use new samples to update memory
        self.update_memory(current_batch)
        # adaptation.
        with timer("test_time_adaptation"):
            if self._meta_conf.model_selection_method == "last_iterate":
                # no model selection (i.e., use the last checkpoint)
                with torch.no_grad():
                    self._model.eval()
                    yhat = self._model(current_batch._x)

            self._model.train()
            memory_sampled_feats, _ = self.memory.get_memory()  # get pseudo iid batch
            memory_sampled_feats = torch.stack(memory_sampled_feats)
            memory_sampled_feats = memory_sampled_feats.to(self._meta_conf.device)

            nbsteps = self._get_adaptation_steps(index=len(previous_batches))
            log(f"\tadapt the model for {nbsteps} steps with lr={self._meta_conf.lr}.")
            self.run_multiple_steps(
                model=self._model,
                optimizer=self._optimizer,
                memory_sampled_feats=memory_sampled_feats,
                current_batch=current_batch,
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
                yhat = optimal_state["yhat"]
                # oracle model selection needs to save steps
                self.oracle_adaptation_steps.append(optimal_state["step"])
                # update optimizer.
                self._optimizer.load_state_dict(optimal_state["optimizer"])

        with timer("evaluate_adaptation_result"):
            metrics.eval(current_batch._y, yhat)
            if self._meta_conf.base_data_name in ["waterbirds"]:
                self.tta_loss_computer.loss(
                    yhat, current_batch._y, current_batch._g, is_training=False
                )

        # stochastic restore part of model parameters if enabled.
        if self._meta_conf.stochastic_restore_model:
            self.stochastic_restore()

    @property
    def name(self):
        return "note"


class FIFO:
    def __init__(self, capacity):
        self.data = [[], []]
        self.capacity = capacity
        pass

    def get_memory(self):
        return self.data

    def get_occupancy(self):
        return len(self.data[0])

    def add_instance(self, instance):
        # waterbirds: [x, y, g], otherwise [x, y]
        assert len(instance) in [2, 3]

        if self.get_occupancy() >= self.capacity:
            self.remove_instance()

        for i, dim in enumerate(self.data):
            dim.append(instance[i])

    def remove_instance(self):
        for dim in self.data:
            dim.pop(0)
        pass


class Reservoir:  # Time uniform
    def __init__(self, capacity):
        super(Reservoir, self).__init__(capacity)
        self.data = [[], []]
        self.capacity = capacity
        self.counter = 0

    def get_memory(self):
        return self.data

    def get_occupancy(self):
        return len(self.data[0])

    def add_instance(self, instance):
        assert len(instance) == 2
        is_add = True
        self.counter += 1

        if self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance()

        if is_add:
            for i, dim in enumerate(self.data):
                dim.append(instance[i])

    def remove_instance(self):

        m = self.get_occupancy()
        n = self.counter
        u = random.uniform(0, 1)
        if u <= m / n:
            tgt_idx = random.randrange(0, m)  # target index to remove
            for dim in self.data:
                dim.pop(tgt_idx)
        else:
            return False
        return True


class PBRS:
    def __init__(self, capacity, num_class):
        self.data = [
            [[], []] for _ in range(num_class)
        ]  # feat, pseudo_cls, domain, cls, loss
        self.counter = [0] * num_class
        self.marker = [""] * num_class
        self.num_class = num_class
        self.capacity = capacity
        pass

    def print_class_dist(self):

        print(self.get_occupancy_per_class())

    def print_real_class_dist(self):

        occupancy_per_class = [0] * self.num_class
        for i, data_per_cls in enumerate(self.data):
            for cls in data_per_cls[2]:
                occupancy_per_class[cls] += 1
        print(occupancy_per_class)

    def get_memory(self):

        data = self.data

        tmp_data = [[], []]
        for data_per_cls in data:
            feats, cls = data_per_cls
            tmp_data[0].extend(feats)
            tmp_data[1].extend(cls)

        return tmp_data

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls[0])
        return occupancy

    def get_occupancy_per_class(self):
        occupancy_per_class = [0] * self.num_class
        for i, data_per_cls in enumerate(self.data):
            occupancy_per_class[i] = len(data_per_cls[0])
        return occupancy_per_class

    def update_loss(self, loss_list):
        for data_per_cls in self.data:
            feats, cls, dls, _, losses = data_per_cls
            for i in range(len(losses)):
                losses[i] = loss_list.pop(0)

    def add_instance(self, instance):
        assert len(instance) == 4
        cls = instance[1]
        self.counter[cls] += 1
        is_add = True

        if self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance(cls)

        if is_add:
            for i, dim in enumerate(self.data[cls]):
                dim.append(instance[i])

    def get_largest_indices(self):

        occupancy_per_class = self.get_occupancy_per_class()
        max_value = max(occupancy_per_class)
        largest_indices = []
        for i, oc in enumerate(occupancy_per_class):
            if oc == max_value:
                largest_indices.append(i)
        return largest_indices

    def remove_instance(self, cls):
        largest_indices = self.get_largest_indices()
        if (
            cls not in largest_indices
        ):  #  instance is stored in the place of another instance that belongs to the largest class
            largest = random.choice(largest_indices)  # select only one largest class
            tgt_idx = random.randrange(
                0, len(self.data[largest][0])
            )  # target index to remove
            for dim in self.data[largest]:
                dim.pop(tgt_idx)
        else:  # replaces a randomly selected stored instance of the same class
            m_c = self.get_occupancy_per_class()[cls]
            n_c = self.counter[cls]
            u = random.uniform(0, 1)
            if u <= m_c / n_c:
                tgt_idx = random.randrange(
                    0, len(self.data[cls][0])
                )  # target index to remove
                for dim in self.data[cls]:
                    dim.pop(tgt_idx)
            else:
                return False
        return True
