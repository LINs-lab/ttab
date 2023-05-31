# -*- coding: utf-8 -*-
import copy
import functools
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ttab.loads.datasets.loaders as loaders
import ttab.model_adaptation.utils as adaptation_utils
from scipy.spatial.distance import cdist
from ttab.api import Batch
from ttab.model_adaptation.base_adaptation import BaseAdaptation
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import Metrics
from ttab.utils.auxiliary import fork_rng_with_seed
from ttab.utils.logging import Logger
from ttab.utils.timer import Timer


class SHOT(BaseAdaptation):
    """
    Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation,
    https://arxiv.org/abs/2002.08546,
    https://github.com/tim-learn/SHOT
    """

    def __init__(self, meta_conf, model: nn.Module):
        super(SHOT, self).__init__(meta_conf, model)

    def _prior_safety_check(self):

        assert (
            self._meta_conf.auxiliary_batch_size > 0
        ), "The batch_size of auxiliary dataloaders requires >= 1"
        assert (
            self._meta_conf.debug is not None
        ), "The state of debug should be specified"
        assert self._meta_conf.n_train_steps > 0, "adaptation steps requires >= 1."

    def _initialize_model(self, model: nn.Module):
        """
        Configure model for adaptation.

        configure target modules for adaptation method updates: enable grad + ...
        """
        # TODO: make this more general
        # Problem description: the naming and structure of classifier layers may change.
        # Be careful: the following list may not cover all cases when using model outside of this library.
        self._freezed_module_names = ["fc", "classifier", "head"]

        model.train()
        for index, (name_module, module) in enumerate(model.named_children()):
            if name_module in self._freezed_module_names:
                module.requires_grad_(False)

        return model.to(self._meta_conf.device)

    def _initialize_trainable_parameters(self):
        """
        SHOT only updates params in the feature extractor.
        Params in classifier layer is freezed.
        """
        self._adapt_module_names = []
        classifier_param_names = []
        adapt_params = []
        adapt_param_names = []

        for name_module, module in self._model.named_children():
            if name_module in self._freezed_module_names:
                for name_param, param in module.named_parameters():
                    classifier_param_names.append(f"{name_module}.{name_param}")
            else:
                self._adapt_module_names.append(name_module)
                for name_param, param in module.named_parameters():
                    adapt_params.append(param)
                    adapt_param_names.append(f"{name_module}.{name_param}")

        assert len(adapt_param_names) > 0, "SHOT needs some adaptable model parameters."
        assert (
            len(classifier_param_names) > 0
        ), "Cannot find the classifier. Please check the model structure."

        return adapt_params, adapt_param_names

    def _initialize_optimizer(self, params) -> torch.optim.Optimizer:
        """
        Set up optimizer for adaptation process.
        """
        optimizer = adaptation_utils.define_optimizer(
            self._meta_conf, params, lr=self._meta_conf.lr
        )

        for param_group in optimizer.param_groups:
            param_group["lr0"] = param_group["lr"]  # for lr scheduler
        return optimizer

    def make_extractor_train(self, model: nn.Module):
        """set the extractor to training mode."""
        for index, (name_module, module) in enumerate(model.named_children()):
            if name_module in self._adapt_module_names:
                module.train()
        return model

    def make_extractor_eval(self, model):
        """set the extractor to eval mode."""
        for index, (name_module, module) in enumerate(model.named_children()):
            if name_module in self._adapt_module_names:
                module.eval()
        return model

    def one_adapt_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        timer: Timer,
        random_seed: int = None,
    ):
        """adapt the model in one step."""
        # some check
        if not hasattr(self._meta_conf, "cls_par"):
            self._meta_conf.cls_par = 0.3
        if not hasattr(self._meta_conf, "ent_par"):
            self._meta_conf.ent_par = 1.0
        assert (
            self._meta_conf.cls_par > 0 and self._meta_conf.ent_par > 0
        ), "coefficients in the objective function should be positive."

        # optimize.
        with timer("forward"):
            with fork_rng_with_seed(random_seed):
                y_hat = model(batch._x)

            # pseudo label.
            if self._meta_conf.offline_pre_adapt:
                # offline version.
                y_label = batch._y
                classifier_loss = self._meta_conf.cls_par * nn.CrossEntropyLoss()(
                    y_hat, y_label
                )
            else:
                # online version.
                py, y_prime = F.softmax(y_hat, dim=-1).max(1)
                reliable_labels = py > self._meta_conf.threshold_shot
                classifier_loss = F.cross_entropy(
                    y_hat[reliable_labels], y_prime[reliable_labels]
                )

            # entropy loss
            entropy_loss = adaptation_utils.softmax_entropy(y_hat).mean(0)
            # divergence loss
            softmax_out = F.softmax(y_hat, dim=-1)
            msoftmax = softmax_out.mean(dim=0)
            div_loss = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))

            loss = (
                self._meta_conf.cls_par * classifier_loss
                + self._meta_conf.ent_par * (entropy_loss + div_loss)
            )

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
            "yhat": y_hat,
        }

    def eval_per_epoch(
        self,
        auxiliary_loader: loaders.BaseLoader,
        timer: Timer,
        logger: Logger,
    ):
        """evaluate adapted model's performance on the auxiliary dataset."""
        log = functools.partial(logger.log, display=self._meta_conf.debug)

        # some init for evaluation.
        self.make_extractor_eval(self._model)
        metrics = Metrics(self._meta_conf)

        with timer("model_evaluation"):
            for _, _, batch in auxiliary_loader.iterator(
                batch_size=self._meta_conf.auxiliary_batch_size,
                shuffle=False,
                repeat=False,
                ref_num_data=None,
                num_workers=self._meta_conf.num_workers
                if hasattr(self._meta_conf, "num_workers")
                else 2,
                pin_memory=True,
                drop_last=False,
            ):
                with torch.no_grad():
                    y_hat = self._model(batch._x)
                metrics.eval(batch._y, y_hat)
            stats = metrics.tracker()
            log(f"stats of evaluating model={stats}.")

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

    # TODO: refactor offline_adapt
    def offline_adapt(
        self,
        model_selection_method: BaseSelection,
        auxiliary_loader: loaders.BaseLoader,
        timer: Timer,
        logger: Logger,
        random_seed: int = None,
    ):
        """implement offline adaptation given the complete dataset."""
        log = functools.partial(logger.log, display=self._meta_conf.debug)

        # some init for model selection method.
        if model_selection_method.name != "last_iterate":
            raise ValueError(
                f"offline adaptation only supports last_iterate, but got {model_selection_method.name}."
            )
        log(f"\tinitialize selection method={model_selection_method.name}.")
        model_selection_method.initialize()

        log(f"\tOffline adaptation begins.", display=True)
        for i in range(self._meta_conf.offline_nepoch):
            # pseudo labels are generated first and replace the labels of auxiliary loader at the beginning of each epoch.
            self.make_extractor_eval(self._model)
            ps_label = self.obtain_ps_label(
                auxiliary_loader,
                batch_size=self._meta_conf.auxiliary_batch_size,
                random_seed=random_seed + i,
            )
            ps_label = torch.from_numpy(ps_label).to(self._meta_conf.device)
            self.make_extractor_train(self._model)

            num_batches = int(
                len(auxiliary_loader.dataset) / self._meta_conf.auxiliary_batch_size
            )
            drop_last = False
            if not drop_last:
                num_batches += 1

            # Use the same generator as the obtain_ps_label function to control the order of batches after shuffling.
            G = torch.Generator()
            G.manual_seed(random_seed + i)
            for batch_indx, _, batch in auxiliary_loader.iterator(
                batch_size=self._meta_conf.auxiliary_batch_size,
                shuffle=True,
                repeat=False,
                ref_num_data=None,
                num_workers=self._meta_conf.num_workers
                if hasattr(self._meta_conf, "num_workers")
                else 2,
                generator=G,
                pin_memory=True,
                drop_last=drop_last,
            ):
                batch._y = ps_label[
                    self._meta_conf.auxiliary_batch_size
                    * (batch_indx - 1) : self._meta_conf.auxiliary_batch_size
                    * batch_indx
                ]
                with timer("offline_adapt_model"):
                    # grid-search the best adaptation result per test iteration and save required information.
                    nbsteps = self._get_adaptation_steps()
                    log(
                        f"\tadapt the model for {nbsteps} steps with lr={self._meta_conf.lr}."
                    )
                    adaptation_utils.lr_scheduler(
                        self._optimizer,
                        iter_ratio=(batch_indx + num_batches * i)
                        / (self._meta_conf.offline_nepoch * num_batches),
                    )

                    self.run_multiple_steps(
                        model=self._model,
                        optimizer=self._optimizer,
                        batch=batch,
                        model_selection_method=model_selection_method,
                        nbsteps=nbsteps,
                        timer=timer,
                        random_seed=random_seed,
                    )
                # select the optimal checkpoint, and return the corresponding prediction.
                with timer("select_optimal_model"):
                    optimal_state = model_selection_method.select_state()
                    log(
                        f"\tselect the optimal model ({optimal_state['step']}-th step and lr={optimal_state['lr']}) for the current mini-batch.",
                    )
                    self._model.load_state_dict(optimal_state["model"])
                    model_selection_method.clean_up()

            # evaluate model performance at the end of each epoch.
            log(f"\tbegin evaluating the model for the {i}-th epoch.")
            self.eval_per_epoch(
                auxiliary_loader=auxiliary_loader,
                timer=timer,
                logger=logger,
            )

    def online_adapt(
        self,
        model_selection_method: BaseSelection,
        current_batch: Batch,
        previous_batches: List[Batch],
        logger: Logger,
        timer: Timer,
        random_seed: int = None,
    ):
        """
        Implement online SHOT adaptation given the current batch of data.

        https://github.com/matsuolab/T3A/blob/master/domainbed/adapt_algorithms.py#L376
        """
        log = functools.partial(logger.log, display=self._meta_conf.debug)
        log(f"\tinitialize selection method={model_selection_method.name}.")
        model_selection_method.initialize()

        with timer("test_time_adapt"):
            nbsteps = self._get_adaptation_steps(index=len(previous_batches))
            log(f"\tadapt the model for {nbsteps} steps with lr={self._meta_conf.lr}.")
            self.run_multiple_steps(
                model=self._model,
                optimizer=self._optimizer,
                batch=current_batch,
                model_selection_method=model_selection_method,
                nbsteps=nbsteps,
                timer=timer,
                random_seed=random_seed,
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
                # online shot needs to save steps
                self.oracle_adaptation_steps.append(optimal_state["step"])
                # update optimizer.
                self._optimizer.load_state_dict(optimal_state["optimizer"])

        return optimal_state["yhat"]

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
        if not self._meta_conf.offline_pre_adapt and episodic:
            log("\treset model to initial state during the test time.")
            self.reset()

        if self._meta_conf.offline_pre_adapt:
            # offline mode.
            self.make_extractor_eval(self._model)
            with torch.no_grad():
                y_hat = self._model(current_batch._x)
        else:
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

            # online mode.
            y_hat = self.online_adapt(
                model_selection_method=model_selection_method,
                current_batch=current_batch,
                previous_batches=previous_batches,
                logger=logger,
                timer=timer,
                random_seed=self._meta_conf.seed,
            )

        with timer("evaluate_adaptation_result"):
            metrics.eval(current_batch._y, y_hat)
            if self._meta_conf.base_data_name in ["waterbirds"]:
                self.tta_loss_computer.loss(
                    y_hat, current_batch._y, current_batch._g, is_training=False
                )

        ## stochastic restore part of model parameters if enabled.
        if self._meta_conf.stochastic_restore_model:
            self.stochastic_restore()

    # TODO: refactor obtain_ps_label.
    def obtain_ps_label(
        self,
        dataset_loader: loaders.BaseLoader,
        batch_size: int,  # param used to define the batch property of a dataset iterator.
        random_seed: int = None,
    ):
        """
        Apply a self-supervised pseudo-labeling method for each unlabeled data to better supervise
        the target encoding training.

        https://github.com/tim-learn/SHOT/blob/master/digit/uda_digit.py
        """
        if not hasattr(self._meta_conf, "threshold_shot"):
            self._meta_conf.threshold_shot = 0

        start_test = True
        with torch.no_grad():
            G = torch.Generator()
            G.manual_seed(random_seed)
            for step, _, batch in dataset_loader.iterator(
                batch_size=batch_size,
                shuffle=True,
                repeat=False,
                ref_num_data=None,
                num_workers=self._meta_conf.num_workers
                if hasattr(self._meta_conf, "num_workers")
                else 2,
                generator=G,
                pin_memory=True,
                drop_last=False,
            ):
                inputs = batch._x
                labels = batch._y
                children_modules = []
                for name, module in self._model.named_children():
                    if name in self._adapt_module_names:
                        children_modules.append(module)

                feas = inputs
                for i in range(len(children_modules)):
                    feas = children_modules[i](feas)
                feas = feas.view(feas.size(0), -1)

                outputs = self._model(inputs)
                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        all_output = nn.Softmax(dim=1)(all_output)
        _, predict = torch.max(all_output, 1)

        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()

        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])

        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count > self._meta_conf.threshold_shot)
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset], "cosine")
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

        for _ in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_fea, initc[labelset], "cosine")
            pred_label = dd.argmin(axis=1)
            pred_label = labelset[pred_label]

        return pred_label.astype("int")

    @property
    def name(self):
        return "shot"
