# -*- coding: utf-8 -*-
import copy
import functools
import statistics
from typing import List, Optional, Type

import torch
import torch.nn as nn
import ttab.loads.datasets.loaders as loaders
import ttab.loads.define_dataset as define_dataset
import ttab.model_adaptation.utils as adaptation_utils
from ttab.api import Batch
from ttab.model_adaptation.base_adaptation import BaseAdaptation
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import Metrics
from ttab.utils.logging import Logger
from ttab.utils.timer import Timer


class TTTPlusPlus(BaseAdaptation):
    """
    TTT++ introduces a test-time feature alignment strategy utilizing offline feature summarization and online moment matching, which regularizes adaptation without revisiting training data.
    It further scales this strategy in the online setting through batch-queue decoupling to enable robust moment estimates.
    """

    def __init__(self, meta_conf, model):
        super(TTTPlusPlus, self).__init__(meta_conf, model)

    def _prior_safety_check(self):

        assert hasattr(
            self._meta_conf, "entry_of_shared_layers"
        ), "The shared argument must be specified"

        # Used in TTT. Keep it for training process
        # if not hasattr(self._meta_conf, "auxiliary_batch_size"):
        #     if isinstance(self._base_model, ResNetCifar):
        #         self._meta_conf.auxiliary_batch_size = 128
        #     elif isinstance(self._base_model, ResNetImagenet):
        #         self._meta_conf.auxiliary_batch_size = 32

        assert (
            self._meta_conf.batch_size_align > 0
        ), "batch_size used for alignment must be specified >= 1"
        assert self._meta_conf.queue_size > 0, "queue_size must be specified >= 1"
        # first few epochs to update bn stat.
        assert self._meta_conf.bnepoch > 0, "bnepoch must be specified >= 1"
        assert self._meta_conf.offline_nepoch >= 0, "delayepoch must be specified >= 0"
        # In first few epochs after bnepoch, we dont do both ssl and align (only ssl actually).
        assert self._meta_conf.delayepoch >= 0, "delayepoch must be specified >= 0"
        # to decide when to Test-Time Training.
        assert self._meta_conf.stopepoch > 0, "stopepoch must be specified >= 0"
        # scale of align loss on ext
        assert self._meta_conf.scale_ext >= 0, "scale_ext must be specified >= 0"
        # scale of align loss on ssh
        assert self._meta_conf.scale_ssh >= 0, "scale_ssh must be specified >= 0"
        assert (
            self._meta_conf.align_ext is not None
        ), "The state of align_ext argument must be specified."
        assert (
            self._meta_conf.align_ssh is not None
        ), "The state of align_ssh argument must be specified."
        assert (
            self._meta_conf.fix_ssh is not None
        ), "The state of fix_ssh argument must be specified."
        assert self._meta_conf.method in [
            "ssl",
            "align",
            "both",
        ], "The method argument must be specified."
        assert self._meta_conf.divergence in [
            "all",
            "coral",
            "mmd",
        ], "The divergence argument must be specified."
        assert (
            self._meta_conf.debug is not None
        ), "The state of debug should be specified"
        assert self._meta_conf.n_train_steps > 0, "adaptation steps requires >= 1."
        assert (
            self._meta_conf.model_name == "resnet50"
        ), "This benchmark only supports resnet50 for now because of the absence of ssl pretraining module."

    def _initialize_model(self, model):
        """Configure model for use with adaptation method."""

        main_model = model.main_model
        ext = model.ext
        head = model.head
        ssh = model.ssh
        classifier = model.classifier

        return (
            main_model.to(self._meta_conf.device),
            ext.to(self._meta_conf.device),
            head.to(self._meta_conf.device),
            ssh.to(self._meta_conf.device),
            classifier.to(self._meta_conf.device),
        )

    def _initialize_trainable_parameters(self):
        """
        Setup parameters for training and adaptation.
        Different from previous adaptation methods in this benchmark which clarify parameters to adapt and
        parameters to freeze first, then set up the train or eval status. In TTT variants, we need to construct
        the model first and then set up parameters' states.
        """
        self._adapt_module_names = []
        self._adapt_modules = []
        self._model.requires_grad_(False)
        self._ssh.requires_grad_(False)

        for named_module, module in self._ext.named_modules():
            module.requires_grad_(True)
            self._adapt_module_names.append(named_module)
            self._adapt_modules.append(module)

        if self._meta_conf.fix_ssh:
            self._model.eval()
            self._head.eval()
            self._ext.train()
        else:
            for named_module, module in self._head.named_modules():
                module.requires_grad_(True)
                self._adapt_module_names.append(named_module)
                self._adapt_modules.append(module)
            self._model.train()
            self._head.train()

    def _initialize_optimizer(self, ssh) -> torch.optim.Optimizer:
        """Set up optimizer for adaptation process."""

        params = list(ssh.encoder.parameters())
        if not self._meta_conf.fix_ssh:
            params = params + list(ssh.head.parameters())
        optimizer = adaptation_utils.define_optimizer(self._meta_conf, params)
        return optimizer

    def _post_safety_check(self):

        if self._meta_conf.fix_ssh:
            is_training = self._ext.training
            assert (
                is_training
            ), "TTT++ needs train mode in the shared feature extractor: call model.train()."
        else:
            is_training = (self._ext.training) and (self._head.training)
            assert (
                is_training
            ), "TTT++ needs train mode in the shared feature extractor and ssl head: call model.train()."

        param_grads = [p.requires_grad for p in self._ssh.parameters()]
        has_any_params = any(param_grads)
        has_all_params = all(param_grads)
        assert has_any_params, "adaptation needs some trainable params in ssh."
        if self._meta_conf.fix_ssh:
            assert (
                not has_all_params
            ), "not all params are trainable in ssh when fix_ssh=True."
        else:
            assert has_all_params, "all params are trainable in ssh when fix_ssh=False."

    def initialize(self, seed: int):
        """Initialize the benchmark."""

        (
            self._model,
            self._ext,
            self._head,
            self._ssh,
            self._classifier,
        ) = self._initialize_model(model=copy.deepcopy(self._base_model))
        self._source_statistics = self._base_model.source_statistics
        self._initialize_trainable_parameters()
        self._optimizer = self._initialize_optimizer(self._ssh)
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer,
            "min",
            factor=0.5,
            patience=10,
            cooldown=10,
            threshold=0.0001,
            threshold_mode="rel",
            min_lr=0.0001,
            verbose=True,
        )
        self._criterion = adaptation_utils.SupConLoss(
            temperature=(
                self._meta_conf.temperature
                if hasattr(self._meta_conf, "temperature")
                else 0.5
            )
        ).to(self._meta_conf.device)

        self._auxiliary_data_cls = define_dataset.ConstructAuxiliaryDataset(config=self._meta_conf)

    def reset(self):
        self._model.load_state_dict(self._base_model.main_model.state_dict())
        self._head.load_state_dict(self._base_model.head.state_dict())

    def load_source_statistics(self, location: str):
        """load source domain statistics of the certain component"""
        if location == "ext":
            return (
                self._source_statistics["cov_src_ext"],
                self._source_statistics["coral_src_ext"],
                self._source_statistics["mu_src_ext"],
                self._source_statistics["mmd_src_ext"],
            )

        else:
            return (
                self._source_statistics["cov_src_ssh"],
                self._source_statistics["coral_src_ssh"],
                self._source_statistics["mu_src_ssh"],
                self._source_statistics["mmd_src_ssh"],
            )

    def offline(self, trloader, ext, scale):
        """
        This function is used to build an initial dynamic queue based on training set.
        """
        ext.eval()

        mu_src = None
        cov_src = None

        coral_stack = []
        mmd_stack = []
        feat_stack = []

        with torch.no_grad():
            for batch_idx, epoch_fractional, batch in trloader.iterator(
                batch_size=self._meta_conf.batch_size_align,
                shuffle=True,
                repeat=False,
                ref_num_data=None,
                num_workers=self._meta_conf.num_workers
                if hasattr(self._meta_conf, "num_workers")
                else 2,
                pin_memory=True,
                drop_last=True,
            ):

                feat = ext(batch._x)
                cov = adaptation_utils.covariance(feat)
                mu = feat.mean(dim=0)

                if cov_src is None:
                    cov_src = cov
                    mu_src = mu
                else:
                    loss_coral = adaptation_utils.coral(cov_src, cov)
                    loss_mmd = adaptation_utils.linear_mmd(mu_src, mu)
                    coral_stack.append(loss_coral.item())
                    mmd_stack.append(loss_mmd.item())
                    feat_stack.append(feat)

        print(
            "Source loss_mean: mu = {:.4f}, std = {:.4f}".format(
                scale, scale / statistics.mean(mmd_stack) * statistics.stdev(mmd_stack)
            )
        )
        print(
            "Source loss_coral: mu = {:.4f}, std = {:.4f}".format(
                scale,
                scale / statistics.mean(coral_stack) * statistics.stdev(coral_stack),
            )
        )

        feat_all = torch.cat(feat_stack)
        feat_cov = adaptation_utils.covariance(feat_all)
        feat_mean = feat_all.mean(dim=0)
        return (
            feat_cov,
            statistics.mean(coral_stack),
            feat_mean,
            statistics.mean(mmd_stack),
        )

    def initialize_queue(self):
        """
        Initialize the dynamic queue for better statistics estimation.
        """
        if self._meta_conf.queue_size > self._meta_conf.batch_size_align:
            assert self._meta_conf.queue_size % self._meta_conf.batch_size_align == 0

        MMD_SCALE_FACTOR = 0.5

        if self._meta_conf.align_ext:
            self._meta_conf.scale_align = self._meta_conf.scale_ext
            # load source data statistics from checkpoints.
            (
                cov_src_ext,
                coral_src_ext,
                mu_src_ext,
                mmd_src_ext,
            ) = self.load_source_statistics(location="ext")

            scale_coral_ext = self._meta_conf.scale_ext / coral_src_ext
            scale_mmd_ext = self._meta_conf.scale_ext / mmd_src_ext * MMD_SCALE_FACTOR
            self.ext_stat = {
                "cov_src": cov_src_ext,
                "mu_src": mu_src_ext,
                "scale_coral": scale_coral_ext,
                "scale_mmd": scale_mmd_ext,
            }

            # construct queue
            if self._meta_conf.queue_size > self._meta_conf.batch_size_align:
                self.queue_ext = adaptation_utils.FeatureQueue(
                    dim=mu_src_ext.shape[0],
                    length=self._meta_conf.queue_size
                    - self._meta_conf.batch_size_align,
                )

        if self._meta_conf.align_ssh:
            self._meta_conf.scale_align = self._meta_conf.scale_ssh
            (
                cov_src_ssh,
                coral_src_ssh,
                mu_src_ssh,
                mmd_src_ssh,
            ) = self.load_source_statistics(location="ssh")

            scale_coral_ssh = self._meta_conf.scale_ssh / coral_src_ssh
            scale_mmd_ssh = self._meta_conf.scale_ssh / mmd_src_ssh * MMD_SCALE_FACTOR
            self.ssh_stat = {
                "cov_src": cov_src_ssh,
                "mu_src": mu_src_ssh,
                "scale_coral": scale_coral_ssh,
                "scale_mmd": scale_mmd_ssh,
            }

            # construct queue
            if self._meta_conf.queue_size > self._meta_conf.batch_size_align:
                self.queue_ssh = adaptation_utils.FeatureQueue(
                    dim=mu_src_ssh.shape[0],
                    length=self._meta_conf.queue_size
                    - self._meta_conf.batch_size_align,
                )

    def train_source(
        self, auxiliary_loader: Optional[loaders.BaseLoader], logger: Logger
    ):
        """
        In the pre-training stage, the model is trained on both tasks on the same data drawn from source dataset.
        """
        # TODO: add SSL.
        assert (
            self._model.training
        ), "Multi-task training needs self._model in train mode."
        assert self._ssh.training, "Multi-task training needs self._ssh in train mode."

        logger.log("Multi-task pretraining begins...")
        for i in range(self._meta_conf.epoch):
            logger.log(
                f"Begin multi-task training for {i}-th epoch.",
                display=self._meta_conf.debug,
            )
            for batch_indx, epoch_fractional, batch in auxiliary_loader.iterator(
                batch_size=self._meta_conf.auxiliary_batch_size,
                shuffle=True,
                repeat=False,
                ref_num_data=None,
                num_workers=self._meta_conf.num_workers
                if hasattr(self._meta_conf, "num_workers")
                else 2,
                pin_memory=True,
                drop_last=False,
            ):
                self._optimizer.zero_grad()
                inputs_cls, targets_cls = batch._x, batch._y
                targets_cls_hat = self._model(inputs_cls)
                loss = nn.CrossEntropyLoss()(targets_cls_hat, targets_cls)

                if self._meta_conf.entry_of_shared_layers is not None:
                    inputs_ssh, targets_ssh = adaptation_utils.rotate_batch(
                        batch._x, self._meta_conf.device
                    )
                    targets_ssh_hat = self._ssh(inputs_ssh)
                    loss_ssh = nn.CrossEntropyLoss()(targets_ssh_hat, targets_ssh)
                    loss += loss_ssh

                loss.backward()
                self._optimizer.step()
            self._scheduler.step()
        logger.log("Multi-task pretraining ends...")

    # Have no relevance with test-time adaptation. No need to use.
    def test(self, model, teloader, sslabel=None):
        criterion = nn.CrossEntropyLoss(reduction="none").to(self._meta_conf.device)
        model.eval()
        correct = []
        losses = []
        for batch_indx, _, batch in teloader.iterator(
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
            inputs, labels = batch._x, batch._y
            if sslabel is not None:
                inputs, labels = adaptation_utils.rotate_batch(inputs, sslabel)
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                losses.append(loss.cpu())
                _, predicted = outputs.max(1)
                correct.append(predicted.eq(labels).cpu())
        correct = torch.cat(correct).numpy()
        losses = torch.cat(losses).numpy()
        model.train()
        return 1 - correct.mean(), correct, losses

    def get_auxiliary_loader(self, scenario) -> loaders.BaseLoader:
        return self._auxiliary_data_cls.construct_auxiliary_loader(
            scenario, data_augment=True
        )

    def offline_adapt(
        self,
        model_selection_method: Type[BaseSelection],
        auxiliary_loader: Optional[loaders.BaseLoader],
        timer: Timer,
        logger: Logger,
        random_seed=None,
    ):
        """
        This function is used to implement the test-time adaptation procedure of TTT++
        """
        logger.log(f"\tinitialize selection method={model_selection_method.name}.")
        model_selection_method.initialize()

        # Initialize dynamic queue.
        self.initialize_queue()

        # is_both_activated = False
        all_err_cls = []
        all_err_ssh = []
        for i in range(self._meta_conf.offline_nepoch):
            previous_adapted_batches = []
            for batch_indx, _, batch in auxiliary_loader.iterator(
                batch_size=self._meta_conf.batch_size_align,
                shuffle=True,  # False.
                repeat=False,
                ref_num_data=None,
                num_workers=self._meta_conf.num_workers
                if hasattr(self._meta_conf, "num_workers")
                else 2,
                pin_memory=True,
                drop_last=True,
            ):

                logger.log(
                    f"\tadapt the model for {self._meta_conf.n_train_steps} steps.",
                    display=self._meta_conf.debug,
                )
                for _ in range(self._meta_conf.n_train_steps):
                    self._optimizer.zero_grad()
                    self.loss_and_gradient(self._ssh, timer, batch)
                    if i > self._meta_conf.bnepoch:
                        self._optimizer.step()

                    model_selection_method.save_state(
                        {"state_dict": self._model.state_dict()},
                        currrent_batch=batch,
                    )

                    # select the optimal model, and return the corresponding optimal prediction.
                    with timer("select_optimal_model"):
                        logger.log(
                            "\tselect the optimal model for the current mini-batch.",
                            display=self._meta_conf.debug,
                        )
                        optimal_state = model_selection_method.select_state()
                        self._model.load_state_dict(optimal_state["state_dict"])
                        model_selection_method.clean_up()

                    previous_adapted_batches.append(batch)

            err_cls = self.test(self._model, auxiliary_loader)[0]
            all_err_cls.append(err_cls)
            print(
                ("Epoch %d/%d:" % (i, self._meta_conf.offline_nepoch)).ljust(24)
                + "%.2f\t\t" % (err_cls * 100)
            )

            # termination
            if i > (self._meta_conf.stopepoch + 1) and all_err_cls[
                -self._meta_conf.stopepoch
            ] < min(all_err_cls[-self._meta_conf.stopepoch + 1 :]):
                print(
                    "Termination: {:.2f}".format(
                        all_err_cls[-self._meta_conf.stopepoch] * 100
                    )
                )
                break

            self._scheduler.step(err_cls)

    def loss_and_gradient(self, ssh: torch.nn.Module, timer: Timer, batch: Batch):
        """
        This function is used to calculate loss and returns loss, gradient, and sometimes predictions.
        """
        if self._meta_conf.align_ext:
            with timer("forward"):
                loss = 0
                # feat_ext = ssh.ext(batch._x)
                feat_ext = ssh.encoder(batch._x)

                # queue
                if self._meta_conf.queue_size > self._meta_conf.batch_size_align:
                    feat_queue = self.queue_ext.get()
                    self.queue_ext.update(feat_ext)
                    if feat_queue is not None:
                        feat_ext = torch.cat(
                            [feat_ext, feat_queue.to(self._meta_conf.device)]
                        )

                # coral
                if self._meta_conf.divergence in ["coral", "all"]:
                    cov_ext = adaptation_utils.covariance(feat_ext)
                    loss += (
                        adaptation_utils.coral(self.ext_stat["cov_src"], cov_ext)
                        * self.ext_stat["scale_coral"]
                    )

                # mmd
                if self._meta_conf.divergence in ["mmd", "all"]:
                    mu_ext = feat_ext.mean(dim=0)
                    loss += (
                        adaptation_utils.linear_mmd(self.ext_stat["mu_src"], mu_ext)
                        * self.ext_stat["scale_mmd"]
                    )
            with timer("backward"):
                loss.backward()
                del loss

        if self._meta_conf.align_ssh:
            with timer("forward"):
                loss = 0
                # feat_ssh = F.normalize(ssh(batch._x), dim=1)
                feat_ssh = ssh(batch._x)

                # queue
                if self._meta_conf.queue_size > self._meta_conf.batch_size_align:
                    feat_queue = self.queue_ssh.get()
                    self.queue_ssh.update(feat_ssh)
                    if feat_queue is not None:
                        feat_ssh = torch.cat(
                            [feat_ssh, feat_queue.to(self._meta_conf.device)]
                        )

                # coral
                if self._meta_conf.divergence in ["coral", "all"]:
                    cov_ssh = adaptation_utils.covariance(feat_ssh)
                    loss += (
                        adaptation_utils.coral(self.ssh_stat["cov_src"], cov_ssh)
                        * self.ssh_stat["scale_coral"]
                    )

                # mmd
                if self._meta_conf.divergence in ["mmd", "all"]:
                    mu_ssh = feat_ssh.mean(dim=0)
                    loss += (
                        adaptation_utils.linear_mmd(self.ssh_stat["mu_src"], mu_ssh)
                        * self.ssh_stat["scale_mmd"]
                    )

            with timer("backward"):
                loss.backward()
                del loss

    def adapt_and_eval(
        self,
        episodic,
        metrics: Metrics,
        model_selection_method: Type[BaseSelection],
        current_batch,
        previous_batches: List[Batch],
        logger: Logger,
        timer: Timer,
    ):
        """The key entry of test-time adaptation."""
        # some simple initialization.
        log = functools.partial(logger.log, display=self._meta_conf.debug)

        log("Test-time Training++ begins...")
        # TODO: add online TTT++.
        if episodic:
            log("\treset model to initial state during the test time.")
            self.reset()

        with timer("evaluate_adapted_model"):
            y_hat = self._model(current_batch._x)
            metrics.eval(current_batch._y, y_hat)

    @property
    def name(self):
        return "ttt_plus_plus"
