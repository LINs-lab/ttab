# -*- coding: utf-8 -*-
import copy
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
from ttab.utils.logging import Logger
from ttab.utils.timer import Timer


class NoAdaptation(BaseAdaptation):
    """Standard test-time evaluation (no adaptation)."""

    def __init__(self, meta_conf, model: nn.Module):
        super().__init__(meta_conf, model)

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
        model.eval()
        return model.to(self._meta_conf.device)

    def _post_safety_check(self):
        pass

    def initialize(self, seed: int):
        """Initialize the algorithm."""
        self._model = self._initialize_model(model=copy.deepcopy(self._base_model))

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
        with timer("test_time_adaptation"):
            with torch.no_grad():
                y_hat = self._model(current_batch._x)

        with timer("evaluate_adaptation_result"):
            metrics.eval(current_batch._y, y_hat)
            if self._meta_conf.base_data_name in ["waterbirds"]:
                self.tta_loss_computer.loss(
                    y_hat, current_batch._y, current_batch._g, is_training=False
                )

    @property
    def name(self):
        return "no_adaptation"
