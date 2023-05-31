# -*- coding: utf-8 -*-
import copy
from typing import Any, Dict

import torch
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import accuracy_top1, cross_entropy


class OracleModelSelection(BaseSelection):
    """grid-search the best adaptation result per batch (given a sufficiently long adaptation
    steps and a single learning rate in each step, save the best checkpoint
    and its optimizer states after iterating over all adaptation steps)"""

    def __init__(self, meta_conf, model_adaptation_method):
        super().__init__(meta_conf, model_adaptation_method)

    def initialize(self):
        if hasattr(self.model, "ssh"):
            self.model.ssh.eval()
            self.model.main_model.eval()
        else:
            self.model.eval()

        self.optimal_state = None
        self.current_batch_best_acc = 0
        self.current_batch_coupled_ent = None

    def clean_up(self):
        self.optimal_state = None
        self.current_batch_best_acc = 0
        self.current_batch_coupled_ent = None

    def save_state(self, state, current_batch):
        """Selectively save state for current batch of data."""
        batch_best_acc = self.current_batch_best_acc
        coupled_ent = self.current_batch_coupled_ent

        if not hasattr(self.model, "ssh"):
            self.model.load_state_dict(state["model"])
            with torch.no_grad():
                outputs = self.model(current_batch._x)
        else:
            self.model.main_model.load_state_dict(state["main_model"])
            with torch.no_grad():
                outputs = self.model.main_model(current_batch._x)

        current_acc = self.cal_acc(current_batch._y, outputs)
        if (self.optimal_state is None) or (current_acc > batch_best_acc):
            self.current_batch_best_acc = current_acc
            self.current_batch_coupled_ent = self.cal_ent(current_batch._y, outputs)
            state["yhat"] = outputs
            self.optimal_state = state
        elif current_acc == batch_best_acc:
            # compare cross entropy
            assert coupled_ent is not None, "Cross entropy value cannot be none."
            current_ent = self.cal_ent(current_batch._y, outputs)
            if current_ent < coupled_ent:
                self.current_batch_coupled_ent = current_ent
                state["yhat"] = outputs
                self.optimal_state = state

    def cal_acc(self, targets, outputs):
        return accuracy_top1(targets, outputs)

    def cal_ent(self, targets, outputs):
        return cross_entropy(targets, outputs)

    def select_state(self) -> Dict[str, Any]:
        """return the optimal state and sync the model defined in the model selection method."""
        if not hasattr(self.model, "ssh"):
            self.model.load_state_dict(self.optimal_state["model"])
        else:
            self.model.main_model.load_state_dict(self.optimal_state["main_model"])
            self.model.ssh.load_state_dict(self.optimal_state["ssh"])
        return self.optimal_state

    @property
    def name(self):
        return "oracle_model_selection"
