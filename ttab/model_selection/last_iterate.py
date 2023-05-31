# -*- coding: utf-8 -*-
import copy
from typing import Any, Dict

from ttab.model_selection.base_selection import BaseSelection


class LastIterate(BaseSelection):
    """Naively return the model generated from the last iterate of adaptation."""

    def __init__(self, meta_conf, model_adaptation_method):
        super().__init__(meta_conf, model_adaptation_method)

    def initialize(self):
        if hasattr(self.model, "ssh"):
            self.model.ssh.eval()
            self.model.main_model.eval()
        else:
            self.model.eval()

        self.optimal_state = None

    def clean_up(self):
        self.optimal_state = None

    def save_state(self, state, current_batch):
        self.optimal_state = state

    def select_state(self) -> Dict[str, Any]:
        """return the optimal state and sync the model defined in the model selection method."""
        return self.optimal_state

    @property
    def name(self):
        return "last_iterate"
