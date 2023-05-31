# -*- coding: utf-8 -*-
from typing import Any, Dict, List

from ttab.api import Batch


class BaseSelection(object):
    def __init__(self, meta_conf, model_adaptation_method):
        self.meta_conf = meta_conf
        self.model = model_adaptation_method.copy_model()
        self.model.to(self.meta_conf.device)

        self.initialize()

    def initialize(self):
        pass

    def clean_up(self):
        pass

    def save_state(self):
        pass

    def select_state(
        self,
        current_batch: Batch,
        previous_batches: List[Batch],
    ) -> Dict[str, Any]:
        pass
