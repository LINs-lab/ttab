# -*- coding: utf-8 -*-

from .last_iterate import LastIterate
from .oracle_model_selection import OracleModelSelection


def get_model_selection_method(selection_name):
    return {
        "last_iterate": LastIterate,
        "oracle_model_selection": OracleModelSelection,
    }[selection_name]
