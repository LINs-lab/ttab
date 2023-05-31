# -*- coding: utf-8 -*-
import os
import time
import json
from typing import Any

import torch

import ttab.utils.file_io as file_io


def init_checkpoint(conf: Any):
    # init checkpoint dir.
    conf.checkpoint_path = os.path.join(
        conf.root_path,
        conf.model_name,
        conf.job_name,
        # f"{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.model_selection_method}_{int(conf.timestamp if conf.timestamp is not None else time.time())}-seed{conf.seed}",
        f"{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.model_selection_method}_{str(time.time()).replace('.', '_')}-seed{conf.seed}",
    )

    # if the directory does not exists, create them.
    file_io.build_dirs(conf.checkpoint_path)
    return conf.checkpoint_path


def save_arguments(conf: Any, force: bool = False):
    # save the configure file to the checkpoint.
    path = os.path.join(conf.checkpoint_path, "arguments.json")

    if force or not os.path.exists(path):
        with open(path, "w") as fp:
            json.dump(
                dict(
                    [
                        (k, v)
                        for k, v in conf.__dict__.items()
                        if file_io.is_jsonable(v) and type(v) is not torch.Tensor
                    ]
                ),
                fp,
                indent=" ",
            )
