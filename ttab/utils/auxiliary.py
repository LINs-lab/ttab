# -*- coding: utf-8 -*-
import os
import time
import collections

import contextlib

import torch

import ttab.utils.checkpoint as checkpoint


class dict2obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [dict2obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, dict2obj(b) if isinstance(b, dict) else b)


def flatten_nested_dicts(d, parent_key="", sep="_"):
    """Borrowed from
    https://stackoverflow.com/a/6027615
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_nested_dicts(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


@contextlib.contextmanager
def fork_rng_with_seed(seed):
    if seed is None:
        yield
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            yield


@contextlib.contextmanager
def evaluation_monitor(conf):
    conf.status = "started"
    checkpoint.save_arguments(conf)

    yield

    # update the training status.
    job_id = (
        conf.job_id if conf.job_id is not None else f"/tmp/tmp_{str(int(time.time()))}"
    )
    os.system(f"echo {conf.checkpoint_path} >> {job_id}")

    # get updated conf
    conf.status = "finished"
    checkpoint.save_arguments(conf, force=True)
