# -*- coding: utf-8 -*-
from datetime import datetime


def str2time(string, pattern):
    """convert the string to the datetime."""
    return datetime.strptime(string, pattern)


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")


def is_float(value):
    try:
        float(value)
        return True
    except:
        return False


def dict_parser(values):
    local_dict = {}
    if values is None:
        return local_dict
    for kv in values.split(",,"):
        k, v = kv.split("=")
        try:
            local_dict[k] = float(v)
        except ValueError:
            try:
                local_dict[k] = str2bool(v)
            except ValueError:
                local_dict[k] = v
    return local_dict
