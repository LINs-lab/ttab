# -*- coding: utf-8 -*-
import yaml
import os
import time
from tqdm import tqdm


def ossystem(cmds):
    if isinstance(cmds, str):
        print(f"\n=> {cmds}")
        os.system(cmds)
    elif isinstance(cmds, list):
        for cmd in tqdm(cmds):
            ossystem(cmd)
    else:
        raise NotImplementedError(
            "Cmds should be string or list of str. Got {}.".format(cmds)
        )


def environ(env):
    return os.getenv(env)


def load_yaml(file):
    with open(file) as f:
        return yaml.safe_load(f)


def wait_for_file(fn, max_wait_sec=600, check_interval=0.02):
    start_time = time.time()
    while True:
        if time.time() - start_time > max_wait_sec:
            assert False, "Timeout %s exceeded" % (max_wait_sec)
        if not os.path.exists(fn):
            time.sleep(check_interval)
            continue
        else:
            break


if __name__ == "__main__":
    ossystem("ls")
