# -*- coding: utf-8 -*-
import os
import shutil
import json


"""json related."""


def read_json(path):
    """read json file from path."""
    with open(path, "r") as f:
        return json.load(f)


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False


"""operate dir."""


def build_dir(path, force):
    """build directory."""
    if os.path.exists(path) and force:
        shutil.rmtree(path)
        os.mkdir(path)
    elif not os.path.exists(path):
        os.mkdir(path)
    return path


def build_dirs(path):
    try:
        os.makedirs(path)
    except Exception as e:
        print(" encounter error: {}".format(e))


def remove_folder(path):
    try:
        shutil.rmtree(path)
    except Exception as e:
        print(" encounter error: {}".format(e))


def list_files(root_path):
    dirs = os.listdir(root_path)
    return [os.path.join(root_path, path) for path in dirs]
