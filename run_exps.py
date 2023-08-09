# -*- coding: utf-8 -*-
import sys
import random
import six
import time
import importlib
import itertools
import functools

import monitor.tmux_cluster.tmux as tx


def import_string(dotted_path):
    """
    Import a dotted module path and
    return the attribute/class designated by the last name in the path.

    Raise ImportError if the import failed.
    """
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
    except ValueError:
        msg = "%s doesn't look like a module path" % dotted_path
        six.reraise(ImportError, ImportError(msg), sys.exc_info()[2])

    module = importlib.import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        msg = 'Module "%s" does not define a "%s" attribute/class' % (
            module_path,
            class_name,
        )
        six.reraise(ImportError, ImportError(msg), sys.exc_info()[2])


def read_replacements_from_python_class(python_file_path, script_class_name):
    # replace python_file_path.
    if python_file_path is not None:
        python_file_path = python_file_path.replace(".py", "").replace("/", ".") + (
            ".NewConf" if script_class_name is None else script_class_name
        )
        new_conf_object = import_string(python_file_path)

        if hasattr(new_conf_object, "to_be_replaced"):
            return new_conf_object.to_be_replaced
        else:
            return None
    else:
        return None

def split_list_argument(key, values):
    assert isinstance(values, list), "values should be a list here."
    cmd = " --{} ".format(key)
    for i in range(len(values)):
        cmd += "{} ".format(values[i])
    return cmd


def build_script(conf, idx, replacement=None, device="cuda:0"):
    # get prefix_cmd.
    conf.timestamp = str(int(time.time()) + random.randint(0, 1000) + idx)

    # build complete script.
    if replacement is not None:
        cmd = f"OMP_NUM_THREADS={conf.num_cpus} MKL_NUM_THREADS={conf.num_cpus} {conf.python_path if 'python_path' not in replacement else replacement['python_path']} {conf.main_file}"
    else:
        cmd = f"OMP_NUM_THREADS={conf.num_cpus} MKL_NUM_THREADS={conf.num_cpus} {conf.python_path} {conf.main_file}"

    # perform replacement.
    for k, v in conf.__dict__.items():
        if replacement is not None and k in replacement:
            if not isinstance(replacement[k], list):
                cmd += " --{} {} ".format(k, replacement[k])
            else:
                cmd += split_list_argument(key=k, values=replacement[k])
        elif v is not None:
            if not isinstance(v, list):
                cmd += " --{} {} ".format(k, v)
            else:
                cmd += split_list_argument(key=k, values=v)

    cmd += " --device {} ".format(device)
    return cmd


def create_scripts(conf):
    # get the replacement list for each job.
    replacements = read_replacements_from_python_class(
        conf.script_path, conf.script_class_name
    )

    # must specify device in the replacement file.
    if "device" in replacements.keys():
        available_devices = replacements["device"]
        num_available_devices = len(available_devices)
        del replacements["device"]

    if replacements is not None:
        replacement_keys, replacement_values = (
            list(replacements.keys()),
            list(replacements.values()),
        )

        # replace job_name in conf.
        if "job_name" in replacement_keys:
            assert len(replacements["job_name"]) == 1, "cannot implement more than 1 type of experiments at a time."
            conf.job_name = replacements["job_name"][0]
        
        if "main_file" in replacement_keys:
            assert len(replacements["main_file"]) == 1, "cannot implement more than 1 type of python script at a time."
            conf.main_file = replacements["main_file"][0]

        # build replacement combinations.
        if "coupled" not in replacement_keys:
            new_replacements = [
                dict(zip(replacement_keys, v))
                for v in itertools.product(*replacement_values)
            ]
        else:
            # check the job files.
            coupled_keys = replacements["coupled"] + ["coupled"]
            coupled_key_values = [
                (couple, replacements[couple]) for couple in replacements["coupled"]
            ]
            coupled_value_length = [len(values) for key, values in coupled_key_values]
            assert coupled_value_length.count(coupled_value_length[0]) == len(
                coupled_key_values
            )

            # for coupled keys, we ensure they are the same,
            # otherwise we use itertools.product over its values.
            excluded_replacement_keys = [
                key for key, value in replacements.items() if key not in coupled_keys
            ]
            excluded_replacement_values = [
                value for key, value in replacements.items() if key not in coupled_keys
            ]
            excluded_replacements = [
                dict(zip(excluded_replacement_keys, v))
                for v in itertools.product(*excluded_replacement_values)
            ]
            new_replacements = functools.reduce(
                lambda a, b: a + b,
                [
                    [
                        list(excluded_replacement.items())
                        + [
                            (key, values[idx])
                            for key, values in coupled_key_values
                            if key != "coupled"
                        ]
                        for excluded_replacement in excluded_replacements
                    ]
                    for idx in range(coupled_value_length[0])
                ],
            )
            new_replacements = [dict(replacement) for replacement in new_replacements]
    else:
        new_replacements = [None]

    # create job scripts.
    scripts = []

    # update the job_id.
    conf.job_id = f"/tmp/jobrun_logs_{str(int(time.time()))}"

    for idx, new_replacement in enumerate(new_replacements):
        print(f"{idx+1}-th replacement conf: {new_replacement}.")
        device_id = idx % num_available_devices
        device = available_devices[device_id]
        scripts.append(build_script(conf, idx, new_replacement, device))
    return scripts


def create_jobs_on_node(conf, scripts):
    def _query_job_status(log_path):
        try:
            with open(log_path, "rb") as f:
                lines = f.readlines()
            return list(set([line for line in lines if len(line) > 0]))
        except FileNotFoundError:
            return []

    print(f"\n\nRun jobs on the host with job_id={conf.job_id}.")
    is_complete = False
    num_finished_task = 0
    task_count = 0
    current_degree_parallelism = 0
    expected_degree_parallelism = conf.num_jobs_per_node

    while not is_complete:
        if current_degree_parallelism > 0:
            time.sleep(conf.wait_in_seconds_per_job)

        # run one new experiment, and update the counter.
        if (
            current_degree_parallelism < expected_degree_parallelism
            and task_count < len(scripts)
        ):
            new_task_script = scripts[task_count]
            print(
                f"\n\nlaunch new task@{task_count + 1} / {len(scripts)}: {new_task_script}."
            )
            tx.Run(name=f"{conf.job_name}", job_node="localhost").make_job(
                job_name=f"job-{task_count}", task_scripts=[new_task_script]
            )
            current_degree_parallelism += 1
            task_count += 1

        # update the counter.
        cur_num_finished_task = int(len(_query_job_status(conf.job_id)) / conf.num_jobs_per_script)
        if cur_num_finished_task != num_finished_task:
            current_degree_parallelism -= cur_num_finished_task - num_finished_task
            num_finished_task = cur_num_finished_task

        if num_finished_task == len(scripts):
            is_complete = True

    # exit.
    sys.exit(0)


if __name__ == "__main__":
    from parameters import get_args

    conf = get_args()

    """workflow:
    1. we read the experiment setup from one py file,
    2. we create the exact experiment script
        (based on the default hyper-parameters as well as the new hyper-parameters).
    3. launch the experiments by feeding predefined num_jobs_per_node to the experiment queue.
    """
    scripts = create_scripts(conf)
    create_jobs_on_node(conf, scripts)
