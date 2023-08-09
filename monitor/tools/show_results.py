# -*- coding: utf-8 -*-
from __future__ import division
import os
import json
import functools
import numbers
import joblib
import collections
import pandas as pd

from monitor.tools.file_io import list_files
from monitor.tools.file_io import load_pickle, read_json
from monitor.tools.utils import str2time

"""load data from pickled file."""


def get_pickle_info(root_data_path, experiments):
    file_paths = []
    for experiment in experiments:
        file_paths += [
            os.path.join(root_data_path, experiment, file)
            for file in os.listdir(os.path.join(root_data_path, experiment))
            if "pickle" in file
        ]

    results = dict((path, load_pickle(path)) for path in file_paths)
    info = functools.reduce(lambda a, b: a + b, list(results.values()))
    return info


"""load the raw results"""


def load_raw_info_from_experiments(root_path):
    """load experiments.
    root_path: a directory with a list of different trials.
    """
    exp_folder_paths = [
        folder_path
        for folder_path in list_files(root_path)
        if "pickle" not in folder_path
    ]

    info = []
    for folder_path in exp_folder_paths:
        try:
            element_of_info = _get_info_from_the_folder(folder_path)
            info.append(element_of_info)
        except Exception as e:
            print("error: {}".format(e))
    return info


def _get_info_from_the_folder(folder_path):
    print("process the folder: {}".format(folder_path))
    arguments_path = os.path.join(folder_path, "arguments.json")

    # return the information.
    results = {
        "records": dict(
            (key, sorted(values, key=lambda value: value["time"]))
            for key, values in _parse_runtime_infos(folder_path).items()
        )
    }
    return (folder_path, {"arguments": read_json(arguments_path), **results})


def _parse_runtime_infos(file_folder):
    existing_json_files = [
        file for file in os.listdir(file_folder) if "json" in file and "log" in file
    ]
    list_of_dicts = joblib.Parallel(n_jobs=7, backend="multiprocessing")(
        joblib.delayed(_parse_runtime_info)(
            os.path.join(file_folder, existing_json_file), existing_json_file
        )
        for existing_json_file in existing_json_files
    )
    list_of_dicts = [_dict for _id, _dict in sorted(list_of_dicts, key=lambda x: x[0])]
    return functools.reduce(
        lambda dict1, dict2: dict(
            (key, dict1[key] + dict2[key]) for key in dict1.keys()
        ),
        list_of_dicts,
    )


def _parse_runtime_info(json_file_path, json_file_name):
    lines = collections.defaultdict(list)

    with open(json_file_path) as json_file:
        raw_lines = json.load(json_file)
        for line in raw_lines:
            if line["measurement"] != "runtime":
                continue

            try:
                _time = str2time(line["time"], "%Y-%m-%d %H:%M:%S")
            except:
                _time = None
            line["time"] = _time

            #
            key = (
                f"{line['split']}-{line['type']}"
                if "type" in line
                else f"{line['split']}"
            )
            lines[key].append(line)

    file_id = int(json_file_name.split(".json")[0].split("-")[-1])
    return file_id, lines


"""extract the results based on the condition."""


def _is_same(items):
    return len(set(items)) == 1


def is_meet_conditions(args, conditions, threshold=1e-8):
    if conditions is None:
        return True

    # get condition values and have a safety check.
    condition_names = list(conditions.keys())
    condition_values = list(conditions.values())
    assert _is_same([len(values) for values in condition_values]) is True

    # re-build conditions.
    num_condition = len(condition_values)
    num_condition_value = len(condition_values[0])
    condition_values = [
        [condition_values[ind_cond][ind_value] for ind_cond in range(num_condition)]
        for ind_value in range(num_condition_value)
    ]

    # check re-built condition.
    g_flag = False
    try:
        for cond_values in condition_values:
            l_flag = True
            for ind, cond_value in enumerate(cond_values):
                _cond = cond_value == args[condition_names[ind]]

                if isinstance(cond_value, numbers.Number):
                    _cond = (
                        _cond
                        or abs(cond_value - args[condition_names[ind]]) <= threshold
                    )

                l_flag = l_flag and _cond
            g_flag = g_flag or l_flag
        return g_flag
    except:
        return False


def reorganize_records(records):
    # define attributes (parsed one) and map_attributes (appeared inline).
    attributes = ["time", "step", "loss", "accuracy", "worst-group_accuracy", "preadapt_accuracy", "dataset_statistics"]
    map_attributes = ["time", "step", "cross_entropy", "accuracy_top1", "worst_group_accuracy", "preadapted_accuracy_top1", "dataset_statistics"]

    def _parse(name, lines, is_train=True):
        parsed_lines = collections.defaultdict(list)

        for line in lines:
            for attribute, map_attribute in zip(attributes, map_attributes):
                if map_attribute is not None and map_attribute in line:
                    parsed_lines[attribute].append(line[map_attribute])
        return parsed_lines

    # deal with the records.
    record_lines = records["records"]

    # deal with different types of the records.
    parsed_record_lines = dict(
        (key, _parse(key, values, is_train=True if "train" in key else False))
        for key, values in record_lines.items()
    )
    return dict(
        (f"{record_line_name}-{attribute}", record_line_values[attribute])
        for attribute in attributes
        for record_line_name, record_line_values in parsed_record_lines.items()
    )


def extract_list_of_records(list_of_records, conditions, larger_is_better=True):
    # load and filter data.
    records = []

    for path, raw_records in list_of_records:
        # check conditions.
        if len(conditions) > 0 and not is_meet_conditions(
            raw_records["arguments"], conditions
        ):
            continue

        # get parsed records
        records += [(raw_records["arguments"], reorganize_records(raw_records))]

    print("we have {}/{} records.".format(len(records), len(list_of_records)))
    return records


"""summary the results."""


def _summarize_info(record, arg_names, be_groupby, larger_is_better):
    args, info = record
    if be_groupby in info:
        test_performance = (
            max(info[be_groupby]) if larger_is_better else min(info[be_groupby])
        )
    else:
        test_performance = -1
    return [args[arg_name] if arg_name in args else None for arg_name in arg_names] + [
        test_performance
    ]


def reorder_records(records, reorder_on):
    # records is in the form of <args, info>
    conditions = reorder_on.split(",")
    list_of_args = [
        (ind, [args[condition] for condition in conditions])
        for ind, (args, info) in enumerate(records)
    ]
    sorted_list_of_args = sorted(list_of_args, key=lambda x: x[1:])
    return [records[ind] for ind, args in sorted_list_of_args]


def summarize_info(records, arg_names, reorder_on, groupby_on, larger_is_better=True):
    # define header.
    headers = arg_names + [groupby_on]
    # reorder records
    records = reorder_records(records, reorder_on)
    # extract test records
    test_records = [
        _summarize_info(record, arg_names, groupby_on, larger_is_better)
        for record in records
    ]
    if len(test_records) > 0:
        # aggregate test records
        aggregated_records = pd.DataFrame(test_records, columns=headers)
        # average test records
        averaged_records = (
            aggregated_records.fillna(-1)
            .groupby(headers[:-1], as_index=False)
            .agg({groupby_on: ["mean", "std", "max", "min", "count"]})
            .sort_values((groupby_on, "mean"), ascending=not larger_is_better)
        )
        return aggregated_records, averaged_records
    else:
        empty_df = pd.DataFrame([])
        return empty_df, empty_df
