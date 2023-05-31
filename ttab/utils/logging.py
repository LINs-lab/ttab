# -*- coding: utf-8 -*-
import os
import json
import time
import pprint
from typing import Any, Dict

from io import StringIO
import csv


class Logger(object):
    """
    Very simple prototype logger that will store the values to a JSON file
    """

    def __init__(self, folder_path: str) -> None:
        """
        :param filename: ending with .json
        :param auto_save: save the JSON file after every addition
        """
        self.folder_path = folder_path
        self.json_file_path = os.path.join(folder_path, "log-1.json")
        self.txt_file_path = os.path.join(folder_path, "log.txt")
        self.values = []
        self.pp = MyPrettyPrinter(indent=2, depth=3, compact=True)

    def log_metric(
        self,
        name: str,
        values: Dict[str, Any],
        tags: Dict[str, Any],
        display: bool = False,
    ) -> None:
        """
        Store a scalar metric

        :param name: measurement, like 'accuracy'
        :param values: dictionary, like { epoch: 3, value: 0.23 }
        :param tags: dictionary, like { split: train }
        """
        self.values.append({"measurement": name, **values, **tags})

        if display:
            print(
                "{name}: {values} ({tags})".format(name=name, values=values, tags=tags)
            )

    def pretty_print(self, value: Any) -> None:
        self.pp.pprint(value)

    def log(self, value: str, display: bool = True) -> None:
        content = time.strftime("%Y-%m-%d %H:%M:%S") + "\t" + value
        if display:
            print(content)
        self.save_txt(content)

    def save_json(self) -> None:
        """Save the internal memory to a file."""
        with open(self.json_file_path, "w") as fp:
            json.dump(self.values, fp, indent=" ")

        if len(self.values) > 1e4:
            # reset 'values' and redirect the json file to a different path.
            self.values = []
            self.redirect_new_json()

    def save_txt(self, value: str) -> None:
        with open(self.txt_file_path, "a") as f:
            f.write(value + "\n")

    def redirect_new_json(self) -> None:
        """get the number of existing json files under the current folder."""
        existing_json_files = [
            file for file in os.listdir(self.folder_path) if "json" in file
        ]
        self.json_file_path = os.path.join(
            self.folder_path, "log-{}.json".format(len(existing_json_files) + 1)
        )


class MyPrettyPrinter(pprint.PrettyPrinter):
    """Borrowed from
    https://stackoverflow.com/questions/30062384/pretty-print-namedtuple
    """

    def format_namedtuple(self, object, stream, indent, allowance, context, level):
        # Code almost equal to _format_dict, see pprint code
        write = stream.write
        write(object.__class__.__name__ + "(")
        object_dict = object._asdict()
        length = len(object_dict)
        if length:
            # We first try to print inline, and if it is too large then we print it on multiple lines
            inline_stream = StringIO()
            self.format_namedtuple_items(
                object_dict.items(),
                inline_stream,
                indent,
                allowance + 1,
                context,
                level,
                inline=True,
            )
            max_width = self._width - indent - allowance
            if len(inline_stream.getvalue()) > max_width:
                self.format_namedtuple_items(
                    object_dict.items(),
                    stream,
                    indent,
                    allowance + 1,
                    context,
                    level,
                    inline=False,
                )
            else:
                stream.write(inline_stream.getvalue())
        write(")")

    def format_namedtuple_items(
        self, items, stream, indent, allowance, context, level, inline=False
    ):
        # Code almost equal to _format_dict_items, see pprint code
        indent += self._indent_per_level
        write = stream.write
        last_index = len(items) - 1
        if inline:
            delimnl = ", "
        else:
            delimnl = ",\n" + " " * indent
            write("\n" + " " * indent)
        for i, (key, ent) in enumerate(items):
            last = i == last_index
            write(key + "=")
            self._format(
                ent,
                stream,
                indent + len(key) + 2,
                allowance if last else 1,
                context,
                level,
            )
            if not last:
                write(delimnl)

    def _format(self, object, stream, indent, allowance, context, level):
        # We dynamically add the types of our namedtuple and namedtuple like
        # classes to the _dispatch object of pprint that maps classes to
        # formatting methods
        # We use a simple criteria (_asdict method) that allows us to use the
        # same formatting on other classes but a more precise one is possible
        if hasattr(object, "_asdict") and type(object).__repr__ not in self._dispatch:
            self._dispatch[type(object).__repr__] = MyPrettyPrinter.format_namedtuple
        super()._format(object, stream, indent, allowance, context, level)


class CSVBatchLogger:
    """Borrowed from https://github.com/kohpangwei/group_DRO/blob/master/utils.py#L39"""

    def __init__(self, csv_path, n_groups, mode="w"):
        columns = ["epoch", "batch"]
        for idx in range(n_groups):
            columns.append(f"avg_loss_group:{idx}")
            columns.append(f"exp_avg_loss_group:{idx}")
            columns.append(f"avg_acc_group:{idx}")
            columns.append(f"processed_data_count_group:{idx}")
            columns.append(f"update_data_count_group:{idx}")
            columns.append(f"update_batch_count_group:{idx}")
        columns.append("avg_actual_loss")
        columns.append("avg_per_sample_loss")
        columns.append("avg_acc")
        columns.append("model_norm_sq")
        columns.append("reg_loss")

        self.path = csv_path
        self.file = open(csv_path, mode)
        self.columns = columns
        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        if mode == "w":
            self.writer.writeheader()

    def log(self, epoch, batch, stats_dict):
        stats_dict["epoch"] = epoch
        stats_dict["batch"] = batch
        self.writer.writerow(stats_dict)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()
