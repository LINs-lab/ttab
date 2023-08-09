# -*- coding: utf-8 -*-
import numpy as np

from monitor.tools.show_results import reorder_records
from monitor.tools.plot_utils import (
    determine_color_and_lines,
    plot_one_case,
    smoothing_func,
    configure_figure,
    build_legend,
)


"""plot the curve in terms of time."""


def plot_curve_wrt_time(
    ax,
    records,
    x_wrt_sth,
    y_wrt_sth,
    xlabel,
    ylabel,
    title=None,
    markevery_list=None,
    is_smooth=True,
    smooth_space=100,
    l_subset=0.0,
    r_subset=1.0,
    reorder_record_item=None,
    remove_duplicate=True,
    n_by_line=1,
    n_by_color=1,
    font_shift=0,
    has_legend=True,
    legend=None,
    legend_loc="lower right",
    legend_ncol=2,
    bbox_to_anchor=[0, 0],
    ylimit_bottom=None,
    ylimit_top=None,
    use_log=False,
):
    """Each info consists of
    ['tr_loss', 'tr_top1', 'tr_time', 'te_top1', 'te_step', 'te_time'].
    """
    # parse a list of records.
    distinct_conf_set = set()

    # re-order the records.
    if reorder_record_item is not None:
        records = reorder_records(records, reorder_on=reorder_record_item)

    count = 0
    for ind, (args, info) in enumerate(records):
        y_wrt_sth_list = y_wrt_sth.split(",")
        # check.
        if len(y_wrt_sth_list) > 1:
            assert y_wrt_sth_list[-1].endswith("preadapt_accuracy"), "the second item of y_wrt_sth must end with preadapt_accuracy."
            assert len(y_wrt_sth_list) == 2, "only support two arguments in y_wrt_sth."
            assert len(records) <= 4, "only support # records <= 4 when having two arguments in y_wrt_sth" # otherwise, the color and line_style will be a mess.

        for i in range(len(y_wrt_sth_list)):
            y_wrt_sth_i = y_wrt_sth_list[i]
            # build legend.
            _legend = build_legend(args, legend)
            if len(y_wrt_sth_list)>1 and i==0:
                _legend = ", ".join([_legend, "after_adapt"])
            elif i==1:
                _legend = ", ".join([_legend, "before_adapt"])

            if _legend in distinct_conf_set and remove_duplicate:
                continue
            else:
                distinct_conf_set.add(_legend)

            # determine the style of line, color and marker.
            line_style, color_style, mark_style = determine_color_and_lines(
                n_by_line, n_by_color, index=count
            )

            if len(y_wrt_sth_list)>1 and i==0:
                line_style = "-"
            elif i==1:
                line_style = "--"

            if markevery_list is not None:
                mark_every = markevery_list[ind]
            else:
                mark_style = None
                mark_every = None

            # determine if we want to smooth the curve.
            if "train-step" in x_wrt_sth or "train-epoch" in x_wrt_sth:
                info["train-step"] = list(range(1, 1 + len(info["train-loss"])))
            if "train-epoch" == x_wrt_sth:
                x = info["train-step"]
                x = [1.0 * _x / args["num_batches_train_per_device_per_epoch"] for _x in x]
            else:
                x = info[x_wrt_sth]
                if "time" in x_wrt_sth:
                    x = [(time - x[0]).seconds + 1 for time in x]
            y = info[y_wrt_sth_i]

            if is_smooth:
                x, y = smoothing_func(x, y, smooth_space)

            # only plot subtset.
            _l_subset, _r_subset = int(len(x) * l_subset), int(len(x) * r_subset)
            _x = x[_l_subset:_r_subset]
            _y = y[_l_subset:_r_subset]

            # use log scale for y
            if use_log:
                _y = np.log10(_y)

            # plot
            ax = plot_one_case(
                ax,
                x=_x,
                y=_y,
                label=_legend,
                line_style=line_style,
                color_style=color_style,
                mark_style=mark_style,
                mark_every=mark_every,
                remove_duplicate=remove_duplicate,
            )
        count += 1

    ax.set_ylim(bottom=ylimit_bottom, top=ylimit_top)
    ax = configure_figure(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        has_legend=has_legend,
        legend_loc=legend_loc,
        legend_ncol=legend_ncol,
        bbox_to_anchor=bbox_to_anchor,
        font_shift=font_shift,
    )
    return ax
