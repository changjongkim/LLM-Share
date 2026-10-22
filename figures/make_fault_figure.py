#!/usr/bin/env python3
"""Figure 15(b) since the review of 2026-10-22: a sharer with a defect writes the published prefix through the GPU
while eight children generate; the share of children whose tokens still equal those of the control case, by token
after the write (analysis.json of stator_campaign_20261016/tools/summarize_fault_writer.py).

    FAULT_RESULT=<result dir> python3 make_fault_figure.py
"""
import json
import os

from make_paper_figures import DEVICE, NAME, OURS, panel, finish, legend
from make_eval_figures import AXIS


def fault_writer(result):
    modes = json.load(open(os.path.join(result, "analysis.json")))["modes"]
    # every child generated at least 12 tokens before the write, all of them equal to the control case
    before = min(min(m["tokens_before_write"]) for m in modes.values())
    span = min(len(m["curve_percent"]) for m in modes.values())
    # the strip above both panels of the figure: (a) is Device alone
    legend("eval_dev_legend", [("Device", DEVICE), (NAME, OURS)], 2)
    fig, ax = panel()
    for label, colour, mode, marker in (("Device", DEVICE, "device_merged", "s"), (NAME, OURS, "extent", "o")):
        curve = [100.0] * before + modes[mode]["curve_percent"][:span]
        ax.plot(range(-before, span), curve, color=colour, linewidth=2, drawstyle="steps-post", label=label,
                marker=marker, markersize=6, markevery=12, markeredgecolor="white", markeredgewidth=1.0)
    ax.axvline(0, color=AXIS, linewidth=1.0, linestyle="--")
    ax.set_xlim(-before, span - 1)
    finish(fig, ax, "eval_fault", "Tokens after the Write", "Unchanged\nChildren (%)", 115)


if __name__ == "__main__":
    fault_writer(os.environ["FAULT_RESULT"])
