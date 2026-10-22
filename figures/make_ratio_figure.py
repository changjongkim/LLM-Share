#!/usr/bin/env python3
"""Figure of the shared-prefix ratio (review of 2026-10-22, campaign 144): eight agents with prompts of 16,384
tokens, of which the first part is the published BFCL prefix and the rest is their own. The memory that the
children add and their TTFT, for No Sharing (every child computes its whole prompt), Copy and STATOR. Means of
the repetitions with their 95% confidence intervals.

    RATIO_RESULT=<result dir> python3 make_ratio_figure.py
"""
import json
import os

from make_paper_figures import COPY, COPY_PALE, NAME, OURS, finish, legend, panel

# No Sharing takes the pale tint of Copy: the same device cache, without the restored prefix
MODES = (("No Sharing", COPY_PALE, "alone", "D"), ("Copy", COPY, "copy", "s"), (NAME, OURS, "extent", "o"))


def ratio_figure(result):
    groups = json.load(open(os.path.join(result, "analysis_summary.json")))["groups"]
    legend("eval_ratio_legend", [(label, colour) for label, colour, _, _ in MODES], len(MODES))
    for name, key, ylabel, scale, log in (("eval_ratio_mem", "child_increment_peak_gib", "Memory (GiB)", 1.0, False),
                                          ("eval_ratio_ttft", "child_ttft_ms", "TTFT (ms, log)", 1.0, True)):
        fig, ax = panel()
        for label, colour, mode, marker in MODES:
            cells = sorted((g for g in groups if g["mode"] == mode), key=lambda g: g["shared"])
            xs = [g["shared"] for g in cells]
            metric = [g["metrics"][key] for g in cells]
            ys = [m["mean"] / scale for m in metric]
            lo = [(m["mean"] - m["ci95_low"]) / scale if m["ci95_low"] is not None else 0 for m in metric]
            hi = [(m["ci95_high"] - m["mean"]) / scale if m["ci95_high"] is not None else 0 for m in metric]
            # No Sharing holds the memory of Copy: its line is dashed and drawn over that of Copy
            ax.errorbar(xs, ys, yerr=[lo, hi], color=colour, marker=marker, markersize=6, linewidth=2, capsize=2,
                        markeredgecolor="white", markeredgewidth=1.0, linestyle="--" if mode == "alone" else "-",
                        zorder=4 if mode == "alone" else 3)
        ax.set_xticks([2048, 4096, 8192, 12288, 16321], ["2K", "4K", "8K", "12K", "16K"])
        ax.set_xlim(800, 17600)
        finish(fig, ax, name, "Shared Prefix (Tokens)", ylabel, None if log else 24, log=log)


if __name__ == "__main__":
    ratio_figure(os.environ["RATIO_RESULT"])
