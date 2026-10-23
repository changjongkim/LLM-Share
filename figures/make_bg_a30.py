#!/usr/bin/env python3
"""Figure 3(a) and (b) with the A30 (2026-10-22): the same probe (mech_probe2) on an x86 server whose A30 accesses
pageable memory through HMM, next to the Thor values. The colour is the reader (CPU blue, GPU orange) and the light
tint of a colour is the same reader on the A30, a variant of the same measurement on the other platform. The
locked page cache of (b) keeps the colour of its reader and is named on its axis.

    python3 make_bg_a30.py
"""
import csv
import os

import make_paper_figures as paper
from make_paper_figures import (swatch, COW, COW_LIGHT, DEMAND, DEMAND_LIGHT, DENSE, MECHS, TICK2, compact, finish, full,
                                grouped, labels, legend, panel, rows, save, style, tip)
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

A30 = "/home/thor/kcj/portable_bench_a30/results/a30_20261022/node0-20261022-020702/mech/mech_probe_summary.csv"
READERS = (("CPU (Thor)", DEMAND, "thor", "cpu"), ("GPU (Thor)", COW, "thor", "gpu"),
           ("CPU (A30)", DEMAND_LIGHT, "a30", "cpu"), ("GPU (A30)", COW_LIGHT, "a30", "gpu"))


def summary(path_rows):
    data = {}
    for r in path_rows:
        if r["mech"] == "gate" or r["step"] == "events":
            continue
        data[(r["mech"], r["step"], r["access"], r["pressure"])] = r
    return data


def main():
    directory = full("mech-probe")
    thor = {}
    for d in (directory, full("mech-probe-shared")):
        for key, r in summary(rows(d, "mech_probe_summary.csv")).items():
            if d == directory and key[0] == "shared":
                continue   # every shared case of the first campaign failed
            thor[key] = r
    a30 = summary(csv.DictReader(open(A30)))
    platform = {"thor": thor, "a30": a30}
    # the strip stands above (a) and (b) only, the half of the figure whose bars it names, in one row with narrow
    # spacing so that it fits that width
    strip = plt.figure(figsize=(6.6, 0.38))
    strip.legend([swatch(colour) for _, colour, _, _ in READERS],
                 [label for label, _, _, _ in READERS], loc="center", ncol=4, columnspacing=0.8, handlelength=1.0,
                 handletextpad=0.4, fontsize=12.5)
    save(strip, "bg_mech_legend")

    def value(where, mech, step, access, column):
        row = platform[where].get((mech, step, access, ""))
        return float(row[column]) if row and row[column] else 0.0

    fig, ax = panel()
    series = [(label, colour, [value(where, m, "read", access, "seconds") for _, m in MECHS])
              for label, colour, where, access in READERS]
    for group in grouped(ax, [label for label, _ in MECHS], series, width=0.21):
        for bar in group:
            tip(ax, bar, compact(bar.get_height()))
            ax.texts[-1].set_fontsize(DENSE)
            ax.texts[-1].set_rotation(90)
    ax.tick_params(axis="x", labelsize=TICK2)
    top = max(v for _, _, vs in series for v in vs)
    low = min(v for _, _, vs in series for v in vs if v > 0)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylim(low / 3, top * 30)
    ax.set_ylabel("Read Time (s)")
    style(ax)
    fig.tight_layout()
    save(fig, "bg_mech_read")

    # one group per reader, Thor and the A30 side by side
    fig, ax = panel()
    groups = (("CPU", "none", "cpu", DEMAND, DEMAND_LIGHT), ("GPU", "none", "gpu", COW, COW_LIGHT),
              ("GPU\nLocked", "lock", "gpu", COW, COW_LIGHT))
    xs, values, colours = [], [], []
    for index, (_, pin, access, solid, tint) in enumerate(groups):
        for offset, where, colour in ((-0.21, "thor", solid), (0.21, "a30", tint)):
            row = platform[where].get(("reclaim", pin, access, "pressure"))
            xs.append(index + offset)
            values.append(float(row["rate_vs_none"]) if row and row["rate_vs_none"] else 0.0)
            colours.append(colour)
    bars = ax.bar(xs, values, 0.4, color=colours, edgecolor="white", linewidth=1.5)
    ax.set_xticks(range(len(groups)), [label for label, _, _, _, _ in groups])
    ax.tick_params(axis="x", labelsize=TICK2)
    labels(ax, bars, "{:.2f}")
    for text in ax.texts[-len(bars):]:
        text.set_fontsize(DENSE)
        text.set_rotation(90)
    finish(fig, ax, "bg_mech_reclaim", "", "Speed under Reclaim", 1.45)


if __name__ == "__main__":
    main()
