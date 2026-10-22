#!/usr/bin/env python3
"""Figure of eight forks of a running publisher (campaign 133, 2026-10-22): the pause of the parent at each fork and
the peak memory of the system, for Copy, STATOR (Copied File) and STATOR. Means of the repetitions with their
minimum and maximum.

    FORK_RESULT=<result dir> python3 make_fork_figure.py
"""
import json
import os
import statistics

from make_paper_figures import COPY, NAME, OURS, OURS_LIGHT, finish, labels, legend, panel

MODES = (("Copy", COPY, "copy", "s"), (f"{NAME} (Copied File)", OURS_LIGHT, "copy_map", "D"), (NAME, OURS, "extent", "o"))


def fork_figure(result):
    cases = [c for c in json.load(open(os.path.join(result, "cases.json"))) if c["status"] == "ok"]
    legend("eval_fork_legend", [(label, colour) for label, colour, _, _ in MODES], 3)

    fig, ax = panel()
    forks = max(len(c["publishes"]) for c in cases)
    for label, colour, mode, marker in MODES:
        runs = [[float(p["publish_ms"]) for p in c["publishes"]] for c in cases if c["mode"] == mode]
        xs = list(range(1, forks + 1))
        ys = [statistics.mean(r[i] for r in runs) for i in range(forks)]
        lo = [ys[i] - min(r[i] for r in runs) for i in range(forks)]
        hi = [max(r[i] for r in runs) - ys[i] for i in range(forks)]
        ax.errorbar(xs, ys, yerr=[lo, hi], color=colour, marker=marker, markersize=6, linewidth=2, capsize=2,
                    markeredgecolor=OURS if colour == OURS_LIGHT else "white", markeredgewidth=1.0)
    ax.set_xticks(range(1, forks + 1))
    ax.set_yscale("log")
    ax.set_ylim(3, 3000)
    finish(fig, ax, "eval_fork_pause", "Fork", "Pause (ms, log)", log=True, plain=True)

    fig, ax = panel()
    peaks = [[c["total_peak_accounted_bytes"] / 2**30 for c in cases if c["mode"] == mode] for _, _, mode, _ in MODES]
    means = [statistics.mean(p) for p in peaks]
    bars = ax.bar(range(len(MODES)), means, 0.55, color=[colour for _, colour, _, _ in MODES],
                  edgecolor="white", linewidth=1.5,
                  yerr=[[m - min(p) for m, p in zip(means, peaks)], [max(p) - m for m, p in zip(means, peaks)]],
                  error_kw={"capsize": 3, "elinewidth": 1.0, "ecolor": "#5f5d58"})
    ax.set_xticks(range(len(MODES)), ["" for _ in MODES])
    ax.tick_params(axis="x", length=0)
    labels(ax, bars, "{:.1f}")
    for text, p in zip(ax.texts[-len(MODES):], peaks):
        text.set_y(max(p))
    finish(fig, ax, "eval_fork_mem", "", "Peak Memory (GiB)", max(max(p) for p in peaks) * 1.3)


if __name__ == "__main__":
    fork_figure(os.environ["FORK_RESULT"])
