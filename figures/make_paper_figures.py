#!/usr/bin/env python3
"""Half-column panels for the paper, in the figure style of the PHASOR paper.

Reads the packaged summaries under ../results (nothing is typed in by hand)
and writes one PDF and one PNG per panel next to this script. A panel whose
campaign has not been run yet is skipped. Every panel shows one metric with
its unit on the axis and the value on the bar.

  eval_mem_*       memory of the agents on one prefix
  eval_weights_*   weights in place against the device copy
  eval_hand_*      pause of the publisher, attach time, time to first token
  eval_speed_*     host-mapped cache against device memory; read bandwidth
  eval_mode_*      time slicing, MPS, MIG, MPS within each MIG instance
  eval_cow_*       extents against copy-on-write mappings
  eval_dev_*       host extents against shared device memory
  eval_tree_*      extent chains on a tree of agents
  eval_sota_*      five ways to hand a prefix over, in one engine
  eval_attach_*    attach cost of shared device memory by allocation size
  eval_scale_*     more agents; eval_model_*, eval_work_*: other models, workload
"""
import math
import os
import re
import statistics
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_eval_figures as base  # noqa: E402
from make_eval_figures import (AXIS, COPY, COW, DEVICE, GRID, INK, OURS,  # noqa: E402
                               OURS_LIGHT, RES, grouped, rows, save, style, tip)

NAME = "STATOR"
# One palette for every panel, in a fixed order: blue for what copies the state (Copy and its
# variants, 10-23), dark grey for the existing engine and vLLM, then blue, violet and orange for the other
# mechanisms or systems of a panel, and teal for STATOR. A lighter tint of a
# colour is a variant of the same thing (other page size, a part left out).
DEMAND = "#6aa3dd"         # blue: Demand
EXISTING = "#3d3c39"       # dark grey: the existing engine (and the unmodified vLLM server)
OLLAMA = "#8c564b"         # brown: Ollama
SHARED_MAP = "#d9ae1e"     # gold: shared writable mapping
RO_MAP = "#ecd27a"         # light gold: read-only shared mapping
LIGHT_GRAY = "#6f97d4"     # a tint of Copy: CPU offloading of vLLM (copies to host memory)
DEMAND_LIGHT = "#b9cfe9"
DEVICE_LIGHT = "#c9bdf0"   # violet (DEVICE) is the third entry
COPY_LIGHT = "#6f97d4"     # tints of Copy (10-23): 9.0 from teal under CVD simulation
COPY_PALE = "#b8cdee"
OURS_PALE = "#c9e9e5"    # teal (OURS, OURS_LIGHT): a second part left out
# Text sizes in points on a canvas that the paper places at 0.51 of its size
# (a quarter of the page or half a column): a low canvas with large text,
# which takes little height in a column. Every panel is saved as its whole
# canvas, so that all panels print at one scale. A legend that two panels
# share is a strip above them (legend()).
VALUE = 10.5    # a value on a mark
DENSE = 9.5     # a value above one of five bars of a group, set upright
LEGEND = 10.5
TICK2 = 11      # tick labels of two lines
PANEL = (3.2, 1.75)
plt.rcParams.update({"font.size": 13, "axes.labelsize": 13.5, "xtick.labelsize": 12.5,
                     "ytick.labelsize": 12.5, "legend.fontsize": LEGEND})
base.PANEL = PANEL
base.TIP = VALUE
base.LABEL_MIN, base.TICK_MIN, base.VALUE_MIN = 12.5, 10.0, 9.0
base.PAD = 0.3
base.FIXED = True


_save = save


def save(fig, name):
    """Every legend of a panel stands above the plot, never inside it: a row
    at the top of the canvas, set smaller or on two rows if it is wider than
    the canvas, and the plot is laid out in the room below it."""
    for ax in fig.axes:
        inside = ax.get_legend()
        if inside is None:
            continue
        handles = getattr(inside, "legend_handles", None) or inside.legendHandles
        texts = [text.get_text() for text in inside.get_texts()]
        inside.remove()
        size, ncol = LEGEND, len(texts) if len(texts) <= 3 else 2
        while True:
            top = fig.legend(handles, texts, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=ncol,
                             fontsize=size, handlelength=1.0, columnspacing=0.8, handletextpad=0.4,
                             borderaxespad=0.15, labelspacing=0.25)
            fig.canvas.draw()
            if top.get_window_extent().width <= fig.bbox.width - 4:
                break
            if size - 0.5 >= LEGEND - 1.0:
                size -= 0.5
            elif ncol > 2:
                size, ncol = LEGEND, 2
            else:
                print(f"WARN {name}: the legend is wider than the canvas")
                break
            top.remove()
        fig.layout_rect = (0, 0, 1, 1 - top.get_window_extent().height / fig.bbox.height)
    _save(fig, name)


base.save = save


def have(*path):
    return os.path.exists(os.path.join(RES, base.resolve(path[0]), *path[1:]))


def panel(size=PANEL):
    return plt.subplots(figsize=size)


def legend(name, entries, ncol, half=False):
    """A strip that holds only the legend, shared by the panels below it. A strip above one half-column panel
    (half) keeps the text size and spaces its entries closer, so that it fits the width of that panel."""
    fig = plt.figure(figsize=(1.9 * ncol + 0.6, 0.38))
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour) for _, colour in entries]
    spacing = dict(columnspacing=0.7, handlelength=1.0, handletextpad=0.4) if half else \
        dict(columnspacing=1.6, handlelength=1.3)
    fig.legend(handles, [label for label, _ in entries], loc="center", ncol=ncol, fontsize=12.5, **spacing)
    save(fig, name)


def bg_state_legend():
    """The strip above (b) of Figure 1: the two prefix lengths (10-23, in place of the legend inside the panel)."""
    legend("bg_state_legend", [("4,081 Tokens", COPY_LIGHT), ("16,321 Tokens", COPY)], 2, half=True)


def pair_legend():
    legend("eval_pair_legend", [("Copy", COPY), (NAME, OURS)], 2)


def labels(ax, bars, form="{:.1f}", size=VALUE):
    for bar in bars:
        tip(ax, bar, form.format(bar.get_height()))
        ax.texts[-1].set_fontsize(size)


def finish(fig, ax, name, xlabel, ylabel, top=None, log=False, xtick=None, plain=False):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
        if plain:   # 0.1, 1, 10 take less width than powers of ten
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    if top is not None:
        ax.set_ylim(ax.get_ylim()[0] if log else 0, top)
    emphasize(ax)
    style(ax)
    if xtick is not None:
        ax.tick_params(axis="x", labelsize=xtick)
    fig.tight_layout()
    save(fig, name)


def emphasize(ax):
    """The lines of STATOR stand out in every line plot (10-23): drawn over the other lines and thicker. A line of
    another configuration that lies on a line of STATOR (within 4 px at most of their common points) is drawn as a
    thin line over it, so that both show; so is a line marked with the gid "over"."""
    import numpy as np
    from matplotlib.colors import to_hex
    data = [line for line in ax.lines if line.get_linestyle() not in ("None", "") and len(line.get_xdata()) > 1]
    ours = [line for line in data if to_hex(line.get_color()) == OURS]
    for line in ours:
        line.set_linewidth(2.8)
        line.set_zorder(5)

    def pixels(line):
        points = np.column_stack([np.asarray(line.get_xdata(), float), np.asarray(line.get_ydata(), float)])
        return {round(x, 9): y for (x, _), (_, y) in zip(points, ax.transData.transform(points))}

    for line in data:
        if line in ours:
            continue
        mine = pixels(line)
        close = any(sum(abs(mine[x] - theirs[x]) <= 4 for x in common) * 2 >= len(common) > 0
                    for theirs in map(pixels, ours) for common in [set(mine) & set(theirs)])
        if close or line.get_gid() == "over":
            line.set_zorder(6)
            line.set_linewidth(1.4)
            line.set_markersize(min(line.get_markersize(), 5))


# ------------------------------------------------------------------ memory
def mem():
    data = {(r["paragraphs"], r["mode"], int(r["agents"])): r
            for r in rows("20261006-engine-kvshare-v1", "engine_kvshare_summary.csv")}
    series_of = (("Copy", COPY, "restore"), (f"{NAME} (Allocated at Once)", OURS_LIGHT, "extent"),
                 (NAME, OURS, "extent_lazy"))
    legend("eval_mem_legend", [(label, colour) for label, colour, _ in series_of], 3)
    counts = [1, 4, 8]
    for size, tag in (("80", "4k"), ("320", "16k")):
        fig, ax = panel()
        series = [(label, colour, [float(data[size, mode, n]["memory_mib"]) / 1024 for n in counts])
                  for label, colour, mode in series_of]
        for group in grouped(ax, [str(n) for n in counts], series, 0.27):
            labels(ax, group)
        finish(fig, ax, f"eval_mem_{tag}", "Agents", "Memory (GiB)", 23)


def weights():
    agents = {(r["config"], r["mode"], int(r["agents"])): r
              for r in rows("20261005-engine-agents-v1", "engine_agents_summary.csv")}
    # processes in different MIG instances on the first platform, time-sliced
    # on the whole GPU
    config = "mig" if any(k[0] == "mig" for k in agents) else "timeslice"
    counts = sorted({k[2] for k in agents if k[0] == config})
    fig, ax = panel()
    series = [(label, colour, [float(agents[config, mode, n]["memory_total_mib"]) / 1024 for n in counts])
              for label, colour, mode in (("Existing Engine", EXISTING, "copy"), (NAME, OURS, "inplace"))]
    for group in grouped(ax, [str(n) for n in counts], series, 0.38):
        labels(ax, group)
    # the legend is a strip above (a) and (b), as in the other figures (10-23)
    legend("eval_weights_legend", [("Existing Engine", EXISTING), (NAME, OURS)], 2)
    finish(fig, ax, "eval_weights_mem", "Serving Processes", "Memory (GiB)", 47)

    pages = {r["mode"]: r for r in rows("20261005-engine-pages-v1", "engine_pages_summary.csv")}
    order = (("Existing\nEngine", "copy", EXISTING), (f"{NAME}\n(4 KiB)", "inplace_4k", OURS_LIGHT),
             (f"{NAME}\n(2 MiB)", "inplace_thp", OURS))
    fig, ax = panel()
    values = [float(pages[mode]["generation_vs_copy"]) for _, mode, _ in order]
    low = [v - float(pages[mode]["lower_ci95"]) for v, (_, mode, _) in zip(values, order)]
    high = [float(pages[mode]["upper_ci95"]) - v for v, (_, mode, _) in zip(values, order)]
    bars = ax.bar([label for label, _, _ in order], values, 0.6, color=[c for _, _, c in order],
                  edgecolor="white", linewidth=1.5, yerr=[low, high],
                  error_kw={"ecolor": INK, "elinewidth": 1.0, "capsize": 3})
    for bar, high_part in zip(bars, high):
        ax.annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width() / 2,
                                                 bar.get_height() + high_part),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=VALUE, color=INK)
    ax.axhline(1.0, color=AXIS, linewidth=1.0)
    finish(fig, ax, "eval_weights_speed", "", "Relative Speed", 1.18)


# ---------------------------------------------------------------- handover
# CoW Map is the private writable mapping of the kernel without a step of
# STATOR (mode writable_no_read). The mode `cow` of the campaigns adds the CPU
# read pass of STATOR on 2 MiB pages; it is a part of STATOR and is drawn as
# one in cow().
MECHANISMS = (("Copy", COPY, "copy"), ("Demand", DEMAND, "demand"), ("Device", DEVICE, "device_merged"),
              ("CoW Map", COW, "writable_no_read"), (NAME, OURS, "extent"))
# The second campaign runs the engine that puts the entries of a divided 2 MiB
# block back at a publish; the first, which also has the cells with one and
# four children, is kept for the panels by the number of agents.
MECH = "20261010-engine-kvmech-v2"
MECH_COUNTS = "20261007-engine-kvmech-v1"
# The plain mapping was measured later in the cells of MECH, with copy and
# extent repeated as a check (within 5% in memory, gate W4 of the verifier).
MECH_PLAIN = "20261013-engine-kvmech-cowplain-v1"


def mechanisms(directory=None):
    """The campaign that runs every mechanism in one engine, by cell and mode."""
    data = {(r["placement"], int(r["children"]), int(r["paragraphs"]), r["mode"]): r
            for r in rows(directory or MECH, "engine_kvmech_summary.csv")}
    if directory is None and have(MECH_PLAIN, "engine_kvmech_summary.csv"):
        data.update({(r["placement"], int(r["children"]), int(r["paragraphs"]), r["mode"]): r
                     for r in rows(MECH_PLAIN, "engine_kvmech_summary.csv") if r["mode"] == "writable_no_read"})
    return data


def hand():
    """Handing a prefix over with every mechanism: pause, attach, first token."""
    if not have(MECH, "engine_kvmech_summary.csv"):
        return
    data = mechanisms()
    legend("eval_hand_legend", [(label, colour) for label, colour, _ in MECHANISMS], 5)
    sizes = sorted({k[2] for k in data if k[0] == "same" and k[1] == 8})
    counts = sorted({k[1] for k in data if k[0] == "same" and k[2] == max(sizes)})
    long_tokens = int(float(data["same", 8, max(sizes), "copy"]["prefix_tokens"]))

    def bars(name, groups, cells, column, ylabel, scale, form, log, xlabel):
        fig, ax = panel()
        # a mechanism that a campaign does not have is left out of its panels
        series = [(label, colour, [float(data[cell + (m,)][column]) / scale for cell in cells])
                  for label, colour, m in MECHANISMS if all(cell + (m,) in data for cell in cells)]
        for group in grouped(ax, groups, series, 0.17):
            labels(ax, group, form, DENSE)
            for text in ax.texts[-len(group):]:
                text.set_rotation(90)
        top = max(max(values) for _, _, values in series)
        if log:
            ax.set_ylim(1, top * 14)
        finish(fig, ax, name, xlabel, ylabel, None if log else top * 1.3, log)

    bars("eval_hand_pause", [f"{int(float(data['same', 8, size, 'copy']['prefix_tokens'])):,}" for size in sizes],
         [("same", 8, size) for size in sizes], "publish_ms", "Pause (ms, log)", 1.0, "{:.0f}", True,
         "Prefix (Tokens)")
    if have(MECH_COUNTS, "engine_kvmech_summary.csv"):
        data = mechanisms(MECH_COUNTS)
        counts = sorted({k[1] for k in data if k[0] == "same" and k[2] == max(sizes)})
    cells = [("same", n, max(sizes)) for n in counts]
    bars("eval_hand_attach", [str(n) for n in counts], cells, "child_attach_ms", "Attach Time (ms, log)",
         1.0, "{:.0f}", True, f"Agents ({long_tokens:,}-Token Prefix)")
    bars("eval_hand_ttft", [str(n) for n in counts], cells, "child_first_token_ms", "TTFT (s)",
         1000.0, "{:.2f}", False, f"Agents ({long_tokens:,}-Token Prefix)")


# ------------------------------------------------------------------- speed
def speed():
    cells = (("12 SMs", "20261006-engine-kvspeed-v1"), ("12 SMs", "20261006-engine-kvspeed-long-v1"),
             ("8 SMs", "20261006-engine-kvspeed-6sm-v1"), ("8 SMs", "20261006-engine-kvspeed-6sm-long-v1"))
    names, values, low, high = [], [], [], []
    for instance, directory in cells:
        row = next(r for r in rows(directory, "engine_kvspeed_summary.csv")
                   if r["mode"] == "extent_lazy" and r["processes"] == "1")
        names.append(f"{instance}\n{int(row['prefix_tokens']):,}")
        values.append(float(row["generation_vs_device"]))
        low.append(values[-1] - float(row["lower_ci95"]))
        high.append(float(row["upper_ci95"]) - values[-1])
    fig, ax = panel()
    bars = ax.bar(names, values, 0.6, color=OURS, edgecolor="white", linewidth=1.5,
                  yerr=[low, high], error_kw={"ecolor": INK, "elinewidth": 1.0, "capsize": 3})
    for bar, high_part in zip(bars, high):
        ax.annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width() / 2,
                                                 bar.get_height() + high_part),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=VALUE, color=INK)
    ax.axhline(1.0, color=AXIS, linewidth=1.0)
    finish(fig, ax, "eval_speed_host", "MIG Instance and Prefix (Tokens)",
           "Relative Speed", 1.18)


def reach():
    directory = "20261006-read-path-v1"
    if not have(directory, "read_path_summary.csv"):
        return
    data = {(r["instance"], r["kind"], r["size_mib"]): r
            for r in rows(directory, "read_path_summary.csv")}
    size = max({k[2] for k in data}, key=int)
    kinds = (("Device", DEVICE, "device"), ("Host, 2 MiB", OURS, "host_huge"),
             ("Host, 4 KiB", OURS_LIGHT, "host_small"))
    fig, ax = panel()
    series = [(label, colour, [float(data[instance, kind, size]["full_gib_per_s"])
                               for instance in ("12sm", "6sm")])
              for label, colour, kind in kinds]
    for group in grouped(ax, ["12 SMs", "8 SMs"], series, 0.27):
        labels(ax, group, "{:.0f}")
    ax.legend(loc="upper right", handlelength=1.2, fontsize=LEGEND)
    top = max(max(values) for _, _, values in series) * 1.45
    finish(fig, ax, "eval_speed_reach", "MIG Instance", "Bandwidth\n(GiB/s)", top)


# ------------------------------------------------------ GPU sharing modes
def mode():
    directory = "20261010-engine-kvmps-v2"
    if not have(directory, "engine_kvmps_summary.csv"):
        return
    data = {(r["config"], int(r["children"]), r["prefix_tokens"], r["mode"]): r
            for r in rows(directory, "engine_kvmps_summary.csv")}
    children = max(k[1] for k in data)
    tokens = max({k[2] for k in data}, key=int)
    configs = tuple(c for c in (("Time\nSlicing", "timeslice"), ("MPS", "mps"), ("MIG", "mig"),
                                ("MIG\n+MPS", "mig_mps")) if any(k[0] == c[1] for k in data))
    pair = (("Copy", COPY, "copy"), (NAME, OURS, "extent"))
    legend("eval_mode_legend", [(label, colour) for label, colour, _ in pair], 2)
    for name, column, ylabel, scale, form in (
            ("eval_mode_tps", "children_tps_sum", "Throughput (tokens/s)", 1.0, "{:.1f}"),
            ("eval_mode_mem", "children_memory_mib", "Memory (GiB)", 1024.0, "{:.1f}")):
        fig, ax = panel()
        series = [(label, colour, [float(data[config, children, tokens, m][column]) / scale
                                   for _, config in configs])
                  for label, colour, m in pair]
        for group in grouped(ax, [label for label, _ in configs], series, 0.38):
            labels(ax, group, form)
        for text in ax.texts:
            text.set_rotation(90)
        top = max(max(values) for _, _, values in series) * 1.3
        finish(fig, ax, name, "", ylabel, top)


# --------------------------------------------------------- copy-on-write
def cow():
    data = {(r["mode"], int(r["agents"])): r
            for r in rows("20261006-engine-kvcow-v1", "engine_kvcow_summary.csv")}
    # an ablation: the baseline is the mapping as the kernel provides it; the
    # next two bars add the CPU read pass of STATOR on each page size and keep
    # the writable private mapping (tints of STATOR: a part left out); the
    # last bar is STATOR with its extents
    order = (("CoW\nMap", "cow_noread", COW),
             (NAME + "\n(Read,\n4 KiB)", "cow_small", OURS_PALE),
             (NAME + "\n(Read,\n2 MiB)", "cow", OURS_LIGHT), (NAME, "extent_lazy", OURS))
    for name, column, ylabel, scale, form in (
            ("eval_cow_mem", "memory_mib", "Memory (GiB)", 1024.0, "{:.1f}"),
            ("eval_cow_time", "first_token_ms", "TTFT (ms)", 1.0, "{:,.0f}")):
        fig, ax = panel()
        values = [float(data[m, 8][column]) / scale for _, m, _ in order]
        bars = ax.bar([label for label, _, _ in order], values, 0.62,
                      color=[c for _, _, c in order], edgecolor="white", linewidth=1.5)
        labels(ax, bars, form)
        ax.tick_params(axis="x", labelsize=TICK2)
        finish(fig, ax, name, "", ylabel, max(values) * 1.18)


# ------------------------------------------- resident and released agents
def resident():
    """Agents that wait hold memory; the eight that generate keep their rate."""
    directory = "20261012-engine-kvresident-v1"
    if not have(directory, "engine_kvresident_summary.csv"):
        return
    data = rows(directory, "engine_kvresident_summary.csv")
    counts = [r["residents"] for r in data]
    for name, column, ylabel, scale, form in (
            ("eval_resident_mem", "memory_mib", "Memory (GiB)", 1024.0, "{:.1f}"),
            ("eval_resident_tps", "active_tps_sum", "Throughput (tokens/s)", 1.0, "{:.1f}")):
        fig, ax = panel()
        values = [float(r[column]) / scale for r in data]
        bars = ax.bar(counts, values, 0.62, color=OURS, edgecolor="white", linewidth=1.5)
        labels(ax, bars, form)
        finish(fig, ax, name, "Resident Agents", ylabel, max(values) * 1.18)


# --------------------------------------------- response time by placement
def latency():
    """One agent that must answer, with background agents in its MIG instance
    (time-sliced) or in the other one."""
    directory = "20261012-engine-kvlatency-v1"
    if not have(directory, "engine_kvlatency_summary.csv"):
        return
    data = {(r["critical_instance"], r["placement"], r["background"]): r
            for r in rows(directory, "engine_kvlatency_summary.csv")}
    counts = sorted({int(key[2]) for key in data if key[1] != "alone"})
    kinds = (("Same Instance", OURS_LIGHT, "shared"), ("Other Instance", OURS, "split"))
    legend("eval_latency_legend", [(label, colour) for label, colour, _ in kinds], 2)
    for name, instance in (("eval_latency_12sm", "12sm"), ("eval_latency_8sm", "8sm")):
        alone = float(data[instance, "alone", "0"]["generation_tps"])
        fig, ax = panel()
        series = [(label, colour, [alone] + [float(data[instance, placement, str(n)]["generation_tps"])
                                             for n in counts])
                  for label, colour, placement in kinds]
        for group in grouped(ax, ["0"] + [str(n) for n in counts], series, 0.38):
            labels(ax, group, "{:.1f}")
        for text in ax.texts:
            text.set_rotation(90)
        finish(fig, ax, name, "Background Agents", "Speed (tokens/s)", alone * 1.36)


# --------------------------------------------------------- device memory
def dev():
    data = {(r["paragraphs"], r["parent_mig"], r["mode"]): r
            for r in rows("20261006-engine-kvvmm-v1", "engine_kvvmm_summary.csv")}
    instance = "fafc828a"
    kinds = (("Copy", COPY, "copy"), ("Device", DEVICE, "vmm"), (NAME, OURS, "extent"))
    fig, ax = panel()
    series = [(label, colour, [float(data["320", instance, m]["publish_ms"]),
                               float(data["320", instance, m]["child_attach_ms"])])
              for label, colour, m in kinds]
    for group in grouped(ax, ["Parent Pause", "Child Attach"], series, 0.27):
        labels(ax, group, "{:.0f}")
    ax.set_ylim(1, 4000)
    ax.legend(loc="upper center", ncol=3, handlelength=1.0, columnspacing=0.8, fontsize=LEGEND,
              bbox_to_anchor=(0.5, 1.02))
    finish(fig, ax, "eval_dev_time", "", "Time (ms, log)", log=True)

    fig, ax = panel()
    values = [float(data["320", instance, m]["children_memory_mib"]) / 1024 for _, _, m in kinds]
    bars = ax.bar([label for label, _, _ in kinds], values, 0.6, color=[c for _, c, _ in kinds],
                  edgecolor="white", linewidth=1.5)
    labels(ax, bars)
    done = {m: sum(int(data["320", mig, m]["children_finished"]) for mig in ("fafc828a", "31ffbfe4"))
            for m in ("vmm_cross", "extent_cross")}
    started = {m: sum(int(data["320", mig, m]["children_started"]) for mig in ("fafc828a", "31ffbfe4"))
               for m in ("vmm_cross", "extent_cross")}
    ax.text(0.98, 0.97, "Across Instances\n"
            f"Device: {done['vmm_cross']}/{started['vmm_cross']}\n"
            f"{NAME}: {done['extent_cross']}/{started['extent_cross']}",
            transform=ax.transAxes, ha="right", va="top", fontsize=VALUE, color=INK,
            linespacing=1.2)
    finish(fig, ax, "eval_dev_mem", "", "Memory (GiB)", max(values) * 1.25)


# ------------------------------------------------- baselines in one engine
def sota():
    """Every mechanism with eight children, in the parent's instance and across instances."""
    if not have(MECH, "engine_kvmech_summary.csv"):
        return
    data = mechanisms()
    size = max(k[2] for k in data)
    children = 8
    legend("eval_sota_legend", [(label, colour) for label, colour, _ in MECHANISMS], 5)
    if any(k[0] == "cross" for k in data):
        placements = [(label, (key, children, size)) for label, key in
                      (("Same\nInstance", "same"), ("Across\nInstances", "cross"))
                      if (key, children, size, "copy") in data]
        xlabel = ""
    else:
        # the whole GPU: the two prefix lengths, as in the panel of the pause
        sizes = sorted({k[2] for k in data if k[0] == "same" and k[1] == children})
        placements = [(f"{int(float(data['same', children, s, 'copy']['prefix_tokens'])):,}",
                       ("same", children, s)) for s in sizes]
        xlabel = "Prefix (Tokens)"
        # 32 children on the longer prefix, up to the memory of Copy (10-15)
        extra = full("engine-kvmech-32")
        if extra is not None and have(extra, "engine_kvmech_summary.csv"):
            data.update(mechanisms(extra))
            placements = [(f"{label}\n8 Agents", cell) for label, cell in placements]
            placements.append((f"{placements[-1][0].split()[0]}\n32 Agents", ("same", 32, max(sizes))))
    for name, column, ylabel, scale, form, log in (
            ("eval_sota_mem", "children_memory_mib", "Memory (GiB)", 1024.0, "{:.1f}", False),
            ("eval_sota_attach", "child_attach_ms", "Attach Time (ms, log)", 1.0, "{:.0f}", True),
            ("eval_sota_ttft", "child_first_token_ms", "TTFT (s)", 1000.0, "{:.2f}", False),
            ("eval_sota_tps", "children_tps_sum", "Throughput (tokens/s)", 1.0, "{:.1f}", False)):
        fig, ax = panel()
        series = [(label, colour, [float(data[cell + (m,)][column]) / scale for _, cell in placements])
                  for label, colour, m in MECHANISMS]
        groups = grouped(ax, [label for label, _ in placements], series, 0.17)
        for (label, colour, m), group in zip(MECHANISMS, groups):
            for bar, (_, cell) in zip(group, placements):
                row = data[cell + (m,)]
                done, started = int(row["children_finished"]), int(row["children_started"])
                if done < started:
                    # Not every child completed: a value over the children that did
                    # would not compare with the other bars, so a cross stands in its place.
                    bar.set_height(0)
                    ax.annotate("\u00d7", (bar.get_x() + bar.get_width() / 2, 0), xytext=(0, 2),
                                textcoords="offset points", ha="center", va="bottom", fontsize=LEGEND,
                                color=colour, xycoords=("data", "axes fraction"))
                    continue
                tip(ax, bar, form.format(bar.get_height()))
                ax.texts[-1].set_fontsize(DENSE)
                ax.texts[-1].set_rotation(90 if log or name.endswith(("ttft", "tps")) else 0)
        top = max(bar.get_height() for group in groups for bar in group)
        if log:
            ax.set_ylim(1, top * 12)
        finish(fig, ax, name, xlabel, ylabel, None if log else top * 1.22, log)

    # the device-memory baseline by how far it is tuned, against the extents
    stages = (("Device", DEVICE_LIGHT, "device"), ("+ Batched", "#a18fe0", "device_tuned"),
              ("+ One Allocation", DEVICE, "device_merged"), (NAME, OURS, "extent"))
    fig, ax = panel()
    cell = ("same", children, size)
    series = [(label, colour, [float(data[cell + (m,)]["publish_ms"]), float(data[cell + (m,)]["child_attach_ms"])])
              for label, colour, m in stages]
    for group in grouped(ax, ["Parent Pause", "Child Attach"], series, 0.2):
        labels(ax, group, "{:.0f}")
    ax.set_ylim(1, max(max(v) for _, _, v in series) * 30)
    ax.legend(loc="upper left", handlelength=1.1, fontsize=LEGEND, labelspacing=0.25, ncol=2,
              columnspacing=0.9)
    finish(fig, ax, "eval_dev_tune", "", "Time (ms, log)", log=True)


def attach():
    directory = "20261006-vmm-attach-v1"
    if not have(directory, "vmm_attach_summary.csv"):
        return
    data = [r for r in rows(directory, "vmm_attach_summary.csv") if int(r["runs"]) > 0]
    fig, ax = panel()
    sizes = [int(r["granule_mib"]) for r in data]
    ax.plot(sizes, [float(r["attach_ms"]) for r in data], color=DEVICE, marker="s", markersize=6,
            linewidth=2, markeredgecolor="white", markeredgewidth=1.0, label="Device: Import and Map")
    times = [float(r["attach_ms"]) for r in data]
    for index in (0, len(data) - 1):
        ax.annotate(f"{times[index]:.1f} ms\n{data[index]['handles']} handle" + ("s" if index == 0 else ""),
                    (sizes[index], times[index]), xytext=(8 if index == 0 else -6, 4 if index == 0 else 10),
                    textcoords="offset points", ha="left" if index == 0 else "right", va="bottom",
                    fontsize=VALUE, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    # every second size is named; the marks of the others stand between them
    ax.set_xticks(sizes, [str(v) if index % 2 == 0 else "" for index, v in enumerate(sizes)])
    ax.set_yticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
    ax.minorticks_off()
    ax.set_ylim(1, 60)
    finish(fig, ax, "eval_attach_granule", "Allocation Size (MiB)", "Attach (ms, log)", log=True)


def protect():
    """Writes through the GPU that changed state shared with another process."""
    directory = "20261006-protect-v1"
    path = os.path.join(RES, base.resolve(directory), "protect_summary.csv")
    if not os.path.exists(path):
        return
    lines_of = [line.strip().split(",") for line in open(path) if line.strip()]
    host = dict(zip(lines_of[0], lines_of[1]))
    vmm = dict(zip(lines_of[2], lines_of[3]))
    changed = [(f"{NAME}\n(Host Mapping)", int(host["attempts"]) - int(host["shared_state_intact"]),
                int(host["attempts"]), OURS),
               ("Device\n(CUDA VMM)", int(vmm["attempts"]) - int(vmm["exporter_state_intact"]),
                int(vmm["attempts"]), DEVICE)]
    fig, ax = panel()
    bars = ax.bar([label for label, _, _, _ in changed],
                  [100.0 * count / total for _, count, total, _ in changed], 0.5,
                  color=[colour for _, _, _, colour in changed], edgecolor="white", linewidth=1.5)
    for bar, (_, count, total, _) in zip(bars, changed):
        tip(ax, bar, f"{count} of {total}")
    finish(fig, ax, "eval_protect", "", "Shared State\nChanged (%)", 125)


# ------------------------------------------------------------ agent tree
def tree():
    directory = "20261010-engine-kvtree-v2"
    if not have(directory, "engine_kvtree_summary.csv"):
        return
    data = {r["mode"]: r for r in rows(directory, "engine_kvtree_summary.csv")}
    order = (("Copy", "copy", COPY), (f"{NAME}\n(One Level)", "flat", OURS_LIGHT),
             (f"{NAME}\n(Chains)", "chain", OURS))
    fig, ax = panel()
    values = [float(data[m]["tree_memory_mib"]) / 1024 for _, m, _ in order]
    bars = ax.bar([label for label, _, _ in order], values, 0.6, color=[c for _, _, c in order],
                  edgecolor="white", linewidth=1.5)
    labels(ax, bars)
    finish(fig, ax, "eval_tree_mem", "", "Memory (GiB)", max(values) * 1.18)

    fig, ax = panel()
    values = [float(data[m]["leaf_first_token_ms"]) / 1000 for _, m, _ in order]
    bars = ax.bar([label for label, _, _ in order], values, 0.6, color=[c for _, _, c in order],
                  edgecolor="white", linewidth=1.5)
    labels(ax, bars, "{:.2f}")
    finish(fig, ax, "eval_tree_time", "", "TTFT (s)", max(values) * 1.18)


# ------------------------------------------------------------ sensitivity
def lines(name, directory, summary, column, ylabel, scale, form, replace=None):
    data = {(r["mode"], int(r["agents"])): r for r in rows(directory, summary)}
    # A cell that was measured again takes the place of the first measurement.
    if replace is not None and have(replace, summary):
        data.update({(r["mode"], int(r["agents"])): r for r in rows(replace, summary)})
    counts = sorted({k[1] for k in data})
    top = max(float(r[column]) for r in data.values()) / scale * 1.2
    series = (("Copy", COPY, "restore", "s"), (NAME, OURS, "extent_lazy", "o"))
    last = {m: float(data[m, max(n for n in counts if (m, n) in data)][column]) / scale for _, _, m, _ in series}
    fig, ax = panel()
    for label, colour, m, marker in series:
        points = [(n, float(data[m, n][column]) / scale) for n in counts if (m, n) in data]
        ax.plot([p[0] for p in points], [p[1] for p in points], color=colour, marker=marker,
                markersize=7, linewidth=2, label=label, markeredgecolor="white", markeredgewidth=1.2)
        # the value of the lower line stands below it when the two lines end close together
        below = last[m] == min(last.values()) and max(last.values()) - min(last.values()) < 0.2 * top
        ax.annotate(form.format(points[-1][1]), points[-1], xytext=(-4, -8 if below else 7),
                    textcoords="offset points", ha="right", va="top" if below else "baseline",
                    fontsize=VALUE, color=INK)
    ax.set_xticks(counts)
    finish(fig, ax, name, "Agents (16,321-Token Prefix)", ylabel, top)


def scale():
    directory = "20261006-engine-kvscale-v1"
    if not have(directory, "engine_kvscale_summary.csv"):
        return
    # The 32-agent cell is that of the rerun, in which no repetition stalled.
    rerun = "20261007-engine-kvscale-32-v2"
    lines("eval_scale_mem", directory, "engine_kvscale_summary.csv", "memory_mib",
          "Memory (GiB)", 1024.0, "{:.1f}", replace=rerun)
    lines("eval_scale_tps", directory, "engine_kvscale_summary.csv", "generation_tps_total",
          "Throughput (tokens/s)", 1.0, "{:.1f}", replace=rerun)


def prefix_length():
    """Eight agents by the length of the prefix they start from."""
    # 1k and 30k tokens are campaigns of their own; 4k and 16k are the cells
    # with eight agents of the sharing and of the scale campaign
    cells = [(d, f) for d, f in (("20261007-engine-kvscale-p20-v1", "engine_kvscale_summary.csv"),
                                 ("20261006-engine-kvshare-v1", "engine_kvshare_summary.csv"),
                                 ("20261006-engine-kvscale-v1", "engine_kvscale_summary.csv"),
                                 ("20261007-engine-kvscale-p600-v1", "engine_kvscale_summary.csv"))
             if have(d, f)]
    if len(cells) < 4:
        return
    points = {}
    for directory, summary in cells:
        for r in rows(directory, summary):
            if int(r["agents"]) == 8 and r["mode"] in ("restore", "extent_lazy"):
                points.setdefault((int(float(r["prefix_tokens"])), r["mode"]), r)
    tokens = sorted({k[0] for k in points})
    for name, column, ylabel, scale, form in (
            ("eval_prefix_mem", "memory_mib", "Memory (GiB)", 1024.0, "{:.1f}"),
            ("eval_prefix_attach", "attach_ms", "Attach Time (ms)", 1.0, "{:.0f}")):
        fig, ax = panel()
        for label, colour, m, marker in (("Copy", COPY, "restore", "s"), (NAME, OURS, "extent_lazy", "o")):
            values = [float(points[t, m][column]) / scale for t in tokens]
            ax.plot(range(len(tokens)), values, color=colour, marker=marker, markersize=7, linewidth=2,
                    label=label, markeredgecolor="white", markeredgewidth=1.2)
            ax.annotate(form.format(values[-1]), (len(tokens) - 1, values[-1]), xytext=(-4, 7),
                        textcoords="offset points", ha="right", fontsize=VALUE, color=INK)
        ax.set_xticks(range(len(tokens)), [f"{t / 1000:.0f}k" for t in tokens])
        ax.legend(loc="upper left", handlelength=1.6)
        finish(fig, ax, name, "Prefix (Tokens)", ylabel,
               max(float(r[column]) for r in points.values()) / scale * 1.25)


def variants():
    """Eight agents: other models and the agent workload, memory by copy and by extents."""
    cells = (("Qwen\n7B", "20261006-engine-kvshare-v1", "engine_kvshare_summary.csv", "320"),
             ("Llama\n8B", "20261006-engine-kvscale-llama8b-v1", "engine_kvscale_summary.csv", None),
             ("Qwen\n14B", "20261006-engine-kvscale-qwen14b-v1", "engine_kvscale_summary.csv", None),
             ("Agent\nWorkload", "20261006-engine-kvscale-agent-v1", "engine_kvscale_summary.csv", None))
    names, copy, ours = [], [], []
    for label, directory, summary, size in cells:
        if not have(directory, summary):
            continue
        data = {(r["mode"], int(r["agents"])): r for r in rows(directory, summary)
                if size is None or r["paragraphs"] == size}
        names.append(label)
        copy.append(float(data["restore", 8]["memory_mib"]) / 1024)
        ours.append(float(data["extent_lazy", 8]["memory_mib"]) / 1024)
    if len(names) < 2:
        return
    fig, ax = panel()
    for group in grouped(ax, names, [("Copy", COPY, copy), (NAME, OURS, ours)], 0.38):
        labels(ax, group)
    ax.tick_params(axis="x", labelsize=TICK2)
    finish(fig, ax, "eval_model_mem", "", "Memory (GiB)", max(copy) * 1.18)


def inproc():
    """Eight agents: separate processes, batching servers, and Ollama."""
    share = {(r["paragraphs"], r["mode"], int(r["agents"])): r
             for r in rows("20261006-engine-kvshare-v1", "engine_kvshare_summary.csv")}
    publish = {(r["paragraphs"], r["store"]): r
               for r in rows("20261006-engine-kvshare-v1", "engine_kvshare_publish.csv")}
    batch = {r["config"]: r for r in rows("20261010-engine-kvbatch-v2", "engine_kvbatch_summary.csv")}
    file_gib = float(publish["320", "huge"]["file_used_mib"]) / 1024
    gib = lambda row, key="memory_mib": float(row[key]) / 1024
    # (label, memory GiB, tokens/s, marker, colour, label offset in points, alignment)
    points = [
        ("8 Processes", gib(share["320", "restore", 8]),
         float(share["320", "restore", 8]["generation_tps_total"]), "o", COPY, (10, -4), "left"),
        ("8 Processes", gib(share["320", "extent_lazy", 8]) + file_gib,
         float(share["320", "extent_lazy", 8]["generation_tps_total"]), "o", OURS, (-10, -4), "right"),
        ("1 Server", gib(batch["one_server"]), float(batch["one_server"]["generation_tps_total"]),
         "s", COPY, (0, -17), "center"),
        ("2 Servers", gib(batch["two_servers_copy"]), float(batch["two_servers_copy"]["generation_tps_total"]),
         "^", COPY, (9, -4), "left"),
        ("2 Servers", gib(batch["two_servers_extent"]), float(batch["two_servers_extent"]["generation_tps_total"]),
         "^", OURS, (0, 9), "center"),
    ]
    directory = "20261006-ollama-agents-v1"
    if have(directory, "ollama_agents_summary.csv"):
        ollama = {(r["config"], r["round"]): r for r in rows(directory, "ollama_agents_summary.csv")}
        points += [
            ("1 Server", gib(ollama["one", "warm"]), float(ollama["one", "warm"]["generation_tps_sum"]),
             "s", DEMAND, (0, 9), "center"),
            ("2 Servers", gib(ollama["two", "warm"]), float(ollama["two", "warm"]["generation_tps_sum"]),
             "^", DEMAND, (0, 9), "center"),
        ]
    # vLLM is another engine with other weights; the paper reports it in the
    # text, and the figure shows it only on request (INPROC_VLLM=1)
    vllm = next((d for d in ("20261007-ext-vllm-synth-v1", "20261007-ext-vllm-v1")
                 if have(d, "ext_vllm_summary.csv")), None) if os.environ.get("INPROC_VLLM") == "1" else None
    if vllm is not None:
        warm = {r["round"]: r for r in rows(vllm, "ext_vllm_summary.csv")}["warm"]
        points.append(("1 Server", gib(warm, "peak_memory_mib"), float(warm["agents_tps_sum"]),
                       "s", DEVICE, (11, -4), "left"))
    fig, ax = plt.subplots(figsize=(6.4, 2.0))
    for index, (label, memory, speed, marker, colour, offset, align) in enumerate(points):
        if index == 2:
            # one batching server lies next to two servers on STATOR: a hollow,
            # larger square keeps the triangle inside it visible
            ax.scatter([memory], [speed], s=300, marker=marker, facecolors="none", edgecolors=colour,
                       linewidth=2.0, zorder=4)
            ax.annotate(label, (memory, speed), xytext=(0, -21), textcoords="offset points",
                        ha="center", va="center", fontsize=VALUE,
                        arrowprops=dict(arrowstyle="-", color=INK, lw=0.7, shrinkA=1, shrinkB=9))
            continue
        ax.scatter([memory], [speed], s=130, marker=marker, color=colour, edgecolor="white",
                   linewidth=1.5, zorder=3)
        if index == 4:
            ax.annotate(label, (memory, speed), xytext=(0, 19), textcoords="offset points",
                        ha="center", va="center", fontsize=VALUE,
                        arrowprops=dict(arrowstyle="-", color=INK, lw=0.7, shrinkA=1, shrinkB=6))
            continue
        ax.annotate(label, (memory, speed), xytext=offset, textcoords="offset points", ha=align,
                    fontsize=VALUE)
    # Copy to STATOR within one deployment: the same throughput in less memory
    for copy, ours in ((points[0], points[1]), (points[3], points[4])):
        ax.annotate("", xy=(ours[1], ours[2]), xytext=(copy[1], copy[2]),
                    arrowprops=dict(arrowstyle="-|>", color=OURS, lw=1.6, mutation_scale=12,
                                    shrinkA=7, shrinkB=7), zorder=2)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=colour, markersize=9,
                          markeredgecolor="white", label=label)
               for label, colour in (("llama.cpp, Copy", COPY), (f"llama.cpp, {NAME}", OURS), ("Ollama", DEMAND))
               + ((("vLLM", DEVICE),) if vllm is not None else ())]
    if vllm is not None:
        ax.legend(handles=handles, loc="center right", handlelength=1.0, ncol=2, columnspacing=1.0,
                  bbox_to_anchor=(1.0, 0.56))
    else:
        ax.legend(handles=handles, loc="upper right", handlelength=1.0)
    ax.set_xlim(0, 27)
    ax.set_ylim(0, 190 if vllm is not None else 160)
    ax.set_xlabel("Memory (GiB)")
    ax.set_ylabel("Throughput (tokens/s)")
    style(ax, "both")
    fig.tight_layout()
    save(fig, "eval_proc")


def server():
    # The second campaign publishes at a token boundary; the first, which did
    # not, is kept in the record.
    directory = "20261010-engine-kvserver2-v2"
    if not have(directory, "engine_kvserver_summary.csv"):
        directory = "20261006-engine-kvserver-v1"
    if not have(directory, "engine_kvserver_summary.csv"):
        return
    data = {(r["mode"], int(r["agents"])): r for r in rows(directory, "engine_kvserver_summary.csv")}
    pair = (("Copy", COPY, "copy"), (NAME, OURS, "extent"))
    most = max(k[1] for k in data)
    fig, ax = panel()
    steps = (("Slot\nSave", "save_request_ms"), ("Slot\nRestore", "restore_request_ms"),
             ("First\nToken", "first_token_ms"))
    series = [(label, colour, [float(data[m, most][column]) for _, column in steps])
              for label, colour, m in pair]
    for group in grouped(ax, [label for label, _ in steps], series, 0.36):
        labels(ax, group, "{:.0f}")
    ax.set_ylim(1, 3000)
    finish(fig, ax, "eval_server_time", "", "Time (ms, log)", log=True)

    counts = sorted({k[1] for k in data})
    fig, ax = panel()
    series = [(label, colour, [float(data[m, n]["agents_memory_mib"]) / 1024 for n in counts])
              for label, colour, m in pair]
    for group in grouped(ax, [str(n) for n in counts], series, 0.36):
        labels(ax, group)
    finish(fig, ax, "eval_server_mem", "Agent Servers", "Memory (GiB)",
           max(max(values) for _, _, values in series) * 1.18)


def limit():
    directory = "20261006-engine-kvlimit-v1"
    if not have(directory, "engine_kvlimit_summary.csv"):
        return
    data = {(r["phase"], r["mode"]): r for r in rows(directory, "engine_kvlimit_summary.csv")}
    modes = (("Copy", COPY, "restore"), (NAME, OURS, "extent_lazy"))
    # what four agents hold, and what the kernel charges to their control groups
    fig, ax = panel()
    x = np.arange(len(modes))
    held = [float(data["account", m]["mem_available_drop_mib"]) / 1024 for _, _, m in modes]
    charged = [float(data["account", m]["charged_peak_sum_mib"]) / 1024 for _, _, m in modes]
    first = ax.bar(x - 0.19, held, 0.36, color=[c for _, c, _ in modes], edgecolor="white", linewidth=1.5)
    second = ax.bar(x + 0.19, charged, 0.36, color=[c for _, c, _ in modes], edgecolor="white",
                    linewidth=1.5, hatch="////", alpha=0.55)
    labels(ax, first)
    labels(ax, second)
    ax.set_xticks(x, [label for label, _, _ in modes])
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=COPY),
               plt.Rectangle((0, 0), 1, 1, facecolor=COPY, hatch="////", alpha=0.55, edgecolor="white")],
              ["Held", "Charged"], loc="upper right", handlelength=1.4)
    finish(fig, ax, "eval_limit_charge", "", "Memory (GiB)", max(held) * 1.2)

    # agent 0 receives a long task under a limit
    fig, ax = panel()
    cap = float(data["limit", "extent_lazy"]["limit_mib"])
    peaks = [float(data["limit", m]["agent0_peak_mib"]) for _, _, m in modes]
    names = []
    for label, _, m in modes:
        row = data["limit", m]
        done, runs = int(row["agent0_completed"]), int(row["runs"])
        outcome = f"completes {done}/{runs}" if done else f"ended {row['agent0_oom_kills']}/{runs}"
        names.append(f"{label}\n({outcome})")
    bars = ax.bar(names, peaks, 0.5, color=[c for _, c, _ in modes], edgecolor="white", linewidth=1.5)
    labels(ax, bars, "{:.0f}")
    ax.tick_params(axis="x", labelsize=TICK2)
    ax.axhline(cap, color=INK, linewidth=1.2, linestyle=(0, (4, 2)))
    ax.text(0.03, cap * 1.04, f"limit: {cap:.0f} MiB", ha="left", va="bottom", fontsize=VALUE,
            transform=ax.get_yaxis_transform())
    finish(fig, ax, "eval_limit_peak", "", "Peak Charge (MiB)", cap * 1.6)


# ------------------------------------------------------------ whole stack
# The two stacks between the existing engine and STATOR apply one of its two
# mechanisms; they are parts of STATOR and carry its colour in a lighter tint.
STACKS = (("Existing Engine", EXISTING, "none", "s"), (f"{NAME} (KV)", OURS_PALE, "kv", "D"),
          (f"{NAME} (Weights)", OURS_LIGHT, "weights", "^"), (NAME, OURS, "both", "o"))


def ran(directory, mode):
    """The largest count of a stack in a campaign in which none of its agents failed."""
    return max(int(r["agents"]) for r in rows(directory, "engine_kvstack_summary.csv")
               if r["mode"] == mode and int(r["runs"]) > 0 and int(r["failed_agents"]) == 0)


def stack():
    """The memory of the whole serving stack by what the agents share."""
    directory = "20261007-engine-kvstack-v1"
    if not have(directory, "engine_kvstack_summary.csv"):
        return
    data = {(r["mode"], int(r["agents"])): r
            for r in rows(directory, "engine_kvstack_summary.csv") if int(r["runs"]) > 0}
    fit = {r["mode"]: r for r in rows(directory, "engine_kvstack_fit.csv")}
    legend("eval_stack_legend", [(label, colour) for label, colour, _, _ in STACKS], 4)
    # the larger counts of the stack that shares both, run as a campaign of their own
    line = dict(data)
    for more in ("20261008-engine-kvstack-capacity-v2",
                 "20261008-engine-kvstack-capacity-136-v1",
                 "20261008-engine-kvstack-capacity-140-v1",
                 "20261009-engine-kvstack-capacity-copy-v1"):
        if have(more, "engine_kvstack_summary.csv"):
            line.update({(r["mode"], int(r["agents"])): r
                         for r in rows(more, "engine_kvstack_summary.csv") if int(r["runs"]) > 0})

    # memory against the number of agents; the line of the fit goes on to the
    # memory that was available, where the stack stops fitting
    fig, ax = panel()
    room = float(fit["none"]["memory_available_mib"]) / 1024
    for label, colour, m, marker in STACKS:
        counts = sorted(n for mode, n in line if mode == m)
        shared = float(fit[m]["model_file_shared_mib"]) + float(fit[m]["cache_file_shared_mib"])
        # a marker in a light tint has an outline in the colour of STATOR, so that it shows on white
        ax.plot(counts, [(float(line[m, n]["memory_mib"]) + shared) / 1024 for n in counts],
                color=colour, marker=marker, markersize=5.5, linewidth=2,
                markeredgecolor=OURS if colour in (OURS_PALE, OURS_LIGHT) else "white",
                markeredgewidth=0.9, label=label)
    ax.axhline(room, color=INK, linewidth=1.1, linestyle=(0, (4, 2)))
    ax.text(0.03, room * 1.015, f"Available: {room:.0f} GiB", ha="left", va="bottom", fontsize=VALUE,
            color=INK, transform=ax.get_yaxis_transform())
    ax.set_xscale("log", base=2)
    ticks = [1, 4, 16, 64, 256]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(0.8, 256)
    finish(fig, ax, "eval_stack_mem", "Agents (log)", "Memory (GiB)", room * 1.16)

    # what the memory of 8 agents consists of
    count = 8
    none, weights_only, kv_only, both = (float(data[m, count]["memory_mib"]) / 1024
                                         for m in ("none", "weights", "kv", "both"))
    model_file = float(fit["both"]["model_file_shared_mib"]) / 1024
    cache_file = float(fit["both"]["cache_file_shared_mib"]) / 1024
    # greys for what every agent holds for itself, teal for what is held once
    parts = (("Weight Copies", COPY, (none - weights_only, kv_only - both, 0.0, 0.0)),
             ("KV Copies", COPY_LIGHT, (weights_only - both, 0.0, weights_only - both, 0.0)),
             ("Agent State", COPY_PALE, (both, both, both, both)),
             ("Shared Files", OURS, (0.0, cache_file, model_file, model_file + cache_file)))
    fig, ax = panel()
    names = ["Existing\nEngine", f"{NAME}\n(KV)", f"{NAME}\n(Weights)", NAME]
    bottom = np.zeros(4)
    for label, colour, values in parts:
        ax.bar(names, values, 0.54, bottom=bottom, color=colour, edgecolor="white", linewidth=1.5,
               label=label)
        bottom += np.array(values)
    for x, total in enumerate(bottom):
        ax.annotate(f"{total:.1f}", (x, total), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=VALUE, color=INK)
    ax.legend(loc="upper right", handlelength=1.1, fontsize=LEGEND, labelspacing=0.3)
    ax.tick_params(axis="x", labelsize=TICK2 - 1)
    ax.set_xlim(-0.32, len(names) - 0.68)    # the names of two lines need the whole width of the axis
    finish(fig, ax, "eval_stack_parts", "", "Memory (GiB)", bottom.max() * 1.18)

    # how many agents fit by the line of the unmodified stack and of the stack
    # that shares both, for this model and for one with weights twice as large
    larger = "20261008-engine-kvstack-qwen14b-v1"
    models = [("Qwen2.5-7B", fit)]
    if have(larger, "engine_kvstack_fit.csv"):
        models.append(("Qwen2.5-14B", {r["mode"]: r for r in rows(larger, "engine_kvstack_fit.csv")}))
    fig, ax = panel()
    series = [(label, colour, [int(model[m]["agents_that_fit_by_line"]) for _, model in models])
              for label, colour, m, _ in STACKS if m in ("none", "both")]
    # every stack at the count that it runs while 16 GiB remain unused; the model
    # file that the agents map counts as used, which the larger counts of the
    # campaigns -max-v1 and -qwen14b-131-v1 do not satisfy
    measured = {"none": ("20261009-engine-kvstack-capacity-copy-v1", "20261008-engine-kvstack-qwen14b-none8-v1"),
                "both": ("20261008-engine-kvstack-capacity-140-v1", "20261008-engine-kvstack-capacity-qwen14b-118-v1")}
    series = [(label, colour, [ran(d, m) for d in measured[m]][:len(models)])
              for label, colour, m, _ in STACKS if m in measured]
    for group in grouped(ax, [label for label, _ in models], series, 0.36):
        labels(ax, group, "{:.0f}")
    finish(fig, ax, "eval_stack_fit", "", "Agents", max(max(v) for _, _, v in series) * 1.18)

    # generation speed of eight agents relative to the unmodified stack, with
    # the model file in the page cache (4 KiB pages) and on a tmpfs with 2 MiB
    # pages; a stack that copies the weights does not depend on the page size
    huge_dir = "20261007-engine-kvstack-huge-v1"
    huge = ({(r["mode"], int(r["agents"])): r
             for r in rows(huge_dir, "engine_kvstack_summary.csv") if int(r["runs"]) > 0}
            if have(huge_dir, "engine_kvstack_summary.csv") else {})
    fig, ax = panel()
    shown = [(f"{NAME}\n(KV)", "kv", OURS, OURS_LIGHT), (f"{NAME}\n(Weights)", "weights", OURS, OURS_LIGHT),
             (NAME, "both", OURS, OURS_LIGHT)]
    width = 0.36
    for x, (label, m, full, light) in enumerate(shown):
        cells = [(data, light, -width / 2)] + ([(huge, full, width / 2)] if (m, count) in huge else [])
        if len(cells) == 1:
            cells = [(data, full, 0.0)]
        for source, colour, offset in cells:
            row = source[m, count]
            value = float(row["vs_none"])
            bar = ax.bar(x + offset, value, width, color=colour, edgecolor="white", linewidth=1.5,
                         yerr=[[value - float(row["vs_none_low"])], [float(row["vs_none_high"]) - value]],
                         error_kw={"ecolor": INK, "elinewidth": 1.0, "capsize": 2.5})[0]
            ax.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width() / 2, float(row["vs_none_high"])),
                        xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                        fontsize=VALUE, color=INK)
    ax.set_xticks(range(len(shown)), [label for label, _, _, _ in shown])
    ax.tick_params(axis="x", labelsize=TICK2 - 1)   # as in the panel of the parts
    ax.axhline(1.0, color=AXIS, linewidth=1.0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=OURS_LIGHT), plt.Rectangle((0, 0), 1, 1, color=OURS)]
    ax.legend(handles, ["4 KiB Pages", "2 MiB Pages"], loc="upper center", ncol=2,
              handlelength=1.1, fontsize=LEGEND, columnspacing=1.2)
    finish(fig, ax, "eval_stack_speed", "", "Relative\nSpeed", 1.42)


# ------------------------------------------------- loss in the 8-SM instance
def local():
    """Where the loss of speed with host memory comes from, by MIG instance."""
    cells = (("12-SM\nInstance", "20261007-engine-kvlocal-12sm-v1"), ("8-SM\nInstance", "20261007-engine-kvlocal-6sm-v1"))
    if base.resolve(cells[0][1]) != cells[0][1]:
        # the rerun on the whole GPU, which has no instances
        cells = (("Whole\nGPU", cells[0][1]),)
    if all(have(d, "engine_kvmech_summary.csv") for _, d in cells):
        # A private cache in host memory is a copy (grey, lighter on 4 KiB
        # pages). The two variants of the extent page sizes (grow_all,
        # small_pages) equal extent within 0.003; the record has their rows.
        forms = (("Private 2 MiB", COPY, "host_copy_huge"), ("Private 4 KiB", COPY_LIGHT, "host_copy"),
                 (NAME, OURS, "extent"))
        data = {label: {r["mode"]: r for r in rows(d, "engine_kvmech_summary.csv")} for label, d in cells}
        fig, ax = panel()
        width = 0.27
        # generation-window throughput, paired within a repetition (10-20)
        from make_paper_figures_1020 import windows
        T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}
        def window_ratio(directory, m):
            rates = windows(base.resolve(directory), "CHILD", ("mode",))
            r = [rates[(m,)][k] / rates[("copy",)][k] for k in rates[(m,)] if k in rates[("copy",)]]
            g = [math.log(x) for x in r]
            mu = statistics.mean(g)
            h = T95.get(len(g), 1.96) * statistics.stdev(g) / math.sqrt(len(g)) if len(g) > 1 else 0
            return math.exp(mu), math.exp(mu - h), math.exp(mu + h)
        for index, (label, colour, m) in enumerate(forms):
            trio = [window_ratio(d, m) for _, d in cells]
            values = [t[0] for t in trio]
            low = [t[0] - t[1] for t in trio]
            high = [t[2] - t[0] for t in trio]
            x = np.arange(len(cells)) + (index - 1) * width
            ax.bar(x, values, width, color=colour, edgecolor="white", linewidth=1.5, label=label,
                   yerr=[low, high], error_kw={"ecolor": INK, "elinewidth": 1.0, "capsize": 2})
            for position, value in zip(x, values):
                ax.annotate(f"{value:.3f}", (position, value), xytext=(0, 5), textcoords="offset points",
                            ha="center", va="bottom", fontsize=DENSE, color=INK, rotation=90)
        ax.set_xticks(range(len(cells)), [cell for cell, _ in cells])
        ax.tick_params(axis="x", labelsize=TICK2)
        ax.axhline(1.0, color=AXIS, linewidth=1.0)
        ax.set_ylim(0.8, 1.2)
        ax.set_yticks([0.8, 0.9, 1.0, 1.1])
        # the legends of (a) and (b) are strips above their panels, as in the other figures (10-23)
        legend("eval_local_speed_legend", [(label, colour) for label, colour, _ in forms], 2, half=True)
        ax.set_ylabel("Speed vs. Device")
        style(ax)
        fig.tight_layout()
        save(fig, "eval_local_speed")

    directory = "20261007-tlb-probe-v1"
    if have(directory, "tlb_probe_summary.csv"):
        data = rows(directory, "tlb_probe_summary.csv")
        size = max(int(r["size_mib"]) for r in data)
        stride = max(int(r["stride_kib"]) for r in data)
        cell = {(int(r["sms"]), r["kind"]): r for r in data
                if int(r["size_mib"]) == size and int(r["stride_kib"]) == stride}
        sms = sorted({k[0] for k in cell}, reverse=True)
        groups, order = [], []
        for kernel, name in (("gather", "Read"), ("scatter", "Write")):
            for count in sms:
                label = f"{count} SM"
                groups.append(f"{name}\n{label}")
                order.append((kernel, count))
        series = [(label, colour, [float(cell[count, kind][f"{kernel}_vs_device"]) for kernel, count in order])
                  for label, colour, kind in (("Host, 2 MiB Pages", OURS, "host_huge"),
                                               ("Host, 4 KiB Pages", OURS_LIGHT, "host_small"))]
        fig, ax = panel()
        for group in grouped(ax, groups, series, 0.36):
            labels(ax, group, "{:.2f}")
        ax.axhline(1.0, color=AXIS, linewidth=1.0)
        ax.tick_params(axis="x", labelsize=TICK2)
        legend("eval_local_probe_legend", [("2 MiB Pages", OURS), ("4 KiB Pages", OURS_LIGHT)], 2, half=True)
        finish(fig, ax, "eval_local_probe", "", "Rate vs. Device", 1.45)


# ------------------------------------------------------------- the pipeline
def pipe():
    """A root, leaders and leaves on the tools of a public benchmark."""
    directory = "20261010-engine-kvpipe-huge-v2"
    if not have(directory, "engine_kvpipe_summary.csv"):
        return
    data = {r["mode"]: r for r in rows(directory, "engine_kvpipe_summary.csv")}
    stacks = [(label, m, colour) for label, m, colour in (("Existing\nEngine", "stock", COPY), ("Copy", "copy", COPY_LIGHT),
                                                           (NAME, "chain", OURS)) if m in data]
    names = [label for label, _, _ in stacks]
    phases = (("Prefix", "prefix", COPY_PALE), ("Leaders", "leaders", COPY_LIGHT), ("Leaves", "workers", COPY))

    def stacked(name, column, ylabel, scale, form):
        fig, ax = panel()
        bottom = np.zeros(len(stacks))
        for label, key, colour in phases:
            values = np.array([float(data[m][column.format(key)]) / scale for _, m, _ in stacks])
            ax.bar(names, values, 0.55, bottom=bottom, color=colour, edgecolor="white", linewidth=1.5,
                   label=label)
            bottom += values
        for x, total in enumerate(bottom):
            ax.annotate(form.format(total), (x, total), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=VALUE, color=INK)
        ax.legend(loc="upper center", ncol=3, handlelength=1.0, fontsize=LEGEND, columnspacing=1.0)
        finish(fig, ax, name, "", ylabel, bottom.max() * 1.36)

    stacked("eval_pipe_time", "{}_s", "Time (s)", 1.0, "{:.1f}")
    stacked("eval_pipe_energy", "vin_{}_j", "Energy (kJ)", 1000.0, "{:.2f}")
    fig, ax = panel()
    # The drop is taken from before the root starts, so the cache files are in it.
    memory = [float(data[m]["memory_mib"]) / 1024 for _, m, _ in stacks]
    bars = ax.bar(names, memory, 0.55, color=[colour for _, _, colour in stacks], edgecolor="white",
                  linewidth=1.5)
    labels(ax, bars)
    finish(fig, ax, "eval_pipe_mem", "", "Memory (GiB)", max(memory) * 1.18)


# ---------------------------------------------------- deeper and wider trees
def deep():
    cells = [(label, d) for label, d in (("Four Levels\n(23 Processes)", "20261007-engine-kvdeep-v1"),
                                         ("16 Leaves\n(18 Processes)", "20261007-engine-kvdeep-wide-v1"))
             if have(d, "engine_kvdeep_summary.csv")]
    if not cells:
        return
    data = {label: {r["mode"]: r for r in rows(d, "engine_kvdeep_summary.csv")} for label, d in cells}
    for name, column, ylabel, scale, form in (
            ("eval_deep_mem", "tree_memory_mib", "Memory (GiB)", 1024.0, "{:.1f}"),
            ("eval_deep_ttft", "leaf_first_token_ms", "TTFT of a Leaf (s)", 1000.0, "{:.2f}")):
        fig, ax = panel()
        series = [(label, colour, [(float(data[cell][m][column])
                                    + (float(data[cell][m]["files_attached_mib"]) if column == "tree_memory_mib" else 0.0))
                                   / scale for cell, _ in cells])
                  for label, colour, m in (("Copy", COPY, "copy"), (NAME, OURS, "chain"))]
        for group in grouped(ax, [cell for cell, _ in cells], series, 0.36):
            labels(ax, group, form)
        ax.legend(loc="upper right", handlelength=1.1, fontsize=LEGEND)
        ax.tick_params(axis="x", labelsize=TICK2)
        finish(fig, ax, name, "", ylabel, max(max(v) for _, _, v in series) * 1.2)

    # the pages of the segment files while the subtrees of one tree leave
    timeline = os.path.join(RES, cells[0][1], "timeline.chain.1")
    if cells[0][1].endswith("kvdeep-v1") and os.path.exists(timeline):
        points = [[float(v) for v in line.split()] for line in open(timeline) if line.strip()]
        seconds = [p[0] / 1000 for p in points]
        fig, ax = panel()
        ax.plot(seconds, [p[2] / 1024 for p in points], color=OURS, linewidth=2)
        # the levels of the curve: every process, the first subtree gone, all gone
        full = max(p[2] for p in points)
        first = next((i for i, p in enumerate(points) if p[2] < full), None)
        last = next((i for i, p in enumerate(points) if p[2] == 0), len(points) - 1)
        if first is not None and first < last:
            ax.annotate("Every Process\nRuns", (seconds[0], full / 1024), xytext=(0, 3),
                        textcoords="offset points", ha="left", va="bottom", fontsize=VALUE, color=INK,
                        linespacing=1.0)
            ax.annotate("First Subtree\nHas Left", (seconds[last], points[(first + last) // 2][2] / 1024), xytext=(-5, -4),
                        textcoords="offset points", ha="right", va="top", fontsize=VALUE, color=INK,
                        linespacing=1.0)
        finish(fig, ax, "eval_deep_return", "Time (s)", "Segment Files (GiB)", full / 1024 * 1.4)


# ------------------------------------------------ an external way to share weights
def wshare():
    directory = "20261007-ext-weightshare-v1"
    if not have(directory, "ext_weightshare_summary.csv"):
        return
    data = {(r["placement"], r["stack"]): r for r in rows(directory, "ext_weightshare_summary.csv")}
    model_gib = 4.36
    placements = [(label, key) for label, key in (("Same\nInstance", "same"), ("Across\nInstances", "cross"))
                  if (key, "stock") in data]
    stacks = (("Existing llama.cpp Engine", COPY, "stock"), ("CUDA IPC Library", DEVICE, "ipc"), (NAME, OURS, "inplace"))
    fig, ax = panel()
    series = [(label, colour, [(float(data[key, m]["memory_mib"]) / 1024 + (model_gib if m == "inplace" else 0.0))
                               for _, key in placements]) for label, colour, m in stacks]
    groups = grouped(ax, [label for label, _ in placements], series, 0.27)
    for (label, colour, m), group in zip(stacks, groups):
        for bar, (_, key) in zip(group, placements):
            row = data[key, m]
            done, started = int(row["agents_finished"]), int(row["agents_started"])
            if done < started:
                bar.set_hatch("////")
                bar.set_alpha(0.55)
                tip(ax, bar, f"{done // int(row['runs'])}/{started // int(row['runs'])}")
            else:
                tip(ax, bar, f"{bar.get_height():.1f}")
            ax.texts[-1].set_fontsize(VALUE)
    ax.legend(loc="upper center", ncol=3, handlelength=1.0, fontsize=LEGEND, columnspacing=0.8)
    finish(fig, ax, "eval_wshare_mem", "", "Memory (GiB)", max(max(v) for _, _, v in series) * 1.3)


# ------------------------------------- placement of agents (background)
PLACE = "20261013-engine-kvplace-v1"


def cross(ax, bar, colour):
    """A cross in place of a bar whose agents do not all complete."""
    bar.set_height(0)
    ax.annotate("\u00d7", (bar.get_x() + bar.get_width() / 2, 0), xytext=(0, 2),
                textcoords="offset points", ha="center", va="bottom", fontsize=LEGEND,
                color=colour, xycoords=("data", "axes fraction"))


def bg_place():
    """Two half-column panels for the background, both on the unmodified
    engine: the speed of an agent next to seven others by placement, without
    a fault and after a GPU fault of another process, and the memory of eight
    agents whose weights a public library shares in device memory."""
    if have(PLACE, "engine_kvplace_summary.csv"):
        data = {(r["placement"], r["fault"]): r for r in rows(PLACE, "engine_kvplace_summary.csv")
                if r["critical_instance"] == "12sm"}
        placements = (("Time\nSlicing", "slice"), ("MPS", "mps"), ("Other\nInstance", "split"))
        cases = (("No Fault", COPY_LIGHT, "none"), ("After a Fault", COPY, "write"))
        fig, ax = panel()
        series = [(label, colour, [float(data[key, fault]["rate_vs_alone"]) for _, key in placements])
                  for label, colour, fault in cases]
        groups = grouped(ax, [label for label, _ in placements], series, 0.36)
        for (label, colour, fault), group in zip(cases, groups):
            for bar, (_, key) in zip(group, placements):
                if int(data[key, fault]["critical_completed"]) < int(data[key, fault]["runs"]):
                    cross(ax, bar, colour)
                else:
                    tip(ax, bar, f"{bar.get_height():.2f}")
        ax.axhline(1.0, color=AXIS, linewidth=1.0)
        ax.legend(loc="upper center", ncol=2, handlelength=1.0, fontsize=LEGEND, columnspacing=1.0)
        ax.tick_params(axis="x", labelsize=TICK2)
        finish(fig, ax, "bg_place_speed", "", "Relative Speed", 1.25)

    directory = "20261007-ext-weightshare-v1"
    if have(directory, "ext_weightshare_summary.csv"):
        data = {(r["placement"], r["stack"]): r for r in rows(directory, "ext_weightshare_summary.csv")}
        placements = (("Same\nInstance", "same"), ("Across\nInstances", "cross"))
        stacks = (("Existing Engine", COPY, "stock"), ("CUDA IPC Library", DEVICE, "ipc"))
        fig, ax = panel()
        series = [(label, colour, [float(data[key, m]["memory_mib"]) / 1024 for _, key in placements])
                  for label, colour, m in stacks]
        top = max(max(values) for _, _, values in series)
        groups = grouped(ax, [label for label, _ in placements], series, 0.36)
        for (label, colour, m), group in zip(stacks, groups):
            for bar, (_, key) in zip(group, placements):
                if int(data[key, m]["agents_finished"]) < int(data[key, m]["agents_started"]):
                    cross(ax, bar, colour)
                else:
                    tip(ax, bar, f"{bar.get_height():.1f}")
        ax.legend(loc="upper center", ncol=2, handlelength=1.0, fontsize=LEGEND, columnspacing=1.0)
        ax.tick_params(axis="x", labelsize=TICK2)
        finish(fig, ax, "bg_place_share", "", "Memory (GiB)", top * 1.22)


# ------------------------------------------------------- vLLM servers on extents
VLLM_MODES = (("vLLM", EXISTING, "vllm"), ("vLLM-\nCPU", LIGHT_GRAY, "cpu"), ("vLLM-\nLMCache", COPY, "lmcache"),
              ("vLLM-\n" + NAME, OURS, "stator"))
VLLM_TICK = TICK2 - 1


def vshare():
    """Two vLLM servers on one prefix: on the whole GPU when that campaign has
    been run, otherwise one per MIG instance."""
    directory = full("vllm-share") or "20261008-vllm-share-v2"
    if not have(directory, "vllm_share_summary.csv"):
        return
    data = {r["mode"]: r for r in rows(directory, "vllm_share_summary.csv")}
    modes = [(label, colour, m) for label, colour, m in VLLM_MODES if m in data and int(data[m]["failed_cases"]) == 0]
    names = [label for label, _, _ in modes]
    colours = [colour for _, colour, _ in modes]

    def bars(name, values, ylabel, form, log=False):
        fig, ax = panel()
        drawn = ax.bar(names, values, 0.58, color=colours, edgecolor="white", linewidth=1.5)
        labels(ax, drawn, form)
        if log:
            ax.set_yscale("log")
            ax.set_ylim(100 if min(values) >= 100 else 0.1, max(values) * 4.0)
        # the names of two lines need the whole width of the axis
        ax.set_xlim(-0.38, len(names) - 0.62)
        finish(fig, ax, name, "", ylabel, max(values) * (4.0 if log else 1.18), log=log, xtick=VLLM_TICK,
               plain=log and ylabel != "TTFT (ms)")

    bars("eval_vllm_mem", [float(data[m]["memory_mib"]) / 1024 for _, _, m in modes], "Memory (GiB)", "{:.1f}")
    bars("eval_vllm_ttft", [float(data[m]["first_token_ms"]) for _, _, m in modes],
         "TTFT (ms)", "{:,.0f}", log=True)
    fig, ax = panel()
    bottom = np.zeros(len(modes))
    for label, column, colour in (("12-SM Instance", "tps_first_instance", COPY),
                                  ("8-SM Instance", "tps_second_instance", COPY_PALE)):
        values = np.array([float(data[m][column]) for _, _, m in modes])
        ax.bar(names, values, 0.58, bottom=bottom, color=colour, edgecolor="white", linewidth=1.5, label=label)
        for x, (value, base) in enumerate(zip(values, bottom)):
            ax.annotate(f"{value:.0f}", (x, base + value / 2), ha="center", va="center", fontsize=VALUE,
                        color="white" if colour == COPY else INK)
        bottom += values
    for x, total in enumerate(bottom):
        ax.annotate(f"{total:.0f}", (x, total), xytext=(0, 3), textcoords="offset points", ha="center",
                    va="bottom", fontsize=VALUE, color=INK)
    ax.legend(loc="upper center", ncol=2, handlelength=1.0, fontsize=LEGEND, columnspacing=1.0)
    ax.set_xlim(-0.38, len(names) - 0.62)
    finish(fig, ax, "eval_vllm_tps", "", "Tokens/s", bottom.max() * 1.3, xtick=VLLM_TICK)


# --------------------------------------------- the kernel's sharing mechanisms
def full(name):
    """The newest campaign of a runner on the whole GPU (MIG disabled), or None."""
    found = sorted(d for d in os.listdir(RES)
                   if d.endswith(f"-{name}-full-v1") and have(d, "source_hashes.txt"))
    return found[-1] if found else None


# Eight agents on one prefix through the mechanisms of the kernel, none of
# which holds a step of STATOR; CoW Map keeps the colour it has in the other
# panels.
LINUX = (("Copy", COPY, "copy"), ("Shared Map", SHARED_MAP, "shared_rw"), ("CoW Map", COW, "cow_plain"))


def bg_linux():
    """Two half-column panels for the background: the memory of eight agents
    and the page faults that the kernel serves for the GPU."""
    directory = full("engine-kvlinux-mechanisms")
    if directory is None:
        return
    data = {r["mode"]: r for r in rows(directory, "engine_kvlinux_summary.csv") if r["children"] == "8"}
    for name, column, ylabel, scale, form in (
            ("bg_linux_mem", "children_memory_mib", "Memory (GiB)", 1024.0, "{:.1f}"),
            ("bg_linux_faults", "gpu_fault_pages", "GPU Page Faults (M)", 1e6, "{:.2f}")):
        fig, ax = panel()
        values = [float(data[m][column]) / scale for _, _, m in LINUX]
        bars = ax.bar(range(len(LINUX)), values, 0.6, color=[colour for _, colour, _ in LINUX],
                      edgecolor="white", linewidth=1.5)
        ax.set_xticks(range(len(LINUX)), [label.replace(" ", "\n") for label, _, _ in LINUX])
        labels(ax, bars, form)
        ax.tick_params(axis="x", labelsize=TICK2)
        finish(fig, ax, name, "", ylabel, max(values) * 1.22)


# The same agents by their number. Read-Only Map is the read-only shared mapping of the
# kernel: the prefix mapped read-only as STATOR maps it, but without the CPU
# read and the populate-ahead, so every page comes through a fault. It takes
# the light tint of the blue of Shared Map, the other shared mapping.
SCALING = (("Copy", COPY, "copy", "s"), ("CoW Map", COW, "cow_plain", "^"),
           ("Read-Only Map", RO_MAP, "ro_plain", "D"), (NAME, OURS, "extent", "o"))


def linux_scale():
    directory = full("engine-kvlinux-scaling")
    if directory is None:
        return
    data = {(r["mode"], int(r["children"])): r for r in rows(directory, "engine_kvlinux_summary.csv")}
    # 48 to 128 agents, up to the count at which each mode still fits (10-15)
    for large in ("engine-kvlinux-scaling-large", "engine-kvlinux-scaling-large-cow"):
        extra = full(large)
        if extra is not None and have(extra, "engine_kvlinux_summary.csv"):
            data.update({(r["mode"], int(r["children"])): r for r in rows(extra, "engine_kvlinux_summary.csv")
                         if int(r["children"]) > 32})
    counts = sorted({n for _, n in data})
    ticks = [n for n in counts if n & (n - 1) == 0]
    legend("eval_lscale_legend", [(label, colour) for label, colour, _, _ in SCALING], 4)
    for name, column, ylabel, scale in (
            ("eval_lscale_ttft", "child_first_token_ms", "TTFT (ms)", 1.0),
            ("eval_lscale_kernel", "fault_thread_ms", "Fault Service (s)", 1000.0),
            ("eval_lscale_mem", "children_memory_mib", "Memory (GiB)", 1024.0)):
        fig, ax = panel()
        top = 0.0
        for label, colour, m, marker in SCALING:
            points = [(n, float(data[m, n][column]) / scale) for n in counts if (m, n) in data]
            if not points:
                continue
            top = max(top, max(value for _, value in points))
            # Copy is drawn on top: its TTFT lies between those of STATOR and Read-Only Map, and it serves no fault,
            # which puts it on the zero line of STATOR in the fault panel, where it is dashed (10-23)
            on_zero = m == "copy" and all(value == 0 for _, value in points)
            ax.plot([p[0] for p in points], [p[1] for p in points], color=colour, marker=marker,
                    markersize=7, linewidth=2, label=label, markeredgecolor="white", markeredgewidth=1.2,
                    linestyle=(0, (3, 2)) if on_zero else "-", gid="over" if on_zero else None)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks, [str(n) for n in ticks])
        ax.minorticks_off()
        # the TTFT on a log axis, where the lines of the small counts separate
        log = column == "child_first_token_ms"
        finish(fig, ax, name, "Agents", ylabel.replace(")", ", log)") if log else ylabel, None if log else top * 1.15, log=log)


# Model weights read in place under the reclaim of the kernel.
RECLAIM = (("Existing Engine", EXISTING, "device"), (f"{NAME} (4 KiB)", OURS_LIGHT, "pagecache"),
           (f"{NAME} (2 MiB)", OURS, "memfs"))


def reclaim():
    directory = full("engine-reclaim")
    if directory is None:
        return
    data = {(r["mode"], r["pressure"]): r for r in rows(directory, "engine_reclaim_summary.csv")}
    legend("eval_reclaim_legend", [(label, colour) for label, colour, _ in RECLAIM], 3)
    for name, column, ylabel, scale, form in (
            ("eval_reclaim_speed", "tps_vs_none", "Speed under Reclaim", 1.0, "{:.2f}"),
            ("eval_reclaim_faults", "gpu_fault_pages", "GPU Page Faults (M)", 1e6, "{:.1f}")):
        fig, ax = panel()
        values = [float(data[m, "reclaim"][column]) / scale for _, _, m in RECLAIM]
        bars = ax.bar(range(len(RECLAIM)), values, 0.6, color=[colour for _, colour, _ in RECLAIM],
                      edgecolor="white", linewidth=1.5)
        ax.set_xticks(range(len(RECLAIM)), ["" for _ in RECLAIM])
        ax.tick_params(axis="x", length=0)
        labels(ax, bars, form)
        finish(fig, ax, name, "", ylabel, max(max(values), 1e-9) * 1.22)


# Each sharing mechanism of the kernel with the CPU or the GPU as the reader
# of the same 1 GiB of shared pages (mech_probe.cu); the reader is the only
# difference between the two bars of a group. The shared mapping comes from
# the rerun with the file opened read-write (mech-probe-shared).
ACCESS = (("CPU", DEMAND, "cpu"), ("GPU", COW, "gpu"))
COW_LIGHT = "#f6b28f"   # orange tint: the GPU with the page cache locked
MECHS = (("Private", "private"), ("Shared", "shared"), ("fork", "fork"), ("KSM", "ksm"))


def compact(value):
    if value < 0.0005:
        return "0"
    if value < 0.1:
        return f"{value:.3f}"
    return f"{value:.2f}" if value < 10 else f"{value:.1f}"


def bg_mech():
    """Three quarter-width panels for the background: the time of a read of
    every word, the memory that this read copies, and the speed of repeated
    reads of a hot file while the kernel reclaims from its control group."""
    directory = full("mech-probe")
    if directory is None:
        return
    data = {}
    for d in (directory, full("mech-probe-shared")):
        if d is None:
            continue
        for r in rows(d, "mech_probe_summary.csv"):
            if r["mech"] == "gate" or r["step"] == "events":
                continue
            if d == directory and r["mech"] == "shared":
                continue   # every shared case of the first campaign failed
            data[(r["mech"], r["step"], r["access"], r["pressure"])] = r
    legend("bg_mech_legend", [(label, colour) for label, colour, _ in ACCESS], 2)

    def value(mech, step, access, column, scale=1.0):
        row = data.get((mech, step, access, ""))
        return float(row[column]) / scale if row and row[column] else 0.0

    for name, column, ylabel, scale, log in (
            ("bg_mech_read", "seconds", "Read Time (s)", 1.0, True),
            ("bg_mech_copy", "copied_mib", "Copied (GiB)", 1024.0, False)):
        fig, ax = panel()
        series = [(label, colour, [value(m, "read", access, column, scale) for _, m in MECHS])
                  for label, colour, access in ACCESS]
        bars = grouped(ax, [label for label, _ in MECHS], series, width=0.38)
        for group in bars:
            for bar in group:
                tip(ax, bar, compact(bar.get_height()))
                ax.texts[-1].set_fontsize(DENSE)
        ax.tick_params(axis="x", labelsize=TICK2)
        ax.legend()   # save() sets it as a row above the plot of this panel
        top = max(max(v for _, _, vs in series for v in vs), 1e-9)
        if not log:
            finish(fig, ax, name, "", ylabel, top * 1.25)
            continue
        low = min(v for _, _, vs in series for v in vs if v > 0)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_ylim(low / 3, top * 8)
        ax.set_ylabel(ylabel)
        style(ax)
        fig.tight_layout()
        save(fig, name)

    fig, ax = panel()
    entries = (("CPU", DEMAND, "none", "cpu"), ("GPU", COW, "none", "gpu"), ("GPU\nLocked", COW_LIGHT, "lock", "gpu"))
    values = []
    for _, _, pin, access in entries:
        row = data.get(("reclaim", pin, access, "pressure"))
        values.append(float(row["rate_vs_none"]) if row and row["rate_vs_none"] else 0.0)
    bars = ax.bar(range(len(entries)), values, 0.55, color=[c for _, c, _, _ in entries],
                  edgecolor="white", linewidth=1.5)
    ax.set_xticks(range(len(entries)), [label for label, _, _, _ in entries])
    ax.tick_params(axis="x", labelsize=TICK2)
    labels(ax, bars, "{:.2f}")
    finish(fig, ax, "bg_mech_reclaim", "", "Speed under Reclaim", 1.25)


# Eight agents by deployment, Copy against STATOR within each deployment: eight
# separate processes, two batching servers, and one batching server that holds
# all agents (the in-process reference, one bar).
def inproc_bars():
    share = {(r["paragraphs"], r["mode"], int(r["agents"])): r
             for r in rows("20261006-engine-kvshare-v1", "engine_kvshare_summary.csv")}
    publish = {(r["paragraphs"], r["store"]): r
               for r in rows("20261006-engine-kvshare-v1", "engine_kvshare_publish.csv")}
    batch = {r["config"]: r for r in rows("20261010-engine-kvbatch-v2", "engine_kvbatch_summary.csv")}
    ollama = {(r["config"], r["round"]): r for r in rows("20261006-ollama-agents-v1", "ollama_agents_summary.csv")}
    file_gib = float(publish["320", "huge"]["file_used_mib"]) / 1024
    gib = lambda row: float(row["memory_mib"]) / 1024
    tps = lambda row, key="generation_tps_total": float(row[key])
    copy8, ours8 = share["320", "restore", 8], share["320", "extent_lazy", 8]
    groups = ["8 Processes", "2 Servers", "1 Server"]
    w = 0.27
    # (x, memory GiB, tokens/s, colour); Ollama runs with its own code and no STATOR
    bars = [
        (-w / 2, gib(copy8), tps(copy8), COPY),
        (w / 2, gib(ours8) + file_gib, tps(ours8), OURS),
        (1 - w, gib(batch["two_servers_copy"]), tps(batch["two_servers_copy"]), COPY),
        (1, gib(batch["two_servers_extent"]), tps(batch["two_servers_extent"]), OURS),
        (1 + w, gib(ollama["two", "warm"]), tps(ollama["two", "warm"], "generation_tps_sum"), DEVICE),
        (2 - w / 2, gib(batch["one_server"]), tps(batch["one_server"]), DEMAND),
        (2 + w / 2, gib(ollama["one", "warm"]), tps(ollama["one", "warm"], "generation_tps_sum"), DEVICE),
    ]
    legend("eval_proc_legend", [("Copy", COPY), (NAME, OURS), ("One Batching Server", DEMAND), ("Ollama", DEVICE)], 4)
    for name, index, ylabel, form in (("eval_proc_mem", 1, "Memory (GiB)", "{:.1f}"),
                                      ("eval_proc_tps", 2, "Throughput (tokens/s)", "{:.0f}")):
        fig, ax = panel()
        drawn = ax.bar([b[0] for b in bars], [b[index] for b in bars], w,
                       color=[b[3] for b in bars], edgecolor="white", linewidth=1.5)
        for bar in drawn:
            tip(ax, bar, form.format(bar.get_height()))
            ax.texts[-1].set_fontsize(DENSE)
            ax.texts[-1].set_rotation(0)
        ax.set_xticks(range(len(groups)), groups)
        ax.tick_params(axis="x", labelsize=TICK2)
        finish(fig, ax, name, "", ylabel, max(b[index] for b in bars) * 1.22)


# Eight agents in one batching server or split over 2, 4 and 8 servers, one
# campaign (run_engine_kvbatch_n.sh). The throughput is that of the round, from
# the first token of the first server to the last token of the last: summed
# rates would add the publisher's rate while the servers of Copy still copy.
def round_tps(directory):
    path = os.path.join(RES, base.resolve(directory), "raw.log")
    rounds, case = {}, None
    for line in open(path):
        if line.startswith("BEGIN_KVBATCH"):
            run = re.search(r"run=(\d+)", line).group(1)
        elif line.startswith("CASE"):
            case = (re.search(r"config=(\S+)", line).group(1), run)
            rounds[case] = []
        elif line.startswith("SERVER") and case is not None:
            f = dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)
            if f.get("exit") != "0" or not f.get("generation_tps"):
                continue
            start = float(f["started_ms"]) + float(f["first_token_ms"])
            tokens = int(f["sequences"]) * 64
            rounds[case].append((start, start + tokens / float(f["generation_tps"]) * 1000, tokens))
    out = {}
    for (config, _), servers in rounds.items():
        begin, end = min(s[0] for s in servers), max(s[1] for s in servers)
        out.setdefault(config, []).append(sum(s[2] for s in servers) / ((end - begin) / 1000))
    return {config: sum(v) / len(v) for config, v in out.items()}


def inproc_n():
    directory = full("engine-kvbatch-n")
    if directory is None:
        return
    mem = {r["config"]: float(r["memory_mib"]) / 1024
           for r in rows(directory, "engine_kvbatch_n_summary.csv")}
    tps = round_tps(directory)
    ollama = {(r["config"], r["round"]): r for r in rows("20261006-ollama-agents-v1", "ollama_agents_summary.csv")}
    marker = {2: "^", 4: "D", 8: "o"}
    fig, ax = plt.subplots(figsize=(6.4, 2.0))
    # one batching server: a level mark, which keeps the point of two servers on STATOR below it visible
    ax.scatter([mem["1:none"]], [tps["1:none"]], s=360, marker="_", color=COPY, linewidth=3.0, zorder=4)
    ax.annotate("1 Server", (mem["1:none"], tps["1:none"]), xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom", fontsize=VALUE)
    for n in (2, 4, 8):
        copy, ours = (mem[f"{n}:copy"], tps[f"{n}:copy"]), (mem[f"{n}:extent"], tps[f"{n}:extent"])
        ax.scatter([copy[0]], [copy[1]], s=130, marker=marker[n], color=COPY, edgecolor="white", linewidth=1.5, zorder=3)
        ax.scatter([ours[0]], [ours[1]], s=130, marker=marker[n], color=OURS, edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate("", xy=ours, xytext=copy, arrowprops=dict(arrowstyle="-|>", color=OURS, lw=1.6,
                    mutation_scale=12, shrinkA=7, shrinkB=7), zorder=2)
        # the marker shape names the pair, so only the point of Copy carries the label
        ax.annotate(f"{n} Servers", copy, xytext=(10, -4), textcoords="offset points", ha="left", fontsize=VALUE)
    for key, label, mark, offset in ((("one", "warm"), "1 Server", "s", (0, 9)),
                                     (("two", "warm"), "2 Servers", "^", (0, 9))):
        row = ollama[key]
        point = (float(row["memory_mib"]) / 1024, float(row["generation_tps_sum"]))
        ax.scatter([point[0]], [point[1]], s=130, marker=mark, color=DEVICE, edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(label, point, xytext=offset, textcoords="offset points", ha="center", fontsize=VALUE)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=colour, markersize=9, markeredgecolor="white",
                          label=label)
               for label, colour in (("llama.cpp, Copy", COPY), (f"llama.cpp, {NAME}", OURS), ("Ollama", DEVICE))]
    ax.legend(handles=handles, loc="upper right", handlelength=1.0)
    ax.set_xlim(0, 27)
    ax.set_ylim(0, 160)
    ax.set_xlabel("Memory (GiB)")
    ax.set_ylabel("Throughput (tokens/s)")
    style(ax, "both")
    fig.tight_layout()
    save(fig, "eval_proc")


# Eight agents per batching server on 2, 4 and 8 servers (16 to 64 agents),
# Copy or STATOR between the servers, and one server with all agents as the
# in-process reference (run_engine_kvbatch_n.sh, AGENTS = 8 N). Throughput of
# the round, as in inproc_n.
def inproc_per8(reference=True, name="eval_proc8"):
    dirs = {n: full(f"engine-kvbatch-per8-n{n}") for n in (2, 4, 8)}
    if any(d is None for d in dirs.values()):
        return
    marker = {2: "^", 4: "D", 8: "o"}
    fig, ax = plt.subplots(figsize=(6.4, 2.0))
    top = 0.0
    for n, directory in dirs.items():
        mem = {r["config"]: float(r["memory_mib"]) / 1024 for r in rows(directory, "engine_kvbatch_n_summary.csv")}
        tps = round_tps(directory)
        copy, ours = (mem[f"{n}:copy"], tps[f"{n}:copy"]), (mem[f"{n}:extent"], tps[f"{n}:extent"])
        ax.scatter([copy[0]], [copy[1]], s=130, marker=marker[n], color=COPY, edgecolor="white", linewidth=1.5, zorder=3)
        ax.scatter([ours[0]], [ours[1]], s=130, marker=marker[n], color=OURS, edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate("", xy=ours, xytext=copy, arrowprops=dict(arrowstyle="-|>", color=OURS, lw=1.6,
                    mutation_scale=12, shrinkA=7, shrinkB=7), zorder=2)
        ax.annotate(f"{n} Servers", copy, xytext=(10, -4), textcoords="offset points", ha="left", fontsize=VALUE)
        top = max(top, copy[1], ours[1])
        if reference:
            one = (mem["1:none"], tps["1:none"])
            ax.scatter([one[0]], [one[1]], s=130, marker=marker[n], facecolors="none", edgecolors=COPY,
                       linewidth=1.8, zorder=3)
            ax.annotate(f"{8 * n} Agents", one, xytext=(10, -4), textcoords="offset points", ha="left",
                        fontsize=VALUE)
            top = max(top, one[1])
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=COPY, markersize=9, markeredgecolor="white",
                          label="Servers, Copy"),
               plt.Line2D([], [], marker="o", linestyle="", color=OURS, markersize=9, markeredgecolor="white",
                          label=f"Servers, {NAME}")]
    if reference:
        handles.append(plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="none", color=COPY,
                                  markersize=9, markeredgewidth=1.8, label="One Server"))
    ax.legend(handles=handles, loc="upper right", handlelength=1.0)
    ax.set_xlim(0, 23)
    ax.set_ylim(0, top * 1.25)
    ax.set_xlabel("Memory (GiB)")
    ax.set_ylabel("Throughput (tokens/s)")
    style(ax, "both")
    fig.tight_layout()
    save(fig, name)


def inproc_per8_bars():
    dirs = {n: full(f"engine-kvbatch-per8-n{n}") for n in (2, 4, 8)}
    if any(d is None for d in dirs.values()):
        return
    mem, tps = {}, {}
    for n, directory in dirs.items():
        m = {r["config"]: float(r["memory_mib"]) / 1024 for r in rows(directory, "engine_kvbatch_n_summary.csv")}
        t = round_tps(directory)
        mem[n] = (m[f"{n}:copy"], m[f"{n}:extent"], m["1:none"])
        tps[n] = (t[f"{n}:copy"], t[f"{n}:extent"], t["1:none"])
    order = (("Servers, Copy", COPY), (f"Servers, {NAME}", OURS), ("One Server", COPY_LIGHT))
    legend("eval_proc8_legend", list(order), 3)
    groups = [f"{n} ({8 * n})" for n in dirs]
    for name, data, ylabel, form in (("eval_proc8_mem", mem, "Memory (GiB)", "{:.1f}"),
                                     ("eval_proc8_tps", tps, "Throughput (tokens/s)", "{:.0f}")):
        fig, ax = panel()
        series = [(label, colour, [data[n][i] for n in dirs]) for i, (label, colour) in enumerate(order)]
        for group in grouped(ax, groups, series, 0.25):
            labels(ax, group, form, DENSE)
        top = max(max(v) for _, _, v in series)
        finish(fig, ax, name, "Servers (Agents)", ylabel, top * 1.22)


if __name__ == "__main__":
    for figure in (bg_linux, linux_scale, reclaim, bg_mech, inproc_bars, inproc_n):
        figure()
        print("wrote", figure.__name__)
    # the first platform (STATOR_FULL=0) draws every panel; the whole GPU draws
    # the panels of the paper, without those of the MIG instances
    every = (base.bg_state, bg_place, pair_legend, mem, weights, hand, speed, reach, mode, cow, resident, latency,
             dev, sota, attach, protect, tree, scale, variants, inproc, server, limit, stack, local, pipe, deep,
             wshare, prefix_length, vshare)
    paper = (base.bg_state, bg_state_legend, pair_legend, mem, weights, hand, mode, cow, resident, sota, attach, protect, tree,
             scale, variants, inproc, stack, local, vshare)
    for figure in (every if base.MODE == "0" else paper):
        figure()
        print("wrote", figure.__name__)
