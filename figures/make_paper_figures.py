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
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_eval_figures import (AXIS, COPY, COW, DEVICE, GRID, INK, OURS,  # noqa: E402
                               OURS_LIGHT, RES, grouped, rows, save, style, tip)

NAME = "STATOR"
DEMAND = "#4f86c6"   # a copy into device memory that is backed on demand
WEIGHTS_ONLY = "#d99a2b"   # the stack that shares the weights and copies the cache
PANEL = (3.6, 2.7)


def have(*path):
    return os.path.exists(os.path.join(RES, *path))


def panel(size=PANEL):
    return plt.subplots(figsize=size)


def legend(name, entries, ncol):
    """A strip that holds only the legend, shared by the panels below it."""
    fig = plt.figure(figsize=(1.9 * ncol + 0.6, 0.38))
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour) for _, colour in entries]
    fig.legend(handles, [label for label, _ in entries], loc="center", ncol=ncol,
               columnspacing=1.6, handlelength=1.3, fontsize=12.5)
    save(fig, name)


def labels(ax, bars, form="{:.1f}", size=10.5):
    for bar in bars:
        tip(ax, bar, form.format(bar.get_height()))
        ax.texts[-1].set_fontsize(size)


def finish(fig, ax, name, xlabel, ylabel, top=None, log=False):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    if top is not None:
        ax.set_ylim(ax.get_ylim()[0] if log else 0, top)
    style(ax)
    fig.tight_layout()
    save(fig, name)


# ------------------------------------------------------------------ memory
def mem():
    data = {(r["paragraphs"], r["mode"], int(r["agents"])): r
            for r in rows("20261006-engine-kvshare-v1", "engine_kvshare_summary.csv")}
    series_of = (("Copy", COPY, "restore"), ("Extents, Whole Tail", OURS_LIGHT, "extent"),
                 (NAME, OURS, "extent_lazy"))
    legend("eval_mem_legend", [(label, colour) for label, colour, _ in series_of], 3)
    counts = [1, 4, 8]
    for size, tag in (("80", "4k"), ("320", "16k")):
        fig, ax = panel()
        series = [(label, colour, [float(data[size, mode, n]["memory_mib"]) / 1024 for n in counts])
                  for label, colour, mode in series_of]
        for group in grouped(ax, [str(n) for n in counts], series, 0.27):
            labels(ax, group, size=9.5)
        finish(fig, ax, f"eval_mem_{tag}", "Agents", "Memory (GiB)", 23)


def weights():
    agents = {(r["config"], r["mode"], int(r["agents"])): r
              for r in rows("20261005-engine-agents-v1", "engine_agents_summary.csv")}
    counts = sorted({k[2] for k in agents if k[0] == "mig"})
    fig, ax = panel()
    series = [(label, colour, [float(agents["mig", mode, n]["memory_total_mib"]) / 1024 for n in counts])
              for label, colour, mode in (("Device Copy", COPY, "copy"), (NAME, OURS, "inplace"))]
    for group in grouped(ax, [str(n) for n in counts], series, 0.38):
        labels(ax, group, size=9.5)
    ax.legend(loc="upper left", handlelength=1.2)
    finish(fig, ax, "eval_weights_mem", "Serving Processes", "Memory (GiB)", 47)

    pages = {r["mode"]: r for r in rows("20261005-engine-pages-v1", "engine_pages_summary.csv")}
    order = (("Device\nCopy", "copy", COPY), ("In Place\n4 KiB", "inplace_4k", OURS_LIGHT),
             ("In Place\n2 MiB", "inplace_thp", OURS))
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
                    fontsize=10.5, color=INK)
    ax.axhline(1.0, color=AXIS, linewidth=1.0)
    finish(fig, ax, "eval_weights_speed", "", "Relative Speed", 1.18)


# ---------------------------------------------------------------- handover
MECHANISMS = (("Copy", COPY, "copy"), ("Demand", DEMAND, "demand"), ("Device", DEVICE, "device_merged"),
              ("CoW Map", COW, "cow"), (NAME, OURS, "extent"))
MECH = "20261007-engine-kvmech-v1"


def mechanisms():
    """The campaign that runs every mechanism in one engine, by cell and mode."""
    return {(r["placement"], int(r["children"]), int(r["paragraphs"]), r["mode"]): r
            for r in rows(MECH, "engine_kvmech_summary.csv")}


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
        series = [(label, colour, [float(data[cell + (m,)][column]) / scale for cell in cells])
                  for label, colour, m in MECHANISMS]
        for group in grouped(ax, groups, series, 0.17):
            labels(ax, group, form, 8)
            for text in ax.texts[-len(group):]:
                text.set_rotation(90)
        top = max(max(values) for _, _, values in series)
        if log:
            ax.set_ylim(1, top * 14)
        finish(fig, ax, name, xlabel, ylabel, None if log else top * 1.3, log)

    bars("eval_hand_pause", [f"{int(float(data['same', 8, size, 'copy']['prefix_tokens'])):,}" for size in sizes],
         [("same", 8, size) for size in sizes], "publish_ms", "Pause (ms, log)", 1.0, "{:.0f}", True,
         "Prefix (Tokens)")
    cells = [("same", n, max(sizes)) for n in counts]
    bars("eval_hand_attach", [str(n) for n in counts], cells, "child_attach_ms", "Attach Time (ms, log)",
         1.0, "{:.0f}", True, f"Agents ({long_tokens:,}-Token Prefix)")
    bars("eval_hand_ttft", [str(n) for n in counts], cells, "child_first_token_ms", "TTFT (s)",
         1000.0, "{:.2f}", False, f"Agents ({long_tokens:,}-Token Prefix)")


# ------------------------------------------------------------------- speed
def speed():
    cells = (("12 SMs", "20261006-engine-kvspeed-v1"), ("12 SMs", "20261006-engine-kvspeed-long-v1"),
             ("6 SMs", "20261006-engine-kvspeed-6sm-v1"), ("6 SMs", "20261006-engine-kvspeed-6sm-long-v1"))
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
                    fontsize=10.5, color=INK)
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
    for group in grouped(ax, ["12 SMs", "6 SMs"], series, 0.27):
        labels(ax, group, "{:.0f}", 9.5)
    ax.legend(loc="upper right", handlelength=1.2, fontsize=10.5)
    top = max(max(values) for _, _, values in series) * 1.45
    finish(fig, ax, "eval_speed_reach", "MIG Instance", "Read Bandwidth (GiB/s)", top)


# ------------------------------------------------------ GPU sharing modes
def mode():
    directory = "20261006-engine-kvmps-v1"
    if not have(directory, "engine_kvmps_summary.csv"):
        return
    data = {(r["config"], int(r["children"]), r["prefix_tokens"], r["mode"]): r
            for r in rows(directory, "engine_kvmps_summary.csv")}
    children = max(k[1] for k in data)
    tokens = max({k[2] for k in data}, key=int)
    configs = (("Time\nSlicing", "timeslice"), ("MPS", "mps"), ("MIG", "mig"), ("MIG\n+MPS", "mig_mps"))
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
            labels(ax, group, form, 9)
        for text in ax.texts:
            text.set_rotation(90)
        top = max(max(values) for _, _, values in series) * 1.3
        finish(fig, ax, name, "", ylabel, top)


# --------------------------------------------------------- copy-on-write
def cow():
    data = {(r["mode"], int(r["agents"])): r
            for r in rows("20261006-engine-kvcow-v1", "engine_kvcow_summary.csv")}
    order = (("CoW Map\n2M\nNo Read", "cow_noread", COW), ("CoW Map\n4K\nRead", "cow_small", COW),
             ("CoW Map\n2M\nRead", "cow", COW), (NAME, "extent_lazy", OURS))
    for name, column, ylabel, scale, form in (
            ("eval_cow_mem", "memory_mib", "Memory (GiB)", 1024.0, "{:.1f}"),
            ("eval_cow_time", "suffix_ms", "Task Decoding Time (s)", 1000.0, "{:.2f}")):
        fig, ax = panel()
        values = [float(data[m, 8][column]) / scale for _, m, _ in order]
        bars = ax.bar([label for label, _, _ in order], values, 0.62,
                      color=[c for _, _, c in order], edgecolor="white", linewidth=1.5)
        labels(ax, bars, form)
        ax.tick_params(axis="x", labelsize=9.5)
        finish(fig, ax, name, "", ylabel, max(values) * 1.18)


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
        labels(ax, group, "{:.0f}", 9.5)
    ax.set_ylim(1, 4000)
    ax.legend(loc="upper center", ncol=3, handlelength=1.0, columnspacing=0.8, fontsize=10,
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
    ax.text(0.97, 0.95, "Child in the Other MIG Instance\n"
            f"Device: {done['vmm_cross']}/{started['vmm_cross']} Complete\n"
            f"{NAME}: {done['extent_cross']}/{started['extent_cross']} Complete",
            transform=ax.transAxes, ha="right", va="top", fontsize=10, color=INK,
            linespacing=1.35)
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
    placements = [(label, key) for label, key in (("Same\nInstance", "same"), ("Across\nInstances", "cross"))
                  if (key, children, size, "copy") in data]
    for name, column, ylabel, scale, form, log in (
            ("eval_sota_mem", "children_memory_mib", "Memory (GiB)", 1024.0, "{:.1f}", False),
            ("eval_sota_attach", "child_attach_ms", "Attach Time (ms, log)", 1.0, "{:.0f}", True),
            ("eval_sota_ttft", "child_first_token_ms", "TTFT (s)", 1000.0, "{:.2f}", False),
            ("eval_sota_tps", "children_tps_sum", "Throughput (tokens/s)", 1.0, "{:.1f}", False)):
        fig, ax = panel()
        series = [(label, colour, [float(data[key, children, size, m][column]) / scale for _, key in placements])
                  for label, colour, m in MECHANISMS]
        groups = grouped(ax, [label for label, _ in placements], series, 0.17)
        for (label, colour, m), group in zip(MECHANISMS, groups):
            for bar, (_, key) in zip(group, placements):
                row = data[key, children, size, m]
                done, started = int(row["children_finished"]), int(row["children_started"])
                if done < started:
                    # Not every child completed: the bar covers the children that did.
                    bar.set_hatch("////")
                    bar.set_alpha(0.55)
                    tip(ax, bar, f"{done // int(row['runs'])}/{started // int(row['runs'])}")
                else:
                    tip(ax, bar, form.format(bar.get_height()))
                ax.texts[-1].set_fontsize(8)
                ax.texts[-1].set_rotation(90 if log or name.endswith(("ttft", "tps")) else 0)
        top = max(max(values) for _, _, values in series)
        if log:
            ax.set_ylim(1, top * 12)
        finish(fig, ax, name, "", ylabel, None if log else top * 1.22, log)

    # the device-memory baseline by how far it is tuned, against the extents
    stages = (("Device", "#c9bdf0", "device"), ("+ Batched", "#a18fe0", "device_tuned"),
              ("+ One Allocation", DEVICE, "device_merged"), (NAME, OURS, "extent"))
    fig, ax = panel()
    cell = ("same", children, size)
    series = [(label, colour, [float(data[cell + (m,)]["publish_ms"]), float(data[cell + (m,)]["child_attach_ms"])])
              for label, colour, m in stages]
    for group in grouped(ax, ["Parent Pause", "Child Attach"], series, 0.2):
        labels(ax, group, "{:.0f}", 9)
    ax.set_ylim(1, max(max(v) for _, _, v in series) * 30)
    ax.legend(loc="upper left", handlelength=1.1, fontsize=9, labelspacing=0.25, ncol=2,
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
                    fontsize=9.5, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sizes, [str(v) for v in sizes], fontsize=9)
    ax.set_yticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
    ax.minorticks_off()
    ax.set_ylim(1, 60)
    finish(fig, ax, "eval_attach_granule", "Allocation Size (MiB)", "Attach Time (ms, log)", log=True)


def protect():
    """Writes through the GPU that changed state shared with another process."""
    directory = "20261006-protect-v1"
    path = os.path.join(RES, directory, "protect_summary.csv")
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
    finish(fig, ax, "eval_protect", "", "Writes That Changed\nShared State (%)", 125)


# ------------------------------------------------------------ agent tree
def tree():
    directory = "20261006-engine-kvtree-v1"
    if not have(directory, "engine_kvtree_summary.csv"):
        return
    data = {r["mode"]: r for r in rows(directory, "engine_kvtree_summary.csv")}
    order = (("Copy", "copy", COPY), ("Extents\n(One Level)", "flat", OURS_LIGHT),
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
def lines(name, directory, summary, column, ylabel, scale, form):
    data = {(r["mode"], int(r["agents"])): r for r in rows(directory, summary)}
    counts = sorted({k[1] for k in data})
    fig, ax = panel()
    for label, colour, m, marker in (("Copy", COPY, "restore", "s"), (NAME, OURS, "extent_lazy", "o")):
        points = [(n, float(data[m, n][column]) / scale) for n in counts if (m, n) in data]
        ax.plot([p[0] for p in points], [p[1] for p in points], color=colour, marker=marker,
                markersize=7, linewidth=2, label=label, markeredgecolor="white", markeredgewidth=1.2)
        ax.annotate(form.format(points[-1][1]), points[-1], xytext=(-4, 7),
                    textcoords="offset points", ha="right", fontsize=10.5, color=INK)
    ax.set_xticks(counts)
    ax.legend(loc="upper left", handlelength=1.6)
    finish(fig, ax, name, "Agents (16,321-Token Prefix)", ylabel,
           max(float(r[column]) for r in data.values()) / scale * 1.2)


def scale():
    directory = "20261006-engine-kvscale-v1"
    if not have(directory, "engine_kvscale_summary.csv"):
        return
    lines("eval_scale_mem", directory, "engine_kvscale_summary.csv", "memory_mib",
          "Memory (GiB)", 1024.0, "{:.1f}")
    lines("eval_scale_tps", directory, "engine_kvscale_summary.csv", "generation_tps_total",
          "Throughput (tokens/s)", 1.0, "{:.1f}")


def prefix_length():
    """Eight agents by the length of the prefix they start from."""
    cells = [d for d in ("20261007-engine-kvscale-p20-v1", "20261007-engine-kvscale-p80-v1",
                         "20261006-engine-kvscale-v1", "20261007-engine-kvscale-p600-v1")
             if have(d, "engine_kvscale_summary.csv")]
    if len(cells) < 3:
        return
    points = {}
    for directory in cells:
        for r in rows(directory, "engine_kvscale_summary.csv"):
            if int(r["agents"]) == 8:
                points[int(float(r["prefix_tokens"])), r["mode"]] = r
    tokens = sorted({k[0] for k in points})
    for name, column, ylabel, scale, form in (
            ("eval_prefix_mem", "memory_mib", "Memory (GiB)", 1024.0, "{:.1f}"),
            ("eval_prefix_attach", "attach_ms", "Attach Time (ms)", 1.0, "{:.0f}")):
        fig, ax = panel()
        for label, colour, m, marker in (("Copy", COPY, "restore", "s"), (NAME, OURS, "extent_lazy", "o")):
            values = [float(points[t, m][column]) / scale for t in tokens]
            ax.plot(range(len(tokens)), values, color=colour, marker=marker, markersize=7, linewidth=2,
                    label=label, markeredgecolor="white", markeredgewidth=1.2)
            for x, value in enumerate(values):
                ax.annotate(form.format(value), (x, value), xytext=(0, 7 if m == "restore" else -14),
                            textcoords="offset points", ha="center", fontsize=9.5, color=INK)
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
        labels(ax, group, size=9.5)
    ax.tick_params(axis="x", labelsize=10)
    ax.legend(loc="upper left", handlelength=1.2)
    finish(fig, ax, "eval_model_mem", "", "Memory (GiB)", max(copy) * 1.25)


def inproc():
    """Eight agents: separate processes, batching servers, and Ollama."""
    share = {(r["paragraphs"], r["mode"], int(r["agents"])): r
             for r in rows("20261006-engine-kvshare-v1", "engine_kvshare_summary.csv")}
    publish = {(r["paragraphs"], r["store"]): r
               for r in rows("20261006-engine-kvshare-v1", "engine_kvshare_publish.csv")}
    batch = {r["config"]: r for r in rows("20261006-engine-kvbatch-v1", "engine_kvbatch_summary.csv")}
    file_gib = float(publish["320", "huge"]["file_used_mib"]) / 1024
    gib = lambda row, key="memory_mib": float(row[key]) / 1024
    # (label, memory GiB, tokens/s, marker, colour, label offset in points, alignment)
    points = [
        ("8 Processes", gib(share["320", "restore", 8]),
         float(share["320", "restore", 8]["generation_tps_total"]), "o", COPY, (9, 2), "left"),
        ("8 Processes", gib(share["320", "extent_lazy", 8]) + file_gib,
         float(share["320", "extent_lazy", 8]["generation_tps_total"]), "o", OURS, (0, -17), "center"),
        ("1 Server", gib(batch["one_server"]), float(batch["one_server"]["generation_tps_total"]),
         "s", COPY, (9, -4), "left"),
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
             "s", DEMAND, (-9, -11), "right"),
            ("2 Servers", gib(ollama["two", "warm"]), float(ollama["two", "warm"]["generation_tps_sum"]),
             "^", DEMAND, (0, 9), "center"),
        ]
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    for label, memory, speed, marker, colour, offset, align in points:
        ax.scatter([memory], [speed], s=130, marker=marker, color=colour, edgecolor="white",
                   linewidth=1.5, zorder=3)
        ax.annotate(label, (memory, speed), xytext=offset, textcoords="offset points", ha=align,
                    fontsize=10.5)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=colour, markersize=9,
                          markeredgecolor="white", label=label)
               for label, colour in (("llama.cpp, Copy", COPY), (f"llama.cpp, {NAME}", OURS), ("Ollama", DEMAND))]
    ax.legend(handles=handles, loc="upper right", handlelength=1.0)
    ax.set_xlim(0, 27)
    ax.set_ylim(0, 125)
    ax.set_xlabel("Memory (GiB)")
    ax.set_ylabel("Throughput (tokens/s)")
    style(ax, "both")
    fig.tight_layout()
    save(fig, "eval_proc")


def server():
    # The second campaign publishes at a token boundary; the first, which did
    # not, is kept in the record.
    directory = "20261007-engine-kvserver2-v1"
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
        labels(ax, group, "{:.0f}", 9.5)
    ax.set_ylim(1, 6000)
    ax.legend(loc="upper left", handlelength=1.2, ncol=2, columnspacing=1.0, fontsize=10.5)
    finish(fig, ax, "eval_server_time", "", "Time (ms, log)", log=True)

    counts = sorted({k[1] for k in data})
    fig, ax = panel()
    series = [(label, colour, [float(data[m, n]["agents_memory_mib"]) / 1024 for n in counts])
              for label, colour, m in pair]
    for group in grouped(ax, [str(n) for n in counts], series, 0.36):
        labels(ax, group)
    ax.legend(loc="upper left", handlelength=1.2)
    finish(fig, ax, "eval_server_mem", "Agent Servers", "Memory (GiB)",
           max(max(values) for _, _, values in series) * 1.22)


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
    bars = ax.bar([label for label, _, _ in modes], peaks, 0.5, color=[c for _, c, _ in modes],
                  edgecolor="white", linewidth=1.5)
    for bar, (_, _, m) in zip(bars, modes):
        row = data["limit", m]
        done, runs = int(row["agent0_completed"]), int(row["runs"])
        outcome = f"completes {done}/{runs}" if done else f"ended {row['agent0_oom_kills']}/{runs}"
        tip(ax, bar, f"{bar.get_height():.0f}, {outcome}")
        ax.texts[-1].set_fontsize(9.5)
    ax.axhline(cap, color=INK, linewidth=1.2, linestyle=(0, (4, 2)))
    ax.text(0.03, cap * 1.02, f"limit: {cap:.0f} MiB", ha="left", va="bottom", fontsize=10,
            transform=ax.get_yaxis_transform())
    finish(fig, ax, "eval_limit_peak", "", "Peak Charge (MiB)", cap * 1.55)


# ------------------------------------------------------------ whole stack
STACKS = (("Unmodified", COPY, "none", "s"), ("KV Shared", DEMAND, "kv", "D"),
          ("Weights Shared", WEIGHTS_ONLY, "weights", "^"), (NAME, OURS, "both", "o"))


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
    more = "20261007-engine-kvstack-capacity-v1"
    line = dict(data)
    if have(more, "engine_kvstack_summary.csv"):
        line.update({(r["mode"], int(r["agents"])): r
                     for r in rows(more, "engine_kvstack_summary.csv") if int(r["runs"]) > 0})

    # memory against the number of agents; the line of the fit goes on to the
    # memory that was available, where the stack stops fitting
    fig, ax = panel()
    room = float(fit["none"]["memory_available_mib"]) / 1024
    for label, colour, m, marker in STACKS:
        counts = sorted(n for mode, n in line if mode == m)
        slope = float(fit[m]["slope_mib_per_agent"]) / 1024
        shared = float(fit[m]["model_file_shared_mib"]) + float(fit[m]["cache_file_shared_mib"])
        start = (float(fit[m]["intercept_mib"]) + shared) / 1024
        fits = int(fit[m]["agents_that_fit_by_line"])
        ax.plot([counts[-1], fits], [start + slope * counts[-1], start + slope * fits], color=colour,
                linewidth=1.4, linestyle=(0, (3, 2)))
        ax.plot(counts, [(float(line[m, n]["memory_mib"]) + shared) / 1024 for n in counts],
                color=colour, marker=marker, markersize=5.5, linewidth=2, markeredgecolor="white",
                markeredgewidth=0.9, label=label)
    ax.axhline(room, color=INK, linewidth=1.1, linestyle=(0, (4, 2)))
    ax.text(0.03, room * 1.015, f"Available: {room:.0f} GiB", ha="left", va="bottom", fontsize=9.5,
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
    parts = (("Weight Copies", COPY, (none - weights_only, kv_only - both, 0.0, 0.0)),
             ("KV Copies", "#c9c7c1", (weights_only - both, 0.0, weights_only - both, 0.0)),
             ("Shared Files", OURS_LIGHT, (0.0, cache_file, model_file, model_file + cache_file)),
             ("Agent State", OURS, (both, both, both, both)))
    fig, ax = panel()
    names = ["Unmodified", "KV\nShared", "Weights\nShared", NAME]
    bottom = np.zeros(4)
    for label, colour, values in parts:
        ax.bar(names, values, 0.58, bottom=bottom, color=colour, edgecolor="white", linewidth=1.5,
               label=label)
        bottom += np.array(values)
    for x, total in enumerate(bottom):
        ax.annotate(f"{total:.1f}", (x, total), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10.5, color=INK)
    ax.legend(loc="upper right", handlelength=1.1, fontsize=9.5, labelspacing=0.3)
    ax.tick_params(axis="x", labelsize=9.5)
    finish(fig, ax, "eval_stack_parts", "", "Memory (GiB)", bottom.max() * 1.18)

    # how many agents fit by the line of each stack
    fig, ax = panel()
    fits = [int(fit[m]["agents_that_fit_by_line"]) for _, _, m, _ in STACKS]
    bars = ax.bar(names, fits, 0.58, color=[colour for _, colour, _, _ in STACKS], edgecolor="white",
                  linewidth=1.5)
    for bar, (_, _, m, _) in zip(bars, STACKS):
        tip(ax, bar, f"{bar.get_height():.0f}")
    ax.tick_params(axis="x", labelsize=9.5)
    finish(fig, ax, "eval_stack_fit", "", "Agents That Fit", max(fits) * 1.18)

    # generation speed of eight agents relative to the unmodified stack, with
    # the model file in the page cache (4 KiB pages) and on a tmpfs with 2 MiB
    # pages; a stack that copies the weights does not depend on the page size
    huge_dir = "20261007-engine-kvstack-huge-v1"
    huge = ({(r["mode"], int(r["agents"])): r
             for r in rows(huge_dir, "engine_kvstack_summary.csv") if int(r["runs"]) > 0}
            if have(huge_dir, "engine_kvstack_summary.csv") else {})
    fig, ax = panel()
    shown = [("KV\nShared", "kv", DEMAND, "#b9cfe9"), ("Weights\nShared", "weights", WEIGHTS_ONLY, "#f0d6a4"),
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
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=9.5, color=INK)
    ax.set_xticks(range(len(shown)), [label for label, _, _, _ in shown])
    ax.tick_params(axis="x", labelsize=9.5)
    ax.axhline(1.0, color=AXIS, linewidth=1.0)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#c9c7c1"), plt.Rectangle((0, 0), 1, 1, color=COPY)]
    ax.legend(handles, ["4 KiB Pages", "2 MiB Pages"], loc="upper center", ncol=2,
              handlelength=1.1, fontsize=9.5, columnspacing=1.2)
    finish(fig, ax, "eval_stack_speed", "", "Speed vs. Unmodified", 1.42)


# ------------------------------------------------- loss in the 6-SM instance
def local():
    """Where the loss of speed with host memory comes from, by MIG instance."""
    cells = (("12 SMs", "20261007-engine-kvlocal-12sm-v1"), ("6 SMs", "20261007-engine-kvlocal-6sm-v1"))
    tints = ("#c9c7c1", COPY)
    if all(have(d, "engine_kvmech_summary.csv") for _, d in cells):
        modes = (("Private\n2 MiB", "host_copy_huge"), ("Private\n4 KiB", "host_copy"),
                 (NAME, "extent"), ("Tail\n2 MiB", "grow_all"), ("Prefix\n4 KiB", "small_pages"))
        fig, ax = panel()
        width = 0.38
        for index, ((label, directory), colour) in enumerate(zip(cells, tints)):
            data = {r["mode"]: r for r in rows(directory, "engine_kvmech_summary.csv")}
            values = [float(data[m]["children_tps_vs_copy"]) for _, m in modes]
            low = [v - float(data[m]["vs_copy_low"]) for v, (_, m) in zip(values, modes)]
            high = [float(data[m]["vs_copy_high"]) - v for v, (_, m) in zip(values, modes)]
            ax.bar(np.arange(len(modes)) + (index - 0.5) * width, values, width, color=colour,
                   edgecolor="white", linewidth=1.5, label=label, yerr=[low, high],
                   error_kw={"ecolor": INK, "elinewidth": 1.0, "capsize": 2})
        ax.set_xticks(range(len(modes)), [label for label, _ in modes])
        ax.tick_params(axis="x", labelsize=8.5)
        ax.axhline(1.0, color=AXIS, linewidth=1.0)
        ax.set_ylim(0.8, 1.06)
        ax.legend(loc="upper right", ncol=2, handlelength=1.1, fontsize=9.5, columnspacing=1.0)
        ax.set_ylabel("Speed vs. Device Cache")
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
            for count, label in zip(sms, ("12 SMs", "6 SMs")):
                groups.append(f"{name}\n{label}")
                order.append((kernel, count))
        series = [(label, colour, [float(cell[count, kind][f"{kernel}_vs_device"]) for kernel, count in order])
                  for label, colour, kind in (("Host, 2 MiB Pages", OURS, "host_huge"),
                                               ("Host, 4 KiB Pages", OURS_LIGHT, "host_small"))]
        fig, ax = panel()
        for group in grouped(ax, groups, series, 0.36):
            labels(ax, group, "{:.2f}", 9)
        ax.axhline(1.0, color=AXIS, linewidth=1.0)
        ax.tick_params(axis="x", labelsize=9)
        ax.legend(loc="upper right", handlelength=1.1, fontsize=9, labelspacing=0.25)
        finish(fig, ax, "eval_local_probe", "", "Rate vs. Device Memory", 1.45)


# ------------------------------------------------------------- the pipeline
def pipe():
    """A planner, leaders and workers on the tools of a public benchmark."""
    directory = "20261007-engine-kvpipe-v1"
    if not have(directory, "engine_kvpipe_summary.csv"):
        return
    data = {r["mode"]: r for r in rows(directory, "engine_kvpipe_summary.csv")}
    stacks = [(label, m, colour) for label, m, colour in (("Unmodified", "stock", COPY), ("Copy", "copy", "#b7b5af"),
                                                           (NAME, "chain", OURS)) if m in data]
    names = [label for label, _, _ in stacks]
    phases = (("Prefix", "prefix", "#d9d7d1"), ("Leaders", "leaders", "#a9a7a1"), ("Workers", "workers", "#6f6d68"))

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
                        ha="center", va="bottom", fontsize=10.5, color=INK)
        ax.legend(loc="upper center", ncol=3, handlelength=1.0, fontsize=9.5, columnspacing=1.0)
        finish(fig, ax, name, "", ylabel, bottom.max() * 1.36)

    stacked("eval_pipe_time", "{}_s", "Completion Time (s)", 1.0, "{:.1f}")
    stacked("eval_pipe_energy", "vin_{}_j", "Energy (kJ)", 1000.0, "{:.2f}")
    fig, ax = panel()
    memory = [(float(data[m]["memory_mib"]) + float(data[m]["files_mib"])) / 1024 for _, m, _ in stacks]
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
        ax.legend(loc="upper right", handlelength=1.1, fontsize=10)
        ax.tick_params(axis="x", labelsize=9.5)
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
            ax.text((seconds[0] + seconds[first]) / 2, full / 1024, "Every Process\nRuns", ha="center",
                    va="bottom", fontsize=9, color=INK, linespacing=1.0)
            ax.text((seconds[first] + seconds[last]) / 2, points[first][2] / 1024, "First Subtree\nHas Left",
                    ha="center", va="bottom", fontsize=9, color=INK, linespacing=1.0)
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
    stacks = (("Unmodified", COPY, "stock"), ("CUDA IPC Library", DEVICE, "ipc"), (NAME, OURS, "inplace"))
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
            ax.texts[-1].set_fontsize(9.5)
    ax.legend(loc="upper center", ncol=3, handlelength=1.0, fontsize=8.5, columnspacing=0.8)
    finish(fig, ax, "eval_wshare_mem", "", "Memory (GiB)", max(max(v) for _, _, v in series) * 1.3)


if __name__ == "__main__":
    for figure in (mem, weights, hand, speed, reach, mode, cow, dev, sota, attach, protect, tree, scale,
                   variants, inproc, server, limit, stack, local, pipe, deep, wshare, prefix_length):
        figure()
        print("wrote", figure.__name__)
