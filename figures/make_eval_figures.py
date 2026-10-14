#!/usr/bin/env python3
"""Evaluation graphs of STATOR, in the figure style of the PHASOR paper.

Reads the packaged summaries under ../results (nothing is typed in by hand)
and writes one PDF and one PNG per figure next to this script.

  eval_weights    weights in place: memory of N processes, speed by page size
  eval_memory     memory of N agents on one prefix
  eval_handover   handing a prefix over, attaching to it, first token
  eval_cow        extents against copy-on-write mappings
  eval_speed      generation speed of a host-memory cache, by MIG instance
  eval_inproc     eight agents: separate processes, one server, two servers
  eval_vmm        host extents against a device-memory cache and the copy
  bg_state        background: memory of processes that hold the same state
"""
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

# One colour per entity, the same in every figure. The proposed design is
# teal; a lighter step of the same hue is its variant with the whole tail
# allocated. Upstream (copy) is the neutral grey.
OURS, OURS_LIGHT = "#1a9a8f", "#8fd3cb"
COPY = "#898781"
COW = "#eb6834"
DEVICE = "#7b61c9"
INK, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "font.weight": "normal",
    "axes.labelweight": "normal", "axes.labelsize": 12.5, "axes.linewidth": 0.8,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "text.color": INK,
    "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "xtick.color": INK,
    "ytick.color": INK, "legend.fontsize": 11.5, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white",
})


# Half-column panels; make_paper_figures.py sets its own size and text sizes.
PANEL = (3.6, 2.7)
TIP = 10.5
# the smallest sizes that fit_texts sets for an axis label, a tick label and a value
LABEL_MIN, TICK_MIN, VALUE_MIN = 11.0, 10.0, 9.0
PAD = 1.08       # margin of the layout around a plot, in units of the text size
FIXED = False    # a plot is saved as its whole canvas, so that all panels have one size


# The campaigns of the paper were measured again on the whole GPU with MIG
# disabled (2026-10-14, result directories *-<runner>-full-v1). A figure
# names the first campaign; resolve() reads the rerun of the same runner when
# it exists. STATOR_FULL=0 reads the first campaigns; STATOR_FULL=strict fails
# when a rerun is missing, so that no figure mixes the two platforms.
FULL = {
    "20261005-engine-agents-v1": "engine-agents",
    "20261005-engine-pages-v1": "engine-pages",
    "20261006-engine-kvbatch-v1": "engine-kvbatch",
    "20261010-engine-kvbatch-v2": "engine-kvbatch",
    "20261006-engine-kvcow-v1": "engine-kvcow",
    "20261006-engine-kvlimit-v1": "engine-kvlimit",
    "20261006-engine-kvscale-v1": "engine-kvscale",
    "20261007-engine-kvscale-32-v2": "engine-kvscale",
    "20261007-engine-kvscale-p20-v1": "engine-kvscale-p20",
    "20261007-engine-kvscale-p600-v1": "engine-kvscale-p600",
    "20261006-engine-kvscale-agent-v1": "engine-kvscale-agent",
    "20261006-engine-kvscale-llama8b-v1": "engine-kvscale-llama8b",
    "20261006-engine-kvscale-qwen14b-v1": "engine-kvscale-qwen14b",
    "20261006-engine-kvserver-v1": "engine-kvserver2",
    "20261010-engine-kvserver2-v2": "engine-kvserver2",
    "20261006-engine-kvshare-v1": "engine-kvshare",
    "20261006-engine-kvspeed-v1": "engine-kvspeed",
    "20261006-engine-kvspeed-long-v1": "engine-kvspeed-long",
    "20261006-engine-kvvmm-v1": "engine-kvvmm",
    "20261006-ollama-agents-v1": "ollama-agents",
    "20261006-protect-v1": "protect",
    "20261006-read-path-v1": "read-path",
    "20261006-vmm-attach-v1": "vmm-attach",
    "20261007-tlb-probe-v1": "tlb-probe",
    "20261007-engine-kvdeep-v1": "engine-kvdeep",
    "20261007-engine-kvdeep-wide-v1": "engine-kvdeep-wide",
    "20261007-engine-kvlocal-12sm-v1": "engine-kvlocal",
    "20261007-engine-kvmech-v1": "engine-kvmech",
    "20261010-engine-kvmech-v2": "engine-kvmech",
    "20261013-engine-kvmech-cowplain-v1": "engine-kvmech",
    "20261007-engine-kvablate-v1": "engine-kvablate",
    "20261007-engine-kvstack-v1": "engine-kvstack",
    "20261007-engine-kvstack-huge-v1": "engine-kvstack-huge",
    "20261008-engine-kvstack-capacity-v2": "engine-kvstack-capacity",
    "20261008-engine-kvstack-capacity-136-v1": "engine-kvstack-capacity",
    "20261008-engine-kvstack-capacity-140-v1": "engine-kvstack-capacity",
    "20261009-engine-kvstack-capacity-copy-v1": "engine-kvstack-capacity",
    "20261008-engine-kvstack-qwen14b-v1": "engine-kvstack-qwen14b",
    "20261008-engine-kvstack-qwen14b-none8-v1": "engine-kvstack-qwen14b-none8",
    "20261008-engine-kvstack-capacity-qwen14b-118-v1": "engine-kvstack-capacity-qwen14b",
    "20261010-engine-kvstack-qwen14b-huge-v1": "engine-kvstack-qwen14b-huge",
    "20261007-ext-vllm-v1": "ext-vllm",
    "20261007-ext-weightshare-v1": "ext-weightshare",
    "20261008-vllm-share-v2": "vllm-share",
    "20261010-engine-kvmps-v2": "engine-kvmps",
    "20261010-engine-kvpipe-huge-v2": "engine-kvpipe-huge",
    "20261010-engine-kvtree-v2": "engine-kvtree",
    "20261012-engine-kvresident-v1": "engine-kvresident",
    "20261012-engine-kvpipefault-v1": "engine-kvpipefault",
    "20261010-engine-kvfaultcount-v2": "engine-kvfaultcount",
    "20261010-engine-kvfork-v2": "engine-kvfork",
    "20261007-engine-kvscale-gen2k-v1": "engine-kvscale-gen2k",
    "20261005-engine-adapters-v1": "engine-adapters",
}
# Campaigns that only exist with MIG instances; a figure of the whole GPU
# must not read them.
MIG_ONLY = {"20261006-engine-kvspeed-6sm-v1", "20261006-engine-kvspeed-6sm-long-v1",
            "20261007-engine-kvlocal-6sm-v1", "20261012-engine-kvlatency-v1",
            "20261013-engine-kvplace-v1"}
MODE = os.environ.get("STATOR_FULL", "1")


def resolve(directory):
    """The directory to read for a campaign name of the first platform."""
    if MODE == "strict" and directory in MIG_ONLY:
        raise SystemExit(f"{directory} exists only with MIG instances")
    if MODE == "0" or directory not in FULL:
        return directory
    name = FULL[directory]
    found = sorted(d for d in os.listdir(RES) if d.endswith(f"-{name}-full-v1")
                   and os.path.exists(os.path.join(RES, d, "source_hashes.txt")))
    if found:
        return found[-1]
    if MODE == "strict":
        raise SystemExit(f"no rerun on the whole GPU for {directory} ({name})")
    return directory


def rows(*path):
    with open(os.path.join(RES, resolve(path[0]), *path[1:]), newline="") as handle:
        return list(csv.DictReader(handle))


def style(ax, grid="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.tick_params(length=0)


def _overlaps(boxes, apart=0.0):
    """Pairs of boxes that overlap, or that are side by side with less than `apart` pixels between them."""
    pairs = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if min(a.x1, b.x1) - max(a.x0, b.x0) > 1 - apart and min(a.y1, b.y1) - max(a.y0, b.y0) > 1:
                pairs.append((boxes[i][0], boxes[j][0]))
    return pairs


def _raise_top(fig, ax):
    """Raises the upper limit of the value axis until every text ends below it."""
    for _ in range(120):
        fig.canvas.draw()
        box = ax.get_window_extent()
        top = max(text.get_window_extent().y1 for text in ax.texts)
        if top <= box.y1 - 1.0:
            return
        low, high = ax.get_ylim()
        if ax.get_yscale() == "log":
            ax.set_ylim(low, high * 1.25)
        else:
            ax.set_ylim(low, high + 0.04 * (high - low))


def _upright(fig, ax, values):
    for text in values:
        text.set_rotation(90)
    _raise_top(fig, ax)


def _layout(fig):
    rect = getattr(fig, "layout_rect", None)
    if rect is None:
        fig.tight_layout(pad=PAD)
    else:
        fig.tight_layout(pad=PAD, rect=rect)


def _beyond(fig):
    """By how many inches the content of a figure leaves its canvas. The
    labels of the axes count at their whole length, which the layout of
    Matplotlib leaves out for a label that is longer than its axis."""
    renderer = fig.canvas.get_renderer()
    boxes = [ax.get_tightbbox(renderer, for_layout_only=False) for ax in fig.axes]
    boxes += [ax.xaxis.label.get_window_extent() for ax in fig.axes if ax.xaxis.label.get_text()]
    boxes += [ax.yaxis.label.get_window_extent() for ax in fig.axes if ax.yaxis.label.get_text()]
    boxes += [item.get_window_extent() for item in fig.legends]
    tight = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted())
    width, height = fig.get_size_inches()
    return max(-tight.x0, tight.x1 - width, -tight.y0, tight.y1 - height), tight


def _fit_axes(fig, ax, name, report):
    def warn(text):
        if report:
            print(f"WARN {name}: {text}")

    fig.canvas.draw()
    box = ax.get_window_extent()
    label = ax.yaxis.label
    if label.get_text() and "\n" not in label.get_text() and \
            label.get_window_extent().height > box.height + 2:
        words = label.get_text().split(" ")
        if len(words) > 1:
            best = min(range(1, len(words)),
                       key=lambda k: abs(len(" ".join(words[:k])) - len(" ".join(words[k:]))))
            label.set_text(" ".join(words[:best]) + "\n" + " ".join(words[best:]))
            _layout(fig)
    # the label of the value axis is as long as that axis at most, and the
    # label of the x axis as wide as the canvas within its margins
    for axis_label, extent in ((ax.yaxis.label, "height"), (ax.xaxis.label, "width")):
        while axis_label.get_text():
            fig.canvas.draw()
            room = ax.get_window_extent().height if extent == "height" else \
                fig.bbox.width - 2 * PAD * plt.rcParams["font.size"] * fig.dpi / 72
            if getattr(axis_label.get_window_extent(), extent) <= room + 2:
                break
            if axis_label.get_fontsize() - 0.5 < LABEL_MIN:
                warn(f"the label '{axis_label.get_text()}' is longer than its axis")
                break
            axis_label.set_fontsize(axis_label.get_fontsize() - 0.5)
            _layout(fig)
    if FIXED and ax.xaxis.label.get_text():
        fig.canvas.draw()
        margin = PAD * plt.rcParams["font.size"] * fig.dpi / 72
        extent = ax.xaxis.label.get_window_extent()
        shift = min(0.0, fig.bbox.width - margin - extent.x1) + max(0.0, margin - extent.x0)
        if abs(shift) > 0.5:    # centred under the axis unless it would leave the canvas
            ax.xaxis.label.set_x(ax.xaxis.label.get_position()[0] + shift / ax.get_window_extent().width)
    for _ in range(10):
        fig.canvas.draw()
        ticks = [t.get_window_extent() for t in ax.get_xticklabels() if t.get_text()]
        size = ax.get_xticklabels()[0].get_fontsize()
        gap = 0.22 * size * fig.dpi / 72    # two thirds of a space between words
        if not any(a.x1 > b.x0 - gap for a, b in zip(ticks, ticks[1:])):
            break
        if size - 0.5 < TICK_MIN:
            warn("tick labels on the x axis overlap")
            break
        ax.tick_params(axis="x", labelsize=size - 0.5)
        _layout(fig)
    if not ax.texts:
        return
    values = [text for text in ax.texts if re.fullmatch(r"[\d.,]+", text.get_text())]
    _raise_top(fig, ax)
    bars = [patch for patch in ax.patches if isinstance(patch, plt.Rectangle) and patch.get_height() > 0]
    fig.canvas.draw()
    marks = [("bar", bar.get_window_extent()) for bar in bars]
    apart = 0.2 * TIP * fig.dpi / 72    # room between a value and the next value or bar
    if values and any(text.get_rotation() == 0 for text in values) and any(
            text.get_va() != "center" and _overlaps([(text.get_text(), text.get_window_extent()), mark], apart)
            for text in values for mark in marks):
        _upright(fig, ax, values)
    for _ in range(8):
        fig.canvas.draw()
        pairs = _overlaps([(text.get_text(), text.get_window_extent()) for text in values], apart)
        if not pairs:
            break
        if any(text.get_rotation() == 0 for text in values):
            _upright(fig, ax, values)
            continue
        size = values[0].get_fontsize()
        if size - 0.5 < VALUE_MIN:
            break
        for text in values:
            text.set_fontsize(size - 0.5)
        _raise_top(fig, ax)
    fig.canvas.draw()
    box = ax.get_window_extent()
    boxes = [(text.get_text(), text.get_window_extent()) for text in ax.texts if text.get_text()]
    for label_text, b in boxes:
        if b.x0 < box.x0 - 1 or b.x1 > box.x1 + 1 or b.y1 > box.y1 + 1 or b.y0 < box.y0 - 1:
            warn(f"text '{label_text}' leaves the axes")
    for first, second in _overlaps(boxes):
        warn(f"texts '{first}' and '{second}' overlap")
    marks = [("bar", bar.get_window_extent()) for bar in bars]
    for text in values:
        if text.get_va() != "center" and \
                any(_overlaps([(text.get_text(), text.get_window_extent()), mark]) for mark in marks):
            warn(f"value '{text.get_text()}' lies on a bar")


def fit_texts(fig, name):
    """Keeps the text of a plot inside the plot and apart:

    an axis label that is longer than its axis is set on two lines (value
    axis) and then smaller; tick labels of the x axis that touch are set
    smaller; values on marks that overlap one another or a bar of another
    series are set upright and, if they still overlap, smaller; and the upper
    limit of the value axis is raised until the highest text ends below the
    top of the axes. No text is set below LABEL_MIN, TICK_MIN or VALUE_MIN.
    With a fixed canvas the plot is laid out again after these steps, which
    change the width of its labels, until it lies on the canvas. What none of
    this repairs, and content that leaves the canvas, is reported."""
    if FIXED and fig.axes:
        for _ in range(6):
            _layout(fig)
            for ax in fig.axes:
                _fit_axes(fig, ax, name, False)
            fig.canvas.draw()
            if _beyond(fig)[0] <= 0.0:
                break
    for ax in fig.axes:
        _fit_axes(fig, ax, name, True)
    fig.canvas.draw()
    for ax in fig.axes:
        label = ax.yaxis.label
        if label.get_text() and label.get_window_extent().height > ax.get_window_extent().height + 6:
            print(f"WARN {name}: the label of the value axis is longer than the axis")
    if fig.axes:
        over, tight = _beyond(fig)
        width, height = fig.get_size_inches()
        if over > (0.0 if FIXED else 0.01):
            print(f"WARN {name}: content leaves the canvas by {over * 72:.1f} pt "
                  f"(x {tight.x0 * 72:.0f}..{tight.x1 * 72:.0f} of {width * 72:.0f}, "
                  f"y {tight.y0 * 72:.0f}..{tight.y1 * 72:.0f} of {height * 72:.0f})")


def save(fig, name):
    if FIXED:   # the panels of the paper; the figures of the README are saved as drawn
        fit_texts(fig, name)
    # Matplotlib otherwise embeds the wall-clock creation time, which makes
    # a figure rebuilt from identical CSV inputs differ byte-for-byte.
    pdf_metadata = {"CreationDate": None, "ModDate": None}
    if FIXED and fig.axes:
        fig.savefig(os.path.join(HERE, name + ".pdf"), metadata=pdf_metadata)
        fig.savefig(os.path.join(HERE, name + ".png"), dpi=200)
        plt.close(fig)
        return
    fig.savefig(os.path.join(HERE, name + ".pdf"), bbox_inches="tight", pad_inches=0.03,
                metadata=pdf_metadata)
    fig.savefig(os.path.join(HERE, name + ".png"), bbox_inches="tight", pad_inches=0.06, dpi=200)
    plt.close(fig)


def grouped(ax, groups, series, width=0.25):
    """series: list of (label, colour, values); bars of a group touch with a white gap."""
    x = np.arange(len(groups))
    n = len(series)
    bars = []
    for i, (label, colour, values) in enumerate(series):
        offset = (i - (n - 1) / 2) * width
        bars.append(ax.bar(x + offset, values, width, color=colour, label=label,
                           edgecolor="white", linewidth=1.5))
    ax.set_xticks(x, groups)
    return bars


def tip(ax, bar, text, dy=0.0):
    ax.annotate(text, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3 + dy), textcoords="offset points", ha="center",
                va="bottom", fontsize=TIP, color=INK)


# ----------------------------------------------------------------- weights
def eval_weights():
    agents = {(r["config"], r["mode"], int(r["agents"])): r
              for r in rows("20261005-engine-agents-v1", "engine_agents_summary.csv")}
    pages = {r["mode"]: r for r in rows("20261005-engine-pages-v1", "engine_pages_summary.csv")}
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.0), gridspec_kw={"width_ratios": [1.25, 1]})

    counts = [1, 2, 4, 8]
    copy = [float(agents["mig", "copy", n]["memory_total_mib"]) / 1024 for n in counts]
    place = [float(agents["mig", "inplace", n]["memory_total_mib"]) / 1024 for n in counts]
    bars = grouped(left, [str(n) for n in counts],
                   [("Device copy (upstream)", COPY, copy), ("Weights in place", OURS, place)], 0.34)
    tip(left, bars[0][-1], f"{copy[-1]:.1f}")
    tip(left, bars[1][-1], f"{place[-1]:.1f}")
    left.set_xlabel("Serving Processes")
    left.set_ylabel("Memory (GiB)")
    left.set_ylim(0, max(copy) * 1.15)
    left.legend(loc="upper left")
    style(left)

    names = [("inplace_4k", "4 KiB\npage cache"), ("inplace_thp", "2 MiB\ntmpfs"),
             ("inplace_hugetlb", "2 MiB\nhugetlbfs")]
    y = [float(pages[k]["generation_vs_copy"]) for k, _ in names]
    low = [float(pages[k]["lower_ci95"]) for k, _ in names]
    high = [float(pages[k]["upper_ci95"]) for k, _ in names]
    x = np.arange(len(names))
    right.axhline(1.0, color=COPY, linewidth=1.6)
    right.errorbar(x, y, yerr=[np.subtract(y, low), np.subtract(high, y)], fmt="o",
                   color=OURS, markersize=8, linewidth=2, capsize=4)
    for xi, yi in zip(x, y):
        right.annotate(f"{yi:.3f}", (xi, yi), xytext=(9, -3), textcoords="offset points",
                       fontsize=10.5)
    right.annotate("device copy", (-0.45, 1.0), xytext=(0, 4),
                   textcoords="offset points", ha="left", fontsize=10.5, color=MUTED)
    right.set_xticks(x, [label for _, label in names])
    right.set_xlim(-0.5, len(names) - 0.4)
    right.set_ylim(0.88, 1.05)
    right.set_ylabel("Generation Speed\n(relative, 95% CI)")
    right.set_xlabel("Pages of the Mapped Model")
    style(right)
    fig.tight_layout(w_pad=2.0)
    save(fig, "eval_weights")


# ----------------------------------------------------------------- memory
def eval_memory():
    data = {(r["paragraphs"], r["mode"], int(r["agents"])): r
            for r in rows("20261006-engine-kvshare-v1", "engine_kvshare_summary.csv")}
    sizes = sorted({k[0] for k in data}, key=int)
    counts = [1, 4, 8]
    fig, axes = plt.subplots(1, len(sizes), figsize=(9.6, 3.0), sharey=True)
    for ax, size in zip(axes, sizes):
        tokens = int(data[size, "restore", 1]["prefix_tokens"])
        series = []
        for label, colour, mode in (("Copy (upstream)", COPY, "restore"),
                                    ("Extents, whole tail", OURS_LIGHT, "extent"),
                                    ("Extents, tail follows use", OURS, "extent_lazy")):
            series.append((label, colour,
                           [float(data[size, mode, n]["memory_mib"]) / 1024 for n in counts]))
        bars = grouped(ax, [str(n) for n in counts], series, 0.26)
        for group in bars:
            tip(ax, group[-1], f"{group[-1].get_height():.1f}")
        ax.set_title(f"{tokens:,}-Token Prefix", fontsize=12.5, fontweight="bold")
        ax.set_xlabel("Agents")
        style(ax)
    axes[0].set_ylabel("Memory of the Agents (GiB)")
    axes[0].set_ylim(0, 23)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08),
               columnspacing=1.6, handlelength=1.4)
    fig.tight_layout(w_pad=1.5)
    save(fig, "eval_memory")


# --------------------------------------------------------------- handover
def eval_handover():
    publish = {(r["paragraphs"], r["store"]): r
               for r in rows("20261006-engine-kvshare-v1", "engine_kvshare_publish.csv")}
    data = {(r["paragraphs"], r["mode"], int(r["agents"])): r
            for r in rows("20261006-engine-kvshare-v1", "engine_kvshare_summary.csv")}
    sizes = sorted({k[0] for k in publish}, key=int)
    long = sizes[-1]
    counts = [1, 4, 8]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.0))

    ax = axes[0]
    groups = [f"{int(publish[s, 'device']['prefix_tokens']):,}" for s in sizes]
    copy = [float(publish[s, "device"]["publish_ms"]) for s in sizes]
    ours = [float(publish[s, "huge"]["publish_ms"]) for s in sizes]
    bars = grouped(ax, groups, [("Copy (upstream)", COPY, copy), ("Extents", OURS, ours)], 0.34)
    for group in bars:
        for bar in group:
            tip(ax, bar, f"{bar.get_height():.0f}")
    ax.set_xlabel("Prefix (tokens)")
    ax.set_ylabel("Handing Over (ms)")
    ax.set_ylim(0, max(copy) * 1.18)
    style(ax)

    for ax, column, label, scale in ((axes[1], "attach_ms", "Attach (ms)", 1.0),
                                     (axes[2], "first_token_ms", "First Token (s)", 1000.0)):
        copy = [float(data[long, "restore", n][column]) / scale for n in counts]
        ours = [float(data[long, "extent_lazy", n][column]) / scale for n in counts]
        bars = grouped(ax, [str(n) for n in counts],
                       [("Copy (upstream)", COPY, copy), ("Extents", OURS, ours)], 0.34)
        digits = "{:.0f}" if scale == 1.0 else "{:.2f}"
        for group in bars:
            tip(ax, group[-1], digits.format(group[-1].get_height()))
        ax.set_xlabel(f"Agents ({int(data[long, 'restore', 1]['prefix_tokens']):,}-token prefix)")
        ax.set_ylabel(label)
        ax.set_ylim(0, max(copy) * 1.18)
        style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08),
               columnspacing=1.6, handlelength=1.4)
    fig.tight_layout(w_pad=1.6)
    save(fig, "eval_handover")


# -------------------------------------------------------------------- cow
def eval_cow():
    data = {(r["mode"], int(r["agents"])): r
            for r in rows("20261006-engine-kvcow-v1", "engine_kvcow_summary.csv")}
    ways = [("extent_lazy", "Extents", OURS),
            ("cow", "Copy-on-write mapping, 2 MiB pages, read pass", COW),
            ("cow_small", "Copy-on-write mapping, 4 KiB pages, read pass", COW),
            ("cow_noread", "Copy-on-write mapping, 2 MiB pages, no read pass", COW),
            ("cow_small_noread", "Copy-on-write mapping, 4 KiB pages, no read pass", COW)]
    agents = 8
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 2.9), sharey=True)
    y = np.arange(len(ways))[::-1]
    for ax, column, label, scale, digits in (
            (axes[0], "memory_mib", f"Memory of {agents} Agents (GiB)", 1024.0, "{:.1f}"),
            (axes[1], "first_token_ms", "First Token (s)", 1000.0, "{:.1f}")):
        values = [float(data[mode, agents][column]) / scale for mode, _, _ in ways]
        bars = ax.barh(y, values, 0.58, color=[colour for _, _, colour in ways],
                       edgecolor="white", linewidth=1.5)
        for bar, value in zip(bars, values):
            ax.annotate(digits.format(value), (value, bar.get_y() + bar.get_height() / 2),
                        xytext=(4, 0), textcoords="offset points", va="center", fontsize=10.5)
        ax.set_xlabel(label)
        ax.set_xlim(0, max(values) * 1.14)
        style(ax, "x")
    axes[0].set_yticks(y, [label for _, label, _ in ways])
    fig.tight_layout(w_pad=1.5)
    save(fig, "eval_cow")


# ------------------------------------------------------------------ speed
def eval_speed():
    runs = [("20261006-engine-kvspeed-v1", "12-SM"), ("20261006-engine-kvspeed-long-v1", "12-SM"),
            ("20261006-engine-kvspeed-6sm-v1", "8-SM"), ("20261006-engine-kvspeed-6sm-long-v1", "8-SM")]
    rerun = [(tag, "whole GPU") for tag, _ in runs if resolve(tag) != tag]
    if rerun:
        # the reruns on the whole GPU, which has no instances
        runs = rerun
    labels, points = [], {"anon": [], "extent_lazy": []}
    for tag, instance in runs:
        data = {(r["processes"], r["mode"]): r for r in rows(tag, "engine_kvspeed_summary.csv")}
        tokens = int(data["1", "device"]["prefix_tokens"])
        where = instance if rerun else f"{instance} instance"
        labels.append(f"{where[0].upper() + where[1:]}\n{tokens:,}-token prefix")
        for mode in points:
            r = data["1", mode]
            points[mode].append((float(r["generation_vs_device"]), float(r["lower_ci95"]),
                                 float(r["upper_ci95"])))
    fig, ax = plt.subplots(figsize=(7.6, 3.1))
    y = np.arange(len(labels))[::-1]
    ax.axvline(1.0, color=COPY, linewidth=1.6)
    for mode, label, colour, dy in (("anon", "Private host memory (nothing shared)", COPY, 0.14),
                                    ("extent_lazy", "Extents", OURS, -0.14)):
        mid = np.array([p[0] for p in points[mode]])
        low = np.array([p[1] for p in points[mode]])
        high = np.array([p[2] for p in points[mode]])
        ax.errorbar(mid, y + dy, xerr=[mid - low, high - mid], fmt="o", color=colour,
                    markersize=8, linewidth=2, capsize=4, label=label,
                    markeredgecolor="white", markeredgewidth=1.5)
        if mode == "extent_lazy":
            for m, yy in zip(mid, y + dy):
                ax.annotate(f"{m:.3f}", (m, yy), xytext=(0, -15), textcoords="offset points",
                            ha="center", fontsize=10.5)
    ax.annotate("cache in device memory", (1.0, len(labels) - 0.52), xytext=(-5, 0),
                textcoords="offset points", ha="right", va="center", fontsize=10.5, color=MUTED)
    ax.set_yticks(y, labels)
    ax.set_ylim(-0.6, len(labels) - 0.3)
    ax.set_xlim(0.925, 1.015)
    ax.set_xlabel("Generation Speed (relative, 95% CI)")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, columnspacing=1.4,
              handletextpad=0.3)
    style(ax, "x")
    save(fig, "eval_speed")


# ----------------------------------------------------------------- inproc
def eval_inproc():
    share = {(r["paragraphs"], r["mode"], int(r["agents"])): r
             for r in rows("20261006-engine-kvshare-v1", "engine_kvshare_summary.csv")}
    publish = {(r["paragraphs"], r["store"]): r
               for r in rows("20261006-engine-kvshare-v1", "engine_kvshare_publish.csv")}
    batch = {r["config"]: r for r in rows("20261006-engine-kvbatch-v1", "engine_kvbatch_summary.csv")}
    file_gib = float(publish["320", "huge"]["file_used_mib"]) / 1024
    # (label, memory GiB, tokens/s, marker, colour, label offset in points)
    configs = [
        ("8 processes,\neach copies", float(share["320", "restore", 8]["memory_mib"]) / 1024,
         float(share["320", "restore", 8]["generation_tps_total"]), "o", COPY, (-8, 8, "right")),
        ("8 processes\non extents", float(share["320", "extent_lazy", 8]["memory_mib"]) / 1024 + file_gib,
         float(share["320", "extent_lazy", 8]["generation_tps_total"]), "o", OURS, (10, -4, "left")),
        ("1 server,\n8 sequences", float(batch["one_server"]["memory_mib"]) / 1024,
         float(batch["one_server"]["generation_tps_total"]), "s", COPY, (10, -12, "left")),
        ("2 servers: each computes,\nor the second copies", float(batch["two_servers_compute"]["memory_mib"]) / 1024,
         float(batch["two_servers_compute"]["generation_tps_total"]), "^", COPY, (11, -6, "left")),
        ("", float(batch["two_servers_copy"]["memory_mib"]) / 1024,
         float(batch["two_servers_copy"]["generation_tps_total"]), "^", COPY, (0, 0, "left")),
        ("2 servers\non extents", float(batch["two_servers_extent"]["memory_mib"]) / 1024,
         float(batch["two_servers_extent"]["generation_tps_total"]), "^", OURS, (-2, 10, "center")),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    for label, memory, speed, marker, colour, (dx, dy, align) in configs:
        ax.scatter([memory], [speed], s=150, marker=marker, color=colour, edgecolor="white",
                   linewidth=1.5, zorder=3)
        ax.annotate(label, (memory, speed), xytext=(dx, dy), textcoords="offset points",
                    ha=align, fontsize=10.5, linespacing=0.95)
    ax.set_xlabel("Memory of All Processes (GiB)")
    ax.set_ylabel("Generation (tokens/s)")
    ax.set_title("Eight Agents on a 16,321-Token Prefix", fontsize=12.5, fontweight="bold")
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 128)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=OURS, markersize=10,
                          markeredgecolor="white", label="Prefix on extents"),
               plt.Line2D([], [], marker="o", linestyle="", color=COPY, markersize=10,
                          markeredgecolor="white", label="Prefix copied or computed")]
    ax.legend(handles=handles, loc="center right")
    style(ax, "both")
    save(fig, "eval_inproc")


# -------------------------------------------------------------------- vmm
def eval_vmm():
    data = {(r["paragraphs"], r["parent_mig"], r["mode"]): r
            for r in rows("20261006-engine-kvvmm-v1", "engine_kvvmm_summary.csv")}
    instances = [("fafc828a", "12-SM"), ("31ffbfe4", "8-SM")]
    modes = [("copy", "Copy (upstream)", COPY), ("vmm", "Device-memory cache", DEVICE),
             ("extent", "Host extents", OURS)]
    size = "320"
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.0))
    panels = [("child_attach_ms", "Child Attach (ms)", 1.0, "{:.0f}"),
              ("children_memory_mib", "Memory of 4 Children (GiB)", 1024.0, "{:.1f}"),
              ("child_generation_tps", "Child Generation\n(relative to copy)", None, "{:.2f}")]
    for ax, (column, label, scale, digits) in zip(axes, panels):
        series = []
        for mode, name, colour in modes:
            values = []
            for mig, _ in instances:
                value = float(data[size, mig, mode][column])
                if scale is None:
                    value /= float(data[size, mig, "copy"][column])
                else:
                    value /= scale
                values.append(value)
            series.append((name, colour, values))
        bars = grouped(ax, [name for _, name in instances], series, 0.27)
        for group in bars:
            for bar in group:
                ax.annotate(digits.format(bar.get_height()),
                            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points", ha="center",
                            va="bottom", fontsize=9.5, color=INK)
        ax.set_xlabel("MIG Instance")
        ax.set_ylabel(label)
        top = max(max(values) for _, _, values in series)
        ax.set_ylim(0, top * 1.18)
        style(ax)
    tokens = int(data[size, instances[0][0], "copy"]["prefix_tokens"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=1.6)
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.6, handlelength=1.4)
    fig.suptitle(f"A Parent and Four Children in One MIG Instance, {tokens:,}-Token Prefix",
                 y=1.2, fontsize=12.5, fontweight="bold")
    save(fig, "eval_vmm")


# ------------------------------------------------------------- background
def bg_state():
    """Two half-column panels: what the unmodified engine holds per process."""
    agents = {(r["config"], r["mode"], int(r["agents"])): r
              for r in rows("20261005-engine-agents-v1", "engine_agents_summary.csv")}
    # processes in different MIG instances on the first platform, time-sliced
    # on the whole GPU
    config = "mig" if any(k[0] == "mig" for k in agents) else "timeslice"
    counts = sorted({k[2] for k in agents if k[0] == config and k[1] == "copy"})
    memory = [float(agents[config, "copy", n]["memory_total_mib"]) / 1024 for n in counts]
    fig, ax = plt.subplots(figsize=PANEL)
    bars = ax.bar([str(n) for n in counts], memory, 0.6, color=COPY, edgecolor="white",
                  linewidth=1.5)
    for bar in bars:
        tip(ax, bar, f"{bar.get_height():.1f}")
    model = float(agents[config, "inplace", 1]["model_mapped_mib"]) / 1024
    ax.axhline(model, color=OURS, linewidth=1.6, linestyle=(0, (4, 2)))
    ax.text(0.04, 0.72, f"dashed: model file\n({model:.1f} GiB)", color=OURS, fontsize=11,
            ha="left", va="center", transform=ax.transAxes)
    ax.set_xlabel("Serving Processes")
    ax.set_ylabel("Memory (GiB)")
    ax.set_ylim(0, max(memory) * 1.15)
    style(ax)
    fig.tight_layout()
    save(fig, "bg_state_weights")

    share = {(r["paragraphs"], r["mode"], int(r["agents"])): r
             for r in rows("20261006-engine-kvshare-v1", "engine_kvshare_summary.csv")}
    sizes = sorted({k[0] for k in share}, key=int)
    counts = [1, 4, 8]
    fig, ax = plt.subplots(figsize=PANEL)
    series = []
    for size, colour in zip(sizes, ("#c3c2b7", COPY)):
        tokens = int(share[size, "restore", 1]["prefix_tokens"])
        series.append((f"{tokens:,} tokens", colour,
                       [float(share[size, "restore", n]["memory_mib"]) / 1024 for n in counts]))
    groups = grouped(ax, [str(n) for n in counts], series, 0.36)
    for group in groups:
        for bar in group:
            tip(ax, bar, f"{bar.get_height():.1f}")
    ax.set_xlabel("Agents on One Prefix")
    ax.set_ylabel("Memory (GiB)")
    ax.set_ylim(0, 23)
    ax.legend(loc="upper left", title="Prefix", title_fontsize=11.5, handlelength=1.2)
    style(ax)
    fig.tight_layout()
    save(fig, "bg_state_kv")


if __name__ == "__main__":
    # bg_state() is drawn by make_paper_figures.py, on the canvas of the paper
    for figure in (eval_weights, eval_memory, eval_handover, eval_cow, eval_speed,
                   eval_inproc, eval_vmm):
        figure()
        print("wrote", figure.__name__)
