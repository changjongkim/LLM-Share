#!/usr/bin/env python3
"""Paper figures from the 10-16..10-20 campaigns, and the generation-window
throughput of the earlier campaigns.

Throughput everywhere is the generation window: the tokens that all agents of
a case generate, divided by the time from the earliest start of generation to
the latest end. For the campaigns that print one RESULT per agent and launch
all agents together, the start of an agent is its first_token_ms and its end
first_token_ms + generation_ms. TTFT is in milliseconds.

    python3 make_paper_figures_1020.py
"""
import csv
import json
import math
import os
import re
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import make_paper_figures as base
from make_eval_figures import AXIS
from make_paper_figures import (COPY, COW, DEMAND, DEVICE, OURS, OURS_LIGHT, COPY_LIGHT, NAME, DENSE, VALUE, EXISTING, OLLAMA, STACKS,
                                panel, legend, labels, finish, grouped, save, style)

KCJ = "/home/thor/kcj"
RES = KCJ + "/llm_share/results"
REPORTS = KCJ + "/portable_bench_a30/validation/formal_reports"
E4B = KCJ + "/portable_bench_a30/validation/thor_e4b/analysis"
E4BX = KCJ + "/portable_bench_a30/validation/thor_e4b_ext/analysis"
MECHS = (("Copy", COPY, "copy"), ("Demand", DEMAND, "demand"), ("Device", DEVICE, "device_merged"),
         ("CoW Map", COW, "writable_no_read"), (NAME, OURS, "extent"))
# review of 2026-10-21 (130): STATOR with its conversion replaced by a copy of the rows into a file, which the
# children map read-only with the other steps of STATOR. It carries steps of STATOR, so it is an ablation of STATOR
# in a tint of its colour, just before the full system.
MECHS6 = (("Copy", COPY, "copy"), ("Demand", DEMAND, "demand"), ("Device", DEVICE, "device_merged"),
          ("CoW Map", COW, "writable_no_read"), (f"{NAME} (Copied File)", OURS_LIGHT, "copy_map"), (NAME, OURS, "extent"))


def geo(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def csv_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


# ------------------------------------------------ generation window, legacy
def windows(directory, kind, key_fields):
    """{(case key): {run: window tokens/s}} from AGENT/CHILD lines of raw.log."""
    out = defaultdict(dict)
    run, case = None, None
    for line in open(os.path.join(RES, directory, "raw.log")):
        f = dict(re.findall(r"(\w+)=(\S+)", line))
        if line.startswith("BEGIN_"):
            run = f.get("run", run)
        elif line.startswith("CASE"):
            case = tuple(f.get(k, "") for k in key_fields)
            out[case].setdefault(run, [])
        elif line.startswith(kind + " ") and case is not None and f.get("exit", "0") in ("0",) \
                and f.get("generation_ms") and f.get("first_token_ms"):
            if f.get("role") not in (None, "child"):
                continue
            start = float(f["first_token_ms"])
            out[case][run].append((start, start + float(f["generation_ms"]), int(f.get("generated", "64"))))
    rates = defaultdict(dict)
    for case, by_run in out.items():
        for r, agents in by_run.items():
            if agents:
                begin, end = min(a[0] for a in agents), max(a[1] for a in agents)
                rates[case][r] = sum(a[2] for a in agents) / ((end - begin) / 1000.0)
    return rates


def paired(rates, case, base_case):
    a, b = rates.get(case, {}), rates.get(base_case, {})
    common = [r for r in a if r in b]
    return geo([a[r] / b[r] for r in common]) if common else float("nan")


def stack_speed():
    """eval_stack_d: whole stacks of eight agents against the existing engine."""
    # the same order and colors as the other panels of the stack figure (STACKS)
    stacks = tuple((lab, col, m) for lab, col, m, _ in STACKS)
    groups = (("4 KiB", "20261015-engine-kvstack-full-v1"), ("2 MiB", "20261015-engine-kvstack-huge-full-v1"))
    values = {}
    for label, directory in groups:
        rates = windows(directory, "AGENT", ("mode", "agents"))
        values[label] = [paired(rates, (m, "8"), ("none", "8")) for _, _, m in stacks]
    fig, ax = panel()
    series = [(lab, col, [values[g][i] for g, _ in groups]) for i, (lab, col, _) in enumerate(stacks)]
    for group in grouped(ax, [g for g, _ in groups], series, 0.2):
        labels(ax, group, "{:.2f}", DENSE)
        for text in ax.texts[-len(group):]:
            text.set_rotation(90)
    ax.axhline(1.0, color=AXIS, linewidth=1.0)
    finish(fig, ax, "eval_stack_speed", "Model Page Size", "Relative Speed", 1.35)
    return values


def ablation_speed():
    directory = base.base.resolve("20261007-engine-kvablate-v1")
    rates = windows(directory, "CHILD", ("mode",))
    return {m[0]: paired(rates, m, ("copy",)) for m in rates}


# ------------------------------------------------------------- hand-over
def sota_e1(summary=REPORTS + "/107-live-publish-formal/analysis_summary.csv", mechs=MECHS, long_only=False):
    """Figure of the hand-over: pause, child memory, attach, child TTFT (107; with HANDOVER_RESULT, the
    campaign 130 of the review, which adds the copied file)."""
    data = {(r["prefix_tokens"], r["children"], r["mode"]): r for r in csv_rows(summary)}
    MECHS = mechs
    legend("eval_sota_legend", [(lab, col) for lab, col, _ in MECHS], len(MECHS))
    cells = [("4081", "8"), ("16321", "8"), ("16321", "32")]
    pause_cells = cells[:2]
    names = ["4,081\n8 Agents", "16,321\n8 Agents", "16,321\n32 Agents"]
    if long_only:
        # six mechanisms: the values of three groups do not fit, and the text reads (b)-(d) on 16,321 tokens only
        cells, names = cells[1:], names[1:]
    for name, column, ylabel, scale, form, log, which in (
            ("eval_hand_pause", "publish_ms", "Pause (ms, log)", 1.0, "{:.0f}", True, pause_cells),
            ("eval_sota_mem", "child_increment_peak_gib", "Memory (GiB)", 1.0, "{:.1f}", False, cells),
            ("eval_sota_attach", "child_attach_ms", "Attach Time (ms, log)", 1.0, "{:.0f}", True, cells),
            ("eval_sota_ttft", "child_ttft_ms", "TTFT (ms)", 1.0, "{:.0f}", False, cells)):
        fig, ax = panel()
        def value(row):
            if column == "child_attach_ms":
                # attach as in the earlier campaigns: context creation and state load
                return float(row["child_context_ms"]) + float(row["child_attach_ms"])
            return float(row[column]) / scale
        series = [(lab, col, [value(data[c + (m,)]) for c in which]) for lab, col, m in MECHS]
        group_names = ["4,081", "16,321"] if which is pause_cells else names
        for group in grouped(ax, group_names, series, 0.85 / len(MECHS)):
            labels(ax, group, form, DENSE)
            for text in ax.texts[-len(group):]:
                text.set_rotation(90)
        top = max(max(v) for _, _, v in series)
        if log:
            ax.set_ylim(1, top * 14)
        finish(fig, ax, name, "Prefix (Tokens)" if which is pause_cells else "", ylabel,
               None if log else top * 1.3, log)
    return data


# ----------------------------------------------------------------- trees
def tree_e2():
    rows = csv_rows(REPORTS + "/109-tree11-formal/analysis_summary.csv") + \
        csv_rows(REPORTS + "/110-tree23-31-formal/analysis_summary.csv")
    data = {(r["agents"], r["mode"]): r for r in rows}
    order = (("Copy", COPY, "copy"), (f"{NAME} (One Level)", OURS_LIGHT, "flat"), (f"{NAME} (Chains)", OURS, "chain"))
    sizes = sorted({a for a, _ in data}, key=int)
    legend("eval_tree_legend", [(lab, col) for lab, col, _ in order], 3)
    for name, column, ylabel, scale, form in (("eval_tree_mem", "total_peak_gib", "Memory (GiB)", 1.0, "{:.1f}"),
                                              ("eval_tree_time", "leaf_ttft_without_harness_wait_ms", "Leaf TTFT (ms)",
                                               1.0, "{:.0f}")):
        fig, ax = panel()
        series = [(lab, col, [float(data[(s, m)][column]) / scale for s in sizes]) for lab, col, m in order]
        for group in grouped(ax, [f"{s}" for s in sizes], series, 0.25):
            labels(ax, group, form, DENSE)
            for text in ax.texts[-len(group):]:
                text.set_rotation(90)
        finish(fig, ax, name, "Processes", ylabel, max(max(v) for _, _, v in series) * 1.35)
    return data


# -------------------------------------------------------------- resident
def resident_117(D=RES + "/20261020-e3-resident-formal-v1"):
    # 124 (20261021-e3-resident-tmpfs-v1) repeats this sweep with the model file on the
    # memory file system of STATOR and Copy up to its own limit of 56 agents
    cases = json.load(open(D + "/cases.json"))
    agg = defaultdict(lambda: defaultdict(list))
    for c in cases:
        if c["status"] != "ok":
            continue
        key = (c["mode"], c["count"])
        rel, starts, ends, tok, resume = c["active_release_ns"], [], [], 0, []
        for f in os.listdir(f"{D}/{c['case']}/timing"):
            path = f"{D}/{c['case']}/timing/{f}"
            tokens = [json.loads(l)["mono_ns"] for l in open(path) if '"event":"token"' in l]
            if not tokens:
                continue
            gen = [json.loads(l)["mono_ns"] for l in open(path) if '"event":"generation_start"' in l]
            starts.append(gen[0]); ends.append(tokens[-1]); tok += len(tokens); resume.append((tokens[0] - rel) / 1e6)
        agg[key]["mem"].append(c["resident_accounted_bytes"] / 2 ** 30)
        agg[key]["tps"].append(tok / ((max(ends) - min(starts)) / 1e9))
        agg[key]["resume"] += resume
    modes = (("Copy", COPY, "copy", "s"), (NAME, OURS, "extent", "o"))
    # the legend strip: the colours of the modes and the line of each statistic
    fig = plt.figure(figsize=(1.9 * 4 + 0.6, 0.38))
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for _, col, _, _ in modes] + \
        [plt.Line2D([], [], color=base.INK, linewidth=2), plt.Line2D([], [], color=base.INK, linewidth=2, linestyle=(0, (3, 2)))]
    fig.legend(handles, [lab for lab, _, _, _ in modes] + ["Median", "95th Percentile"], loc="center", ncol=4,
               columnspacing=1.6, handlelength=1.6, fontsize=12.5)
    save(fig, "eval_resident_legend")
    out = {}
    for name, metric, ylabel in (("eval_resident_mem", "mem", "Memory (GiB)"),):
        fig, ax = panel()
        top = 0
        for lab, col, m, mk in modes:
            pts = sorted((n, statistics.mean(v[metric])) for (mm, n), v in agg.items() if mm == m)
            out[(name, m)] = pts
            top = max(top, max(p[1] for p in pts))
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker=mk, markersize=7, linewidth=2,
                    markeredgecolor="white", markeredgewidth=1.2)
        ax.set_xticks([8, 32, 64, 96, 128])
        finish(fig, ax, name, "Resident Agents", ylabel, top * (1.15 if metric == "mem" else 1.3))
    resume = {k: (sorted(v["resume"])[len(v["resume"]) // 2],
                  sorted(v["resume"])[int(round(0.95 * (len(v["resume"]) - 1)))]) for k, v in agg.items()}
    # the time from the release of an active agent (its tool result) to its first
    # token, over the eight active agents of every repetition: median and 95th percentile
    fig, ax = panel()
    top = 0
    # STATOR first, so that Copy, which ends at fewer agents, stays visible on top of it
    for lab, col, m, mk in reversed(modes):
        pts = sorted((n, resume[(mm, n)]) for (mm, n) in resume if mm == m)
        for index, style_ in ((0, "-"), (1, (0, (3, 2)))):
            ys = [p[1][index] for p in pts]
            top = max(top, max(ys))
            ax.plot([p[0] for p in pts], ys, color=col, marker=mk, markersize=6 if index else 7, linewidth=2,
                    linestyle=style_, markeredgecolor="white", markeredgewidth=1.2)
    ax.set_xticks([8, 32, 64, 96, 128])
    finish(fig, ax, "eval_resident_resume", "Resident Agents", "Next Token (ms)", top * 1.15)
    return out, resume


# ---------------------------------------------------------- Figure 20
def proc8_bars():
    dirs = {n: base.full(f"engine-kvbatch-per8-n{n}") for n in (2, 4, 8)}
    mem, tps = {}, {}
    for n, directory in dirs.items():
        m = {r["config"]: float(r["memory_mib"]) / 1024 for r in base.rows(directory, "engine_kvbatch_n_summary.csv")}
        t = base.round_tps(directory)
        mem[n] = (m[f"{n}:copy"], m[f"{n}:extent"])
        tps[n] = (t[f"{n}:copy"], t[f"{n}:extent"])
    # Ollama: eight agents in one or two servers, its own code, window throughput of the round
    oll = ollama_window()
    order = (("Copy", COPY), (NAME, OURS), ("Ollama", OLLAMA))
    legend("eval_proc8_legend", list(order), 3)
    for name, data, odata, ylabel, form in (("eval_proc8_mem", mem, oll["mem"], "Memory (GiB)", "{:.1f}"),
                                            ("eval_proc8_tps", tps, oll["tps"], "Throughput (tokens/s)", "{:.0f}")):
        fig, ax = panel()
        groups = [f"{n} ({8 * n})" for n in dirs] + ["1 (8)", "2 (8)"]
        width = 0.3
        xs = range(len(groups))
        for i, n in enumerate(dirs):
            for j, (lab, col) in enumerate(order[:2]):
                b = ax.bar(i + (j - 0.5) * width, data[n][j], width, color=col, edgecolor="white", linewidth=1.5)
                labels(ax, b, form, DENSE)
                ax.texts[-1].set_rotation(90)
        for k, servers in enumerate(("one", "two")):
            b = ax.bar(len(dirs) + k, odata[servers], width, color=OLLAMA, edgecolor="white", linewidth=1.5)
            labels(ax, b, form, DENSE)
            ax.texts[-1].set_rotation(90)
        ax.set_xticks(list(xs), groups)
        ax.axvline(len(dirs) - 0.5, color=AXIS, linewidth=0.8)
        top = max(max(max(v) for v in data.values()), max(odata.values()))
        finish(fig, ax, name, "Servers (Agents)", ylabel, top * 1.3)
    return mem, tps, oll


def ollama_window():
    directory = base.full("ollama-agents") or base.base.resolve("20261006-ollama-agents-v1")
    rows = {(r["config"], r["round"]): r for r in base.rows(directory, "ollama_agents_summary.csv")}
    mem = {k: float(rows[(k, "warm")]["memory_mib"]) / 1024 for k in ("one", "two")}
    # window throughput: tokens of the warm round over first token to last request end (client common start)
    tps = {}
    raw = os.path.join(RES, directory, "raw.log")
    rounds = defaultdict(list)
    cfg = run = None
    for line in open(raw):
        f = dict(re.findall(r"(\w+)=(\S+)", line))
        if line.startswith("BEGIN_OLLAMA"):
            run = f.get("run", run)
        elif line.startswith("CASE"):
            cfg = f.get("config")
        elif line.startswith("AGENT") and f.get("exit") == "0":
            rounds[(cfg, f["round"], run)].append((float(f["first_token_ms"]), float(f["request_ms"]),
                                                   int(f["generated"])))
    for k in ("one", "two"):
        vals = []
        for (c, r, _), reqs in rounds.items():
            if c == k and r == "warm" and reqs:
                vals.append(sum(q[2] for q in reqs) / ((max(q[1] for q in reqs) - min(q[0] for q in reqs)) / 1000))
        tps[k] = statistics.mean(vals) if vals else float(rows[(k, "warm")]["generation_tps_sum"])
    return {"mem": mem, "tps": tps}


# --------------------------------------------------------- dynamic load
def dynamic_e4b():
    """Agents over time: peak memory and TTFT P95 by mean tool wait; the existing
    engine is drawn at the waits it was run (thor_e4b_none, thor_e4b_none120)."""
    means = (60, 120, 240, 480, 960)
    NONE = KCJ + "/portable_bench_a30/validation/thor_e4b_none/analysis"
    NONE120 = KCJ + "/portable_bench_a30/validation/thor_e4b_none120/analysis"
    stats = defaultdict(lambda: defaultdict(list))
    for mean in means:
        for seed in (1, 2, 3):
            for system in ("none", "weights", "both"):
                if system == "none":
                    folder = NONE120 if mean == 120 else NONE
                else:
                    folder = E4B if mean <= 240 else E4BX
                path = f"{folder}/e4b-seed{seed}-mean{mean}-{system}-v1.json"
                if not os.path.exists(path):
                    continue
                v = json.load(open(path))
                stats[(system, mean)]["mem"].append(v["memory_peak_accounted_gib"])
                stats[(system, mean)]["p95"].append(v["ttft_p95_s"] * 1000)
                stats[(system, mean)]["p50"].append(v["ttft_p50_s"] * 1000)
                stats[(system, mean)]["live"].append(v["live_peak"])
                stats[(system, mean)]["queue"].append(v["queue_peak"])
    modes = (("Existing Engine", EXISTING, "none", "s"), ("Copy", COPY, "weights", "D"), (NAME, OURS, "both", "o"))
    present = [m for m in modes if any((m[2], x) in stats for x in means)]
    legend("eval_dyn_legend", [(lab, col) for lab, col, _, _ in present], len(present))
    for name, metric, ylabel, log in (("eval_dyn_mem", "mem", "Peak Memory (GiB)", False),
                                      ("eval_dyn_p95", "p95", "TTFT P95 (ms, log)", True)):
        fig, ax = panel()
        top = 0
        for lab, col, m, mk in present:
            xs = [i for i, x in enumerate(means) if len(stats[(m, x)][metric]) == 3]
            ys = [statistics.mean(stats[(m, means[i])][metric]) for i in xs]
            lo = [ys[k] - min(stats[(m, means[i])][metric]) for k, i in enumerate(xs)]
            hi = [max(stats[(m, means[i])][metric]) - ys[k] for k, i in enumerate(xs)]
            top = max([top] + [max(stats[(m, means[i])][metric]) for i in xs])
            ax.errorbar(xs, ys, yerr=[lo, hi], color=col, marker=mk, markersize=7, linewidth=2,
                        capsize=3, markeredgecolor="white", markeredgewidth=1.2)
        if metric == "mem":
            ax.axhline(96, color=AXIS, linewidth=1.0, linestyle="--")
        ax.set_xticks(range(len(means)), [str(x) for x in means])
        if log:
            ax.set_yscale("log")
            ax.set_ylim(500, top * 4)
            finish(fig, ax, name, "Mean Tool Wait (s)", ylabel, None, True)
        else:
            finish(fig, ax, name, "Mean Tool Wait (s)", ylabel, 110)
    return stats


if __name__ == "__main__":
    print("stack speed", stack_speed())
    print("ablation speed", ablation_speed())
    if os.environ.get("HANDOVER_RESULT"):
        sota_e1(os.environ["HANDOVER_RESULT"] + "/analysis_summary.csv", MECHS6, long_only=True)
    else:
        sota_e1()
    tree_e2()
    print("resident", resident_117(*filter(None, [os.environ.get("RESIDENT_RESULT")]))[1])
    print("proc8", proc8_bars())
    s = dynamic_e4b()
    for k in sorted(s):
        print(k, {m: round(statistics.mean(v), 1) for m, v in s[k].items() if v})
