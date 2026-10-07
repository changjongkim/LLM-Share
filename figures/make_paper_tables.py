#!/usr/bin/env python3
"""Tables of the paper, written as LaTeX from the packaged summaries.

Reads ../results (nothing is typed in by hand) and writes one tab_NAME.tex
next to this script for every table whose campaign has been run:

  tab_ablation   the extent mechanism with one step changed
  tab_fault      agents that survive a fault of a co-tenant, by GPU sharing mode
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def rows(*path):
    with open(os.path.join(RES, *path), newline="") as handle:
        return list(csv.DictReader(handle))


def have(*path):
    return os.path.exists(os.path.join(RES, *path))


def write(name, lines):
    with open(os.path.join(HERE, name + ".tex"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("wrote", name)


def ablation():
    directory = "20261007-engine-kvablate-v1"
    if not have(directory, "engine_kvmech_summary.csv"):
        return
    data = {r["mode"]: r for r in rows(directory, "engine_kvmech_summary.csv")}
    order = (("extent", r"\STATOR"), ("no_read", "No CPU read of the prefix"),
             ("no_populate", "No populate-ahead of the tail"),
             ("writable", "Writable private mapping"),
             ("writable_no_read", "Writable, no CPU read"),
             ("grow_1024", "Tail in steps of 1,024 rows"),
             ("grow_4096", "Tail in steps of 4,096 rows"),
             ("grow_all", "Tail allocated at once"),
             ("small_pages", "Cache file on 4~KiB pages"),
             ("copy", r"\textit{Copy}"))
    lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"Configuration & \makecell[r]{Attach\\(ms)} & \makecell[r]{TTFT\\(s)} & "
             r"\makecell[r]{Memory\\(GiB)} & \makecell[r]{Speed\\vs. \textit{Copy}} & "
             r"\makecell[r]{Equal\\Texts} \\", r"\midrule"]
    for mode, label in order:
        if mode not in data:
            continue
        r = data[mode]
        lines.append(
            f"{label} & {float(r['child_attach_ms']):.0f} & "
            f"{float(r['child_first_token_ms']) / 1000:.2f} & "
            f"{float(r['children_memory_mib']) / 1024:.1f} & "
            f"{float(r['children_tps_vs_copy']):.3f} & "
            f"{r['texts_equal_to_copy']}/{r['children_finished']} \\\\")
        if mode == "extent":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_ablation", lines)


def fault():
    directory = "20261007-engine-kvfault-v1"
    if not have(directory, "engine_kvfault_summary.csv"):
        return
    data = {(r["config"], r["fault"]): r for r in rows(directory, "engine_kvfault_summary.csv")}
    names = (("timeslice", "Time slicing"), ("mps", "MPS"), ("mig", "MIG"),
             ("mig_mps", "MIG with MPS"))
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"GPU Sharing & \makecell[r]{No\\Fault} & \makecell[r]{GPU Write\\by a Co-Tenant} & "
             r"\makecell[r]{Agent\\Killed} & \makecell[r]{Writes\\Refused} \\", r"\midrule"]
    for config, label in names:
        if (config, "none") not in data:
            continue
        none, write_case, kill = (data[config, f] for f in ("none", "write", "kill"))
        lines.append(
            f"{label} & {none['agents_completed']}/{none['agents_started']} & "
            f"{write_case['agents_completed']}/{write_case['agents_started']} & "
            f"{kill['others_completed']}/{kill['others_started']} & "
            f"{write_case['writes_refused']}/{write_case['writes_injected']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_fault", lines)


if __name__ == "__main__":
    ablation()
    fault()
