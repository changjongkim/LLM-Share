#!/usr/bin/env python3
"""Per-repetition throughput ratios of a campaign, from its raw log.

A mean over repetitions is moved by one repetition in which a process
stalled. For every case of a raw log this prints the summed generation speed
of every mode relative to the baseline mode of the same repetition and the
same other case fields: the ratio of every repetition, their geometric mean
and their median.

Usage: throughput_ratios.py RAW_LOG [BASELINE_MODE]

The log is one with BEGIN_* run=N lines, CASE lines that have a mode= field,
and process lines (AGENT, CHILD or LEAF) that have generation_tps= and
exit= fields. The baseline is the first of none, copy, restore that occurs,
unless one is named.
"""
import math
import statistics
import sys


def fields(line):
    out = {}
    for part in line.split()[1:]:
        name, sep, value = part.partition("=")
        if sep:
            out[name] = value
    return out


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    run = None
    cell = mode = None
    speed = {}
    with open(sys.argv[1], encoding="utf-8", errors="replace") as log:
        for line in log:
            if line.startswith("BEGIN_"):
                run = int(fields(line).get("run", 0))
            elif line.startswith("CASE "):
                case = fields(line)
                mode = case.pop("mode", None)
                cell = " ".join(f"{k}={v}" for k, v in sorted(case.items()))
            elif line.split(" ", 1)[0] in ("AGENT", "CHILD", "LEAF"):
                process = fields(line)
                if process.get("exit") != "0" or "generation_tps" not in process:
                    continue
                key = (cell, mode, run)
                speed[key] = speed.get(key, 0.0) + float(process["generation_tps"])
    modes = []
    for _, name, _ in speed:
        if name not in modes:
            modes.append(name)
    if len(sys.argv) == 3:
        baseline = sys.argv[2]
    else:
        baseline = next((m for m in ("none", "copy", "restore") if m in modes), None)
    if baseline not in modes:
        sys.exit("no baseline mode in the log")
    print("case,mode,baseline,repetitions,geometric_mean,median,lowest,highest,ratios")
    cells = []
    for name, _, _ in speed:
        if name not in cells:
            cells.append(name)
    for name in cells:
        for candidate in modes:
            if candidate == baseline:
                continue
            ratios = [
                speed[(name, candidate, r)] / speed[(name, baseline, r)]
                for (c, m, r) in sorted(speed, key=lambda k: k[2])
                if c == name and m == candidate and speed.get((name, baseline, r), 0) > 0
            ]
            if not ratios:
                continue
            geometric = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
            print(
                f"{name},{candidate},{baseline},{len(ratios)},{geometric:.4f},"
                f"{statistics.median(ratios):.4f},{min(ratios):.4f},{max(ratios):.4f},"
                + " ".join(f"{x:.4f}" for x in ratios)
            )


if __name__ == "__main__":
    main()
