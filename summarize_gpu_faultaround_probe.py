#!/usr/bin/env python3
"""Summary of run_gpu_faultaround_probe.sh: one row per fault-around size and
step, with the mean time to prepare, the mean time of the GPU pass, the mean
GPU faults, and the number of cases whose GPU sum was correct."""
import collections
import re
import statistics
import sys

cases = collections.defaultdict(list)
label = None
for line in open(sys.argv[1]):
    line = line.strip()
    if line.startswith("CASE "):
        label = dict(kv.split("=") for kv in line[5:].split())
        rec = {}
    elif line.startswith("PREPARE") and label is not None:
        rec.update(dict(kv.split("=") for kv in line.split()[1:] if "=" in kv))
    elif line.startswith("KERNEL") and label is not None:
        rec.update(dict(kv.split("=") for kv in line.split()[1:]))
        cases[(label["fab"], label["step"])].append(rec)
        label = None

print("fault_around_bytes,step,runs,failed,checked,prepare_ms,gpu_ms,gpu_fault_pages,fault_thread_ms")
for (fab, step), recs in sorted(cases.items(), key=lambda k: (-int(k[0][0]), ["none", "touch", "populate"].index(k[0][1]))):
    ok = [r for r in recs if r.get("failed") != "1" and "gpu_ms" in r]
    mean = lambda key: statistics.mean(float(r[key]) for r in ok) if ok else float("nan")
    print(f"{fab},{step},{len(recs)},{len(recs) - len(ok)},{sum(r.get('sum_ok') == '1' for r in ok)},"
          f"{mean('prepare_ms'):.2f},{mean('gpu_ms'):.2f},{mean('gpu_fault_pages'):.0f},{mean('fault_thread_ms'):.0f}")
