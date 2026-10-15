"""Summarizes run_attn_profile.sh: total kernel time per model, placement and
kernel class (decode attention, prefill attention, matrix-vector, matrix-matrix,
other), averaged over the repetitions, with the ratio host/device."""
import csv
import glob
import os
import re
import sys
from collections import defaultdict

d = sys.argv[1]
totals = defaultdict(float)
runs = defaultdict(set)


def kind(name):
    n = name.lower()
    if "flash_attn" in n or "fattn" in n:
        return "attn_vec" if "vec" in n else "attn_other"
    if "mul_mat_vec" in n or "mmvq" in n or "mmvf" in n:
        return "matvec"
    if "mul_mat_q" in n or "mmq" in n or "gemm" in n or "cutlass" in n:
        return "matmul"
    return "other"


for path in glob.glob(os.path.join(d, "*_cuda_gpu_kern_sum.csv")):
    base = os.path.basename(path).replace("_cuda_gpu_kern_sum.csv", "")
    m = re.match(r"(.+)\.(device|host)\.(\d+)$", base)
    if not m:
        continue
    model, kv, run = m.groups()
    runs[(model, kv)].add(run)
    with open(path) as f:
        for row in csv.DictReader(f):
            ns = float(row.get("Total Time (ns)", 0) or 0)
            totals[(model, kv, kind(row.get("Name", "")))] += ns
            totals[(model, kv, "all")] += ns
print("model,kind,device_ms,host_ms,host_vs_device")
for model in sorted({k[0] for k in totals}):
    for k in ("attn_vec", "attn_other", "matvec", "matmul", "other", "all"):
        dev = totals[(model, "device", k)] / max(1, len(runs[(model, "device")])) / 1e6
        host = totals[(model, "host", k)] / max(1, len(runs[(model, "host")])) / 1e6
        print(f"{model},{k},{dev:.1f},{host:.1f},{(host / dev if dev else 0):.4f}")
