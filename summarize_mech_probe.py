#!/usr/bin/env python3
"""Summary of run_mech_probe.sh: one row per mechanism, step and accessor.

  summarize_mech_probe.py RAW_LOG SIZE_MIB > mech_probe_summary.csv

Timed repetitions give the mean and the 95% confidence interval (Student t)
of each metric. The traced repetition gives the kernel events of each case
(rows with step=events). The gates M1 and M2 of run_mech_probe.sh are the last
rows (mech=gate).

copied_mib is the private memory that the step adds: anonymous memory for the
file mappings, private dirty memory for the child of a fork, and the pages that
are no longer mapped to a merged page for KSM.
"""
import math
import re
import sys
from collections import defaultdict

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
       9: 2.262, 10: 2.228, 11: 2.201}
EVENTS = ("notifier_invalidate", "notifier_young", "smmu_invalidate", "wp_fault", "unmap",
          "referenced_scan", "fork_copy", "ksm_merge", "cpu_fault", "gpu_fault")


def fields(line):
    return dict(re.findall(r"(\w+)=(\S+)", line))


def mean_ci(values):
    if not values:
        return "", ""
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    v = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return m, T95.get(len(values) - 1, 1.96) * math.sqrt(v / len(values))


def main():
    raw, size_mib = sys.argv[1], float(sys.argv[2])
    timed = defaultdict(lambda: defaultdict(list))  # key -> metric -> values
    failed = defaultdict(int)
    events = {}
    case = None
    for line in open(raw):
        line = line.strip()
        if line.startswith("CASE "):
            case = fields(line)
            case["rows"] = []
            case["events"] = None
        elif case is None:
            continue
        elif line.startswith("STEP "):
            case["rows"].append(("step", fields(line)))
        elif line.startswith("KSMSTEP "):
            case["rows"].append(("ksm", fields(line)))
        elif line.startswith("RECLAIM "):
            case["rows"].append(("reclaim", fields(line)))
        elif line.startswith("EVENTS"):
            case["events"] = fields(line)
        elif line.startswith("END_CASE"):
            status = int(fields(line).get("status", "1"))
            traced = case["traced"] == "1"
            if traced:
                events[(case["mech"], case["access"], case["variant"])] = case["events"] or {}
            for kind, f in case["rows"]:
                if kind == "step":
                    key = (f["mech"], f["step"], f["access"], "")
                    copied = float(f["dirty_mib"] if f["mech"] == "fork" else f["anon_mib"])
                    metrics = {"seconds": float(f["seconds"]), "copied_mib": copied,
                               "cpu_faults": int(f["minor"]) + int(f["major"]),
                               "gpu_fault_pages": int(f["gpu_fault_pages"]),
                               "gpu_fault_ms": int(f["gpu_fault_ms"])}
                    bad = status != 0 or f["right"] != "1"
                    rows = [(key, metrics, bad)]
                elif kind == "ksm":
                    before = int(f["merged_before"])
                    rows = []
                    for step, after, sec, minor, gpu in (
                            ("read", "merged_after_read", "read_s", "read_minor", "read_gpu_fault_pages"),
                            ("write", "merged_after_write", "write_s", "write_minor", "write_gpu_fault_pages")):
                        metrics = {"seconds": float(f[sec]),
                                   "copied_mib": (before - int(f[after])) * 4 / 1024.0,
                                   "cpu_faults": int(f[minor]), "gpu_fault_pages": int(f[gpu]),
                                   "merged_before": before, "merged_after": int(f[after])}
                        rows.append((("ksm", step, f["access"], ""), metrics,
                                     status != 0 or f["right"] != "1"))
                else:
                    # the pin of the hot file: none, lock, or memfs (the runner's variant)
                    pin = case["variant"].split("-")[0]
                    key = ("reclaim", pin, f["access"], f["pressure"])
                    metrics = {"pass_ms": float(f["pass_ms"]), "passes": int(f["passes"]),
                               "cpu_faults": int(f["minor"]) + int(f["major"]),
                               "major_faults": int(f["major"]),
                               "gpu_fault_pages": int(f["gpu_fault_pages"]),
                               "gpu_fault_ms": int(f["gpu_fault_ms"]),
                               "reclaim_succeeded": int(f["reclaim_succeeded"]),
                               "hot_resident": float(f["hot_resident"]),
                               "cold_resident": float(f["cold_resident"]),
                               "memcg_freed_mib": int(f["memcg_before_mib"]) - int(f["memcg_after_mib"]),
                               "reclaim_write_max_ms": int(f["reclaim_write_max_ms"]),
                               "lock_s": float(f["lock_s"])}
                    rows = [(key, metrics, status != 0 or f["right"] != "1")]
                for key, metrics, bad in rows:
                    if traced:
                        continue
                    if bad:
                        failed[key] += 1
                        continue
                    for name, value in metrics.items():
                        timed[key][name].append(value)
            # a case that printed no step failed before its first step
            if not case["rows"] and not traced:
                failed[(case["mech"], "case", case["access"], case["variant"])] += 1
            case = None

    metrics = ("seconds", "copied_mib", "cpu_faults", "gpu_fault_pages", "gpu_fault_ms",
               "merged_before", "merged_after", "pass_ms", "passes", "major_faults",
               "reclaim_succeeded", "hot_resident", "cold_resident", "memcg_freed_mib",
               "reclaim_write_max_ms", "lock_s")
    header = ["mech", "step", "access", "pressure", "runs", "failed"]
    for name in metrics:
        header += [name, name + "_ci95"]
    header += ["rate_vs_none"] + list(EVENTS)
    print(",".join(header))
    keys = sorted(set(timed) | set(failed))
    for key in keys:
        row = list(key) + [str(len(next(iter(timed[key].values()), []))), str(failed[key])]
        for name in metrics:
            m, ci = mean_ci(timed[key].get(name, []))
            row += ["" if m == "" else f"{m:.4f}", "" if ci == "" else f"{ci:.4f}"]
        rate = ""
        if key[0] == "reclaim" and key[3] == "pressure":
            base = timed.get(("reclaim", key[1], key[2], "none"), {}).get("pass_ms")
            here = timed[key].get("pass_ms")
            if base and here:
                rate = f"{(sum(base) / len(base)) / (sum(here) / len(here)):.4f}"
        row.append(rate)
        row += [""] * len(EVENTS)
        print(",".join(row))
    for (mech, access, variant), counts in sorted(events.items()):
        row = [mech, "events", access, variant, "1", "0"] + [""] * (2 * len(metrics)) + [""]
        row += [counts.get(name, "0") for name in EVENTS]
        print(",".join(row))

    m1 = not any(failed.values())
    m2 = True
    for mech in ("private", "fork"):
        copied = timed.get((mech, "read", "cpu", ""), {}).get("copied_mib", [])
        m2 = m2 and bool(copied) and max(copied) < 0.01 * size_mib
    m3 = True
    for key, values in timed.items():
        if key[0] == "reclaim" and key[3] == "none":
            m3 = m3 and min(values["hot_resident"]) >= 0.99 and min(values["cold_resident"]) >= 0.99
    tail = [""] * (len(header) - 6)
    print(",".join(["gate", "M1", "", "", "", "PASS" if m1 else "FAIL"] + tail))
    print(",".join(["gate", "M2", "", "", "", "PASS" if m2 else "FAIL"] + tail))
    print(",".join(["gate", "M3", "", "", "", "PASS" if m3 else "FAIL"] + tail))


if __name__ == "__main__":
    main()
