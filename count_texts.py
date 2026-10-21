#!/usr/bin/env python3
"""The number of texts that agents on STATOR write equal to their counterpart on
Copy or on the existing engine, over the campaigns on the whole GPU (names
ending in -full-v1 or -full-v2, and the formal runs e1, e2, e3). Only the
STATOR configuration of each campaign counts, not its variants (a step
removed, a cache allocated at once). The hand-over campaign of the review (130) counts as well. Gemma-2 is left out, since STATOR does not
support its sliding-window cache (0 of 2 texts in its smoke run).

Usage: count_texts.py [RESULTS_DIR]

Prints one line per campaign (equal, compared) and the totals. A campaign whose
summary has no column of compared texts counts its equal texts only, and the
script reports any STATOR text that differs.
"""
import collections
import csv
import json
import os
import sys

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# campaign suffix -> (summary file, mode of STATOR, column of equal texts, column of compared texts)
SUMMARY = [
    ("engine-kvlinux-mechanisms-full-v1", "engine_kvlinux_summary.csv", ("extent", "extent_write"), "texts_equal_to_copy", "texts_compared"),
    ("engine-kvlinux-scaling-full-v1", "engine_kvlinux_summary.csv", ("extent",), "texts_equal_to_copy", "texts_compared"),
    ("engine-kvdiverge-full-v1", "engine_kvlinux_summary.csv", ("extent",), "texts_equal_to_copy", "texts_compared"),
    ("engine-kvcow-full-v1", "engine_kvcow_summary.csv", ("extent_lazy",), "texts_equal_to_restore", "texts_compared"),
    ("engine-kvablate-full-v1", "engine_kvmech_summary.csv", ("extent",), "texts_equal_to_copy", "texts_compared"),
    ("engine-kvshare-full-v1", "engine_kvshare_summary.csv", ("extent_lazy",), "texts_equal_to_restore", "texts_compared"),
    ("engine-kvmech-full-v1", "engine_kvmech_summary.csv", ("extent",), "texts_equal_to_copy", "texts_compared"),
    ("engine-kvmech-32-full-v1", "engine_kvmech_summary.csv", ("extent",), "texts_equal_to_copy", "texts_compared"),
    ("engine-kvlocal-full-v1", "engine_kvmech_summary.csv", ("extent",), "texts_equal_to_copy", "texts_compared"),
    ("engine-kvlocal-gqa-full-v1", "engine_kvmech_summary.csv", ("extent",), "texts_equal_to_copy", "texts_compared"),
    ("engine-kvlocal-llama8b-full-v1", "engine_kvmech_summary.csv", ("extent",), "texts_equal_to_copy", "texts_compared"),
    ("engine-kvlocal-llama8b-gqa-full-v1", "engine_kvmech_summary.csv", ("extent",), "texts_equal_to_copy", "texts_compared"),
    ("engine-kvlocal-llama8b-upstream-full-v1", "engine_kvmech_summary.csv", ("extent",), "texts_equal_to_copy", "texts_compared"),
    ("engine-kvfaultcount-full-v2", "engine_kvfaultcount_summary.csv", ("extent",), "texts_equal_to_copy", "texts_compared"),
    ("engine-kvfork-full-v1", "engine_kvfork_summary.csv", ("extent",), "child_texts_equal_to_copy", None),
    ("engine-kvmps-full-v1", "engine_kvmps_summary.csv", ("extent",), "texts_equal_to_copy", None),
    ("engine-kvvmm-full-v1", "engine_kvvmm_summary.csv", ("extent", "extent_cross"), "child_texts_equal_to_copy", None),
    ("engine-kvserver2-full-v1", "engine_kvserver_summary.csv", ("extent",), "texts_equal_to_copy", None),
    ("engine-kvbatch-full-v1", "engine_kvbatch_summary.csv", ("two_servers_extent",), "texts_equal_to_copy", "texts_compared"),
    ("engine-kvbatch-n-full-v1", "engine_kvbatch_n_summary.csv", ("extent",), "texts_equal_to_copy", "texts_compared"),
    ("engine-kvbatch-one-full-v1", "engine_kvbatch_n_summary.csv", ("extent",), "texts_equal_to_copy", "texts_compared"),
    ("engine-kvbatch-per8-n2-full-v1", "engine_kvbatch_n_summary.csv", ("extent",), "texts_equal_to_copy", "texts_compared"),
    ("engine-kvbatch-per8-n4-full-v1", "engine_kvbatch_n_summary.csv", ("extent",), "texts_equal_to_copy", "texts_compared"),
    ("engine-kvbatch-per8-n8-full-v1", "engine_kvbatch_n_summary.csv", ("extent",), "texts_equal_to_copy", "texts_compared"),
    ("engine-kvtree-full-v1", "engine_kvtree_summary.csv", ("chain",), "leaf_texts_equal_to_copy", None),
    ("engine-kvdeep-full-v1", "engine_kvdeep_summary.csv", ("chain",), "leaf_texts_equal_to_copy", "leaf_texts_compared"),
    ("engine-kvdeep-wide-full-v1", "engine_kvdeep_summary.csv", ("chain",), "leaf_texts_equal_to_copy", "leaf_texts_compared"),
    ("engine-kvtree-big-226-full-v1", "engine_kvdeep_summary.csv", ("chain",), "leaf_texts_equal_to_copy", "leaf_texts_compared"),
    ("engine-kvpipe-huge-full-v1", "engine_kvpipe_summary.csv", ("chain",), "worker_texts_equal_to_stock", "worker_texts_compared"),
]
# the sensitivity campaigns, every one of the runner engine_kvscale except Gemma-2
SCALE = "engine_kvscale_summary.csv"


def summary_rows(directory, name):
    path = os.path.join(RES, directory, name)
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def count(rows, modes, equal, compared):
    e = c = 0
    for r in rows:
        if (r.get("mode") or r.get("config")) in modes:
            e += int(float(r.get(equal) or 0))
            c += int(float(r.get(compared) or 0)) if compared else 0
    return e, c


def newest(suffix):
    found = sorted(d for d in os.listdir(RES) if d.endswith(suffix) and os.path.isdir(os.path.join(RES, d)))
    return found[-1] if found else None


def e1():
    """Children and parents of the live fork against Copy of the same run, children and prefix."""
    cases = json.load(open(os.path.join(RES, newest("e1-live-publish-formal-v1"), "cases.json")))
    by = collections.defaultdict(dict)
    for c in cases:
        if c.get("status") == "ok":
            by[(c["run"], c["children"], c["paragraphs"], c["trace_on"])][c["mode"]] = c
    e = n = 0
    for m in by.values():
        if "copy" in m and "extent" in m:
            pairs = list(zip(m["copy"]["child_text_sha256"], m["extent"]["child_text_sha256"]))
            pairs.append((m["copy"]["parent_text_sha256"], m["extent"]["parent_text_sha256"]))
            e += sum(a == b for a, b in pairs)
            n += len(pairs)
    return e, n


def handover_review():
    """STATOR (extent) against Copy of the same run in the hand-over campaign of the review (130): children and
    parent of every case."""
    cases = json.load(open(os.path.join(RES, newest("review-handover-all-v1"), "cases.json")))
    by = collections.defaultdict(dict)
    for c in cases:
        if c.get("status") == "ok":
            by[(c["run"], c["children"], c["paragraphs"])][c["mode"]] = c
    e = n = 0
    for m in by.values():
        if "copy" in m and "extent" in m:
            pairs = list(zip(m["copy"]["child_text_sha256"], m["extent"]["child_text_sha256"]))
            pairs.append((m["copy"]["parent_text_sha256"], m["extent"]["parent_text_sha256"]))
            e += sum(a == b for a, b in pairs)
            n += len(pairs)
    return e, n


def validation(suffix, mode):
    v = json.load(open(os.path.join(RES, newest(suffix), "validation.json")))
    rows = [x for x in v["comparisons"] if x["case"].split("-")[1] == mode]
    return sum(bool(x["text_match"]) for x in rows), len(rows)


def main():
    lines = []
    for suffix, name, modes, equal, compared in SUMMARY:
        directory = newest(suffix)
        if directory:
            lines.append((directory, *count(summary_rows(directory, name), modes, equal, compared)))
    for directory in sorted(os.listdir(RES)):
        if directory.endswith("-full-v1") and "kvscale" in directory and "gemma" not in directory:
            rows = summary_rows(directory, SCALE)
            if rows:
                lines.append((directory, *count(rows, ("extent_lazy",), "texts_equal_to_restore", "texts_compared")))
    lines.append(("e1-live-publish-formal-v1", *e1()))
    lines.append(("review-handover-all-v1", *handover_review()))
    lines.append(("e2-tree11-formal-v1", *validation("e2-tree11-formal-v1", "chain")))
    lines.append(("e2-tree23-31-formal-v1", *validation("e2-tree23-31-formal-v1", "chain")))
    lines.append(("e3-resident-tmpfs-v1", *validation("e3-resident-tmpfs-v1", "extent")))
    total_e = total_c = 0
    for directory, e, c in lines:
        flag = "  DIFFERS" if c and e != c else ""
        print(f"{directory}: {e} equal, {c if c else '-'} compared{flag}")
        total_e += e
        total_c += c if c else e
    print(f"TOTAL equal {total_e} of {total_c}")


if __name__ == "__main__":
    main()
