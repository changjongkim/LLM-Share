#!/usr/bin/env python3
"""Eviction while an agent waits for a tool (review of 2026-10-26, E2).

The review asks whether the results of Section 4.6 depend on the policy that a
waiting agent holds its state. Here an agent of the workload of
run_engine_kvdynamic.py is either resident or evicted during its wait, on the
same three stacks, with the prefix published in the same way:

  resident  one process: load the prefix, decode the task, generate, wait,
            then decode the tool result; its resume is the time from the
            release to the decoded tool result (RELEASED to DECODED).
  evicted   a first process loads the prefix, decodes the task, generates and
            exits, so that it holds no memory while the tool runs; at the tool
            result a new process loads the prefix and decodes the task, the
            text that the first process generated and the tool result; its
            resume is the time from its launch to the decoded input.

The GPU runs nothing else while a resume is measured, so that the two differ
only in the process that the evicted agent starts again.

  PRODUCER_MIG=<gpu> REPS=6 SYSTEMS="both weights none" python3 run_engine_evict.py
"""
import json
import os
import statistics
import subprocess
import sys
import time

import run_engine_kvdynamic as D

REPS = int(os.environ.get("REPS", "6"))
WAIT_S = float(os.environ.get("WAIT_S", "2"))
SYSTEMS = os.environ.get("SYSTEMS", "both weights none").split()
RESULT_TAG = os.environ.get("RESULT_TAG", time.strftime("%Y%m%d") + "-engine-evict-v1")


def unescape(line):
    out, i = [], 0
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line):
            out.append({"n": "\n", "r": "\r", "\\": "\\"}.get(line[i + 1], line[i + 1]))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def launch(run, steps, out):
    e = run.base_env()
    e.update(run.agent_env())
    argv = [run.driver(), D.MODEL, str(D.CONTEXT), str(D.N_GEN), *steps]
    handle, err = open(out, "w"), open(out + ".err", "w")
    return subprocess.Popen(argv, env=e, stdout=handle, stderr=err), handle, err


def lines(path):
    try:
        return open(path).read().split("\n")
    except FileNotFoundError:
        return []


def wait_for(path, prefix, popen, after=None, limit_s=600):
    """The first whole line that starts with prefix (after the line `after`), and the time it was seen."""
    end = time.monotonic() + limit_s
    while time.monotonic() < end:
        seen = lines(path)[:-1]
        if after is not None:
            idx = next((i for i, l in enumerate(seen) if l.startswith(after)), None)
            seen = seen[idx + 1:] if idx is not None else []
        for line in seen:
            if line.startswith(prefix):
                return line, time.monotonic()
        if popen.poll() is not None and not any(l.startswith(prefix) for l in seen):
            raise RuntimeError(f"{path}: exited before {prefix}")
        time.sleep(0.001)
    raise RuntimeError(f"{path}: no {prefix} within {limit_s} s")


def resident(run, work, sid, task, log):
    go = os.path.join(work, f"go.{sid}")
    out = os.path.join(work, f"out.{sid}.resident")
    steps = run.first_steps() + [f"say:{D.task_text(sid, task)}", "gen", "mark:ready", f"wait:{go}",
                                 f"say:{D.tool_text(sid, 1)}", "gen"]
    popen, h, e = launch(run, steps, out)
    wait_for(out, "MARK", popen)
    time.sleep(WAIT_S)
    open(go, "w").close()
    released, _ = wait_for(out, "RELEASED", popen)
    decoded, _ = wait_for(out, "DECODED", popen, after="RELEASED")
    popen.wait()
    h.close(), e.close()
    resume = float(D.fields(decoded)["since_start_ms"]) - float(D.fields(released)["since_start_ms"])
    record = {"system": run.system, "sid": sid, "policy": "resident", "resume_ms": round(resume, 2),
              "exit": popen.returncode}
    log.write(json.dumps(record) + "\n")
    return record


def evicted(run, work, sid, task, log):
    first = os.path.join(work, f"out.{sid}.first")
    popen, h, e = launch(run, run.first_steps() + [f"say:{D.task_text(sid, task)}", "gen"], first)
    popen.wait()
    h.close(), e.close()
    text = next(unescape(l[5:]) for l in lines(first) if l.startswith("TEXT "))
    time.sleep(WAIT_S)
    again = os.path.join(work, f"out.{sid}.again")
    combined = D.task_text(sid, task) + text + D.tool_text(sid, 1)
    start = time.monotonic()
    popen, h, e = launch(run, run.first_steps() + [f"say:{combined}"], again)
    decoded, seen = wait_for(again, "DECODED", popen)
    popen.wait()
    h.close(), e.close()
    result = D.fields(next(l for l in lines(again) if l.startswith("RESULT")))
    record = {"system": run.system, "sid": sid, "policy": "evicted", "resume_ms": round((seen - start) * 1000, 1),
              "model_ms": float(result["model_ms"]), "context_ms": float(result["context_ms"]),
              "state_ms": float(result["state_ms"]), "decode_ms": float(result["decode_ms"]),
              "decoded_tokens": int(D.fields(decoded)["tokens"]), "exit": popen.returncode}
    log.write(json.dumps(record) + "\n")
    return record


def p95(values):
    ordered = sorted(values)
    return ordered[int(round(0.95 * (len(ordered) - 1)))]


def main():
    if not D.GPU:
        sys.exit("PRODUCER_MIG (the GPU) is not set")
    result_dir = os.path.join(D.RESULT_ROOT, RESULT_TAG)
    if os.path.exists(result_dir):
        sys.exit(f"result directory already exists: {result_dir}")
    os.makedirs(result_dir)
    huge = os.path.join(D.MOUNT_ROOT, "kv_evict")
    os.makedirs(huge, exist_ok=True)
    D.sudo("mount", "-t", "tmpfs", "-o", f"huge=always,size={D.TMPFS_MIB}m,uid={os.getuid()},gid={os.getgid()}",
           "tmpfs", huge)
    work = os.path.join(result_dir, "work")
    os.makedirs(work)
    tasks = [t for t in open(D.TASKS_FILE).read().split("\n") if t.strip()]
    with open(os.path.join(result_dir, "metadata.txt"), "w") as meta:
        meta.write(f"reps={REPS}\nwait_s={WAIT_S}\nsystems={' '.join(SYSTEMS)}\nn_gen={D.N_GEN}\ncontext={D.CONTEXT}\n")
        meta.write(f"model={os.path.basename(D.MODEL)}\nprefix_file={os.path.basename(D.PREFIX_FILE)}\n")
        meta.write(subprocess.run(["sha256sum", D.DRIVER, D.STOCK_DRIVER, D.CHAIN_LIB + "/libllama.so",
                                   os.path.abspath(__file__), os.path.abspath(D.__file__)],
                                  capture_output=True, text=True).stdout)
        meta.write("boot_id=" + open("/proc/sys/kernel/random/boot_id").read())
    log = open(os.path.join(result_dir, "raw.jsonl"), "a", buffering=1)
    records = []
    try:
        for system in SYSTEMS:
            for path in [os.path.join(huge, f) for f in os.listdir(huge)] + [os.path.join(work, f) for f in os.listdir(work)]:
                os.unlink(path)
            subprocess.run(["sync"])
            D.sudo("sh", "-c", "echo 1 >/proc/sys/vm/drop_caches")
            with open(D.MODEL, "rb") as handle:
                while handle.read(1 << 24):
                    pass
            run = D.Run(system, 0.0, [], work, huge, log)
            run.root_tokens = D.publish(system, work, huge)
            for rep in range(REPS):
                sid = 1000 + rep
                task = tasks[rep % len(tasks)]
                # alternate the order, so that neither policy always runs first
                order = (resident, evicted) if rep % 2 == 0 else (evicted, resident)
                for policy in order:
                    records.append(policy(run, work, sid, task, log))
    finally:
        log.close()
        subprocess.run(["sudo", "-n", "umount", huge])
        try:
            os.rmdir(huge)
        except OSError:
            pass
    with open(os.path.join(result_dir, "evict_summary.csv"), "w") as summary:
        summary.write("system,policy,runs,resume_median_ms,resume_p95_ms,model_ms,context_ms,state_ms,decode_ms\n")
        for system in SYSTEMS:
            for policy in ("resident", "evicted"):
                rows = [r for r in records if r["system"] == system and r["policy"] == policy]
                if not rows:
                    continue
                res = [r["resume_ms"] for r in rows]
                mean = lambda k: round(statistics.mean(r[k] for r in rows), 1) if k in rows[0] else ""
                summary.write(f"{D.NAMES[system]},{policy},{len(rows)},{round(statistics.median(res), 1)},"
                              f"{round(p95(res), 1)},{mean('model_ms')},{mean('context_ms')},{mean('state_ms')},"
                              f"{mean('decode_ms')}\n")
    print(open(os.path.join(result_dir, "evict_summary.csv")).read())


if __name__ == "__main__":
    main()
