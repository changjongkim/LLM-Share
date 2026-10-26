#!/usr/bin/env python3
"""Agents that arrive, call tools, fork and leave over time, on three stacks.

A publisher computes the key-value cache of the agent prefix and exits. Agent
sessions then arrive as a Poisson process of rate RATE during WINDOW_S, from
one trace per rate that every stack replays (the same arrivals, kinds, tasks
and tool latencies). A session is one of three kinds:

  single  one turn: attach to the prefix, decode the task, generate N_GEN
          tokens, leave;
  tool    three turns: after the first and the second, the agent waits for
          the result of a tool call (exponential latency of mean TOOL_MEAN_S,
          clipped to TOOL_MIN_S..TOOL_MAX_S) and holds its state while it waits;
  fork    the agent decodes its task and generates, then forks FORK_K
          sub-agents from its state: it saves (Copy, Existing) or publishes
          (STATOR) its context, and each sub-agent attaches to it, decodes a
          subtask, generates, waits for one tool call and generates again; the
          parent waits for all of them, decodes their results and generates.

Every agent is one process of kv_agent (steps load, say, gen, save, mark,
wait). The stacks are those of run_engine_kvstack.sh:

  none     the engine at the upstream commit: every process copies the weights
           to device memory and restores the prefix from the state file
  weights  (Copy) the weights are read in place; the prefix and every fork are
           restored from state files
  both     (STATOR) the weights are read in place; the prefix is mapped
           read-only from a file on a tmpfs with 2 MiB pages, a fork publishes
           the rows of the parent as a segment of a chain that the sub-agents
           map, and the file of a segment is removed as soon as every
           sub-agent has attached

A process starts when the memory allows it: when the available memory less the
model file, less ESTIMATE_<stack> MiB for every process that has not decoded
its first input yet, still leaves RESERVE_GIB after its own estimate.
Otherwise it waits in a FIFO queue; the sub-agents of a fork go to the front.
The time a session waits there is part of its TTFT. When the available memory
less the model file falls below GUARD_GIB, the newest process is ended (logged
as a guard event; the estimates are set so that this does not happen). After
WINDOW_S no session arrives; a run ends when every session has left or at
LIMIT_S, when the sessions that remain are ended and logged as unfinished.

Measured per run: the TTFT of every session (arrival to the decoded task, so
that the next token can be chosen), the latency of every turn after a tool
result (release to decoded input), the pause of a forking parent and the time
from its publish to the decoded subtask of each sub-agent, MemAvailable, the
live processes and the queue every SAMPLE_S, and the memory left after the
last session (with the prefix file) against the memory before the first.

Kill gates, fixed before the campaign:
  D1  no process fails, and no guard event occurs;
  D2  every process that completes writes the same text on every stack (same
      session, role and index), so the stacks differ only in where the state
      lives;
  D3  after the last session of a STATOR run has left, the tmpfs that holds
      the prefix and the segments of the forks uses no more pages than before
      the first session: the segment of every fork is freed with its last
      mapping. (Redefined on 2026-10-16 with the user: the first definition,
      the system-wide MemAvailable within 0.5 GiB of its start, failed in the
      pilots by 615 MiB for STATOR while Copy left 594 MiB and the existing
      engine 381 MiB, so it measured the system rather than the extents. The
      MemAvailable difference is still reported for every stack.)
  D4  in every run of the lowest rate, every session of `both` and `weights`
      completes.
Reported, not gated: TTFT P50/P99, turn latency P50/P99, peak and mean memory,
peak live processes and queue, unfinished sessions, the tmpfs pages over time
and the MemAvailable difference after the last session.

  run_engine_kvdynamic.py            (parameters from the environment)
  SUMMARIZE=RESULT_DIR run_engine_kvdynamic.py
"""
import csv
import glob
import hashlib
import json
import math
import os
import random
import signal
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def env(name, default):
    return os.environ.get(name, default)


SYSTEMS = env("SYSTEMS", "both weights none").split()
RATES = [float(x) for x in env("RATES", "0.07 0.14 0.21").split()]
WINDOW_S = float(env("WINDOW_S", "480"))
LIMIT_S = float(env("LIMIT_S", "1080"))
SEED = int(env("SEED", "1"))
N_GEN = int(env("N_GEN", "32"))
CONTEXT = int(env("CONTEXT", "16384"))
MIX = [float(x) for x in env("MIX", "0.2 0.6 0.2").split()]  # single tool fork
TOOL_MEAN_S = float(env("TOOL_MEAN_S", "120"))
TOOL_MIN_S = float(env("TOOL_MIN_S", "10"))
TOOL_MAX_S = float(env("TOOL_MAX_S", "300"))
FORK_K = int(env("FORK_K", "3"))
RESERVE_GIB = float(env("RESERVE_GIB", "16"))
GUARD_GIB = float(env("GUARD_GIB", "12"))
SAMPLE_S = float(env("SAMPLE_S", "0.5"))
GROW = env("GROW_ROWS", "256")
ESTIMATE = {"none": float(env("ESTIMATE_NONE", "6400")),
            "weights": float(env("ESTIMATE_WEIGHTS", "1800")),
            "both": float(env("ESTIMATE_BOTH", "850"))}
GPU = env("PRODUCER_MIG", "")
MODEL = env("MODEL", os.path.join(HERE, "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"))
PREFIX_FILE = env("PREFIX_FILE", os.path.join(HERE, "workloads/agent_prefix.txt"))
TASKS_FILE = env("TASKS_FILE", os.path.join(HERE, "workloads/agent_tasks.txt"))
CHAIN_LIB = env("CHAIN_LIB", os.path.join(HERE, "llama.cpp-chain2/build/bin"))
DRIVER = env("DRIVER", os.path.join(HERE, "kv_agent2"))
STOCK_DRIVER = env("STOCK_DRIVER", os.path.join(HERE, "kv_agent_stock"))
MOUNT_ROOT = env("MOUNT_ROOT", os.path.join(HERE, "mounts"))
RESULT_ROOT = env("RESULT_ROOT", os.path.join(HERE, "results"))
RESULT_TAG = env("RESULT_TAG", time.strftime("%Y%m%d-%H%M%S") + "-engine-kvdynamic")
TMPFS_MIB = int(env("TMPFS_MIB", "16384"))
NAMES = {"none": "Existing", "weights": "Copy", "both": "STATOR"}


def meminfo(name):
    with open("/proc/meminfo") as handle:
        for line in handle:
            if line.startswith(name + ":"):
                return int(line.split()[1]) // 1024
    raise KeyError(name)


def tmpfs_mib(path):
    """Pages that the tmpfs at path holds, in MiB."""
    st = os.statvfs(path)
    return (st.f_blocks - st.f_bfree) * st.f_frsize // (1 << 20)


def sudo(*argv):
    subprocess.run(["sudo", "-n", *argv], check=True)


def fields(line):
    out = {}
    for part in line.split()[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return out


# ------------------------------------------------------------------ trace
def make_trace(rate, seed):
    """One trace per rate that every stack replays. The sessions of a lower
    rate are a thinned subset of those of the highest rate in RATES (each kept
    with probability rate / highest), so that a higher rate only adds
    sessions."""
    top = max(RATES)
    rng = random.Random(seed)
    tasks = [line.rstrip("\n") for line in open(TASKS_FILE) if line.strip()]

    def tool():
        return min(TOOL_MAX_S, max(TOOL_MIN_S, rng.expovariate(1.0 / TOOL_MEAN_S)))

    sessions, t, sid = [], 0.0, 0
    while True:
        t += rng.expovariate(top)
        if t >= WINDOW_S:
            break
        u, keep = rng.random(), rng.random()
        kind = "single" if u < MIX[0] else ("tool" if u < MIX[0] + MIX[1] else "fork")
        session = {"sid": sid, "arrival": t, "kind": kind, "task": tasks[sid % len(tasks)]}
        session["tools"] = [tool(), tool()]
        session["child_tools"] = [tool() for _ in range(FORK_K)]
        if keep < rate / top:
            sessions.append(session)
        sid += 1
    return sessions


def task_text(sid, task):
    return f"\n\nTask for agent {sid}: {task}\n"


def tool_text(sid, turn):
    return (f'\n\nTool result {turn} for agent {sid}: {{"status": "ok", "items": {sid % 7 + turn}}}\n'
            "Next call or final answer:\n")


def subtask_text(sid, index):
    return f"\n\nSubtask {index + 1} of {FORK_K} for agent {sid}: carry out step {index + 1} of the plan above.\n"


def merge_text(sid):
    return f"\n\nThe {FORK_K} subtasks of agent {sid} returned their results.\nFinal answer:\n"


# ------------------------------------------------------------------ run
class Proc:
    def __init__(self, session, role, index, argv, envs, out):
        self.session, self.role, self.index = session, role, index
        self.argv, self.envs, self.out = argv, envs, out
        self.popen = None
        self.launch = None      # scheduler time
        self.offset = 0
        self.events = []        # (kind, fields)
        self.exit = None
        self.decoded_first = False


class Run:
    def __init__(self, system, rate, trace, work, huge, log):
        self.system, self.rate, self.trace = system, rate, trace
        self.work, self.huge, self.log = work, huge, log
        self.queue = []          # Proc not started
        self.live = []           # Proc started and not finished
        self.done = []
        self.timers = []         # (time, path)
        self.guard_events = 0
        self.model_mib = os.path.getsize(MODEL) // (1 << 20)
        self.root_tokens = 0
        self.t0 = None
        self.session_state = {}

    # --- processes
    def base_env(self):
        e = dict(os.environ)
        e["CUDA_VISIBLE_DEVICES"] = GPU
        for key in ("LLAMA_KV_HOST", "LLAMA_KV_CHAIN", "LLAMA_KV_PREFIX", "LLAMA_KV_GROW",
                    "GGML_CUDA_HOST_PTR", "LD_LIBRARY_PATH"):
            e.pop(key, None)
        if self.system != "none":
            e["GGML_CUDA_HOST_PTR"] = "1"
            e["LD_LIBRARY_PATH"] = CHAIN_LIB
        return e

    def driver(self):
        return STOCK_DRIVER if self.system == "none" else DRIVER

    def proc(self, session, role, index, steps, extra):
        name = f"{session['sid']}.{role}{index}"
        argv = [self.driver(), MODEL, str(CONTEXT), str(N_GEN), *steps]
        return Proc(session, role, index, argv, extra, os.path.join(self.work, "out." + name))

    def start(self, p, now):
        e = self.base_env()
        e.update(p.envs)
        with open(p.out, "w") as out, open(p.out + ".err", "w") as err:
            p.popen = subprocess.Popen(p.argv, env=e, stdout=out, stderr=err)
        p.launch = now
        self.live.append(p)

    def admit(self, now):
        while self.queue:
            avail = meminfo("MemAvailable") - self.model_mib
            pending = sum(ESTIMATE[self.system] for p in self.live if not p.decoded_first)
            if avail - pending - ESTIMATE[self.system] < RESERVE_GIB * 1024:
                return
            self.start(self.queue.pop(0), now)

    def read(self, p):
        try:
            with open(p.out) as handle:
                handle.seek(p.offset)
                data = handle.read()
        except FileNotFoundError:
            return []
        # only whole lines
        cut = data.rfind("\n")
        if cut < 0:
            return []
        p.offset += cut + 1
        new = []
        for line in data[:cut].split("\n"):
            kind = line.split(" ", 1)[0]
            if kind in ("ATTACHED", "DECODED", "PUBLISHED", "MARK", "RELEASED", "GENERATED", "RESULT"):
                f = fields(line)
                p.events.append((kind, f))
                new.append((kind, f))
            elif kind == "TEXT":
                p.events.append(("TEXT", {"hash": hashlib.sha256(line.encode()).hexdigest()[:16]}))
        return new

    # --- sessions
    def first_steps(self):
        return [f"load:{self.work}/state.root"]

    def agent_env(self):
        if self.system == "both":
            return {"LLAMA_KV_CHAIN": f"{self.huge}/r:{self.root_tokens}", "LLAMA_KV_HOST": "anon",
                    "LLAMA_KV_GROW": GROW}
        return {}

    def go(self, sid, role, index, turn):
        return f"{self.work}/go.{sid}.{role}{index}.{turn}"

    def arrive(self, s):
        sid, kind = s["sid"], s["kind"]
        steps = self.first_steps() + [f"say:{task_text(sid, s['task'])}", "gen"]
        extra = self.agent_env()
        if kind == "tool":
            for turn in (1, 2):
                steps += [f"mark:t{turn}", f"wait:{self.go(sid, 'a', 0, turn)}",
                          f"say:{tool_text(sid, turn)}", "gen"]
        if kind == "fork":
            steps += [f"save:{self.work}/state.s{sid}", "mark:forked", f"wait:{self.go(sid, 'a', 0, 9)}",
                      f"say:{merge_text(sid)}", "gen"]
            if self.system == "both":
                extra = {"LLAMA_KV_CHAIN": f"{self.huge}/r:{self.root_tokens}",
                         "LLAMA_KV_HOST": f"{self.huge}/m{sid}", "LLAMA_KV_GROW": GROW}
        self.session_state[sid] = {"children": [], "attached": 0, "removed": False}
        self.queue.append(self.proc(s, "a", 0, steps, extra))

    def fork(self, parent, published, now):
        s = parent.session
        sid = s["sid"]
        tokens = published.get("tokens", "0")
        state = self.session_state[sid]
        state["publish_seen"] = now
        state["publish_ms"] = float(published.get("publish_ms", "0"))
        children = []
        for index in range(FORK_K):
            steps = [f"load:{self.work}/state.s{sid}", f"say:{subtask_text(sid, index)}", "gen",
                     "mark:t1", f"wait:{self.go(sid, 'c', index, 1)}", f"say:{tool_text(sid, 1)}", "gen"]
            extra = {}
            if self.system == "both":
                extra = {"LLAMA_KV_CHAIN": f"{self.huge}/r:{self.root_tokens},{self.huge}/m{sid}:{tokens}",
                         "LLAMA_KV_HOST": "anon", "LLAMA_KV_GROW": GROW}
            children.append(self.proc(s, "c", index, steps, extra))
        state["children"] = children
        self.queue[0:0] = children

    def remove_fork_files(self, sid):
        state = self.session_state[sid]
        if state["removed"]:
            return
        state["removed"] = True
        for path in glob.glob(f"{self.huge}/m{sid}") + glob.glob(f"{self.huge}/m{sid}.*") + \
                glob.glob(f"{self.work}/state.s{sid}"):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    def handle(self, p, kind, f, now):
        s = p.session
        sid = s["sid"]
        if kind == "DECODED":
            p.decoded_first = True
        if kind == "MARK":
            name = f.get("name", "")
            at = p.launch + float(f.get("since_start_ms", "0")) / 1000.0
            if p.role == "a" and name in ("t1", "t2"):
                turn = int(name[1])
                self.timers.append((at + s["tools"][turn - 1], self.go(sid, "a", 0, turn)))
            elif p.role == "c" and name == "t1":
                self.timers.append((at + s["child_tools"][p.index], self.go(sid, "c", p.index, 1)))
        if kind == "PUBLISHED" and p.role == "a" and s["kind"] == "fork":
            self.fork(p, f, now)
        if kind == "ATTACHED" and p.role == "c":
            state = self.session_state[sid]
            state["attached"] += 1
            if state["attached"] == FORK_K:
                self.remove_fork_files(sid)

    def finish(self, p, now):
        p.exit = p.popen.returncode
        self.live.remove(p)
        self.done.append(p)
        s = p.session
        if p.role == "c":
            state = self.session_state[s["sid"]]
            if all(c.exit is not None for c in state["children"]):
                self.remove_fork_files(s["sid"])
                open(self.go(s["sid"], "a", 0, 9), "w").close()

    def guard(self):
        avail = meminfo("MemAvailable") - self.model_mib
        if avail < GUARD_GIB * 1024 and self.live:
            newest = max(self.live, key=lambda q: q.launch)
            newest.popen.kill()
            self.guard_events += 1
            self.write({"event": "guard", "t": time.monotonic() - self.t0, "avail_mib": avail,
                        "sid": newest.session["sid"], "role": newest.role, "index": newest.index})

    def write(self, record):
        self.log.write(json.dumps(record) + "\n")

    def run(self):
        trace = sorted(self.trace, key=lambda s: s["arrival"])
        before = meminfo("MemAvailable")
        tmpfs_before = tmpfs_mib(self.huge)
        tmpfs_peak = tmpfs_before
        self.t0 = time.monotonic()
        next_sample, arrived, ended = 0.0, 0, "drained"
        while True:
            now = time.monotonic()
            t = now - self.t0
            while arrived < len(trace) and trace[arrived]["arrival"] <= t:
                self.arrive(trace[arrived])
                arrived += 1
            for p in list(self.live):
                for kind, f in self.read(p):
                    self.handle(p, kind, f, now)
                if p.popen.poll() is not None:
                    for kind, f in self.read(p):
                        self.handle(p, kind, f, now)
                    self.finish(p, now)
            due = [x for x in self.timers if x[0] <= now]
            if due:
                self.timers = [x for x in self.timers if x[0] > now]
                for _, path in due:
                    open(path, "w").close()
            self.guard()
            self.admit(now)
            if t >= next_sample:
                used = tmpfs_mib(self.huge)
                tmpfs_peak = max(tmpfs_peak, used)
                self.write({"event": "sample", "t": round(t, 2), "avail_mib": meminfo("MemAvailable"),
                            "drop_mib": before - meminfo("MemAvailable"), "live": len(self.live),
                            "queue": len(self.queue), "tmpfs_mib": used})
                next_sample = t + SAMPLE_S
            if arrived == len(trace) and not self.live and not self.queue:
                break
            if t > LIMIT_S:
                ended = "limit"
                break
            time.sleep(0.05)
        unfinished_live = list(self.live)
        for p in unfinished_live:
            p.popen.kill()
        for p in unfinished_live:
            p.popen.wait()
            self.read(p)
            p.exit = "unfinished"
            self.live.remove(p)
            self.done.append(p)
        for p in self.queue:
            p.exit = "queued"
            self.done.append(p)
        self.queue = []
        for sid in list(self.session_state):
            self.remove_fork_files(sid)
        time.sleep(5)
        after = meminfo("MemAvailable")
        tmpfs_after = tmpfs_mib(self.huge)
        for p in self.done:
            record = {"event": "proc", "sid": p.session["sid"], "kind": p.session["kind"],
                      "arrival": p.session["arrival"], "role": p.role, "index": p.index,
                      "launch": None if p.launch is None else round(p.launch - self.t0, 3),
                      "exit": p.exit, "events": p.events}
            if p.role == "c":
                state = self.session_state[p.session["sid"]]
                record["publish_seen"] = round(state.get("publish_seen", self.t0) - self.t0, 3)
                record["publish_ms"] = state.get("publish_ms", 0.0)
            self.write(record)
        self.write({"event": "end", "ended": ended, "t": round(time.monotonic() - self.t0, 2),
                    "before_mib": before, "after_mib": after, "residual_mib": before - after,
                    "tmpfs_before_mib": tmpfs_before, "tmpfs_peak_mib": tmpfs_peak,
                    "tmpfs_after_mib": tmpfs_after,
                    "guard_events": self.guard_events, "sessions": len(trace)})


def publish(system, work, huge):
    env_ = dict(os.environ)
    env_["CUDA_VISIBLE_DEVICES"] = GPU
    for key in ("LLAMA_KV_HOST", "LLAMA_KV_CHAIN", "LLAMA_KV_PREFIX", "LLAMA_KV_GROW", "GGML_CUDA_HOST_PTR",
                "LD_LIBRARY_PATH"):
        env_.pop(key, None)
    driver = STOCK_DRIVER
    if system != "none":
        env_.update({"GGML_CUDA_HOST_PTR": "1", "LD_LIBRARY_PATH": CHAIN_LIB})
        driver = DRIVER
    if system == "both":
        env_.update({"LLAMA_KV_HOST": f"{huge}/r", "LLAMA_KV_GROW": GROW})
    out = os.path.join(work, "publish.out")
    with open(out, "w") as o, open(out + ".err", "w") as e:
        subprocess.run([driver, MODEL, str(CONTEXT), "1", f"text:{PREFIX_FILE}", f"save:{work}/state.root"],
                       env=env_, stdout=o, stderr=e, check=True)
    tokens = 0
    for line in open(out):
        if line.startswith("PUBLISHED"):
            tokens = int(fields(line).get("tokens", "0"))
    if system == "both":
        for path in glob.glob(f"{huge}/r*"):
            os.chmod(path, 0o400)
    return tokens


def main():
    result_dir = os.path.join(RESULT_ROOT, RESULT_TAG)
    if os.path.exists(result_dir):
        sys.exit(f"result directory already exists: {result_dir}")
    for path in (DRIVER, STOCK_DRIVER, MODEL, PREFIX_FILE, TASKS_FILE):
        if not os.path.exists(path):
            sys.exit(f"missing {path}")
    if not GPU:
        sys.exit("PRODUCER_MIG (the GPU) is not set")
    os.makedirs(result_dir)
    huge = os.path.join(MOUNT_ROOT, "kv_dynamic")
    os.makedirs(huge, exist_ok=True)
    sudo("mount", "-t", "tmpfs", "-o", f"huge=always,size={TMPFS_MIB}m,uid={os.getuid()},gid={os.getgid()}",
         "tmpfs", huge)
    work = os.path.join(result_dir, "work")
    os.makedirs(work)
    with open(os.path.join(result_dir, "metadata.txt"), "w") as meta:
        for key in ("SYSTEMS", "RATES", "WINDOW_S", "LIMIT_S", "SEED", "N_GEN", "CONTEXT", "MIX", "TOOL_MEAN_S",
                    "TOOL_MIN_S", "TOOL_MAX_S", "FORK_K", "RESERVE_GIB", "GUARD_GIB", "ESTIMATE", "TMPFS_MIB"):
            meta.write(f"{key.lower()}={globals()[key]}\n")
        meta.write(f"model={os.path.basename(MODEL)}\nprefix_file={os.path.basename(PREFIX_FILE)}\n")
        meta.write(f"mem_total_mib={meminfo('MemTotal')}\n")
        meta.write(subprocess.run(["sha256sum", DRIVER, STOCK_DRIVER, CHAIN_LIB + "/libllama.so",
                                   os.path.abspath(__file__)], capture_output=True, text=True).stdout)
        meta.write("boot_id=" + open("/proc/sys/kernel/random/boot_id").read())
    log = open(os.path.join(result_dir, "raw.jsonl"), "a", buffering=1)
    current = []

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        for r_index, rate in enumerate(RATES):
            trace = make_trace(rate, SEED)
            order = SYSTEMS if r_index % 2 == 0 else list(reversed(SYSTEMS))
            for system in order:
                for path in glob.glob(f"{huge}/*") + glob.glob(f"{work}/*"):
                    os.unlink(path)
                subprocess.run(["sync"])
                sudo("sh", "-c", "echo 1 >/proc/sys/vm/drop_caches")
                with open(MODEL, "rb") as handle:
                    while handle.read(1 << 24):
                        pass
                tokens = publish(system, work, huge)
                log.write(json.dumps({"event": "begin", "system": system, "rate": rate, "root_tokens": tokens,
                                      "sessions": len(trace), "huge_used_mib": int(subprocess.run(
                                          ["du", "-sm", huge], capture_output=True, text=True).stdout.split()[0]),
                                      "kreclaimable_mib": meminfo("KReclaimable")}) + "\n")
                run = Run(system, rate, trace, work, huge, log)
                current[:] = [run]
                run.root_tokens = tokens
                run.run()
                current[:] = []
    finally:
        for run in current:
            for p in run.live:
                try:
                    p.popen.kill()
                except Exception:
                    pass
        log.close()
        subprocess.run(["sudo", "-n", "umount", huge])
        try:
            os.rmdir(huge)
        except OSError:
            pass
        for path in glob.glob(f"{work}/*"):
            try:
                os.unlink(path)
            except OSError:
                pass
    summarize(result_dir)
    print(f"engine_kvdynamic_result_dir={result_dir}")


# ------------------------------------------------------------------ summary
def pct(values, q):
    if not values:
        return ""
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def summarize(result_dir):
    runs, current = [], None
    for line in open(os.path.join(result_dir, "raw.jsonl")):
        r = json.loads(line)
        if r["event"] == "begin":
            current = {"begin": r, "samples": [], "procs": [], "guards": 0}
            runs.append(current)
        elif r["event"] == "sample":
            current["samples"].append(r)
        elif r["event"] == "proc":
            current["procs"].append(r)
        elif r["event"] == "guard":
            current["guards"] += 1
        elif r["event"] == "end":
            current["end"] = r
    rows, texts = [], {}
    for run in runs:
        system, rate = run["begin"]["system"], run["begin"]["rate"]
        ttft, turn, child, pause, failed, unfinished, completed = [], [], [], [], 0, 0, set()
        # a session that never started counts with the time it waited, a lower bound of its TTFT
        censored = []
        sessions = {}
        for p in run["procs"]:
            sid = p["sid"]
            ev = p["events"]
            ok = p["exit"] == 0
            if p["exit"] in ("unfinished", "queued"):
                unfinished += 1
            elif not ok:
                failed += 1
            sessions.setdefault(sid, []).append(ok)
            decoded = [f for k, f in ev if k == "DECODED"]
            if p["role"] == "a" and decoded and p["launch"] is not None:
                ttft.append(p["launch"] - p["arrival"] + float(decoded[0]["since_start_ms"]) / 1000)
            elif p["role"] == "a":
                censored.append(float(run.get("end", {}).get("t", LIMIT_S)) - p["arrival"])
            if p["role"] == "c" and decoded and p["launch"] is not None:
                child.append(p["launch"] - p["publish_seen"] + float(decoded[0]["since_start_ms"]) / 1000)
            released = [float(f["since_start_ms"]) for k, f in ev if k == "RELEASED"]
            for r_ms in released:
                after = [float(f["since_start_ms"]) for f in decoded if float(f["since_start_ms"]) > r_ms]
                if after:
                    turn.append((after[0] - r_ms) / 1000)
            pause += [float(f["publish_ms"]) for k, f in ev if k == "PUBLISHED"]
            text = [f["hash"] for k, f in ev if k == "TEXT"]
            if ok and text:
                texts.setdefault((rate, sid, p["role"], p["index"]), {})[system] = text[0]
        for sid, oks in sessions.items():
            if all(oks):
                completed.add(sid)
        samples = run["samples"]
        drops = [s["drop_mib"] for s in samples]
        end = run.get("end", {})
        rows.append({
            "system": system, "name": NAMES[system], "rate": rate, "sessions": run["begin"]["sessions"],
            "completed": len(completed), "failed_procs": failed, "unfinished_procs": unfinished,
            "guard_events": run["guards"], "ended": end.get("ended", ""), "wall_s": end.get("t", ""),
            "ttft_p50_s": pct(ttft, 0.5), "ttft_p99_s": pct(ttft, 0.99),
            "ttft_p99_with_unserved_s": pct(ttft + censored, 0.99), "unserved_sessions": len(censored),
            "ttft_mean_s":
                statistics.mean(ttft) if ttft else "",
            "turn_p50_s": pct(turn, 0.5), "turn_p99_s": pct(turn, 0.99),
            "child_p50_s": pct(child, 0.5), "child_p99_s": pct(child, 0.99),
            "fork_pause_ms": statistics.mean(pause) if pause else "",
            "mem_peak_gib": max(drops) / 1024 if drops else "", "mem_mean_gib": statistics.mean(drops) / 1024
            if drops else "", "live_peak": max(s["live"] for s in samples) if samples else "",
            "live_mean": statistics.mean(s["live"] for s in samples) if samples else "",
            "queue_peak": max(s["queue"] for s in samples) if samples else "",
            "residual_gib": end.get("residual_mib", 0) / 1024,
            "tmpfs_before_mib": end.get("tmpfs_before_mib", ""), "tmpfs_peak_mib": end.get("tmpfs_peak_mib", ""),
            "tmpfs_after_mib": end.get("tmpfs_after_mib", "")})
    for row in rows:
        same = compared = 0
        for key, by_system in texts.items():
            if key[0] != row["rate"] or row["system"] not in by_system or len(by_system) < 2:
                continue
            others = [v for s, v in by_system.items() if s != row["system"]]
            compared += 1
            same += all(v == by_system[row["system"]] for v in others)
        row["texts_equal"] = f"{same}/{compared}"
    path = os.path.join(result_dir, "engine_kvdynamic_summary.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["system"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in row.items()})
    gates = {"D1": all(r["failed_procs"] == 0 and r["guard_events"] == 0 for r in rows),
             "D2": all(r["texts_equal"].split("/")[0] == r["texts_equal"].split("/")[1] for r in rows),
             "D3": all(r["tmpfs_after_mib"] != "" and r["tmpfs_after_mib"] - r["tmpfs_before_mib"] < 2
                       for r in rows if r["system"] == "both"),
             "D4": all(r["completed"] == r["sessions"] for r in rows
                       if r["system"] in ("both", "weights") and rows and r["rate"] == min(x["rate"] for x in rows))}
    with open(os.path.join(result_dir, "gates.txt"), "w") as handle:
        for key, value in gates.items():
            handle.write(f"{key} {'PASS' if value else 'FAIL'}\n")


if __name__ == "__main__":
    if os.environ.get("SUMMARIZE"):
        summarize(os.environ["SUMMARIZE"])
    else:
        main()
