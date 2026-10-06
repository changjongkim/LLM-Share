#!/usr/bin/env python3
"""Client of the server campaign (run_engine_kvserver.sh). Talks to stock
llama-server processes through their HTTP interface and prints one line per
request in the key=value form of the other campaigns.

  kv_server_client.py wait    PORT [SECONDS]
  kv_server_client.py publish PORT PREFIX_FILE STATE_NAME
  kv_server_client.py agents  PREFIX_FILE TASKS_FILE N_GEN STATE_NAME PORT...

wait     until the server answers /health.
publish  evaluates the prefix in slot 0 and saves the slot.
agents   for every port, in parallel: restores the slot from STATE_NAME (if
         it is not "-") and requests the completion of prefix + task with
         the prompt cache on. The task of agent i is line i+1 of TASKS_FILE.
"""
import hashlib
import json
import sys
import threading
import time
import urllib.error
import urllib.request


def call(port, path, body=None, timeout=1800):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def wait(port, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            if call(port, "/health", timeout=2).get("status") == "ok":
                return 0
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(0.05)
    return 1


def completion(port, prompt, n_predict):
    return call(port, "/completion", {
        "prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "top_k": 1,
        "seed": 1, "cache_prompt": True, "id_slot": 0, "stream": False})


def publish(port, prefix_file, state_name):
    prefix = open(prefix_file).read()
    start = time.time()
    answer = completion(port, prefix, 0)
    prefix_ms = (time.time() - start) * 1000
    start = time.time()
    saved = call(port, "/slots/0?action=save", {"filename": state_name})
    print(f"PUBLISHED prefix_tokens={saved['n_saved']} prefix_ms={prefix_ms:.1f} "
          f"publish_ms={(time.time() - start) * 1000:.2f} "
          f"server_save_ms={saved['timings']['save_ms']:.2f} state_bytes={saved['n_written']} "
          f"evaluated={answer['timings']['prompt_n']}")
    return 0


def agent(index, port, prefix, task, n_gen, state_name, lines, started):
    try:
        restore_ms = server_restore_ms = 0.0
        restored = 0
        if state_name != "-":
            start = time.time()
            answer = call(port, "/slots/0?action=restore", {"filename": state_name})
            restore_ms = (time.time() - start) * 1000
            server_restore_ms = answer["timings"]["restore_ms"]
            restored = answer["n_restored"]
        start = time.time()
        answer = completion(port, prefix + task, n_gen)
        request_ms = (time.time() - start) * 1000
        timings = answer["timings"]
        text = hashlib.sha256(answer["content"].encode()).hexdigest()[:16]
        lines[index] = (
            f"AGENT index={index} port={port} exit=0 text={text} restored_tokens={restored} "
            f"restore_ms={restore_ms:.2f} server_restore_ms={server_restore_ms:.2f} "
            f"prompt_evaluated={timings['prompt_n']} prompt_ms={timings['prompt_ms']:.1f} "
            f"first_token_ms={(start - started) * 1000 + timings['prompt_ms']:.1f} "
            f"generated={timings['predicted_n']} generation_tps={timings['predicted_per_second']:.3f} "
            f"request_ms={request_ms:.1f}")
    except Exception as error:  # the line records the failure; the runner counts it
        lines[index] = f"AGENT index={index} port={port} exit=1 error={type(error).__name__}"


def agents(prefix_file, tasks_file, n_gen, state_name, ports):
    prefix = open(prefix_file).read()
    tasks = [line.rstrip("\n") for line in open(tasks_file) if line.strip()]
    lines = [None] * len(ports)
    started = time.time()
    threads = []
    for index, port in enumerate(ports):
        task = f"\n\nTask for agent {index}: {tasks[index % len(tasks)]}\n"
        threads.append(threading.Thread(target=agent, args=(
            index, port, prefix, task, n_gen, state_name, lines, started)))
        threads[-1].start()
    for thread in threads:
        thread.join()
    print("\n".join(lines))
    return 0


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "wait":
        return wait(int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 120.0)
    if len(sys.argv) == 5 and sys.argv[1] == "publish":
        return publish(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    if len(sys.argv) >= 7 and sys.argv[1] == "agents":
        return agents(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5],
                      [int(port) for port in sys.argv[6:]])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
