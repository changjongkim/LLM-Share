#!/usr/bin/env python3
"""Client of a server with the OpenAI completions interface (run_ext_vllm.sh).

  openai_client.py wait PORT [SECONDS]
  openai_client.py agents MODEL PREFIX_FILE TASKS_FILE N_GEN PORT COUNT

agents sends COUNT requests to the server at PORT, all at the same time:
agent i asks for the continuation of PREFIX + task i (line i+1 of
TASKS_FILE; raw prompt, greedy decoding, N_GEN tokens). Prints one line per
agent in the key=value form of the other campaigns. The prompt tokens that
the server reports as served from its cache are printed when it reports them.
"""
import hashlib
import json
import sys
import threading
import time
import urllib.error
import urllib.request


def wait(port, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as answer:
                if answer.status == 200:
                    return 0
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    return 1


def agent(index, port, model, prompt, n_gen, lines, started):
    body = {"model": model, "prompt": prompt, "max_tokens": n_gen, "temperature": 0,
            "seed": 1, "stream": True, "stream_options": {"include_usage": True}}
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        text, first, usage = "", None, {}
        with urllib.request.urlopen(request, timeout=3600) as answer:
            for raw in answer:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                piece = json.loads(payload)
                if "error" in piece:
                    raise RuntimeError(str(piece["error"]))
                for choice in piece.get("choices", []):
                    if first is None and choice.get("text"):
                        first = time.time()
                    text += choice.get("text", "")
                if piece.get("usage"):
                    usage = piece["usage"]
        done = time.time()
        generated = int(usage.get("completion_tokens", 0))
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", -1)
        generation = done - (first or done)
        lines[index] = (
            f"AGENT index={index} port={port} exit=0 "
            f"text={hashlib.sha256(text.encode()).hexdigest()[:16]} "
            f"prompt_tokens={usage.get('prompt_tokens', 0)} prompt_cached={cached} "
            f"first_token_ms={((first or done) - started) * 1000:.1f} "
            f"generated={generated} "
            f"generation_tps={((generated - 1) / generation) if generation > 0 and generated > 1 else 0:.3f} "
            f"request_ms={(done - started) * 1000:.1f}")
    except Exception as error:  # the line records the failure; the runner counts it
        lines[index] = (f"AGENT index={index} port={port} exit=1 "
                        f"error={type(error).__name__}:{str(error)[:80].replace(' ', '_')}")


def agents(model, prefix_file, tasks_file, n_gen, port, count):
    prefix = open(prefix_file).read()
    tasks = [line.rstrip("\n") for line in open(tasks_file)]
    lines = [None] * count
    started = time.time()
    threads = []
    for index in range(count):
        task = f"\n\nTask for agent {index}: {tasks[index % len(tasks)]}\n"
        threads.append(threading.Thread(target=agent, args=(
            index, port, model, prefix + task, n_gen, lines, started)))
        threads[-1].start()
    for thread in threads:
        thread.join()
    print("\n".join(lines))
    return 0


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "wait":
        return wait(int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 60.0)
    if len(sys.argv) == 8 and sys.argv[1] == "agents":
        return agents(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]),
                      int(sys.argv[6]), int(sys.argv[7]))
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
