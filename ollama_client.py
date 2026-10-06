#!/usr/bin/env python3
"""Client of the Ollama campaign (run_ollama_agents.sh).

  ollama_client.py wait PORT [SECONDS]
  ollama_client.py status PORT
  ollama_client.py agents MODEL PREFIX_FILE N_GEN CONTEXT PORT...

agents sends one request per listed port entry, all at the same time: agent
i asks the server at its port to continue PREFIX + task i (raw prompt, greedy
decoding, N_GEN tokens, a context of CONTEXT tokens). A port may be listed
several times: its server then holds as many requests in parallel. Prints one
line per agent in the key=value form of the other campaigns.
"""
import hashlib
import json
import sys
import threading
import time
import urllib.error
import urllib.request


def call(port, path, timeout=10):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as answer:
        return json.loads(answer.read().decode())


def wait(port, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/version", timeout=2) as answer:
                if answer.status == 200:
                    return 0
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.1)
    return 1


def status(port):
    models = call(port, "/api/ps").get("models", [])
    vram = sum(int(model.get("size_vram", 0)) for model in models)
    print(f"SERVER port={port} models={len(models)} vram_bytes={vram} gpu={int(vram > 0)}")
    return 0


def agent(index, port, model, prompt, n_gen, context, lines, started):
    body = {"model": model, "prompt": prompt, "raw": True, "stream": True,
            "options": {"num_predict": n_gen, "temperature": 0, "top_k": 1, "seed": 1,
                        "num_ctx": context}}
    request = urllib.request.Request(f"http://127.0.0.1:{port}/api/generate",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        text, first, last = "", None, {}
        with urllib.request.urlopen(request, timeout=3600) as answer:
            for raw in answer:
                piece = json.loads(raw.decode())
                if "error" in piece:
                    raise RuntimeError(piece["error"])
                if first is None and piece.get("response"):
                    first = time.time()
                text += piece.get("response", "")
                if piece.get("done"):
                    last = piece
        done = time.time()
        generation = last.get("eval_duration", 0) / 1e9
        lines[index] = (
            f"AGENT index={index} port={port} exit=0 "
            f"text={hashlib.sha256(text.encode()).hexdigest()[:16]} "
            f"prompt_evaluated={last.get('prompt_eval_count', 0)} "
            f"prompt_ms={last.get('prompt_eval_duration', 0) / 1e6:.1f} "
            f"load_ms={last.get('load_duration', 0) / 1e6:.1f} "
            f"first_token_ms={((first or done) - started) * 1000:.1f} "
            f"generated={last.get('eval_count', 0)} "
            f"generation_tps={(last.get('eval_count', 0) / generation) if generation > 0 else 0:.3f} "
            f"request_ms={(done - started) * 1000:.1f}")
    except Exception as error:  # the line records the failure; the runner counts it
        lines[index] = (f"AGENT index={index} port={port} exit=1 "
                        f"error={type(error).__name__}:{str(error)[:80].replace(' ', '_')}")


def agents(model, prefix_file, n_gen, context, ports):
    prefix = open(prefix_file).read()
    lines = [None] * len(ports)
    started = time.time()
    threads = []
    for index, port in enumerate(ports):
        task = f"\n\nTask for agent {index}: summarize the text above in {index + 2} sentences.\n"
        threads.append(threading.Thread(target=agent, args=(
            index, port, model, prefix + task, n_gen, context, lines, started)))
        threads[-1].start()
    for thread in threads:
        thread.join()
    print("\n".join(lines))
    return 0


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "wait":
        return wait(int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 60.0)
    if len(sys.argv) == 3 and sys.argv[1] == "status":
        return status(int(sys.argv[2]))
    if len(sys.argv) >= 7 and sys.argv[1] == "agents":
        return agents(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]),
                      [int(port) for port in sys.argv[6:]])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
