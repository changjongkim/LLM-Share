#!/usr/bin/env python3
"""Client of several vLLM servers that serve the same prefix (run_vllm_share.sh).

  vllm_share_client.py wait PORT [SECONDS]
  vllm_share_client.py prefix MODEL PREFIX_FILE PORT
  vllm_share_client.py agents MODEL PREFIX_FILE TASKS_FILE N_GEN COUNT PORT[,PORT...] [FIRST_TASK]

prefix asks the server at PORT for one token after the prefix alone, which
leaves the cache of the prefix in that server. agents sends COUNT requests
at the same time, agent i to the i-th port in turn; agent i asks for the
continuation of PREFIX + task FIRST_TASK + i (raw prompt, greedy decoding,
N_GEN tokens). Each agent prints one line in the key=value form of the
other campaigns, with the prompt tokens that the server reports as served
from its cache.
"""
import sys
import threading
import time

from openai_client import agent, wait


def prefix(model, prefix_file, port):
    lines = [None]
    agent(0, port, model, open(prefix_file).read(), 1, lines, time.time())
    print(lines[0].replace("AGENT index=0", "PREFIX", 1))
    return 0


def agents(model, prefix_file, tasks_file, n_gen, count, ports, first_task):
    shared = open(prefix_file).read()
    tasks = [line.rstrip("\n") for line in open(tasks_file)]
    lines = [None] * count
    started = time.time()
    threads = []
    for index in range(count):
        task = f"\n\nTask for agent {first_task + index}: {tasks[(first_task + index) % len(tasks)]}\n"
        threads.append(threading.Thread(target=agent, args=(
            index, ports[index % len(ports)], model, shared + task, n_gen, lines, started)))
        threads[-1].start()
    for thread in threads:
        thread.join()
    print("\n".join(lines))
    return 0


def main():
    arguments = sys.argv[1:]
    if len(arguments) >= 2 and arguments[0] == "wait":
        return wait(int(arguments[1]), float(arguments[2]) if len(arguments) > 2 else 60.0)
    if len(arguments) == 4 and arguments[0] == "prefix":
        return prefix(arguments[1], arguments[2], int(arguments[3]))
    if len(arguments) in (7, 8) and arguments[0] == "agents":
        return agents(arguments[1], arguments[2], arguments[3], int(arguments[4]), int(arguments[5]),
                      [int(port) for port in arguments[6].split(",")],
                      int(arguments[7]) if len(arguments) == 8 else 0)
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
