#!/usr/bin/env python3
"""Samples the memory counters of the kernel with CLOCK_MONOTONIC times.

The run of 128 resident agents in which the GPU took 9.2 million faults left
24 GiB available, and kernel reclaimable memory fell by 2.6 GiB at the
release of the active agents. The runner records MemAvailable and
KReclaimable only, so this sampler records the free pages, the file LRU and
the scan and steal counts of reclaim, which can be placed against the
release times of a run (the runner uses the same clock). It also records
the page faults and CPU ticks of the UVM bottom-half threads, the counter of
GPU faults of the runner, so that the faults of a run can be placed in time.

Usage: vmstat_sampler.py OUT_JSONL [PERIOD_S]

One line per sample, until SIGTERM or SIGINT: mono_ns, then the fields below
in bytes (meminfo) or counts (vmstat), and bh, the page faults and CPU ticks
of each UVM bottom-half thread by its pid. A thread exists while the GPU is
registered, so the threads are looked up again every second.
"""
import json
import os
import signal
import sys
import time

MEMINFO = ('MemFree', 'MemAvailable', 'KReclaimable', 'Active(file)', 'Inactive(file)', 'Shmem',
           'ShmemPmdMapped', 'FilePmdMapped')
VMSTAT = ('nr_free_pages', 'pgscan_kswapd', 'pgscan_direct', 'pgsteal_kswapd', 'pgsteal_direct',
          'pgrefill', 'pgdeactivate', 'workingset_refault_file', 'allocstall_normal',
          'allocstall_movable', 'kswapd_low_wmark_hit_quickly', 'compact_stall', 'pgmigrate_success',
          'thp_split_pmd')


def read(path, keys, scale):
    row = {}
    with open(path) as f:
        for line in f:
            name, value = line.split()[:2]
            name = name.rstrip(':')
            if name in keys:
                row[name] = int(value) * scale
    return row


def bh_threads():
    threads = []
    for pid in filter(str.isdigit, os.listdir('/proc')):
        try:
            line = open(f'/proc/{pid}/stat').read()
        except OSError:
            continue
        name = line[line.find('(') + 1:line.rfind(')')]
        if name.startswith('UVM GPU') and name.endswith(' BH'):
            threads.append(f'/proc/{pid}/stat')
    return threads


def bh_counts(threads):
    counts = {}
    for path in threads:
        try:
            line = open(path).read()
        except OSError:
            continue
        s = line[line.rfind(')') + 2:].split()
        counts[path.split('/')[2]] = [int(s[7]) + int(s[9]), int(s[11]) + int(s[12])]
    return {'bh': counts}


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    period = float(sys.argv[2]) if len(sys.argv) == 3 else 0.05
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    threads, samples = [], 0
    with open(sys.argv[1], 'w') as out:
        while not stop:
            if samples % max(1, round(1 / period)) == 0:
                threads = bh_threads()
            samples += 1
            row = {'mono_ns': time.monotonic_ns()}
            row.update(read('/proc/meminfo', MEMINFO, 1024))
            row.update(read('/proc/vmstat', VMSTAT, 1))
            row.update(bh_counts(threads))
            out.write(json.dumps(row) + '\n')
            time.sleep(period)


if __name__ == '__main__':
    main()
