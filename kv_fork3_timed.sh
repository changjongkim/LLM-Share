#!/usr/bin/env bash
# kv_fork3 with the CLOCK_MONOTONIC event recorder of stator_campaign_20261016
# (timing-only copy of the same source, linked to llama.cpp-kv3). Logs one
# launch line per process (ns, pid, role, KV placement) and execs the driver,
# which keeps the pid, so the recorder file <pid>-<start>.jsonl maps to the line.
D=${STATOR_TIMING_BASE:?set STATOR_TIMING_BASE}
mkdir -p "$D"
printf '%s %s %s %s\n' "$(date +%s%N)" "$$" "$1" "${LLAMA_KV_HOST:-device}" >>"$D/launch.log"
export STATOR_TIMING_DIR="$D"
exec /home/thor/kcj/stator_campaign_20261016/bin/kv_fork3 "$@"
