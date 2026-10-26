#!/usr/bin/env bash
# kv_fork2 with the CLOCK_MONOTONIC event recorder of stator_campaign_20261016
# (timing-only copy of the same source). Logs one launch line per process and
# execs the driver, which keeps the pid of the recorder file <pid>-<start>.jsonl.
D=${STATOR_TIMING_BASE:?set STATOR_TIMING_BASE}
mkdir -p "$D"
printf '%s %s %s %s %s\n' "$(date +%s%N)" "$$" "kv_fork2" "$1" "${LLAMA_KV_HOST:-device}" >>"$D/launch.log"
export STATOR_TIMING_DIR="$D"
exec /home/thor/kcj/stator_campaign_20261016/bin/kv_fork2 "$@"
