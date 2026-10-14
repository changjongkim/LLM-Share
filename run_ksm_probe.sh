#!/usr/bin/env bash
# Kernel samepage merging (KSM) on memory that the GPU reads and writes
# (ksm_probe.cu), on the whole GPU. In every case one process holds COPIES
# buffers of SIZE_MIB with the same content, marked mergeable, and ksmd scans
# KSM_PAGES pages every KSM_SLEEP_MS milliseconds until it has merged them. In
# a process that has not initialized CUDA (nocuda) only the merge is timed; a
# process that has (cuda) then reads and writes the merged buffers on the GPU.
# The runner unmerges everything while the process still maps it, and only
# then lets it exit; the settings of ksmd are restored at the end.
#
# Kill gates, fixed before the campaign:
#   Q1  every case merges at least 99% of its pages and exits;
#   Q2  in cuda, a GPU read of merged pages keeps at least 99% of them merged;
#   Q3  in cuda, a GPU write of one word per page leaves at most 10% of the
#       written pages merged.
# Reported, not gated: the merge rate with and without CUDA, the CPU time of
# ksmd, and the time of the GPU write.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
gpu=${PRODUCER_MIG:-0}
repetitions=${REPETITIONS:-6}
size_mib=${SIZE_MIB:-256}
copies=${COPIES:-4}
ksm_pages=${KSM_PAGES:-20000}
ksm_sleep_ms=${KSM_SLEEP_MS:-1}
probe=${PROBE:-"$script_dir/ksm_probe"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-ksm-probe"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"
ksm=/sys/kernel/mm/ksm

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
if [[ ! -x "$probe" ]]; then
  echo "missing probe; run make ksm_probe" >&2
  exit 2
fi
if ! sudo -n true 2>/dev/null || [[ ! -e "$ksm/run" ]]; then
  echo "passwordless sudo and KSM are required" >&2
  exit 2
fi
if [[ "$(cat "$ksm/pages_sharing")" != 0 || "$(cat "$ksm/run")" != 0 ]]; then
  echo "KSM is in use; it must be stopped with no merged pages" >&2
  exit 2
fi
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi

pages_before=$(cat "$ksm/pages_to_scan")
sleep_before=$(cat "$ksm/sleep_millisecs")
ksm_set() { echo "$2" | sudo -n tee "$ksm/$1" >/dev/null; }
ksmd_ticks() {
  local pid line rest
  pid=$(pgrep -x ksmd | head -n 1)
  [[ -n "$pid" ]] || { printf '0'; return; }
  read -r line <"/proc/$pid/stat"
  rest=${line##*) }
  # shellcheck disable=SC2086
  set -- $rest
  printf '%s' "$(( ${12} + ${13} ))"
}
tick_ms=$(( 1000 / $(getconf CLK_TCK) ))
probe_pid=
fifo=
cleanup() {
  ksm_set run 0 || true
  if [[ -n "$probe_pid" ]] && kill -0 "$probe_pid" 2>/dev/null; then
    # Unmerge while the process still maps its pages, then end it.
    ksm_set run 2 || true
    for _ in $(seq 1 600); do [[ "$(cat "$ksm/pages_sharing")" == 0 ]] && break; sleep 0.2; done
    kill "$probe_pid" 2>/dev/null || true
  fi
  ksm_set run 0 || true
  ksm_set pages_to_scan "$pages_before" || true
  ksm_set sleep_millisecs "$sleep_before" || true
  [[ -z "$fifo" ]] || rm -f "$fifo"
}
trap cleanup EXIT
mkdir -p "$result_dir"
{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'gpu=%s\nrepetitions=%s\nsize_mib=%s\ncopies=%s\n' "$gpu" "$repetitions" "$size_mib" \
    "$copies"
  printf 'ksm_pages_to_scan=%s\nksm_sleep_millisecs=%s\nksm_before=pages_to_scan:%s,sleep_millisecs:%s\n' \
    "$ksm_pages" "$ksm_sleep_ms" "$pages_before" "$sleep_before"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$probe"
} >"$result_dir/metadata.txt"

wait_for() {  # word file pid seconds
  local waited=0
  until grep -q "^$1" "$2" 2>/dev/null; do
    kill -0 "$3" 2>/dev/null || return 1
    (( waited >= $4 * 10 )) && return 1
    sleep 0.1
    waited=$((waited + 1))
  done
}

run_case() {  # kind
  local kind=$1 out ticks_before ticks_after status=0
  out=$(mktemp "$result_dir/out.XXXXXX")
  fifo=$(mktemp -u "$result_dir/fifo.XXXXXX")
  mkfifo "$fifo"
  ksm_set run 0
  ksm_set pages_to_scan "$ksm_pages"
  ksm_set sleep_millisecs "$ksm_sleep_ms"
  CUDA_VISIBLE_DEVICES="$gpu" "$probe" "$size_mib" "$copies" "$kind" <"$fifo" >"$out" 2>&1 &
  probe_pid=$!
  exec 8>"$fifo"
  wait_for READY "$out" "$probe_pid" 120 || status=1
  ticks_before=$(ksmd_ticks)
  ksm_set run 1
  wait_for MERGED "$out" "$probe_pid" 960 || status=1
  ksm_set run 0
  ticks_after=$(ksmd_ticks)
  echo >&8
  wait_for DONE "$out" "$probe_pid" 300 || status=1
  # Unmerge while the process still maps the pages.
  ksm_set run 2
  for _ in $(seq 1 600); do [[ "$(cat "$ksm/pages_sharing")" == 0 ]] && break; sleep 0.2; done
  ksm_set run 0
  echo >&8
  exec 8>&-
  wait "$probe_pid" || status=1
  probe_pid=
  rm -f "$fifo"
  fifo=
  {
    printf 'CASE kind=%s status=%s ksmd_ms=%s\n' "$kind" "$status" \
      "$(( (ticks_after - ticks_before) * tick_ms ))"
    grep -E '^(MERGE|GPU) ' "$out" || true
    printf 'END_CASE\n'
  } >>"$raw_log"
  rm -f "$out"
}

for run in $(seq 1 "$repetitions"); do
  printf 'BEGIN_KSM run=%s\n' "$run" >>"$raw_log"
  if (( run % 2 == 1 )); then kinds=(nocuda cuda); else kinds=(cuda nocuda); fi
  for kind in "${kinds[@]}"; do run_case "$kind"; done
  printf 'END_KSM\n' >>"$raw_log"
done

awk -f "$script_dir/summarize_ksm_probe.awk" "$raw_log" >"$result_dir/ksm_probe_summary.csv"
(
  cd "$script_dir"
  sha256sum ksm_probe.cu summarize_ksm_probe.awk run_ksm_probe.sh
) >"$result_dir/source_hashes.txt"
printf 'ksm_probe_result_dir=%s\n' "$result_dir"
