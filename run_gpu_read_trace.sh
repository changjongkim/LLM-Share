#!/usr/bin/env bash
# Where a wrong GPU read of a read-only file mapping comes from
# (gpu_read_trace.cu). Campaign 20261014-gpu-read-check-full-v1 found two
# reader rounds, both with 32 processes on a tmpfs with 2 MiB pages, in which
# the GPU read words of a 4 KiB page that belong to the first 4 KiB page of the
# same 2 MiB page. This campaign repeats the 32-process case more often, adds
# 8 processes, and maps the file once as the kernel chooses (huge: the kernel
# may map the file with 2 MiB entries) and once advised against huge pages
# (nohuge: 4 KiB entries). After a round with a differing word, the reader reads
# the mapping again from the CPU and from the GPU.
#
# A case writes a file of SIZE_MIB on a tmpfs with 2 MiB pages from the CPU and
# has PROCS processes map its first HEAD_MIB and compare every word on the GPU,
# ROUNDS times each, all starting together. Cases: CASES (mapping:read:procs).
#
# Kill gates, fixed before the campaign:
#   T1  every reader reports every round;
#   T2  with the CPU read first, no word differs.
# Reported, not gated: the rounds and words that differ per case, whether the
# CPU and a second GPU pass read the same words right, where the values come
# from in their 2 MiB page, and the file memory mapped with 2 MiB entries.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
repetitions=${REPETITIONS:-16}
size_mib=${SIZE_MIB:-1024}
head_mib=${HEAD_MIB:-896}
rounds=${ROUNDS:-4}
read -r -a cases <<<"${CASES:-huge:gpu:32 nohuge:gpu:32 huge:gpu:8 nohuge:gpu:8 huge:cpu:32}"
checker=${CHECKER:-"$script_dir/gpu_read_trace"}
probe=${PROBE:-"$script_dir/linux_share_probe"}
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-gpu-read-trace"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]]; then
  echo "REPETITIONS must be a positive integer" >&2
  exit 2
fi
if [[ ! -x "$checker" || ! -x "$probe" ]]; then
  echo "missing programs; run make gpu_read_trace linux_share_probe" >&2
  exit 2
fi
if ! sudo -n true 2>/dev/null; then
  echo "passwordless sudo is required for the tmpfs mount" >&2
  exit 2
fi
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi

# Pages that the driver threads serving GPU faults have faulted in, and their
# CPU time in clock ticks, summed over the threads.
fault_counters() {
  local pages=0 ticks=0 stat line rest
  for stat in /proc/[0-9]*/stat; do
    { read -r line <"$stat"; } 2>/dev/null || continue
    [[ "$line" == *"(UVM GPU"*" BH)"* ]] || continue
    rest=${line##*) }
    # shellcheck disable=SC2086
    set -- $rest
    pages=$(( pages + $8 + ${10} ))
    ticks=$(( ticks + ${12} + ${13} ))
  done
  printf '%s %s' "$pages" "$ticks"
}
tick_ms=$(( 1000 / $(getconf CLK_TCK) ))

memfs_dir="$mount_root/read_trace_memfs"
mounted=0
work=
anchor_pid=
cleanup() {
  if [[ -n "$anchor_pid" ]]; then kill "$anchor_pid" 2>/dev/null || true; fi
  pkill -9 -P $$ 2>/dev/null || true
  if (( mounted )); then sudo -n umount "$memfs_dir" || true; fi
  rmdir "$memfs_dir" "$mount_root" 2>/dev/null || true
  if [[ -n "$work" ]]; then rm -rf "${work:?}"; fi
}
trap cleanup EXIT
mkdir -p "$memfs_dir" "$result_dir"
sudo -n mount -t tmpfs -o "huge=always,size=$(( size_mib + 512 ))m,uid=$(id -u),gid=$(id -g)" \
  tmpfs "$memfs_dir"
mounted=1
work=$(mktemp -d)

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig=%s\nrepetitions=%s\nsize_mib=%s\nhead_mib=%s\nrounds=%s\n' "$mig" "$repetitions" \
    "$size_mib" "$head_mib" "$rounds"
  printf 'cases=%s\n' "${cases[*]}"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$checker" "$probe"
} >"$result_dir/metadata.txt"

export CUDA_VISIBLE_DEVICES="$mig"
# The driver keeps the threads that serve GPU faults while a process holds a
# context; this one holds it for the whole campaign, so that their counters
# carry over between cases.
"$probe" hold 1000000 >"$work/anchor" 2>&1 &
anchor_pid=$!
sleep 3

run_case() {  # mapping read procs
  local mapping=$1 read=$2 procs=$3 file="$memfs_dir/data" failed=0 index pids=()
  local before after
  rm -f "$work/go" "$work"/reader.*
  printf 'CASE mapping=%s read=%s procs=%s\n' "$mapping" "$read" "$procs" >>"$raw_log"
  for index in $(seq 0 $((procs - 1))); do
    timeout -s KILL 900 "$checker" check "$file" "$head_mib" "$rounds" "$read" "$mapping" "$work/go" \
      >"$work/reader.$index" 2>&1 &
    pids+=($!)
  done
  sleep 2
  before=$(fault_counters)
  touch "$work/go"
  for index in "${!pids[@]}"; do
    wait "${pids[$index]}" || failed=$((failed + 1))
  done
  after=$(fault_counters)
  for index in "${!pids[@]}"; do
    grep -E '^(CHECK|EXAMPLE|ERROR) ' "$work/reader.$index" | sed "s/^\\([A-Z]*\\) /\\1 proc=$index /" \
      >>"$raw_log" || true
  done
  read -r -a b <<<"$before"
  read -r -a a <<<"$after"
  printf 'KERNEL gpu_fault_pages=%s fault_thread_ms=%s\n' "$(( a[0] - b[0] ))" \
    "$(( (a[1] - b[1]) * tick_ms ))" >>"$raw_log"
  printf 'END_CASE failed=%s\n' "$failed" >>"$raw_log"
}

"$checker" make "$memfs_dir/data" "$size_mib" >/dev/null
for run in $(seq 1 "$repetitions"); do
  printf 'BEGIN_READ_TRACE run=%s\n' "$run" >>"$raw_log"
  for spec in "${cases[@]}"; do
    IFS=: read -r mapping read procs <<<"$spec"
    run_case "$mapping" "$read" "$procs"
  done
  printf 'END_READ_TRACE\n' >>"$raw_log"
done
awk -f "$script_dir/summarize_gpu_read_trace.awk" "$raw_log" >"$result_dir/gpu_read_trace_summary.csv"
(
  cd "$script_dir"
  sha256sum gpu_read_trace.cu summarize_gpu_read_trace.awk run_gpu_read_trace.sh
) >"$result_dir/source_hashes.txt"
printf 'gpu_read_trace_result_dir=%s\n' "$result_dir"
