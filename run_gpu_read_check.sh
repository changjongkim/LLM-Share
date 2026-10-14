#!/usr/bin/env bash
# Whether the GPU reads the data of a file when it faults on the pages of a
# private, read-only mapping of the file (gpu_read_check.cu), as an agent of
# STATOR (Map Only) faults on the published prefix. The engine campaign
# 20261014-engine-kvdiverge-full-v1 found agents whose text differs from copy
# only when the CPU read of the mapped prefix is left out; this campaign checks
# every word that the GPU reads in that case.
#
# A case writes a file of SIZE_MIB and has PROCS processes map its first
# HEAD_MIB and compare every word on the GPU, ROUNDS times each, all starting
# together. It varies:
#   fs      memfs  a tmpfs with 2 MiB pages (the cache file of STATOR)
#           ext4   the page cache of a file on ext4, in 4 KiB pages
#   writer  cpu    the file is written from the CPU before the readers start
#           gpu    a publisher maps the file shared and writable, writes it
#                  through the GPU, makes the head read-only, and writes the
#                  rest through the GPU while the readers run
#   read    gpu    the readers fault on the pages (STATOR (Map Only))
#           cpu    the readers read every page from the CPU first (STATOR)
#   procs   PROCS_SET
#
# Kill gates, fixed before the campaign:
#   G1  every reader reports every round, and every publisher reads its head
#       back on the CPU without a difference;
#   G2  with the CPU read first, no word differs.
# Reported, not gated: the words, pages and rounds that differ when the GPU
# faults on the pages, and where their values come from.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
repetitions=${REPETITIONS:-4}
size_mib=${SIZE_MIB:-1024}
head_mib=${HEAD_MIB:-896}
rounds=${ROUNDS:-4}
read -r -a procs_set <<<"${PROCS_SET:-1 32}"
read -r -a fs_set <<<"${FS_SET:-memfs ext4}"
read -r -a writer_set <<<"${WRITER_SET:-cpu gpu}"
read -r -a read_set <<<"${READ_SET:-gpu cpu}"
checker=${CHECKER:-"$script_dir/gpu_read_check"}
probe=${PROBE:-"$script_dir/linux_share_probe"}
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-gpu-read-check"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]]; then
  echo "REPETITIONS must be a positive integer" >&2
  exit 2
fi
if [[ ! -x "$checker" || ! -x "$probe" ]]; then
  echo "missing programs; run make gpu_read_check linux_share_probe" >&2
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

memfs_dir="$mount_root/read_check_memfs"
ext4_dir="$mount_root/read_check_ext4"
mounted=0
work=
anchor_pid=
cleanup() {
  if [[ -n "$anchor_pid" ]]; then kill "$anchor_pid" 2>/dev/null || true; fi
  pkill -9 -P $$ 2>/dev/null || true
  if (( mounted )); then sudo -n umount "$memfs_dir" || true; fi
  rm -rf "${ext4_dir:?}"
  rmdir "$memfs_dir" "$mount_root" 2>/dev/null || true
  if [[ -n "$work" ]]; then rm -rf "${work:?}"; fi
}
trap cleanup EXIT
mkdir -p "$memfs_dir" "$ext4_dir" "$result_dir"
sudo -n mount -t tmpfs -o "huge=always,size=$(( size_mib + 512 ))m,uid=$(id -u),gid=$(id -g)" \
  tmpfs "$memfs_dir"
mounted=1
work=$(mktemp -d)

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig=%s\nrepetitions=%s\nsize_mib=%s\nhead_mib=%s\nrounds=%s\n' "$mig" "$repetitions" \
    "$size_mib" "$head_mib" "$rounds"
  printf 'procs_set=%s\nfs_set=%s\nwriter_set=%s\nread_set=%s\n' "${procs_set[*]}" \
    "${fs_set[*]}" "${writer_set[*]}" "${read_set[*]}"
  printf 'ext4_fs=%s\n' "$(stat -f -c %T "$ext4_dir")"
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

run_case() {  # fs writer read procs
  local fs=$1 writer=$2 read=$3 procs=$4 dir file publisher_pid= failed=0 index pids=()
  local before after
  dir=$memfs_dir
  [[ "$fs" == ext4 ]] && dir=$ext4_dir
  file="$dir/data"
  rm -f "$file" "$work/ready" "$work/stop" "$work/go" "$work"/reader.* "$work/publisher"
  printf 'CASE fs=%s writer=%s read=%s procs=%s\n' "$fs" "$writer" "$read" "$procs" >>"$raw_log"
  if [[ "$writer" == cpu ]]; then
    "$checker" make "$file" "$size_mib" >/dev/null
  else
    "$checker" publish "$file" "$size_mib" "$head_mib" "$work/ready" "$work/stop" \
      >"$work/publisher" 2>&1 &
    publisher_pid=$!
    until [[ -e "$work/ready" ]]; do
      kill -0 "$publisher_pid" 2>/dev/null || break
      sleep 0.05
    done
  fi
  for index in $(seq 0 $((procs - 1))); do
    timeout -s KILL 900 "$checker" check "$file" "$head_mib" "$rounds" "$read" "$work/go" \
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
  if [[ -n "$publisher_pid" ]]; then
    touch "$work/stop"
    wait "$publisher_pid" || failed=$((failed + 1))
    grep '^PUBLISH ' "$work/publisher" >>"$raw_log" || printf 'PUBLISH failed=1\n' >>"$raw_log"
  fi
  for index in "${!pids[@]}"; do
    grep -E '^(CHECK|EXAMPLE|ERROR) ' "$work/reader.$index" | sed "s/^\\([A-Z]*\\) /\\1 proc=$index /" \
      >>"$raw_log" || true
  done
  read -r -a b <<<"$before"
  read -r -a a <<<"$after"
  printf 'KERNEL gpu_fault_pages=%s fault_thread_ms=%s\n' "$(( a[0] - b[0] ))" \
    "$(( (a[1] - b[1]) * tick_ms ))" >>"$raw_log"
  printf 'END_CASE failed=%s\n' "$failed" >>"$raw_log"
  rm -f "$file"
}

for run in $(seq 1 "$repetitions"); do
  printf 'BEGIN_READ_CHECK run=%s\n' "$run" >>"$raw_log"
  for fs in "${fs_set[@]}"; do
    for writer in "${writer_set[@]}"; do
      for read in "${read_set[@]}"; do
        for procs in "${procs_set[@]}"; do
          run_case "$fs" "$writer" "$read" "$procs"
        done
      done
    done
  done
  printf 'END_READ_CHECK\n' >>"$raw_log"
done
awk -v rounds="$rounds" -f "$script_dir/summarize_gpu_read_check.awk" "$raw_log" \
  >"$result_dir/gpu_read_check_summary.csv"
(
  cd "$script_dir"
  sha256sum gpu_read_check.cu summarize_gpu_read_check.awk run_gpu_read_check.sh
) >"$result_dir/source_hashes.txt"
printf 'gpu_read_check_result_dir=%s\n' "$result_dir"
