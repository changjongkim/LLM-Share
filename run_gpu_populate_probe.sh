#!/usr/bin/env bash
# K2 of PLAN-20261015 (gpu_populate_probe.cu): calls of the kernel against the
# CPU pass of STATOR, for a read-only private mapping of a file (on a tmpfs with
# 2 MiB pages, `memfs`, and in the ext4 page cache, `ext4`) and for private
# anonymous memory that the GPU writes. Every case is one process; the GPU
# faults are read from the driver threads around it (an anchor process keeps
# them). REPETITIONS repetitions, the order of the steps reversed in every
# second one.
#
# Kill gates, fixed before the campaign:
#   Q1  every case exits and the GPU reads the sum of the file (sum_ok=1) or
#       writes every page;
#   Q2  with `touch` (STATOR), the GPU takes at most 1% of the faults of `none`.
# Reported, not gated: the faults, the time to prepare and the time of the GPU
# pass for `populate` and `willneed`.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig=${PRODUCER_MIG:-GPU-a7c66ad2-6dbb-0ab8-c1a2-37ba6dba3600}
repetitions=${REPETITIONS:-6}
size_mib=${SIZE_MIB:-896}
probe=${PROBE:-"$script_dir/gpu_populate_probe"}
anchor_probe=${ANCHOR:-"$script_dir/linux_share_probe"}
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-gpu-populate-probe"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"
[[ -x "$probe" && -x "$anchor_probe" ]] || { echo "missing programs; run make gpu_populate_probe linux_share_probe" >&2; exit 2; }
sudo -n true 2>/dev/null || { echo "passwordless sudo is required" >&2; exit 2; }
[[ -e "$result_dir" ]] && { echo "result directory already exists: $result_dir" >&2; exit 2; }

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
memfs_dir="$mount_root/populate_memfs"
ext4_dir="$mount_root/populate_ext4"
mounted=0
anchor_pid=
cleanup() {
  if [[ -n "$anchor_pid" ]]; then kill "$anchor_pid" 2>/dev/null || true; fi
  pkill -9 -P $$ 2>/dev/null || true
  if (( mounted )); then sudo -n umount "$memfs_dir" || true; fi
  rm -rf "${ext4_dir:?}"
  rmdir "$memfs_dir" "$mount_root" 2>/dev/null || true
}
trap cleanup EXIT
mkdir -p "$memfs_dir" "$ext4_dir" "$result_dir"
sudo -n mount -t tmpfs -o "huge=always,size=$(( size_mib + 512 ))m,uid=$(id -u),gid=$(id -g)" tmpfs "$memfs_dir"
mounted=1
head -c $(( size_mib * 1024 * 1024 )) /dev/urandom >"$memfs_dir/data"
cp "$memfs_dir/data" "$ext4_dir/data"
{
  printf 'timestamp=%s\nmig=%s\nrepetitions=%s\nsize_mib=%s\n' "$(date --iso-8601=seconds)" "$mig" "$repetitions" "$size_mib"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$probe"
} >"$result_dir/metadata.txt"

export CUDA_VISIBLE_DEVICES="$mig"
"$anchor_probe" hold 1000000 >/dev/null 2>&1 &
anchor_pid=$!
sleep 3
run_case() {  # label args...
  local label=$1 before after out
  shift
  before=$(fault_counters)
  out=$(timeout -s KILL 300 "$probe" "$@" 2>&1 || echo "PREPARE failed=1")
  after=$(fault_counters)
  read -r -a b <<<"$before"
  read -r -a a <<<"$after"
  printf 'CASE %s\n%s\nKERNEL gpu_fault_pages=%s fault_thread_ms=%s\n' "$label" \
    "$(grep -E '^(PREPARE|ERROR)' <<<"$out" || echo 'PREPARE failed=1')" \
    "$(( a[0] - b[0] ))" "$(( (a[1] - b[1]) * tick_ms ))" >>"$raw_log"
}
for run in $(seq 1 "$repetitions"); do
  ro_steps=(none touch populate willneed)
  rw_steps=(none touch populate)
  if (( run % 2 == 0 )); then ro_steps=(willneed populate touch none); rw_steps=(populate touch none); fi
  printf 'BEGIN_POPULATE run=%s\n' "$run" >>"$raw_log"
  for fs in memfs ext4; do
    dir=$memfs_dir
    [[ "$fs" == ext4 ]] && dir=$ext4_dir
    for step in "${ro_steps[@]}"; do
      run_case "mode=ro fs=$fs step=$step" ro "$dir/data" "$size_mib" "$step"
    done
  done
  for step in "${rw_steps[@]}"; do
    run_case "mode=rw fs=anon step=$step" rw "$size_mib" "$step"
  done
  printf 'END_POPULATE\n' >>"$raw_log"
done
awk -f "$script_dir/summarize_gpu_populate_probe.awk" "$raw_log" >"$result_dir/gpu_populate_probe_summary.csv"
(
  cd "$script_dir"
  sha256sum gpu_populate_probe.cu summarize_gpu_populate_probe.awk run_gpu_populate_probe.sh
) >"$result_dir/source_hashes.txt"
printf 'gpu_populate_probe_result_dir=%s\n' "$result_dir"
