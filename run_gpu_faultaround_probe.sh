#!/usr/bin/env bash
# B1 of the review of 2026-10-21 (gpu_populate_probe.cu, unchanged): why a
# populate request of the kernel leaves GPU faults on the 4 KiB pages of the
# page cache (K2: 0.47 of 0.92 million). The hypothesis is fault-around: on
# arm64 without a hardware accessed flag, the kernel maps the neighbours of a
# faulting page "old" (arch_wants_old_prefaulted_pte), and the GPU faults on an
# old entry. With /sys/kernel/debug/fault_around_bytes at 4096 the kernel maps
# only the faulting page, so a populate request should leave no fault.
#
# Each repetition runs the ext4 (4 KiB page cache) cases none, touch, populate
# once with the default fault-around size and once with 4096, the order of the
# two sizes and of the steps reversed in every second repetition. The default
# size is restored on every exit.
#
# Kill gates, fixed before the campaign:
#   Q1  every case exits and the GPU reads the sum of the file (sum_ok=1);
#   Q2  with `touch` (STATOR), the GPU takes at most 1% of the faults of `none`
#       at both sizes.
# Reported, not gated: the faults of `populate` at both sizes.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
gpu=${GPU_UUID:-GPU-a7c66ad2-6dbb-0ab8-c1a2-37ba6dba3600}
repetitions=${REPETITIONS:-6}
size_mib=${SIZE_MIB:-896}
probe=${PROBE:-"$script_dir/gpu_populate_probe"}
anchor_probe=${ANCHOR:-"$script_dir/linux_share_probe"}
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-gpu-faultaround-probe"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"
fab_file=/sys/kernel/debug/fault_around_bytes
[[ -x "$probe" && -x "$anchor_probe" ]] || { echo "missing programs; run make gpu_populate_probe linux_share_probe" >&2; exit 2; }
sudo -n true 2>/dev/null || { echo "passwordless sudo is required" >&2; exit 2; }
[[ -e "$result_dir" ]] && { echo "result directory already exists: $result_dir" >&2; exit 2; }
default_fab=$(sudo -n cat "$fab_file")

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
ext4_dir="$mount_root/faultaround_ext4"
anchor_pid=
cleanup() {
  echo "$default_fab" | sudo -n tee "$fab_file" >/dev/null || true
  if [[ -n "$anchor_pid" ]]; then kill "$anchor_pid" 2>/dev/null || true; fi
  pkill -9 -P $$ 2>/dev/null || true
  rm -rf "${ext4_dir:?}"
  rmdir "$mount_root" 2>/dev/null || true
}
trap cleanup EXIT
mkdir -p "$ext4_dir" "$result_dir"
head -c $(( size_mib * 1024 * 1024 )) /dev/urandom >"$ext4_dir/data"
{
  printf 'timestamp=%s\ngpu=%s\nrepetitions=%s\nsize_mib=%s\ndefault_fault_around_bytes=%s\n' \
    "$(date --iso-8601=seconds)" "$gpu" "$repetitions" "$size_mib" "$default_fab"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$probe"
} >"$result_dir/metadata.txt"

export CUDA_VISIBLE_DEVICES="$gpu"
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
  sizes=("$default_fab" 4096)
  steps=(none touch populate)
  if (( run % 2 == 0 )); then sizes=(4096 "$default_fab"); steps=(populate touch none); fi
  printf 'BEGIN_FAULTAROUND run=%s\n' "$run" >>"$raw_log"
  for fab in "${sizes[@]}"; do
    echo "$fab" | sudo -n tee "$fab_file" >/dev/null
    for step in "${steps[@]}"; do
      cat "$ext4_dir/data" >/dev/null  # the file is in the page cache, as in K2
      run_case "mode=ro fs=ext4 fab=$fab step=$step" ro "$ext4_dir/data" "$size_mib" "$step"
    done
  done
  printf 'END_FAULTAROUND\n' >>"$raw_log"
done
echo "$default_fab" | sudo -n tee "$fab_file" >/dev/null
python3 "$script_dir/summarize_gpu_faultaround_probe.py" "$raw_log" >"$result_dir/gpu_faultaround_probe_summary.csv"
(
  cd "$script_dir"
  sha256sum gpu_populate_probe.cu run_gpu_faultaround_probe.sh summarize_gpu_faultaround_probe.py
) >"$result_dir/source_hashes.txt"
printf 'gpu_faultaround_probe_result_dir=%s\n' "$result_dir"
