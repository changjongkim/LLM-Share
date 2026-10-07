#!/usr/bin/env bash
# Reads and writes that jump across a buffer, by the kind of memory, in each
# MIG instance. The probe (cuda_tlb_probe) reads consecutive words, reads
# words at pseudo-random strides and writes words at the same strides, in
# device memory and in pageable host memory with 2 MiB and with 4 KiB pages.
#
# Question, fixed before the campaign: a key-value cache in host memory
# costs up to 5.8% of the generation speed in the 6-SM instance and nothing
# in the 12-SM instance, and a kernel that scans a buffer reads host memory
# as fast as device memory in both. If the address translation of scattered
# accesses explains the loss, the gather or the scatter on host memory is
# slower than on device memory in the 6-SM instance by more than in the
# 12-SM instance (H1), and the two page sizes differ (H2). If the rates do
# not differ between the kinds of memory, scattered accesses do not explain
# it either, and the record says so.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
sizes=${SIZES_MIB:-"1024"}
strides=${STRIDES_KIB:-"4 64"}
probe="$script_dir/cuda_tlb_probe"
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-tlb-probe"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

[[ -x "$probe" ]] || { echo "missing $probe; run make cuda_tlb_probe" >&2; exit 2; }
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi
mkdir -p "$result_dir"
{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\nsizes_mib=%s\nstrides_kib=%s\n' \
    "$mig_a" "$mig_b" "$repetitions" "$sizes" "$strides"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$probe"
} >"$result_dir/metadata.txt"

for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then
    migs=("$mig_a" "$mig_b") kinds=(device host_huge host_small)
  else
    migs=("$mig_b" "$mig_a") kinds=(host_small host_huge device)
  fi
  for mig in "${migs[@]}"; do
    for size in $sizes; do
      for stride in $strides; do
        for kind in "${kinds[@]}"; do
          line=$(CUDA_VISIBLE_DEVICES="$mig" "$probe" "$kind" "$size" "$stride" 2>&1 || true)
          printf 'PROBE run=%s mig=%s %s\n' "$run" "${mig:4:8}" "${line#RESULT }" >>"$raw_log"
        done
      done
    done
  done
done

awk -f "$script_dir/summarize_tlb_probe.awk" "$raw_log" >"$result_dir/tlb_probe_summary.csv"
(
  cd "$script_dir"
  sha256sum cuda_tlb_probe.cu summarize_tlb_probe.awk run_tlb_probe.sh
) >"$result_dir/source_hashes.txt"
printf 'tlb_probe_result_dir=%s\n' "$result_dir"
