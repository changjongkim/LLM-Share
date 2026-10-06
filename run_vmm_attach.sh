#!/usr/bin/env bash
# What it costs to hand a prefix-sized range of device memory to another
# process through the CUDA virtual memory management interface, by the size
# of the allocations that the range is made of. The device-memory baseline
# of the engine (kv_vmm.patch) uses 2 MiB allocations, the smallest that the
# interface accepts, so that it shares all of a prefix but its last 2 MiB
# per tensor. Larger allocations need fewer handles and share less.
#
# TOTAL_MIB is the size of the 16,321-token prefix of the other campaigns
# (896 MiB). Both processes run in the 12-SM MIG instance.
#
# Question, fixed before the campaign: does any allocation size bring the
# attach of device memory to the attach of host extents (48 ms for this
# prefix, context creation included), and what is the cost per handle?
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
repetitions=${REPETITIONS:-6}
total=${TOTAL_MIB:-896}
granules=${GRANULES_MIB:-"2 4 8 16 32 64 128 448 896"}
probe="$script_dir/cuda_vmm_attach_probe"
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-vmm-attach"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

[[ -x "$probe" ]] || { echo "missing $probe; run make cuda_vmm_attach_probe" >&2; exit 2; }
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi
mkdir -p "$result_dir"
{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig=%s\nrepetitions=%s\ntotal_mib=%s\ngranules_mib=%s\n' "$mig" \
    "$repetitions" "$total" "$granules"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$probe"
} >"$result_dir/metadata.txt"

for run in $(seq 1 "$repetitions"); do
  order=$granules
  if (( run % 2 == 0 )); then order=$(tr ' ' '\n' <<<"$granules" | tac | tr '\n' ' '); fi
  for granule in $order; do
    line=$(CUDA_VISIBLE_DEVICES="$mig" "$probe" "$total" "$granule" \
      2>"$result_dir/probe.err" || true)
    if [[ "$line" == RESULT* ]]; then
      printf 'PROBE run=%s %s\n' "$run" "${line#RESULT }"
    else
      printf 'FAILED run=%s granule_mib=%s %s\n' "$run" "$granule" \
        "$(tail -n 1 "$result_dir/probe.err" | tr ' ' '_')"
    fi >>"$raw_log"
  done
done
rm -f "$result_dir/probe.err"

awk -f "$script_dir/summarize_vmm_attach.awk" "$raw_log" \
  >"$result_dir/vmm_attach_summary.csv"
(
  cd "$script_dir"
  sha256sum cuda_vmm_attach_probe.cu summarize_vmm_attach.awk run_vmm_attach.sh
) >"$result_dir/source_hashes.txt"
printf 'vmm_attach_result_dir=%s\n' "$result_dir"
