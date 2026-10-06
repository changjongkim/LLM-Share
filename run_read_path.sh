#!/usr/bin/env bash
# The read path of the GPU by the kind of memory, in each MIG instance. The
# probe reads one buffer with three kernels (every word; one word per 4 KiB
# page; one word per 2 MiB block) from device memory and from pageable host
# memory with 2 MiB and with 4 KiB pages.
#
# Question, fixed before the campaign: a key-value cache in host memory
# costs 2.2% to 5.8% of the generation speed in the 6-SM instance and
# nothing in the 12-SM instance, with either page size. If the read path
# explains it, the full pass over host memory is slower than over device
# memory in the 6-SM instance and not in the 12-SM instance (H1). If the
# cost is per translation, the page pass or the block pass shows it and the
# two page sizes differ (H2). If neither pass differs between the kinds of
# memory, the cause is not in the read path of a kernel that scans a buffer,
# and the record says so.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
sizes=${SIZES_MIB:-"256 1024"}
passes=${PASSES:-30}
probe="$script_dir/cuda_reach_probe"
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-read-path"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

[[ -x "$probe" ]] || { echo "missing $probe; run make cuda_reach_probe" >&2; exit 2; }
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi
mkdir -p "$result_dir"
{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\nsizes_mib=%s\npasses=%s\n' \
    "$mig_a" "$mig_b" "$repetitions" "$sizes" "$passes"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$probe"
} >"$result_dir/metadata.txt"

forward=(device host_huge host_small)
backward=(host_small host_huge device)
for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then kinds=("${forward[@]}"); else kinds=("${backward[@]}"); fi
  for instance in 12sm 6sm; do
    if [[ "$instance" == 12sm ]]; then mig=$mig_a; else mig=$mig_b; fi
    for size in $sizes; do
      for kind in "${kinds[@]}"; do
        line=$(CUDA_VISIBLE_DEVICES="$mig" "$probe" "$kind" "$size" "$passes" \
          2>"$result_dir/probe.err" || true)
        if [[ "$line" == RESULT* ]]; then
          printf 'PROBE run=%s instance=%s %s\n' "$run" "$instance" "${line#RESULT }"
        else
          printf 'FAILED run=%s instance=%s kind=%s size_mib=%s\n' "$run" \
            "$instance" "$kind" "$size"
        fi >>"$raw_log"
      done
    done
  done
done
rm -f "$result_dir/probe.err"

awk -f "$script_dir/summarize_read_path.awk" "$raw_log" \
  >"$result_dir/read_path_summary.csv"
(
  cd "$script_dir"
  sha256sum cuda_reach_probe.cu summarize_read_path.awk run_read_path.sh
) >"$result_dir/source_hashes.txt"
printf 'read_path_result_dir=%s\n' "$result_dir"
