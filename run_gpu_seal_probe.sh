#!/usr/bin/env bash
# K1 of PLAN-20261015 (gpu_seal_probe.cu): what memfd seals allow next to the
# read-only mappings of STATOR, and a GPU read of a sealed buffer. REPETITIONS
# runs of the probe on SIZE_MIB.
#
# Kill gates, fixed before the campaign:
#   L1  every run reports every step, and the GPU reads the sum of the sealed
#       buffer (sum_ok=1).
# Reported, not gated: which steps the kernel allows.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig=${PRODUCER_MIG:-GPU-a7c66ad2-6dbb-0ab8-c1a2-37ba6dba3600}
repetitions=${REPETITIONS:-3}
size_mib=${SIZE_MIB:-896}
probe=${PROBE:-"$script_dir/gpu_seal_probe"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-gpu-seal-probe"}
result_dir="$result_root/$result_tag"
[[ -x "$probe" ]] || { echo "missing probe; run make gpu_seal_probe" >&2; exit 2; }
[[ -e "$result_dir" ]] && { echo "result directory already exists: $result_dir" >&2; exit 2; }
mkdir -p "$result_dir"
{
  printf 'timestamp=%s\nmig=%s\nrepetitions=%s\nsize_mib=%s\n' "$(date --iso-8601=seconds)" "$mig" "$repetitions" "$size_mib"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  uname -r
  sha256sum "$probe"
} >"$result_dir/metadata.txt"
for run in $(seq 1 "$repetitions"); do
  printf 'BEGIN_SEAL run=%s\n' "$run" >>"$result_dir/raw.log"
  CUDA_VISIBLE_DEVICES="$mig" timeout -s KILL 300 "$probe" "$size_mib" >>"$result_dir/raw.log" 2>&1 ||
    printf 'SEAL step=run result=failed errno=0\n' >>"$result_dir/raw.log"
  printf 'END_SEAL\n' >>"$result_dir/raw.log"
done
awk '/^SEAL / { split($2, s, "="); split($3, r, "="); key = s[2]; if (!(key in n)) order[++m] = key; n[key]++; res[key, r[2]]++
                if ($0 ~ /sum_ok=1/) ok++ }
     END { print "step,runs,allowed,refused,other"; for (i = 1; i <= m; ++i) { k = order[i];
             printf "%s,%d,%d,%d,%d\n", k, n[k], res[k, "allowed"] + 0, res[k, "refused"] + 0, n[k] - res[k, "allowed"] - res[k, "refused"] }
           printf "gpu_sum_ok,%d,%d,0,0\n", ok + 0, ok + 0 }' "$result_dir/raw.log" >"$result_dir/gpu_seal_probe_summary.csv"
( cd "$script_dir" && sha256sum gpu_seal_probe.cu run_gpu_seal_probe.sh ) >"$result_dir/source_hashes.txt"
printf 'gpu_seal_probe_result_dir=%s\n' "$result_dir"
