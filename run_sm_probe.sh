#!/usr/bin/env bash
# How many streaming multiprocessors each MIG instance gives to a kernel that
# only computes. nvidia-smi lists the one-slice profile with 6 SMs, and the
# CUDA runtime reports 8 for the instance that is created from it; the probe
# decides between the two by execution.
#
# cuda_sm_probe starts kernels of N blocks of 1,024 threads for N = 1..32. An
# SM holds 1,536 threads, so one such block occupies an SM, and the time of
# the kernel steps up at the first N that exceeds the SMs of the instance.
# It also reports the rate of a kernel of 8,192 blocks of 256 threads.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
nvcc=${NVCC:-/usr/local/cuda/bin/nvcc}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-sm-probe"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"
mkdir -p "$result_dir"
probe="$result_dir/cuda_sm_probe"
"$nvcc" -O2 -std=c++17 -o "$probe" "$script_dir/cuda_sm_probe.cu"

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\n' "$mig_a" "$mig_b" "$repetitions"
  nvidia-smi -L
  sudo -n nvidia-smi mig -lgip 2>/dev/null || true
  sudo -n nvidia-smi mig -lgi 2>/dev/null || true
} >"$result_dir/metadata.txt"

: >"$raw_log"
for run in $(seq 1 "$repetitions"); do
  for mig in "$mig_a" "$mig_b"; do
    printf 'BEGIN_SM_PROBE run=%s mig=%s\n' "$run" "${mig:4:8}" >>"$raw_log"
    CUDA_VISIBLE_DEVICES="$mig" "$probe" >>"$raw_log"
    printf 'END_SM_PROBE\n' >>"$raw_log"
  done
done
rm -f "$probe"

awk -f "$script_dir/summarize_sm_probe.awk" "$raw_log" >"$result_dir/sm_probe_summary.csv"
(
  cd "$script_dir"
  sha256sum cuda_sm_probe.cu summarize_sm_probe.awk run_sm_probe.sh
) >"$result_dir/source_hashes.txt"
printf 'sm_probe_result_dir=%s\n' "$result_dir"
