#!/usr/bin/env bash
# The device-memory counterpart of a cache made of a shared prefix and a
# private tail: can a process map memory that another process exports
# read-only next to memory of its own, as one contiguous device range, with
# CUDA's virtual memory interface? Checked for each placement of the two
# processes: one MIG instance, one process in each instance, and two clients
# of one MPS server.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-vmm-routes"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi
if pgrep -f '(^|/)nvidia-cuda-mps-control( |$)' >/dev/null; then
  echo "an MPS daemon is already running; refusing to reuse it" >&2
  exit 2
fi
[[ -x "$script_dir/cuda_vmm_probe" ]] || { echo "missing cuda_vmm_probe; run make" >&2; exit 2; }
if [[ -n "${MPS_ROOT:-}" ]]; then
  mps_root=$MPS_ROOT
  mkdir -p "$mps_root"
else
  mps_root=$(mktemp -d "${TMPDIR:-/tmp}/hm.XXXXXX")
fi
(( ${#mps_root} <= 84 )) || { echo "MPS directory path too long" >&2; exit 2; }
pipe="$mps_root/a"
mkdir -p "$pipe" "$mps_root/la"
running=0
start_mps() {
  env CUDA_VISIBLE_DEVICES="$mig_a" CUDA_MPS_PIPE_DIRECTORY="$pipe" \
    CUDA_MPS_LOG_DIRECTORY="$mps_root/la" nvidia-cuda-mps-control -d
  running=1
}
stop_mps() {
  if (( running )); then
    env CUDA_MPS_PIPE_DIRECTORY="$pipe" CUDA_MPS_LOG_DIRECTORY="$mps_root/la" \
      bash -c 'echo quit | nvidia-cuda-mps-control' >/dev/null 2>&1 || true
    running=0
  fi
}
cleanup() {
  stop_mps
  rm -rf "$mps_root"
}
trap cleanup EXIT INT TERM

mkdir -p "$result_dir"
{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\n' "$mig_a" "$mig_b" "$repetitions"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$script_dir/cuda_vmm_probe"
} >"$result_dir/metadata.txt"

attempt() {  # placement command...
  local placement=$1 status=0
  shift
  printf 'BEGIN_VMM placement=%s\n' "$placement" >>"$raw_log"
  timeout 120 "$@" >>"$raw_log" 2>&1 || status=$?
  printf 'END_VMM placement=%s exit_status=%s\n' "$placement" "$status" >>"$raw_log"
}

for run in $(seq 1 "$repetitions"); do
  printf 'RUN %s\n' "$run" >>"$raw_log"
  attempt same_instance "$script_dir/cuda_vmm_probe" "$mig_a" "$mig_a"
  attempt across_instances "$script_dir/cuda_vmm_probe" "$mig_a" "$mig_b"
  start_mps
  attempt same_mps_server env CUDA_MPS_PIPE_DIRECTORY="$pipe" \
    "$script_dir/cuda_vmm_probe" "$mig_a" "$mig_a"
  stop_mps
done

awk -f "$script_dir/summarize_vmm_routes.awk" "$raw_log" \
  >"$result_dir/vmm_routes_summary.csv"
(
  cd "$script_dir"
  sha256sum run_vmm_routes.sh summarize_vmm_routes.awk cuda_vmm_probe.cu
) >"$result_dir/source_hashes.txt"
printf 'vmm_routes_result_dir=%s\n' "$result_dir"
