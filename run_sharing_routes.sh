#!/usr/bin/env bash
# Which route can share GPU-visible memory between two serving processes on
# this device, for each placement of the processes: CUDA IPC (the route of
# device-memory sharing) and the host page table (a mapped object that both
# GPUs read in place).
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
hostmm_dir="$script_dir/../thor_hostmm"
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-5}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-sharing-routes"}
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
make -C "$script_dir" cuda_ipc_probe >/dev/null
for binary in "$script_dir/cuda_ipc_probe" "$hostmm_dir/sharing_modes_probe"; do
  [[ -x "$binary" ]] || { echo "missing $binary" >&2; exit 2; }
done
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
  sha256sum "$script_dir/cuda_ipc_probe" "$hostmm_dir/sharing_modes_probe"
} >"$result_dir/metadata.txt"

attempt() {  # route placement command...
  local route=$1 placement=$2 status=0
  shift 2
  printf 'BEGIN_ROUTE route=%s placement=%s\n' "$route" "$placement" >>"$raw_log"
  timeout 120 "$@" >>"$raw_log" 2>&1 || status=$?
  printf 'END_ROUTE route=%s placement=%s exit_status=%s\n' "$route" \
    "$placement" "$status" >>"$raw_log"
}

for run in $(seq 1 "$repetitions"); do
  printf 'RUN %s\n' "$run" >>"$raw_log"
  # Two processes in one MIG instance, and one in each instance.
  attempt cuda_ipc same_instance "$script_dir/cuda_ipc_probe" "$mig_a" "$mig_a"
  attempt cuda_ipc across_instances "$script_dir/cuda_ipc_probe" "$mig_a" "$mig_b"
  attempt host_page_table same_instance "$hostmm_dir/sharing_modes_probe" \
    perf routes 1024 1 ro_preread "$mig_a" "$mig_a"
  attempt host_page_table across_instances "$hostmm_dir/sharing_modes_probe" \
    perf routes 1024 1 ro_preread "$mig_a" "$mig_b"
  # Two clients of one MPS server.
  start_mps
  attempt cuda_ipc same_mps_server env CUDA_MPS_PIPE_DIRECTORY="$pipe" \
    "$script_dir/cuda_ipc_probe" "$mig_a" "$mig_a"
  attempt host_page_table same_mps_server "$hostmm_dir/sharing_modes_probe" \
    perf routes 1024 1 ro_preread "$mig_a@$pipe" "$mig_a@$pipe"
  stop_mps
done

awk -f "$script_dir/summarize_sharing_routes.awk" "$raw_log" \
  >"$result_dir/sharing_routes_summary.csv"
(
  cd "$script_dir"
  sha256sum run_sharing_routes.sh summarize_sharing_routes.awk cuda_ipc_probe.cu \
    ../thor_hostmm/sharing_modes_probe.cu ../thor_hostmm/shared_model_bench.cu
) >"$result_dir/source_hashes.txt"
printf 'sharing_routes_result_dir=%s\n' "$result_dir"
