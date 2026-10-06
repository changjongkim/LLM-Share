#!/usr/bin/env bash
# N serving processes of one model at the same time, with weights copied to
# device memory or read in place, under each way of sharing the GPU's
# compute: tenants in different MIG instances, time-sliced in one instance,
# clients of one MPS server, and one MPS server per MIG instance. Measures
# tokens per second per process, the sum, and the memory the processes hold.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
agent_counts=${AGENT_COUNTS:-"1 2 4 8"}
n_prompt=${N_PROMPT:-512}
n_gen=${N_GEN:-128}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
bin_dir=${BIN_DIR:-"$script_dir/llama.cpp/build/bin"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-agents"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]]; then
  echo "REPETITIONS must be a positive integer" >&2
  exit 2
fi
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi
if pgrep -f '(^|/)nvidia-cuda-mps-control( |$)' >/dev/null ||
   pgrep -f '(^|/)nvidia-cuda-mps-server( |$)' >/dev/null; then
  echo "an MPS daemon is already running; refusing to reuse it" >&2
  exit 2
fi
for file in "$model" "$bin_dir/llama-bench"; do
  [[ -e "$file" ]] || { echo "missing $file" >&2; exit 2; }
done
# Eight processes with device copies are the largest case.
model_kb=$(( $(stat -c %s "$model") / 1024 ))
needed_kb=$(( 10 * model_kb + 24 * 1024 * 1024 ))
available_kb=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
(( available_kb > needed_kb )) || {
  echo "not enough available memory for eight device copies" >&2
  exit 2
}

# The MPS control socket needs a short path; MPS_ROOT names the directory.
if [[ -n "${MPS_ROOT:-}" ]]; then
  mps_root=$MPS_ROOT
  mkdir -p "$mps_root"
  [[ -z "$(ls -A "$mps_root")" ]] || { echo "MPS_ROOT is not empty" >&2; exit 2; }
else
  mps_root=$(mktemp -d "${TMPDIR:-/tmp}/hm.XXXXXX")
fi
(( ${#mps_root} <= 84 )) || { echo "MPS directory path too long" >&2; exit 2; }
pipe_a="$mps_root/a"
pipe_b="$mps_root/b"
mkdir -p "$pipe_a" "$pipe_b" "$mps_root/la" "$mps_root/lb"
running_a=0
running_b=0
start_mps() {
  if [[ "$1" == a && "$running_a" -eq 0 ]]; then
    env CUDA_VISIBLE_DEVICES="$mig_a" CUDA_MPS_PIPE_DIRECTORY="$pipe_a" \
      CUDA_MPS_LOG_DIRECTORY="$mps_root/la" nvidia-cuda-mps-control -d
    running_a=1
  elif [[ "$1" == b && "$running_b" -eq 0 ]]; then
    env CUDA_VISIBLE_DEVICES="$mig_b" CUDA_MPS_PIPE_DIRECTORY="$pipe_b" \
      CUDA_MPS_LOG_DIRECTORY="$mps_root/lb" nvidia-cuda-mps-control -d
    running_b=1
  fi
}
stop_mps() {
  if [[ "$1" == a && "$running_a" -eq 1 ]]; then
    env CUDA_MPS_PIPE_DIRECTORY="$pipe_a" CUDA_MPS_LOG_DIRECTORY="$mps_root/la" \
      bash -c 'echo quit | nvidia-cuda-mps-control' >/dev/null 2>&1 || true
    running_a=0
  elif [[ "$1" == b && "$running_b" -eq 1 ]]; then
    env CUDA_MPS_PIPE_DIRECTORY="$pipe_b" CUDA_MPS_LOG_DIRECTORY="$mps_root/lb" \
      bash -c 'echo quit | nvidia-cuda-mps-control' >/dev/null 2>&1 || true
    running_b=0
  fi
}
cleanup() {
  stop_mps a
  stop_mps b
  rm -rf "$mps_root"
}
trap cleanup EXIT INT TERM
prepare() {
  case $1 in
    mig | timeslice) stop_mps a; stop_mps b ;;
    mps) start_mps a; stop_mps b ;;
    mig_mps) start_mps a; start_mps b ;;
  esac
}
# Prints "MIG PIPE_DIR" for agent `index` of a configuration; PIPE_DIR is "-"
# for a process that is not an MPS client.
placement() {
  local config=$1 index=$2
  case $config in
    mig) if (( index % 2 == 0 )); then echo "$mig_a -"; else echo "$mig_b -"; fi ;;
    timeslice) echo "$mig_a -" ;;
    mps) echo "$mig_a $pipe_a" ;;
    mig_mps)
      if (( index % 2 == 0 )); then echo "$mig_a $pipe_a"; else echo "$mig_b $pipe_b"; fi ;;
  esac
}

mkdir -p "$result_dir"
work=$(mktemp -d "$result_dir/work.XXXXXX")
{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\nagent_counts=%s\n' \
    "$mig_a" "$mig_b" "$repetitions" "$agent_counts"
  printf 'n_prompt=%s\nn_gen=%s\n' "$n_prompt" "$n_gen"
  printf 'model=%s\nmodel_bytes=%s\n' "$(basename "$model")" "$(stat -c %s "$model")"
  printf 'engine_commit=%s\n' "$(git -C "$script_dir/llama.cpp" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$bin_dir/llama-bench"
} >"$result_dir/metadata.txt"

run_case() {  # run config agents mode
  local run=$1 config=$2 agents=$3 mode=$4 index mig pipe pids=() failed=0
  local extra=()
  if [[ "$mode" == inplace ]]; then extra=(GGML_CUDA_HOST_PTR=1); fi
  local before lowest sample mapped=0
  before=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
  lowest=$before
  for index in $(seq 0 $((agents - 1))); do
    read -r mig pipe <<<"$(placement "$config" "$index")"
    local mps_env=()
    if [[ "$pipe" != - ]]; then mps_env=(CUDA_MPS_PIPE_DIRECTORY="$pipe"); fi
    env -u CUDA_MPS_PIPE_DIRECTORY CUDA_VISIBLE_DEVICES="$mig" "${mps_env[@]}" \
      "${extra[@]}" "$bin_dir/llama-bench" -m "$model" -ngl 99 -p "$n_prompt" \
      -n "$n_gen" -r 1 -o jsonl >"$work/agent.$index" 2>/dev/null &
    pids+=($!)
  done
  local smaps=()
  for index in "${pids[@]}"; do smaps+=("/proc/$index/smaps"); done
  # Lowest available memory while the agents run, and the model pages that
  # the agents map, sampled twice a second.
  while kill -0 "${pids[@]}" 2>/dev/null; do
    sample=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
    (( sample < lowest )) && lowest=$sample
    # An agent may leave between the liveness test and the read; cat then
    # skips its file and the sample covers the agents that remain.
    # shellcheck disable=SC2002
    sample=$(cat "${smaps[@]}" 2>/dev/null |
      awk -v file="$model" '
        /^[0-9a-f]+-[0-9a-f]+ / { inside = ($6 == file) }
        inside && /^Pss:/ { pss += $2 }
        END { printf "%d", pss / 1024 }' || true)
    (( ${sample:-0} > mapped )) && mapped=$sample
    sleep 0.5
  done
  for index in "${!pids[@]}"; do
    wait "${pids[$index]}" || failed=$((failed + 1))
  done
  {
    printf 'BEGIN_ENGINE_AGENTS run=%s config=%s agents=%s mode=%s\n' \
      "$run" "$config" "$agents" "$mode"
    for index in $(seq 0 $((agents - 1))); do
      sed "s/^/AGENT $index /" "$work/agent.$index"
    done
    printf 'MEMORY mem_available_drop_mib=%s model_mapped_mib=%s\n' \
      "$(( (before - lowest) / 1024 ))" "$mapped"
    printf 'END_ENGINE_AGENTS failed_agents=%s\n' "$failed"
  } >>"$raw_log"
}

for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then
    configs=(mig timeslice mps mig_mps)
    modes=(copy inplace)
  else
    configs=(mig_mps mps timeslice mig)
    modes=(inplace copy)
  fi
  for config in "${configs[@]}"; do
    prepare "$config"
    for agents in $agent_counts; do
      for mode in "${modes[@]}"; do
        run_case "$run" "$config" "$agents" "$mode"
      done
    done
  done
done
stop_mps a
stop_mps b
rm -rf "$work"

awk -f "$script_dir/summarize_engine_agents.awk" "$raw_log" \
  >"$result_dir/engine_agents_summary.csv"
(
  cd "$script_dir"
  sha256sum inplace_weights.patch summarize_engine_agents.awk run_engine_agents.sh
) >"$result_dir/source_hashes.txt"
printf 'engine_agents_result_dir=%s\n' "$result_dir"
