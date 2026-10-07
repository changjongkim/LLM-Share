#!/usr/bin/env bash
# An existing way to share weights between serving processes as a baseline:
# cuda-llm-weight-share (github.com/pontostroy/cuda-llm-weight-share), a
# preloaded library that exports the allocation of the weights of the first
# process through CUDA IPC and maps it in the processes that follow. It runs
# with the unmodified engine. A master computes the prefix, saves the
# engine's state file and remains alive; N agents restore the prefix from the
# file and generate. Three stacks, which differ only in the weights:
#
#   stock    the unmodified engine: every process copies the weights
#   ipc      the unmodified engine with the library preloaded: the agents map
#            the weights of the master in device memory
#   inplace  this repository: every process reads the weights in place
#
# and two placements of the agents:
#
#   same    every agent in the MIG instance of the master (12 SMs)
#   cross   every second agent in the other MIG instance (6 SMs)
#
# The size of the allocation that the library shares is read from one run of
# the master in the library's reconnaissance mode. Memory is the largest
# drop of MemAvailable from before the master starts; the model file is in
# the page cache in every stack and is counted in `inplace` as in use.
#
# Validity criteria, fixed before the campaign:
#   W1  no process fails in stock and inplace;
#   W2  every agent that completes writes the text of the agent of stock in
#       the same placement and repetition;
#   W3  the roles that the library reports (master, worker, fallback) and
#       the agents that complete are counted in both placements; whether the
#       library shares across MIG instances is a result, not a gate.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
agents=${AGENTS:-8}
placements=${PLACEMENTS:-"same cross"}
prefix_file=${PREFIX_FILE:-"$script_dir/workloads/agent_prefix.txt"}
tasks_file=${TASKS_FILE:-"$script_dir/workloads/agent_tasks.txt"}
context=${CONTEXT:-16384}
n_gen=${N_GEN:-64}
limit_s=${CASE_LIMIT_S:-600}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
engine=${KV_ENGINE:-"$script_dir/llama.cpp-kv"}
stock_engine=${STOCK_ENGINE:-"$script_dir/llama.cpp-stock"}
driver=${DRIVER:-"$script_dir/kv_fork"}
stock_driver=${STOCK_DRIVER:-"$script_dir/kv_fork_stock"}
library=${WEIGHT_SHARE:-"$script_dir/ext/cuda-llm-weight-share/cuda-llm-weight-share.so"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-ext-weightshare"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"
shm_name="/stator_ws_$$"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
if [[ ! -x "$driver" || ! -x "$stock_driver" || ! -r "$library" || ! -r "$model" ]]; then
  echo "missing driver, library or model; see README.md" >&2
  exit 2
fi
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi

work=
live_pids=()
cleanup() {
  if (( ${#live_pids[@]} )); then kill -9 "${live_pids[@]}" 2>/dev/null || true; fi
  rm -f "/dev/shm${shm_name}"
  if [[ -n "$work" ]]; then rm -rf "$work"; fi
}
trap cleanup EXIT
mkdir -p "$result_dir"
work=$(mktemp -d "$result_dir/work.XXXXXX")
task_lines=$(wc -l <"$tasks_file")
suffix_of() {  # agent index
  printf '\n\nTask for agent %s: %s\n' "$1" \
    "$(sed -n "$(( $1 % task_lines + 1 ))p" "$tasks_file")"
}
text_of() { { grep '^TEXT ' "$1" || true; } | sha256sum | cut -c1-16; }

# The allocation of the weights: the largest one that the library logs.
env CUDA_VISIBLE_DEVICES="$mig_a" LD_PRELOAD="$library" \
  "$stock_driver" parent "$model" "$context" "$prefix_file" "$work/state" "" 0 \
  >"$work/recon.out" 2>"$work/recon.err" || true
model_size=$(sed -n 's/.*cudaMalloc normal request: .*(\([0-9]*\) bytes).*/\1/p' "$work/recon.err" |
  sort -n | tail -n 1)
if ! [[ "${model_size:-}" =~ ^[1-9][0-9]*$ ]]; then
  echo "the library logged no allocation; see $work/recon.err" >&2
  trap - EXIT
  exit 3
fi

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\nagents=%s\nplacements=%s\n' \
    "$mig_a" "$mig_b" "$repetitions" "$agents" "$placements"
  printf 'context=%s\nn_gen=%s\nmodel=%s\nshared_allocation_bytes=%s\n' "$context" \
    "$n_gen" "$(basename "$model")" "$model_size"
  printf 'library_commit=%s\n' "$(git -C "$(dirname "$library")" rev-parse HEAD)"
  printf 'engine_commit=%s\nstock_engine_commit=%s\n' \
    "$(git -C "$engine" rev-parse HEAD)" "$(git -C "$stock_engine" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$driver" "$stock_driver" "$library"
} >"$result_dir/metadata.txt"

run_case() {  # placement stack
  local placement=$1 stack=$2 index mig program=$stock_driver envs=() pids=() migs=()
  case $stack in
    stock) ;;
    ipc) envs=(LD_PRELOAD="$library" MODEL_SIZE="$model_size" CUDA_VRAM_IPC_NAME="$shm_name") ;;
    inplace) program=$driver envs=(GGML_CUDA_HOST_PTR=1) ;;
    *) echo "unknown stack $stack" >&2; exit 2 ;;
  esac
  rm -f "$work"/state "$work/go" "$work"/agent.* "$work"/master* "/dev/shm${shm_name}"
  local before lowest sample started
  before=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
  lowest=$before
  started=$(date +%s)
  env CUDA_VISIBLE_DEVICES="$mig_a" "${envs[@]}" \
    "$program" parent "$model" "$context" "$prefix_file" "$work/state" \
    $'\n\nTask for the master: name the tools above.\n' "$n_gen" "$work/go" \
    >"$work/master" 2>"$work/master.err" &
  local master_pid=$!
  live_pids=("$master_pid")
  until grep -q '^PUBLISHED ' "$work/master" 2>/dev/null; do
    kill -0 "$master_pid" 2>/dev/null || break
    sleep 0.05
  done
  for index in $(seq 0 $((agents - 1))); do
    if [[ "$placement" == cross ]] && (( index % 2 == 1 )); then mig=$mig_b; else mig=$mig_a; fi
    migs+=("$mig")
    env CUDA_VISIBLE_DEVICES="$mig" "${envs[@]}" \
      "$program" child "$model" "$context" "$work/state" "$(suffix_of "$index")" \
      "$n_gen" >"$work/agent.$index" 2>"$work/agent.$index.err" &
    pids+=($!)
  done
  live_pids+=("${pids[@]}")
  local alive=1 timed_out=0
  while (( alive > 0 )); do
    alive=0
    for index in "${!pids[@]}"; do
      if kill -0 "${pids[$index]}" 2>/dev/null; then alive=$((alive + 1)); fi
    done
    sample=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
    (( sample < lowest )) && lowest=$sample
    if (( $(date +%s) - started > limit_s )); then
      timed_out=1
      kill -9 "${pids[@]}" 2>/dev/null || true
    fi
    sleep 0.05
  done
  touch "$work/go"
  local master_code=0
  wait "$master_pid" || master_code=$?
  live_pids=()
  {
    printf 'CASE placement=%s stack=%s agents=%s\n' "$placement" "$stack" "$agents"
    printf 'MASTER exit=%s %s\n' "$master_code" \
      "$(grep '^PUBLISHED ' "$work/master" | sed 's/^PUBLISHED //' || true)"
    local workers=0 masters=0 fallbacks=0
    for index in "${!pids[@]}"; do
      local code=0 result role=none
      wait "${pids[$index]}" || code=$?
      result=$(grep '^RESULT ' "$work/agent.$index" || true)
      if grep -q 'WORKER cudaIpcOpenMemHandle done' "$work/agent.$index.err"; then
        role=worker; workers=$((workers + 1))
      elif grep -q 'Assuming MASTER role' "$work/agent.$index.err"; then
        role=master; masters=$((masters + 1))
      elif grep -q 'fallback cudaMalloc' "$work/agent.$index.err"; then
        role=fallback; fallbacks=$((fallbacks + 1))
      fi
      if (( code != 0 )) || [[ -z "$result" ]]; then
        sed 's/^/STDERR /' "$work/agent.$index.err" | tail -n 4
      fi
      printf 'AGENT index=%s own_instance=%s exit=%s role=%s text=%s %s\n' "$index" \
        "$([[ "${migs[$index]}" == "$mig_a" ]] && echo 1 || echo 0)" "$code" "$role" \
        "$(text_of "$work/agent.$index")" "${result#RESULT }"
    done
    printf 'MEMORY mem_available_drop_mib=%s\n' "$(( (before - lowest) / 1024 ))"
    printf 'END_CASE timed_out=%s workers=%s masters=%s fallbacks=%s\n' "$timed_out" \
      "$workers" "$masters" "$fallbacks"
  } >>"$raw_log"
}

for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then stacks=(stock ipc inplace); else stacks=(inplace ipc stock); fi
  printf 'BEGIN_WEIGHTSHARE run=%s\n' "$run" >>"$raw_log"
  for placement in $placements; do
    for stack in "${stacks[@]}"; do
      run_case "$placement" "$stack"
    done
  done
  printf 'END_WEIGHTSHARE\n' >>"$raw_log"
done

awk -f "$script_dir/summarize_ext_weightshare.awk" "$raw_log" \
  >"$result_dir/ext_weightshare_summary.csv"
(
  cd "$script_dir"
  sha256sum kv_extents.patch kv_fork.cpp summarize_ext_weightshare.awk \
    run_ext_weightshare.sh
) >"$result_dir/source_hashes.txt"
printf 'ext_weightshare_result_dir=%s\n' "$result_dir"
