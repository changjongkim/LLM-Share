#!/usr/bin/env bash
# What a fault of one process does to the agents that share a prefix with
# it. A publisher computes the key-value cache of a prefix, publishes it as
# extents and exits. N agents map the prefix and generate. While they
# generate, one of three things happens:
#
#   none   nothing; the texts of this case are the reference of the others
#   write  a further process maps the file of the prefix read-only, as an
#          agent does, and launches a GPU kernel that writes into it
#          (cuda_protect_probe host); it runs where agent 0 runs, as a client
#          of the same MPS server when there is one
#   kill   agent 0 is killed with SIGKILL
#
# in each of the four ways in which the processes share the GPU:
#
#   timeslice  every process in the 12-SM MIG instance, time-sliced
#   mps        every process a client of one MPS server in that instance
#   mig        every second agent in the 6-SM instance, time-sliced
#   mig_mps    the same placement, every process a client of the MPS server
#              of its instance
#
# An MPS server is started anew for every case, because a fault of a client
# can leave it unusable.
#
# Kill gates, fixed before the campaign:
#   F1  without a fault every agent completes in every configuration;
#   F2  the GPU write is refused in every configuration, and the file of the
#       prefix has the same contents after every case as before it;
#   F3  in timeslice and mig, every agent completes after the write and
#       after the kill of agent 0 (agent 0 excepted), with the text of the
#       case without a fault.
# Counted and reported, not gated: the agents that complete after the write
# and after the kill in mps and mig_mps, where the processes of an instance
# are clients of one server.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
agents=${AGENTS:-8}
config_list=${CONFIGS:-"timeslice mps mig mig_mps"}
prefix_file=${PREFIX_FILE:-"$script_dir/workloads/agent_prefix.txt"}
tasks_file=${TASKS_FILE:-"$script_dir/workloads/agent_tasks.txt"}
context=${CONTEXT:-16384}
n_gen=${N_GEN:-192}
grow=${GROW_ROWS:-256}
fault_after_s=${FAULT_AFTER_S:-10}
probe_mib=${PROBE_MIB:-64}
limit_s=${CASE_LIMIT_S:-600}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
engine=${KV_ENGINE:-"$script_dir/llama.cpp-kv"}
driver=${DRIVER:-"$script_dir/kv_fork"}
probe=${PROBE:-"$script_dir/cuda_protect_probe"}
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-kvfault"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
if [[ ! -x "$driver" || ! -x "$probe" || ! -r "$model" ]]; then
  echo "missing driver, probe or model; run make kv_fork cuda_protect_probe" >&2
  exit 2
fi
for file in "$prefix_file" "$tasks_file"; do
  if [[ ! -r "$file" ]]; then echo "cannot read $file" >&2; exit 2; fi
done
if ! sudo -n true 2>/dev/null; then
  echo "passwordless sudo is required for the tmpfs mount" >&2
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

# The MPS control socket needs a short path; MPS_ROOT names the directory.
if [[ -n "${MPS_ROOT:-}" ]]; then
  mps_root=$MPS_ROOT
  mkdir -p "$mps_root"
  [[ -z "$(ls -A "$mps_root")" ]] || { echo "MPS_ROOT is not empty" >&2; exit 2; }
else
  mps_root=$(mktemp -d "${TMPDIR:-/tmp}/kf.XXXXXX")
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
# Every case starts from servers that have seen no client.
prepare() {  # configuration
  stop_mps a
  stop_mps b
  local waited=0
  while pgrep -f '(^|/)nvidia-cuda-mps-server( |$)' >/dev/null && (( waited < 100 )); do
    sleep 0.1
    waited=$((waited + 1))
  done
  case $1 in
    mps) start_mps a ;;
    mig_mps) start_mps a; start_mps b ;;
  esac
}
# Prints "MIG PIPE_DIR" for a process of a configuration. PIPE_DIR is "-" for
# a process that is not an MPS client.
placement() {  # configuration index
  local config=$1 index=$2
  case $config in
    timeslice) echo "$mig_a -" ;;
    mps) echo "$mig_a $pipe_a" ;;
    mig) if (( index % 2 != 1 )); then echo "$mig_a -"; else echo "$mig_b -"; fi ;;
    mig_mps)
      if (( index % 2 != 1 )); then echo "$mig_a $pipe_a"; else echo "$mig_b $pipe_b"; fi ;;
  esac
}

huge_dir="$mount_root/kv_huge"
mounted_huge=0
work=
live_pids=()
cleanup() {
  if (( ${#live_pids[@]} )); then kill -9 "${live_pids[@]}" 2>/dev/null || true; fi
  stop_mps a
  stop_mps b
  rm -rf "$mps_root"
  if (( mounted_huge )); then sudo -n umount "$huge_dir" || true; fi
  rmdir "$huge_dir" "$mount_root" 2>/dev/null || true
  if [[ -n "$work" ]]; then rm -rf "$work"; fi
}
trap cleanup EXIT
mkdir -p "$huge_dir" "$result_dir"
sudo -n mount -t tmpfs -o "huge=always,size=4096m,uid=$(id -u),gid=$(id -g)" \
  tmpfs "$huge_dir"
mounted_huge=1
work=$(mktemp -d "$result_dir/work.XXXXXX")
task_lines=$(wc -l <"$tasks_file")

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\nagents=%s\nconfigs=%s\n' "$mig_a" \
    "$mig_b" "$repetitions" "$agents" "$config_list"
  printf 'context=%s\nn_gen=%s\ngrow_rows=%s\nfault_after_s=%s\nprobe_mib=%s\n' \
    "$context" "$n_gen" "$grow" "$fault_after_s" "$probe_mib"
  printf 'prefix_file=%s\nprefix_sha256=%s\ntasks_file=%s\n' \
    "$(basename "$prefix_file")" "$(sha256sum "$prefix_file" | cut -d' ' -f1)" \
    "$(basename "$tasks_file")"
  printf 'model=%s\n' "$(basename "$model")"
  printf 'engine_commit=%s\n' "$(git -C "$engine" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$driver" "$probe" "$engine/build/bin/libllama.so" \
    "$engine/build/bin/libggml-cuda.so"
} >"$result_dir/metadata.txt"

suffix_of() {  # agent index
  printf '\n\nTask for agent %s: %s\n' "$1" \
    "$(sed -n "$(( $1 % task_lines + 1 ))p" "$tasks_file")"
}
text_of() { { grep '^TEXT ' "$1" || true; } | sha256sum | cut -c1-16; }
# Runs a program as a process of a configuration, in the background.
launch() {  # configuration index output env-and-arguments...
  local mig pipe output=$3
  read -r mig pipe <<<"$(placement "$1" "$2")"
  shift 3
  local mps_env=()
  if [[ "$pipe" != - ]]; then mps_env=(CUDA_MPS_PIPE_DIRECTORY="$pipe"); fi
  env -u CUDA_MPS_PIPE_DIRECTORY CUDA_VISIBLE_DEVICES="$mig" \
    GGML_CUDA_HOST_PTR=1 "${mps_env[@]}" "$@" >"$output" 2>"$output.err" &
}

run_case() {  # configuration fault prefix-tokens
  local config=$1 fault=$2 tokens=$3 index pids=() migs=() mig pipe
  prepare "$config"
  rm -f "$work"/agent.* "$work"/probe*
  local hash_before hash_after started
  hash_before=$(sha256sum "$huge_dir/kv.0" | cut -c1-16)
  started=$(date +%s)
  for index in $(seq 0 $((agents - 1))); do
    read -r mig pipe <<<"$(placement "$config" "$index")"
    migs+=("$mig")
    launch "$config" "$index" "$work/agent.$index" \
      LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_PREFIX="$tokens" LLAMA_KV_GROW="$grow" \
      "$driver" child "$model" "$context" "$work/state" "$(suffix_of "$index")" "$n_gen"
    pids+=($!)
  done
  live_pids=("${pids[@]}")
  local probe_line=none probe_exit=none alive_at_fault=0
  if [[ "$fault" != none ]]; then
    sleep "$fault_after_s"
    for index in "${!pids[@]}"; do
      if kill -0 "${pids[$index]}" 2>/dev/null; then alive_at_fault=$((alive_at_fault + 1)); fi
    done
    if [[ "$fault" == write ]]; then
      launch "$config" 0 "$work/probe" "$probe" host "$huge_dir/kv.0" "$probe_mib"
      local probe_pid=$!
      probe_exit=0
      wait "$probe_pid" || probe_exit=$?
      probe_line=$(grep '^RESULT ' "$work/probe" | head -n 1 || true)
      probe_line=${probe_line#RESULT }
      probe_line=${probe_line// /,}
      [[ -n "$probe_line" ]] || probe_line=no_result
    else
      kill -9 "${pids[0]}" 2>/dev/null || true
    fi
  fi
  local alive=1 timed_out=0
  while (( alive > 0 )); do
    alive=0
    for index in "${!pids[@]}"; do
      if kill -0 "${pids[$index]}" 2>/dev/null; then alive=$((alive + 1)); fi
    done
    if (( $(date +%s) - started > limit_s )); then
      timed_out=1
      kill -9 "${pids[@]}" 2>/dev/null || true
    fi
    sleep 0.1
  done
  live_pids=()
  hash_after=$(sha256sum "$huge_dir/kv.0" | cut -c1-16)
  {
    printf 'CASE config=%s fault=%s agents=%s\n' "$config" "$fault" "$agents"
    for index in "${!pids[@]}"; do
      local code=0 result
      wait "${pids[$index]}" || code=$?
      result=$(grep '^RESULT ' "$work/agent.$index" || true)
      if (( code != 0 )) || [[ -z "$result" ]]; then
        sed 's/^/STDERR /' "$work/agent.$index.err" | tail -n 3
      fi
      printf 'AGENT index=%s mig=%s exit=%s completed=%s text=%s %s\n' "$index" \
        "${migs[$index]:4:8}" "$code" "$([[ -n "$result" ]] && echo 1 || echo 0)" \
        "$(text_of "$work/agent.$index")" "${result#RESULT }"
    done
    printf 'FAULT alive_at_fault=%s probe_exit=%s probe=%s\n' "$alive_at_fault" \
      "$probe_exit" "$probe_line"
    if [[ "$fault" == write ]]; then sed 's/^/PROBE_STDERR /' "$work/probe.err" | tail -n 3; fi
    printf 'FILE before=%s after=%s\n' "$hash_before" "$hash_after"
    printf 'END_CASE timed_out=%s mps_servers=%s wall_s=%s\n' "$timed_out" \
      "$(pgrep -fc '(^|/)nvidia-cuda-mps-server( |$)' || true)" \
      "$(( $(date +%s) - started ))"
  } >>"$raw_log"
}

for run in $(seq 1 "$repetitions"); do
  configs=()
  for config in $config_list; do
    if (( run % 2 == 1 )); then configs+=("$config"); else configs=("$config" "${configs[@]}"); fi
  done
  stop_mps a
  stop_mps b
  rm -f "$huge_dir/kv.0" "$work/state"
  env CUDA_VISIBLE_DEVICES="$mig_a" GGML_CUDA_HOST_PTR=1 \
    LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_GROW="$grow" \
    "$driver" parent "$model" "$context" "$prefix_file" "$work/state" "" 0 \
    >"$work/publish.out" 2>"$work/publish.err" || true
  chmod 0400 "$huge_dir/kv.0"
  tokens=$(sed -n 's/^PUBLISHED prefix_tokens=\([0-9]*\).*/\1/p' "$work/publish.out")
  printf 'BEGIN_KVFAULT run=%s prefix_tokens=%s\n' "$run" "${tokens:-0}" >>"$raw_log"
  for config in "${configs[@]}"; do
    for fault in none write kill; do
      run_case "$config" "$fault" "${tokens:-0}"
    done
  done
  printf 'END_KVFAULT\n' >>"$raw_log"
done

awk -f "$script_dir/summarize_engine_kvfault.awk" "$raw_log" \
  >"$result_dir/engine_kvfault_summary.csv"
(
  cd "$script_dir"
  sha256sum kv_extents.patch kv_fork.cpp cuda_protect_probe.cu \
    summarize_engine_kvfault.awk run_engine_kvfault.sh
) >"$result_dir/source_hashes.txt"
printf 'engine_kvfault_result_dir=%s\n' "$result_dir"
