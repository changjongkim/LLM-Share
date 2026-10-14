#!/usr/bin/env bash
# The variant of run_engine_kvpipefault.sh for a GPU without MIG partitions
# (PRODUCER_MIG and CONSUMER_MIG name the same device): configuration
# timeslice runs every process time-sliced on the whole GPU, and mps runs every
# process as a client of one MPS server.
#
# A pipeline of agents in which one process fails. The pipeline is that of
# run_engine_kvpipe.sh on extent chains: a planner computes the prompt prefix
# (a system prompt and the schemas of the tools, workloads/bfcl_prefix.txt)
# and publishes it; G leaders attach to it, append the brief and the plan of
# their group and publish; L workers per leader attach to what their leader
# published and work through TURNS turns of a request and tool results. The
# planner and the leaders generate while the workers run. FAULT_AFTER_S
# seconds after the workers have been released, one of four things happens:
#
#   none         nothing; the texts of this case are the reference of the
#                others
#   write        a further process maps the cache file of the planner
#                read-only, as an agent does, and launches a GPU kernel that
#                writes into it (cuda_protect_probe host); it runs where
#                worker 0 runs, as a client of the same MPS server when there
#                is one
#   kill_worker  worker 0 is killed with SIGKILL
#   kill_leader  leader 0 is killed with SIGKILL while its workers, which map
#                the segment that it published, generate
#
# in two ways in which the processes share the GPU:
#
#   mig      the placement of run_engine_kvpipe.sh: the planner and the even
#            leaders in the 12-SM MIG instance, the odd leaders in the other
#            instance, the workers of a leader alternate between the two;
#            every process is time-sliced
#   mig_mps  the same placement, every process a client of the MPS server of
#            its instance
#
# An MPS server is started anew for every case, because a fault of a client
# can leave it unusable.
#
# Kill gates, fixed before the campaign:
#   Q1  without a fault every process completes in both configurations;
#   Q2  the GPU write is refused in every configuration;
#   Q3  in mig, after the write, after the kill of worker 0 and after the
#       kill of leader 0, every other process completes, and every worker
#       (worker 0 excepted after its kill) writes the text of the case
#       without a fault in the same repetition, over all turns;
#   Q4  in mig, the pipeline with a fault completes within 1.10 of the time
#       of the pipeline without a fault in the same repetition.
# Counted and reported, not gated: the processes that complete after a fault
# in mig_mps, where the processes of an instance are clients of one server.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
config_list=${CONFIGS:-"timeslice mps"}
fault_list=${FAULTS:-"none write kill_worker kill_leader"}
groups=${GROUPS_COUNT:-2}
workers=${WORKERS:-4}
turns=${TURNS:-3}
prefix_file=${PREFIX_FILE:-"$script_dir/workloads/bfcl_prefix.txt"}
tasks_file=${TASKS_FILE:-"$script_dir/workloads/bfcl_tasks.txt"}
context=${CONTEXT:-16384}
n_gen=${N_GEN:-64}
grow=${GROW_ROWS:-256}
fault_after_s=${FAULT_AFTER_S:-10}
probe_mib=${PROBE_MIB:-64}
limit_s=${CASE_LIMIT_S:-600}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
engine=${CHAIN_ENGINE:-"$script_dir/llama.cpp-chain2"}
driver=${DRIVER:-"$script_dir/kv_tree2"}
probe=${PROBE:-"$script_dir/cuda_protect_probe"}
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-kvpipefault"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
if [[ ! -x "$driver" || ! -x "$probe" || ! -r "$model" ]]; then
  echo "missing driver, probe or model; run make kv_tree2 cuda_protect_probe" >&2
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
  mps_root=$(mktemp -d "${TMPDIR:-/tmp}/kq.XXXXXX")
fi
(( ${#mps_root} <= 84 )) || { echo "MPS directory path too long" >&2; exit 2; }
pipe_a="$mps_root/a"
pipe_b="$mps_root/b"
mkdir -p "$pipe_a" "$pipe_b" "$mps_root/la" "$mps_root/lb"
running_a=0
running_b=0
daemon_pids=()
# A control daemon accepts commands once its control pipe exists.
await_control() {  # pipe-directory
  local waited=0
  until [[ -e "$1/control" ]] || (( waited >= 100 )); do
    sleep 0.1
    waited=$((waited + 1))
  done
}
start_mps() {
  if [[ "$1" == a && "$running_a" -eq 0 ]]; then
    env CUDA_VISIBLE_DEVICES="$mig_a" CUDA_MPS_PIPE_DIRECTORY="$pipe_a" \
      CUDA_MPS_LOG_DIRECTORY="$mps_root/la" nvidia-cuda-mps-control -d
    running_a=1
    await_control "$pipe_a"
  elif [[ "$1" == b && "$running_b" -eq 0 ]]; then
    env CUDA_VISIBLE_DEVICES="$mig_b" CUDA_MPS_PIPE_DIRECTORY="$pipe_b" \
      CUDA_MPS_LOG_DIRECTORY="$mps_root/lb" nvidia-cuda-mps-control -d
    running_b=1
    await_control "$pipe_b"
  fi
  mapfile -t daemon_pids < <(pgrep -f '(^|/)nvidia-cuda-mps-control( |$)' || true)
}
# A control daemon leaves some time after it was told to quit, and a new one
# for the same directory cannot start before that. Daemons that do not leave
# are ended by their process id.
await_no_mps() {
  local waited=0 pid
  while { pgrep -f '(^|/)nvidia-cuda-mps-server( |$)' >/dev/null ||
          pgrep -f '(^|/)nvidia-cuda-mps-control( |$)' >/dev/null; } &&
        (( waited < 300 )); do
    sleep 0.1
    waited=$((waited + 1))
  done
  for pid in "${daemon_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null || true; sleep 1; fi
  done
  daemon_pids=()
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
  await_no_mps
  if [[ "$1" == mig_mps ]]; then start_mps a; start_mps b; fi
  if [[ "$1" == mps ]]; then start_mps a; fi
}

huge_dir="$mount_root/kv_huge"
mounted_huge=0
work=
live_pids=()
cleanup() {
  if (( ${#live_pids[@]} )); then kill -9 "${live_pids[@]}" 2>/dev/null || true; fi
  stop_mps a
  stop_mps b
  await_no_mps
  rm -rf "${mps_root:?}"
  if (( mounted_huge )); then sudo -n umount "$huge_dir" || true; fi
  rmdir "$huge_dir" "$mount_root" 2>/dev/null || true
  if [[ -n "$work" ]]; then rm -rf "$work"; fi
}
trap cleanup EXIT
mkdir -p "$huge_dir" "$result_dir"
sudo -n mount -t tmpfs -o "huge=always,size=8192m,uid=$(id -u),gid=$(id -g)" \
  tmpfs "$huge_dir"
mounted_huge=1
work=$(mktemp -d "$result_dir/work.XXXXXX")
task_lines=$(wc -l <"$tasks_file")

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\nconfigs=%s\nfaults=%s\n' "$mig_a" \
    "$mig_b" "$repetitions" "$config_list" "$fault_list"
  printf 'groups=%s\nworkers=%s\nturns=%s\n' "$groups" "$workers" "$turns"
  printf 'context=%s\nn_gen=%s\ngrow_rows=%s\nfault_after_s=%s\nprobe_mib=%s\n' \
    "$context" "$n_gen" "$grow" "$fault_after_s" "$probe_mib"
  printf 'prefix_file=%s\nprefix_sha256=%s\ntasks_file=%s\ntasks_sha256=%s\n' \
    "$(basename "$prefix_file")" "$(sha256sum "$prefix_file" | cut -d' ' -f1)" \
    "$(basename "$tasks_file")" "$(sha256sum "$tasks_file" | cut -d' ' -f1)"
  printf 'model=%s\n' "$(basename "$model")"
  printf 'engine_commit=%s\n' "$(git -C "$engine" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$driver" "$probe" "$engine/build/bin/libllama.so" \
    "$engine/build/bin/libggml-cuda.so"
} >"$result_dir/metadata.txt"

planner_task=$'\n\nRequest to the planner: name the three tools above that a travel assistant needs most.\n'
group_brief() {  # group
  printf '\n\nBrief for the leader of group %s: your workers answer user requests with the tools above. Write the rules they follow when they choose a tool and fill its parameters.\n' "$1"
}
leader_task() {  # group
  printf '\n\nRequest to the leader of group %s: state in one sentence what your workers must check before a call.\n' "$1"
}
worker_turn() {  # worker turn
  if (( $2 == 0 )); then
    printf '\n\nUser request for worker %s: %s\nCall:\n' "$1" \
      "$(sed -n "$(( $1 % task_lines + 1 ))p" "$tasks_file")"
  else
    printf '\n\nTool result %s for worker %s: {"status": "ok", "items": %s}\nNext call or final answer:\n' \
      "$2" "$1" "$(( $1 + $2 ))"
  fi
}
leader_mig() { if (( $1 % 2 == 0 )); then echo "$mig_a"; else echo "$mig_b"; fi; }
worker_mig() {  # group worker
  if (( ($1 + $2) % 2 == 0 )); then echo "$mig_a"; else echo "$mig_b"; fi
}
text_of() { { grep '^TEXT ' "$1" || true; } | sha256sum | cut -c1-16; }
field_of() {  # name line-prefix file
  { grep "^$2 " "$3" || true; } | head -n 1 |
    sed -n "s/^.* $1=\\([^ ]*\\).*\$/\\1/p"
}
wait_line() {  # line-prefix file pid: until the file has the line or the process left
  until grep -q "^$1 " "$2" 2>/dev/null; do
    kill -0 "$3" 2>/dev/null || return 0
    sleep 0.02
  done
}
# Runs a program in a MIG instance, in the background, as a client of the MPS
# server of the instance when the configuration has one.
launch() {  # configuration mig output env-and-arguments...
  local config=$1 mig=$2 output=$3 mps_env=()
  shift 3
  if [[ "$config" == mps ]]; then
    mps_env=(CUDA_MPS_PIPE_DIRECTORY="$pipe_a")
  elif [[ "$config" == mig_mps ]]; then
    if [[ "$mig" == "$mig_a" ]]; then
      mps_env=(CUDA_MPS_PIPE_DIRECTORY="$pipe_a")
    else
      mps_env=(CUDA_MPS_PIPE_DIRECTORY="$pipe_b")
    fi
  fi
  env -u CUDA_MPS_PIPE_DIRECTORY CUDA_VISIBLE_DEVICES="$mig" \
    GGML_CUDA_HOST_PTR=1 "${mps_env[@]}" "$@" >"$output" 2>"$output.err" &
}

run_case() {  # configuration fault
  local config=$1 fault=$2 group worker number mig turn
  prepare "$config"
  rm -f "$work"/state.* "$work/go" "$work"/out.* "$work"/probe* "$huge_dir"/*
  local started t_go=0
  started=$(date +%s%N)

  launch "$config" "$mig_a" "$work/out.root" \
    LLAMA_KV_HOST="$huge_dir/r" LLAMA_KV_GROW="$grow" \
    "$driver" "$model" "$context" "$n_gen" "text:$prefix_file" \
    "save:$work/state.root" "wait:$work/go" "say:$planner_task" gen
  local root_pid=$!
  live_pids=("$root_pid")
  wait_line PUBLISHED "$work/out.root" "$root_pid"
  local root_tokens root_file
  root_tokens=$(field_of tokens PUBLISHED "$work/out.root")
  root_file=$(find "$huge_dir" -maxdepth 1 -type f -name 'r*' | sort | head -n 1)

  local leader_pids=() leader_tokens=()
  for group in $(seq 0 $((groups - 1))); do
    launch "$config" "$(leader_mig "$group")" "$work/out.leader.$group" \
      LLAMA_KV_CHAIN="$huge_dir/r:${root_tokens:-0}" \
      LLAMA_KV_HOST="$huge_dir/m$group" LLAMA_KV_GROW="$grow" \
      "$driver" "$model" "$context" "$n_gen" \
      "load:$work/state.root" "say:$(group_brief "$group")" gen \
      "save:$work/state.leader.$group" "wait:$work/go" \
      "say:$(leader_task "$group")" gen
    leader_pids+=($!)
  done
  live_pids+=("${leader_pids[@]}")
  for group in $(seq 0 $((groups - 1))); do
    wait_line PUBLISHED "$work/out.leader.$group" "${leader_pids[$group]}"
    leader_tokens+=("$(field_of tokens PUBLISHED "$work/out.leader.$group")")
  done

  local worker_pids=() worker_groups=() worker_migs=() worker_files=()
  number=0
  for group in $(seq 0 $((groups - 1))); do
    for worker in $(seq 0 $((workers - 1))); do
      mig=$(worker_mig "$group" "$worker")
      local steps=("load:$work/state.leader.$group")
      for turn in $(seq 0 $((turns - 1))); do
        steps+=("say:$(worker_turn "$number" "$turn")" gen)
      done
      launch "$config" "$mig" "$work/out.worker.$number" \
        LLAMA_KV_CHAIN="$huge_dir/r:${root_tokens:-0},$huge_dir/m$group:${leader_tokens[$group]:-0}" \
        LLAMA_KV_HOST=anon LLAMA_KV_GROW="$grow" \
        "$driver" "$model" "$context" "$n_gen" "${steps[@]}"
      worker_pids+=($!)
      worker_groups+=("$group")
      worker_migs+=("$mig")
      worker_files+=("$work/out.worker.$number")
      number=$((number + 1))
    done
  done
  live_pids+=("${worker_pids[@]}")
  touch "$work/go"
  t_go=$(date +%s%N)

  local victim=none victim_alive=0 alive_at_fault=0 probe_line=none probe_exit=none pid
  if [[ "$fault" != none ]]; then
    sleep "$fault_after_s"
    for pid in "${live_pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then alive_at_fault=$((alive_at_fault + 1)); fi
    done
    case $fault in
      write)
        victim=probe
        victim_alive=1
        launch "$config" "${worker_migs[0]}" "$work/probe" \
          "$probe" host "$root_file" "$probe_mib"
        local probe_pid=$!
        probe_exit=0
        wait "$probe_pid" || probe_exit=$?
        probe_line=$(grep '^RESULT ' "$work/probe" | head -n 1 || true)
        probe_line=${probe_line#RESULT }
        probe_line=${probe_line// /,}
        [[ -n "$probe_line" ]] || probe_line=no_result ;;
      kill_worker)
        victim=worker0
        if kill -0 "${worker_pids[0]}" 2>/dev/null; then victim_alive=1; fi
        kill -9 "${worker_pids[0]}" 2>/dev/null || true ;;
      kill_leader)
        victim=leader0
        if kill -0 "${leader_pids[0]}" 2>/dev/null; then victim_alive=1; fi
        kill -9 "${leader_pids[0]}" 2>/dev/null || true ;;
      *) echo "unknown fault $fault" >&2; exit 2 ;;
    esac
  fi

  local alive=1 timed_out=0
  while (( alive > 0 )); do
    alive=0
    for pid in "${live_pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then alive=$((alive + 1)); fi
    done
    if (( ($(date +%s%N) - started) / 1000000000 > limit_s )); then
      timed_out=1
      kill -9 "${live_pids[@]}" 2>/dev/null || true
    fi
    sleep 0.1
  done
  local ended
  ended=$(date +%s%N)
  {
    printf 'CASE config=%s fault=%s groups=%s workers=%s turns=%s\n' "$config" \
      "$fault" "$groups" "$workers" "$turns"
    local code=0 completed
    wait "$root_pid" 2>/dev/null || code=$?
    completed=0
    if (( code == 0 )) && grep -q '^RESULT ' "$work/out.root"; then completed=1; fi
    if (( completed == 0 )); then sed 's/^/STDERR /' "$work/out.root.err" | tail -n 3; fi
    printf 'PLANNER mig=%s exit=%s completed=%s text=%s tokens=%s publish_ms=%s\n' \
      "${mig_a:4:8}" "$code" "$completed" "$(text_of "$work/out.root")" \
      "${root_tokens:-0}" "$(field_of publish_ms PUBLISHED "$work/out.root")"
    for group in $(seq 0 $((groups - 1))); do
      code=0
      wait "${leader_pids[$group]}" 2>/dev/null || code=$?
      completed=0
      if (( code == 0 )) && grep -q '^RESULT ' "$work/out.leader.$group"; then completed=1; fi
      if (( completed == 0 )); then
        sed 's/^/STDERR /' "$work/out.leader.$group.err" | tail -n 3
      fi
      printf 'LEADER group=%s mig=%s exit=%s completed=%s text=%s tokens=%s publish_ms=%s\n' \
        "$group" "$(leader_mig "$group" | cut -c5-12)" "$code" "$completed" \
        "$(text_of "$work/out.leader.$group")" "${leader_tokens[$group]:-0}" \
        "$(field_of publish_ms PUBLISHED "$work/out.leader.$group")"
    done
    for number in "${!worker_pids[@]}"; do
      code=0
      wait "${worker_pids[$number]}" 2>/dev/null || code=$?
      completed=0
      if (( code == 0 )) && grep -q '^RESULT ' "${worker_files[$number]}"; then completed=1; fi
      if (( completed == 0 )); then
        sed 's/^/STDERR /' "${worker_files[$number]}.err" | tail -n 3
      fi
      printf 'WORKER index=%s group=%s mig=%s exit=%s completed=%s text=%s generated=%s generation_tps=%s first_token_ms=%s total_ms=%s\n' \
        "$number" "${worker_groups[$number]}" "${worker_migs[$number]:4:8}" "$code" \
        "$completed" "$(text_of "${worker_files[$number]}")" \
        "$(field_of generated RESULT "${worker_files[$number]}")" \
        "$(field_of generation_tps RESULT "${worker_files[$number]}")" \
        "$(field_of first_token_ms RESULT "${worker_files[$number]}")" \
        "$(field_of total_ms RESULT "${worker_files[$number]}")"
    done
    printf 'FAULT victim=%s victim_alive=%s alive_at_fault=%s probe_exit=%s probe=%s\n' \
      "$victim" "$victim_alive" "$alive_at_fault" "$probe_exit" "$probe_line"
    if [[ "$fault" == write ]]; then sed 's/^/PROBE_STDERR /' "$work/probe.err" | tail -n 3; fi
    printf 'END_CASE timed_out=%s mps_servers=%s wall_ms=%s workers_ms=%s\n' "$timed_out" \
      "$(pgrep -fc '(^|/)nvidia-cuda-mps-server( |$)' || true)" \
      "$(( (ended - started) / 1000000 ))" "$(( (ended - t_go) / 1000000 ))"
  } >>"$raw_log"
  live_pids=()
  rm -f "$work"/state.* "$work/go" "$work"/out.* "$work"/probe* "$huge_dir"/*
}

read -r -a faults <<<"$fault_list"
for run in $(seq 1 "$repetitions"); do
  configs=()
  for config in $config_list; do
    if (( run % 2 == 1 )); then configs+=("$config"); else configs=("$config" "${configs[@]}"); fi
  done
  printf 'BEGIN_KVPIPEFAULT run=%s context=%s\n' "$run" "$context" >>"$raw_log"
  for config in "${configs[@]}"; do
    for fault in "${faults[@]}"; do
      run_case "$config" "$fault"
    done
  done
  printf 'END_KVPIPEFAULT\n' >>"$raw_log"
done
stop_mps a
stop_mps b
await_no_mps

awk -f "$script_dir/summarize_engine_kvpipefault.awk" "$raw_log" \
  >"$result_dir/engine_kvpipefault_summary.csv"
(
  cd "$script_dir"
  sha256sum kv_chain2.patch kv_tree.cpp cuda_protect_probe.cu \
    summarize_engine_kvpipefault.awk run_engine_kvpipefault.sh run_engine_kvpipefault_gpu.sh \
    "$(realpath --relative-to=. "$prefix_file")" \
    "$(realpath --relative-to=. "$tasks_file")"
) >"$result_dir/source_hashes.txt"
printf 'engine_kvpipefault_result_dir=%s\n' "$result_dir"
