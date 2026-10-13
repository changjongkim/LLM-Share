#!/usr/bin/env bash
# Where an agent of the unmodified engine can run next to other agents. One
# agent is latency-critical: it waits for an input and then decodes it and
# generates N_GEN tokens. K background agents generate at the same time, in
# one of three placements:
#
#   slice   in the MIG instance of the critical agent, time-sliced
#   mps     in the same instance, every process a client of one MPS server
#   split   in the other MIG instance
#
# and `alone` is the critical agent without a background agent. Every agent
# is a separate process of the unmodified engine: it copies the weights to
# device memory and restores the prefix from a state file by copy.
#
# Each placement runs without a fault and with one that takes place while the
# background agents generate and the critical agent still waits:
#
#   write   a process in the instance of the background agents (a client of
#           the same MPS server under mps) writes through the GPU to a
#           read-only host mapping
#   kill    one background agent is ended with SIGKILL
#
# AFTER_FAULT_S seconds after the fault, the critical agent is released; one
# that has not finished CRITICAL_LIMIT_S seconds later is ended and counts as
# not completed. A case without a fault counts only if every background agent
# was still generating when the critical agent finished.
#
# Measured for the critical agent: whether it completes, the time from its
# release until its input is decoded, and the time to generate N_GEN tokens;
# for the background agents: how many of those that the fault did not name
# still generate when the critical agent has finished or has left.
#
# An MPS server is started anew for every case under mps, because a fault of
# a client ends the server.
#
# Kill gates, fixed before the campaign:
#   P1  without a fault, no critical agent fails, no case is ended by the
#       time limit or the memory guard, and every background agent is still
#       generating when the critical agent finishes;
#   P2  under slice and under split, with either fault, the critical agent
#       completes in every repetition and every background agent that the
#       fault did not name is still generating when it finishes;
#   P3  under mps with the write fault, the critical agent completes in no
#       repetition and no background agent is still generating;
#   P4  under slice without a fault, the critical agent takes at least 2.0
#       times as long to generate as alone, and under split it generates at
#       no less than 0.90 of its rate alone (paired within a repetition,
#       geometric mean over the repetitions).
# Reported, not gated: the generation time under mps without a fault, the
# outcome of the kill fault under mps, the time from the release to the
# decoded input, the memory, and whether the texts of the critical agent
# are equal to its text alone.
set -euo pipefail
# The clients of an MPS server that a fault ends abort; they leave no core file.
ulimit -c 0

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
critical_list=${CRITICAL:-"a"}
placement_list=${PLACEMENTS:-"slice mps split"}
fault_list=${FAULTS:-"none write kill"}
background=${BACKGROUND:-7}
prefix_file=${PREFIX_FILE:-"$script_dir/workloads/agent_prefix.txt"}
tasks_file=${TASKS_FILE:-"$script_dir/workloads/agent_tasks.txt"}
context=${CONTEXT:-16384}
n_gen=${N_GEN:-64}
bg_gen=${BG_GEN:-480}
probe_mib=${PROBE_MIB:-64}
limit_s=${CASE_LIMIT_S:-600}
critical_limit_s=${CRITICAL_LIMIT_S:-180}
settle_s=${SETTLE_S:-2}
warm_s=${WARM_S:-1}
after_fault_s=${AFTER_FAULT_S:-3}
fault_limit_s=${FAULT_LIMIT_S:-120}
guard_gib=${GUARD_GIB:-16}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
engine=${STOCK_ENGINE:-"$script_dir/llama.cpp-stock"}
driver=${DRIVER:-"$script_dir/kv_agent_stock"}
probe=${PROBE:-"$script_dir/cuda_protect_probe"}
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-kvplace"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
if ! [[ "$background" =~ ^[1-9][0-9]*$ ]]; then
  echo "BACKGROUND must be a positive integer" >&2
  exit 2
fi
if [[ ! -x "$driver" || ! -x "$probe" || ! -r "$model" ]]; then
  echo "missing driver, probe or model; run make kv_agent_stock cuda_protect_probe" >&2
  exit 2
fi
if [[ -n "$(git -C "$engine" status --porcelain --untracked-files=no)" ]]; then
  echo "the unmodified engine has local changes: $engine" >&2
  exit 2
fi
for file in "$prefix_file" "$tasks_file"; do
  if [[ ! -r "$file" ]]; then echo "cannot read $file" >&2; exit 2; fi
done
if ! sudo -n true 2>/dev/null; then
  echo "passwordless sudo is required for the tmpfs mount and the cache drop" >&2
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
  mps_root=$(mktemp -d "${TMPDIR:-/tmp}/kp.XXXXXX")
fi
(( ${#mps_root} <= 84 )) || { echo "MPS directory path too long" >&2; exit 2; }
pipe="$mps_root/p"
mkdir -p "$pipe" "$mps_root/log"
mps_running=0
daemon_pids=()
# A control daemon accepts commands once its control pipe exists.
await_control() {
  local waited=0
  until [[ -e "$pipe/control" ]] || (( waited >= 100 )); do
    sleep 0.1
    waited=$((waited + 1))
  done
}
start_mps() {  # mig
  env CUDA_VISIBLE_DEVICES="$1" CUDA_MPS_PIPE_DIRECTORY="$pipe" \
    CUDA_MPS_LOG_DIRECTORY="$mps_root/log" nvidia-cuda-mps-control -d
  mps_running=1
  await_control
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
  if (( mps_running )); then
    env CUDA_MPS_PIPE_DIRECTORY="$pipe" CUDA_MPS_LOG_DIRECTORY="$mps_root/log" \
      bash -c 'echo quit | nvidia-cuda-mps-control' >/dev/null 2>&1 || true
    mps_running=0
  fi
  await_no_mps
}

huge_dir="$mount_root/kv_huge"
mounted_huge=0
work=
live_pids=()
cleanup() {
  if (( ${#live_pids[@]} )); then kill -9 "${live_pids[@]}" 2>/dev/null || true; fi
  stop_mps
  rm -rf "${mps_root:?}"
  if (( mounted_huge )); then sudo -n umount "$huge_dir" || true; fi
  rmdir "$huge_dir" "$mount_root" 2>/dev/null || true
  if [[ -n "$work" ]]; then rm -rf "$work"; fi
}
trap cleanup EXIT
mkdir -p "$huge_dir" "$result_dir"
sudo -n mount -t tmpfs -o "huge=always,size=1024m,uid=$(id -u),gid=$(id -g)" \
  tmpfs "$huge_dir"
mounted_huge=1
work=$(mktemp -d "$result_dir/work.XXXXXX")
prefix="$work/prefix"
cp "$prefix_file" "$prefix"
# The read-only host mapping that the write fault addresses.
fault_file="$huge_dir/readonly"
head -c "$((probe_mib << 20))" /dev/zero >"$fault_file"
chmod 0400 "$fault_file"
task_lines=$(wc -l <"$tasks_file")
model_kib=$(( $(stat -c %s "$model") / 1024 ))

suffix_of() {  # agent index
  printf '\n\nTask for agent %s: %s\n' "$1" \
    "$(sed -n "$(( $1 % task_lines + 1 ))p" "$tasks_file")"
}
input_of() {  # agent index
  printf '\n\nTool result for agent %s: {"status": "ok", "items": %s}\nNext call or final answer:\n' \
    "$1" "$(( $1 + 1 ))"
}
field_of() {  # name file: value of name=VALUE on the first line that has it
  sed -n "s/^.* $1=\\([^ ]*\\).*\$/\\1/p" "$2" | head -n 1
}
last_of() {  # name line-prefix file: value on the last line of a kind
  { grep "^$2 " "$3" || true; } | tail -n 1 | sed -n "s/^.* $1=\\([^ ]*\\).*\$/\\1/p"
}
meminfo_kib() {  # name
  awk -v name="$1:" '$1 == name { print $2 }' /proc/meminfo
}

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\ncritical=%s\nplacements=%s\nfaults=%s\n' \
    "$mig_a" "$mig_b" "$repetitions" "$critical_list" "$placement_list" "$fault_list"
  printf 'background=%s\ncontext=%s\nn_gen=%s\nbg_gen=%s\nprobe_mib=%s\n' \
    "$background" "$context" "$n_gen" "$bg_gen" "$probe_mib"
  printf 'settle_s=%s\nwarm_s=%s\nafter_fault_s=%s\nfault_limit_s=%s\nguard_gib=%s\n' \
    "$settle_s" "$warm_s" "$after_fault_s" "$fault_limit_s" "$guard_gib"
  printf 'case_limit_s=%s\ncritical_limit_s=%s\n' "$limit_s" "$critical_limit_s"
  printf 'prefix_file=%s\nprefix_sha256=%s\ntasks_file=%s\ntasks_sha256=%s\n' \
    "$(basename "$prefix_file")" "$(sha256sum "$prefix" | cut -d' ' -f1)" \
    "$(basename "$tasks_file")" "$(sha256sum "$tasks_file" | cut -d' ' -f1)"
  printf 'model=%s\nmodel_bytes=%s\n' "$(basename "$model")" "$(stat -c %s "$model")"
  printf 'mem_total_kib=%s\n' "$(meminfo_kib MemTotal)"
  printf 'engine_commit=%s\n' "$(git -C "$engine" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$driver" "$probe" "$engine/build/bin/libllama.so" \
    "$engine/build/bin/libggml-cuda.so"
} >"$result_dir/metadata.txt"

# Runs a program in a MIG instance, in the background, as a client of the MPS
# server when the placement has one; $! is its process.
launch() {  # placement mig output arguments...
  local placement=$1 mig=$2 output=$3 mps_env=()
  shift 3
  if [[ "$placement" == mps ]]; then mps_env=(CUDA_MPS_PIPE_DIRECTORY="$pipe"); fi
  env -u CUDA_MPS_PIPE_DIRECTORY CUDA_VISIBLE_DEVICES="$mig" "${mps_env[@]}" "$@" \
    >"$output" 2>"$output.err" &
}
start_agent() {  # placement mig index tokens-to-generate go-file output
  launch "$1" "$2" "$6" "$driver" "$model" "$context" "$4" \
    "load:$work/state.stock" "say:$(suffix_of "$3")" mark:ready \
    "wait:$5" "say:$(input_of "$3")" gen
}

run_case() {  # critical-mig placement fault
  local critical_mig=$1 placement=$2 fault=$3 other_mig background_mig index count
  if [[ "$critical_mig" == "$mig_a" ]]; then other_mig=$mig_b; else other_mig=$mig_a; fi
  if [[ "$placement" == split ]]; then background_mig=$other_mig; else background_mig=$critical_mig; fi
  if [[ "$placement" == alone ]]; then count=0; else count=$background; fi
  stop_mps
  if [[ "$placement" == mps ]]; then start_mps "$critical_mig"; fi
  rm -f "$work"/agent.* "$work/go.critical" "$work/go.background" "$work"/probe*
  local before lowest sample case_start
  before=$(meminfo_kib MemAvailable)
  lowest=$before
  case_start=$(date +%s%N)
  # Agent 0 is the critical one, so that it has one task in every case.
  start_agent "$placement" "$critical_mig" 0 "$n_gen" "$work/go.critical" "$work/agent.0"
  local critical_pid=$! background_pids=()
  for index in $(seq 1 "$count"); do
    start_agent "$placement" "$background_mig" "$index" "$bg_gen" "$work/go.background" \
      "$work/agent.$index"
    background_pids+=($!)
  done
  live_pids=("$critical_pid" "${background_pids[@]}")

  # phase: 0 until every agent waits, 1 until the background agents generate,
  # 2 while the critical agent runs
  local phase=0 guard=0 timed_out=0 died=0 ready decoded pid
  local resident_lowest=0 generating=0 t_go=0 t_done=0
  local victim=none victim_index=0 probe_exit=none probe_line=none fault_ms=0
  local critical_alive=1 background_alive=0 t_fault
  while :; do
    sample=$(meminfo_kib MemAvailable)
    (( sample < lowest )) && lowest=$sample
    if (( (sample - model_kib) / 1024 < guard_gib * 1024 )); then guard=1; break; fi
    if (( ($(date +%s%N) - case_start) / 1000000000 > limit_s )); then timed_out=1; break; fi
    if (( phase < 2 )); then
      for pid in "${live_pids[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then died=1; fi
      done
      if (( died )); then break; fi
    fi
    if (( phase == 0 )); then
      ready=$( { grep -l '^MARK ' "$work"/agent.*[0-9] 2>/dev/null || true; } | wc -l)
      if (( ready == count + 1 )); then
        sleep "$settle_s"
        sample=$(meminfo_kib MemAvailable)
        (( sample < lowest )) && lowest=$sample
        resident_lowest=$lowest
        touch "$work/go.background"
        phase=1
      fi
    elif (( phase == 1 )); then
      # An agent that has decoded its input has two DECODED lines.
      decoded=0
      for index in $(seq 1 "$count"); do
        if (( $(grep -c '^DECODED ' "$work/agent.$index" || true) >= 2 )); then
          decoded=$((decoded + 1))
        fi
      done
      if (( decoded == count )); then
        sleep "$warm_s"
        if [[ "$fault" != none ]]; then
          t_fault=$(date +%s%N)
          case $fault in
            write)
              victim=probe
              launch "$placement" "$background_mig" "$work/probe" \
                timeout -s KILL "$fault_limit_s" "$probe" host "$fault_file" "$probe_mib"
              probe_exit=0
              wait $! || probe_exit=$?
              probe_line=$(grep '^RESULT ' "$work/probe" | head -n 1 || true)
              probe_line=${probe_line#RESULT }
              probe_line=${probe_line// /,}
              [[ -n "$probe_line" ]] || probe_line=no_result ;;
            kill)
              victim=agent1
              victim_index=1
              kill -9 "${background_pids[0]}" 2>/dev/null || true ;;
            *) echo "unknown fault $fault" >&2; exit 2 ;;
          esac
          fault_ms=$(( ($(date +%s%N) - t_fault) / 1000000 ))
          sleep "$after_fault_s"
          kill -0 "$critical_pid" 2>/dev/null || critical_alive=0
          for index in $(seq 1 "$count"); do
            if (( index != victim_index )) &&
               kill -0 "${background_pids[$((index - 1))]}" 2>/dev/null; then
              background_alive=$((background_alive + 1))
            fi
          done
        fi
        t_go=$(date +%s%N)
        touch "$work/go.critical"
        phase=2
      fi
    else
      if kill -0 "$critical_pid" 2>/dev/null &&
         (( ($(date +%s%N) - t_go) / 1000000000 > critical_limit_s )); then
        timed_out=1
        kill -9 "$critical_pid" 2>/dev/null || true
        sleep 0.2
      fi
      if ! kill -0 "$critical_pid" 2>/dev/null; then
        t_done=$(date +%s%N)
        for index in $(seq 1 "$count"); do
          if (( index != victim_index )) &&
             kill -0 "${background_pids[$((index - 1))]}" 2>/dev/null &&
             ! grep -q '^GENERATED ' "$work/agent.$index"; then
            generating=$((generating + 1))
          fi
        done
        break
      fi
    fi
    sleep 0.02
  done
  kill -9 "${live_pids[@]}" 2>/dev/null || true
  live_pids=()
  {
    printf 'CASE critical_mig=%s placement=%s fault=%s background=%s\n' \
      "${critical_mig:4:8}" "$placement" "$fault" "$count"
    if [[ "$fault" != none ]]; then
      printf 'FAULT victim=%s fault_ms=%s critical_alive=%s background_alive=%s probe_exit=%s probe=%s\n' \
        "$victim" "$fault_ms" "$critical_alive" "$background_alive" "$probe_exit" "$probe_line"
    fi
    local code=0 output="$work/agent.0" text result released decoded_ms
    wait "$critical_pid" 2>/dev/null || code=$?
    result=$(grep '^RESULT ' "$output" || true)
    text=$({ grep '^TEXT ' "$output" || true; } | sha256sum | cut -c1-16)
    if (( code != 0 )) || [[ -z "$result" ]]; then
      text=none
      sed 's/^/STDERR /' "$output.err" | tail -n 3
    fi
    released=$(last_of since_start_ms RELEASED "$output")
    decoded_ms=$(last_of since_start_ms DECODED "$output")
    printf 'CRITICAL exit=%s completed=%s text=%s load_ms=%s released_ms=%s decoded_ms=%s input_ms=%s generation_ms=%s generated=%s\n' \
      "$code" "$([[ $code -eq 0 && -n "$result" ]] && echo 1 || echo 0)" "$text" \
      "$(field_of load_ms "$output")" "${released:-0}" "${decoded_ms:-0}" \
      "$(last_of step_ms DECODED "$output")" "$(last_of step_ms GENERATED "$output")" \
      "$(last_of tokens GENERATED "$output")"
    for index in $(seq 1 "$count"); do
      wait "${background_pids[$((index - 1))]}" 2>/dev/null || true
    done
    printf 'BACKGROUND started=%s others=%s generating_at_end=%s mig=%s\n' "$count" \
      "$(( count - (victim_index > 0 ? 1 : 0) ))" "$generating" "${background_mig:4:8}"
    printf 'MEMORY resident_drop_mib=%s lowest_drop_mib=%s mem_available_before_mib=%s\n' \
      "$(( resident_lowest > 0 ? (before - resident_lowest) / 1024 : 0 ))" \
      "$(( (before - lowest) / 1024 ))" "$(( before / 1024 ))"
    printf 'END_CASE guard=%s timed_out=%s died_early=%s mps_servers=%s critical_s=%s wall_s=%s\n' \
      "$guard" "$timed_out" "$died" \
      "$(pgrep -fc '(^|/)nvidia-cuda-mps-server( |$)' || true)" \
      "$(awk -v a="$t_go" -v b="$t_done" 'BEGIN { printf "%.2f", ((b > 0 && a > 0) ? (b - a) / 1e9 : 0) }')" \
      "$(awk -v a="$case_start" -v b="$(date +%s%N)" 'BEGIN { printf "%.1f", (b - a) / 1e9 }')"
  } >>"$raw_log"
  stop_mps
  rm -f "$work"/agent.* "$work/go.critical" "$work/go.background" "$work"/probe*
}

# The cases of one repetition, in an order that is reversed in every second.
cases=()
for critical in $critical_list; do
  cases+=("$critical alone none")
  for placement in $placement_list; do
    for fault in $fault_list; do
      cases+=("$critical $placement $fault")
    done
  done
done

for run in $(seq 1 "$repetitions"); do
  order=()
  for item in "${cases[@]}"; do
    if (( run % 2 == 1 )); then order+=("$item"); else order=("$item" "${order[@]}"); fi
  done
  rm -f "$work/state.stock"
  sync
  sudo -n sh -c 'echo 1 >/proc/sys/vm/drop_caches'
  cat "$model" >/dev/null
  # The prefix is computed once and saved as a state file of the engine.
  env -u CUDA_MPS_PIPE_DIRECTORY CUDA_VISIBLE_DEVICES="$mig_a" \
    "$driver" "$model" "$context" 0 "text:$prefix" "save:$work/state.stock" \
    >"$work/publish.out" 2>"$work/publish.err" || true
  tokens=$(field_of tokens "$work/publish.out")
  printf 'BEGIN_KVPLACE run=%s context=%s prefix_tokens=%s state_mib=%s\n' \
    "$run" "$context" "${tokens:-0}" \
    "$(( $(stat -c %s "$work/state.stock" 2>/dev/null || echo 0) >> 20 ))" >>"$raw_log"
  for item in "${order[@]}"; do
    read -r critical placement fault <<<"$item"
    if [[ "$critical" == a ]]; then critical_mig=$mig_a; else critical_mig=$mig_b; fi
    run_case "$critical_mig" "$placement" "$fault"
  done
  printf 'END_KVPLACE\n' >>"$raw_log"
  rm -f "$work/state.stock"
done

awk -f "$script_dir/summarize_engine_kvplace.awk" "$raw_log" \
  >"$result_dir/engine_kvplace_summary.csv"
(
  cd "$script_dir"
  sha256sum kv_agent.cpp cuda_protect_probe.cu summarize_engine_kvplace.awk \
    run_engine_kvplace.sh \
    "$(realpath --relative-to=. "$prefix_file")" \
    "$(realpath --relative-to=. "$tasks_file")"
) >"$result_dir/source_hashes.txt"
printf 'engine_kvplace_result_dir=%s\n' "$result_dir"
