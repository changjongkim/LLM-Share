#!/usr/bin/env bash
# What a GPU partition gives an agent that must answer in time. One agent is
# latency-critical: it waits for an input and then decodes it and generates
# N_GEN tokens. K background agents generate at the same time, either in the
# MIG instance of the critical agent, where the processes are time-sliced, or
# in the other MIG instance. All agents are separate processes on one
# published prefix; the weights are read in place and the prefix is mapped
# (the stack `both` of run_engine_kvstack.sh), so the placement changes where
# an agent computes and not what it holds.
#
# Cases, for the critical agent in the 12-SM instance and in the other one:
#
#   alone        no background agent
#   shared K     K background agents in the instance of the critical agent
#   split K      K background agents in the other instance
#
# A case runs as follows. Every agent attaches to the prefix, decodes its
# task and waits. The background agents are released first with an input and
# generate BG_GEN tokens; WARM_S seconds after all of them have decoded their
# input, the critical agent is released. When it has finished, the background
# agents are ended by the runner; they only produce load. A case counts only
# if every background agent was still generating when the critical agent
# finished.
#
# Measured for the critical agent: the time from its release until its input
# is decoded (the next token can then be chosen), and the time to generate
# N_GEN tokens. The memory of a case is the largest drop of MemAvailable
# while its agents wait.
#
# Kill gates, fixed before the campaign:
#   L1  no critical agent fails, no case is ended by the time limit or the
#       memory guard, and in every case every background agent is still
#       generating when the critical agent finishes;
#   L2  within a repetition, the critical agent writes the same text in every
#       case in which it runs in the same instance;
#   L3  with the largest K in its own instance, the critical agent takes at
#       least 2.0 times as long to generate as alone, in both instances
#       (paired within a repetition, geometric mean over the repetitions);
#   L4  with the largest K in the other instance, the critical agent
#       generates at no less than 0.90 of its rate alone, in both instances
#       (paired in the same way).
# Reported, not gated: the time from the release to the decoded input, and
# the memory by placement.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
background_counts=${BACKGROUND_COUNTS:-"1 3 7"}
prefix_file=${PREFIX_FILE:-"$script_dir/workloads/agent_prefix.txt"}
tasks_file=${TASKS_FILE:-"$script_dir/workloads/agent_tasks.txt"}
context=${CONTEXT:-16384}
n_gen=${N_GEN:-64}
bg_gen=${BG_GEN:-320}
grow=${GROW_ROWS:-256}
limit_s=${CASE_LIMIT_S:-600}
settle_s=${SETTLE_S:-2}
warm_s=${WARM_S:-1}
guard_gib=${GUARD_GIB:-16}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
engine=${KV_ENGINE:-"$script_dir/llama.cpp-kv2"}
publisher=${PUBLISHER:-"$script_dir/kv_fork2"}
driver=${DRIVER:-"$script_dir/kv_agent2"}
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-kvlatency"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
if [[ ! -x "$driver" || ! -x "$publisher" || ! -r "$model" ]]; then
  echo "missing driver or model; run make kv_agent2 kv_fork2" >&2
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

huge_dir="$mount_root/kv_huge"
mounted_huge=0
work=
live_pids=()
cleanup() {
  if (( ${#live_pids[@]} )); then kill -9 "${live_pids[@]}" 2>/dev/null || true; fi
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
prefix="$work/prefix"
cp "$prefix_file" "$prefix"
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
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\nbackground_counts=%s\n' "$mig_a" \
    "$mig_b" "$repetitions" "$background_counts"
  printf 'context=%s\nn_gen=%s\nbg_gen=%s\ngrow_rows=%s\nsettle_s=%s\nwarm_s=%s\n' \
    "$context" "$n_gen" "$bg_gen" "$grow" "$settle_s" "$warm_s"
  printf 'prefix_file=%s\nprefix_sha256=%s\ntasks_file=%s\ntasks_sha256=%s\n' \
    "$(basename "$prefix_file")" "$(sha256sum "$prefix" | cut -d' ' -f1)" \
    "$(basename "$tasks_file")" "$(sha256sum "$tasks_file" | cut -d' ' -f1)"
  printf 'guard_gib=%s\n' "$guard_gib"
  printf 'model=%s\nmodel_bytes=%s\n' "$(basename "$model")" "$(stat -c %s "$model")"
  printf 'mem_total_kib=%s\n' "$(meminfo_kib MemTotal)"
  printf 'engine_commit=%s\n' "$(git -C "$engine" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$driver" "$publisher" "$engine/build/bin/libllama.so" \
    "$engine/build/bin/libggml-cuda.so"
} >"$result_dir/metadata.txt"

# Starts one agent in the background and prints nothing; $! is its process.
start_agent() {  # mig index tokens-to-generate go-file output
  env CUDA_VISIBLE_DEVICES="$1" GGML_CUDA_HOST_PTR=1 \
    LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_PREFIX="$tokens" \
    LLAMA_KV_GROW="$grow" "$driver" "$model" "$context" "$3" \
    "load:$work/state.huge" "say:$(suffix_of "$2")" mark:ready \
    "wait:$4" "say:$(input_of "$2")" gen >"$5" 2>"$5.err" &
}

run_case() {  # critical-mig placement background-count
  local critical_mig=$1 placement=$2 background=$3 other_mig background_mig index
  if [[ "$critical_mig" == "$mig_a" ]]; then other_mig=$mig_b; else other_mig=$mig_a; fi
  if [[ "$placement" == split ]]; then background_mig=$other_mig; else background_mig=$critical_mig; fi
  rm -f "$work"/agent.* "$work/go.critical" "$work/go.background"
  local before lowest sample case_start
  before=$(meminfo_kib MemAvailable)
  lowest=$before
  case_start=$(date +%s%N)
  # Agent 0 is the critical one, so that it has one task in every case.
  start_agent "$critical_mig" 0 "$n_gen" "$work/go.critical" "$work/agent.0"
  local critical_pid=$! background_pids=()
  for index in $(seq 1 "$background"); do
    start_agent "$background_mig" "$index" "$bg_gen" "$work/go.background" "$work/agent.$index"
    background_pids+=($!)
  done
  live_pids=("$critical_pid" "${background_pids[@]}")

  # phase: 0 until every agent waits, 1 until the background agents generate,
  # 2 while the critical agent runs
  local phase=0 guard=0 timed_out=0 died=0 ready decoded tick=0 pid
  local resident_lowest=0 generating=0 t_go=0 t_done=0
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
      if (( ready == background + 1 )); then
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
      for index in $(seq 1 "$background"); do
        if (( $(grep -c '^DECODED ' "$work/agent.$index" || true) >= 2 )); then
          decoded=$((decoded + 1))
        fi
      done
      if (( decoded == background )); then
        sleep "$warm_s"
        t_go=$(date +%s%N)
        touch "$work/go.critical"
        phase=2
      fi
    else
      if ! kill -0 "$critical_pid" 2>/dev/null; then
        t_done=$(date +%s%N)
        for index in $(seq 1 "$background"); do
          if kill -0 "${background_pids[$((index - 1))]}" 2>/dev/null &&
             ! grep -q '^GENERATED ' "$work/agent.$index"; then
            generating=$((generating + 1))
          fi
        done
        break
      fi
    fi
    tick=$((tick + 1))
    sleep 0.02
  done
  kill -9 "${live_pids[@]}" 2>/dev/null || true
  live_pids=()
  {
    printf 'CASE critical_mig=%s placement=%s background=%s\n' "${critical_mig:4:8}" \
      "$placement" "$background"
    local code=0 output="$work/agent.0" text result released decoded_ms
    wait "$critical_pid" 2>/dev/null || code=$?
    result=$(grep '^RESULT ' "$output" || true)
    text=$({ grep '^TEXT ' "$output" || true; } | sha256sum | cut -c1-16)
    if (( code != 0 )) || [[ -z "$result" ]]; then
      text=none
      sed 's/^/STDERR /' "$output.err" | tail -n 5
    fi
    released=$(last_of since_start_ms RELEASED "$output")
    decoded_ms=$(last_of since_start_ms DECODED "$output")
    printf 'CRITICAL exit=%s completed=%s text=%s attach_ms=%s released_ms=%s decoded_ms=%s input_ms=%s generation_ms=%s generated=%s\n' \
      "$code" "$([[ -n "$result" ]] && echo 1 || echo 0)" "$text" \
      "$(field_of load_ms "$output")" "${released:-0}" "${decoded_ms:-0}" \
      "$(last_of step_ms DECODED "$output")" "$(last_of step_ms GENERATED "$output")" \
      "$(last_of tokens GENERATED "$output")"
    for index in $(seq 1 "$background"); do
      wait "${background_pids[$((index - 1))]}" 2>/dev/null || true
    done
    printf 'BACKGROUND started=%s generating_at_end=%s mig=%s\n' "$background" \
      "$generating" "${background_mig:4:8}"
    printf 'MEMORY resident_drop_mib=%s lowest_drop_mib=%s mem_available_before_mib=%s\n' \
      "$(( resident_lowest > 0 ? (before - resident_lowest) / 1024 : 0 ))" \
      "$(( (before - lowest) / 1024 ))" "$(( before / 1024 ))"
    printf 'END_CASE guard=%s timed_out=%s died_early=%s critical_s=%s wall_s=%s\n' \
      "$guard" "$timed_out" "$died" \
      "$(awk -v a="$t_go" -v b="$t_done" 'BEGIN { printf "%.2f", (b > 0 ? (b - a) / 1e9 : 0) }')" \
      "$(awk -v a="$case_start" -v b="$(date +%s%N)" 'BEGIN { printf "%.1f", (b - a) / 1e9 }')"
  } >>"$raw_log"
  rm -f "$work"/agent.* "$work/go.critical" "$work/go.background"
}

# The cases of one repetition, in an order that is reversed in every second.
cases=()
for critical in a b; do
  cases+=("$critical alone 0")
  for count in $background_counts; do
    cases+=("$critical shared $count" "$critical split $count")
  done
done

for run in $(seq 1 "$repetitions"); do
  order=()
  for item in "${cases[@]}"; do
    if (( run % 2 == 1 )); then order+=("$item"); else order=("$item" "${order[@]}"); fi
  done
  rm -f "$huge_dir/kv.0" "$work/state.huge"
  sync
  sudo -n sh -c 'echo 1 >/proc/sys/vm/drop_caches'
  cat "$model" >/dev/null
  env CUDA_VISIBLE_DEVICES="$mig_a" GGML_CUDA_HOST_PTR=1 \
    LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_GROW="$grow" \
    "$publisher" parent "$model" "$context" "$prefix" "$work/state.huge" "" 0 \
    >"$work/publish.out" 2>"$work/publish.err" || true
  # The published rows are not written again by anyone.
  chmod 0400 "$huge_dir/kv.0"
  tokens=$(field_of prefix_tokens "$work/publish.out")
  printf 'BEGIN_KVLATENCY run=%s context=%s prefix_tokens=%s file_used_mib=%s\n' \
    "$run" "$context" "${tokens:-0}" \
    "$(( $(du -k "$huge_dir/kv.0" | cut -f1) / 1024 ))" >>"$raw_log"
  for item in "${order[@]}"; do
    read -r critical placement count <<<"$item"
    if [[ "$critical" == a ]]; then critical_mig=$mig_a; else critical_mig=$mig_b; fi
    run_case "$critical_mig" "$placement" "$count"
  done
  printf 'END_KVLATENCY\n' >>"$raw_log"
  rm -f "$huge_dir/kv.0" "$work/state.huge"
done

awk -f "$script_dir/summarize_engine_kvlatency.awk" "$raw_log" \
  >"$result_dir/engine_kvlatency_summary.csv"
(
  cd "$script_dir"
  sha256sum kv_extents2.patch kv_agent.cpp kv_fork.cpp \
    summarize_engine_kvlatency.awk run_engine_kvlatency.sh \
    "$(realpath --relative-to=. "$prefix_file")" \
    "$(realpath --relative-to=. "$tasks_file")"
) >"$result_dir/source_hashes.txt"
printf 'engine_kvlatency_result_dir=%s\n' "$result_dir"
