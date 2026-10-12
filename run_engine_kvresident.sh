#!/usr/bin/env bash
# Resident agents and released agents. A publisher computes the key-value
# cache of a prefix and exits. N agent processes, every second one in the
# other MIG instance, attach to the prefix, decode their task and then wait:
# they are resident. The runner releases ACTIVE of them with an input (the
# result of a tool call); each of these decodes its input and generates N_GEN
# tokens while the other N - ACTIVE agents keep waiting. The weights are read
# in place and the prefix is mapped, as in the stack `both` of
# run_engine_kvstack.sh.
#
# The question is what the resident agents cost the ones that generate: the
# capacity of run_engine_kvstack.sh counts agents that all generate at once,
# and the GPU divides its time among them. Here the count of resident agents
# grows and the count of generating agents stays at ACTIVE.
#
# Measured for every count N:
#   the memory that the N resident agents hold before any of them is
#   released (the largest drop of MemAvailable; the model file and the cache
#   file exist before the case and are reported apart), with the sums of
#   Pss_Anon, Pss_File and Pss_Shmem over the agents and their device memory
#   by nvidia-smi, sampled once when all of them wait;
#   for the released agents, the time from the release until the input is
#   decoded (the next token can then be chosen) and the generation rate.
#
# Before every repetition the page cache is dropped and the model file is
# read again, as in run_engine_kvstack.sh. A case is skipped, and logged as
# skipped, when starting it would leave less than RESERVE_GIB of available
# memory by the estimate; while a case runs, it is ended when the available
# memory without the model file falls below GUARD_GIB.
#
# Kill gates, fixed before the campaign:
#   R1  no process fails in any case, no case is skipped, and no case is
#       ended by the memory guard or by the time limit;
#   R2  within a repetition, a released agent writes the same text at every
#       count of resident agents;
#   R3  the summed generation rate of the released agents at the largest
#       count is at least 0.95 of that at the smallest count (paired within
#       a repetition, geometric mean over the repetitions);
#   R4  at least RESERVE_GIB of memory remain available in every case, with
#       the mapped model file counted as used.
# Reported, not gated: the time from the release to the decoded input, and
# the generation rate by MIG instance.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
counts=${COUNTS:-"8 32 64 140"}
active=${ACTIVE:-8}
prefix_file=${PREFIX_FILE:-"$script_dir/workloads/agent_prefix.txt"}
tasks_file=${TASKS_FILE:-"$script_dir/workloads/agent_tasks.txt"}
context=${CONTEXT:-16384}
n_gen=${N_GEN:-64}
grow=${GROW_ROWS:-256}
limit_s=${CASE_LIMIT_S:-1200}
settle_s=${SETTLE_S:-2}
reserve_gib=${RESERVE_GIB:-16}
guard_gib=${GUARD_GIB:-13}
# Estimate of the memory of one resident agent, in MiB, for the skip only.
estimate=${ESTIMATE_MIB:-730}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
engine=${KV_ENGINE:-"$script_dir/llama.cpp-kv2"}
publisher=${PUBLISHER:-"$script_dir/kv_fork2"}
driver=${DRIVER:-"$script_dir/kv_agent2"}
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-kvresident"}
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
for count in $counts; do
  if (( count < active )); then echo "a count is below ACTIVE" >&2; exit 2; fi
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
# value of name=VALUE on the last line of a kind in a file
last_of() {  # name line-prefix file
  { grep "^$2 " "$3" || true; } | tail -n 1 | sed -n "s/^.* $1=\\([^ ]*\\).*\$/\\1/p"
}
meminfo_kib() {  # name
  awk -v name="$1:" '$1 == name { print $2 }' /proc/meminfo
}

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\ncounts=%s\nactive=%s\n' "$mig_a" \
    "$mig_b" "$repetitions" "$counts" "$active"
  printf 'context=%s\nn_gen=%s\ngrow_rows=%s\nsettle_s=%s\n' "$context" "$n_gen" \
    "$grow" "$settle_s"
  printf 'prefix_file=%s\nprefix_sha256=%s\ntasks_file=%s\ntasks_sha256=%s\n' \
    "$(basename "$prefix_file")" "$(sha256sum "$prefix" | cut -d' ' -f1)" \
    "$(basename "$tasks_file")" "$(sha256sum "$tasks_file" | cut -d' ' -f1)"
  printf 'reserve_gib=%s\nguard_gib=%s\nestimate_mib=%s\n' "$reserve_gib" \
    "$guard_gib" "$estimate"
  printf 'model=%s\nmodel_bytes=%s\n' "$(basename "$model")" "$(stat -c %s "$model")"
  printf 'mem_total_kib=%s\n' "$(meminfo_kib MemTotal)"
  printf 'engine_commit=%s\n' "$(git -C "$engine" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$driver" "$publisher" "$engine/build/bin/libllama.so" \
    "$engine/build/bin/libggml-cuda.so"
} >"$result_dir/metadata.txt"

# The sums over the agents: three lines of smaps_rollup, and the device
# memory of the processes by nvidia-smi.
sample_agents() {  # pid...
  local rollups=() pid
  for pid in "$@"; do rollups+=("/proc/$pid/smaps_rollup"); done
  # shellcheck disable=SC2002
  cat "${rollups[@]}" 2>/dev/null | awk '
    /^Pss_Anon:/ { anon += $2 }
    /^Pss_File:/ { file += $2 }
    /^Pss_Shmem:/ { shmem += $2 }
    END { printf "%d %d %d", anon / 1024, file / 1024, shmem / 1024 }' || true
  printf ' '
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
    2>/dev/null | awk -F', *' -v list="$*" '
    BEGIN { n = split(list, pids, " "); for (i = 1; i <= n; ++i) mine[pids[i]] = 1 }
    ($1 in mine) && $2 ~ /^[0-9]+$/ { total += $2 }
    END { printf "%d", total }' || true
}

run_case() {  # residents prefix-tokens first-mig other-mig
  local residents=$1 tokens=$2 first=$3 other=$4
  local index mig pids=() migs=() roles=() failed=0
  local before lowest sample
  before=$(meminfo_kib MemAvailable)
  if (( before / 1024 - residents * estimate < reserve_gib * 1024 )); then
    printf 'SKIPPED residents=%s mem_available_mib=%s\n' "$residents" \
      "$(( before / 1024 ))" >>"$raw_log"
    return 0
  fi
  lowest=$before
  rm -f "$work"/agent.* "$work/go.active" "$work/go.idle"
  local case_start
  case_start=$(date +%s%N)
  for index in $(seq 0 $((residents - 1))); do
    if (( index % 2 == 0 )); then mig=$first; else mig=$other; fi
    migs+=("$mig")
    local steps=("load:$work/state.huge" "say:$(suffix_of "$index")" mark:ready)
    if (( index < active )); then
      roles+=(active)
      steps+=("wait:$work/go.active" "say:$(input_of "$index")" gen)
    else
      roles+=(idle)
      steps+=("wait:$work/go.idle")
    fi
    env CUDA_VISIBLE_DEVICES="$mig" GGML_CUDA_HOST_PTR=1 \
      LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_PREFIX="$tokens" \
      LLAMA_KV_GROW="$grow" "$driver" "$model" "$context" "$n_gen" \
      "${steps[@]}" >"$work/agent.$index" 2>"$work/agent.$index.err" &
    pids+=($!)
  done
  live_pids=("${pids[@]}")

  # phase: 0 until every agent waits, 1 while the released agents run,
  # 2 while the others leave
  local phase=0 guard=0 timed_out=0 alive ready running tick=0
  local t_ready=0 t_go=0 t_done=0 resident_lowest=0
  local parts=() anon=0 file=0 shmem=0 device=0
  while :; do
    alive=0
    for index in "${!pids[@]}"; do
      if kill -0 "${pids[$index]}" 2>/dev/null; then alive=$((alive + 1)); fi
    done
    sample=$(meminfo_kib MemAvailable)
    (( sample < lowest )) && lowest=$sample
    if (( (sample - model_kib) / 1024 < guard_gib * 1024 )); then
      guard=1
      kill -9 "${pids[@]}" 2>/dev/null || true
      break
    fi
    if (( ($(date +%s%N) - case_start) / 1000000000 > limit_s )); then
      timed_out=1
      kill -9 "${pids[@]}" 2>/dev/null || true
      break
    fi
    if (( phase == 0 )); then
      if (( alive < residents )); then break; fi
      if (( tick % 5 == 0 )); then
        ready=$( { grep -l '^MARK ' "$work"/agent.*[0-9] 2>/dev/null || true; } | wc -l)
        if (( ready == residents )); then
          t_ready=$(date +%s%N)
          sleep "$settle_s"
          sample=$(meminfo_kib MemAvailable)
          (( sample < lowest )) && lowest=$sample
          resident_lowest=$lowest
          read -r -a parts <<<"$(sample_agents "${pids[@]}")"
          if (( ${#parts[@]} == 4 )); then
            anon=${parts[0]} file=${parts[1]} shmem=${parts[2]} device=${parts[3]}
          fi
          t_go=$(date +%s%N)
          touch "$work/go.active"
          phase=1
        fi
      fi
    elif (( phase == 1 )); then
      running=0
      for index in $(seq 0 $((active - 1))); do
        if kill -0 "${pids[$index]}" 2>/dev/null; then running=$((running + 1)); fi
      done
      if (( running == 0 )); then
        t_done=$(date +%s%N)
        touch "$work/go.idle"
        phase=2
      fi
    else
      if (( alive == 0 )); then break; fi
    fi
    tick=$((tick + 1))
    sleep 0.02
  done
  # Whatever is left is ended before the results are read.
  kill -9 "${pids[@]}" 2>/dev/null || true
  live_pids=()
  {
    printf 'CASE residents=%s active=%s\n' "$residents" "$active"
    for index in "${!pids[@]}"; do
      local code=0 output="$work/agent.$index" text result released decoded
      wait "${pids[$index]}" 2>/dev/null || code=$?
      result=$(grep '^RESULT ' "$output" || true)
      text=$({ grep '^TEXT ' "$output" || true; } | sha256sum | cut -c1-16)
      if (( code != 0 )) || [[ -z "$result" ]]; then
        failed=$((failed + 1))
        text=none
        sed 's/^/STDERR /' "$output.err" | tail -n 5
      fi
      released=$(last_of since_start_ms RELEASED "$output")
      decoded=$(last_of since_start_ms DECODED "$output")
      printf 'AGENT index=%s mig=%s role=%s exit=%s text=%s attach_ms=%s ready_ms=%s released_ms=%s decoded_ms=%s input_ms=%s generation_ms=%s generated=%s\n' \
        "$index" "${migs[$index]:4:8}" "${roles[$index]}" "$code" "$text" \
        "$(field_of load_ms "$output")" "$(last_of since_start_ms MARK "$output")" \
        "${released:-0}" "${decoded:-0}" \
        "$(last_of step_ms DECODED "$output")" \
        "$(last_of step_ms GENERATED "$output")" \
        "$(last_of tokens GENERATED "$output")"
    done
    printf 'MEMORY resident_drop_mib=%s lowest_drop_mib=%s mem_available_before_mib=%s remaining_without_model_mib=%s ' \
      "$(( resident_lowest > 0 ? (before - resident_lowest) / 1024 : 0 ))" \
      "$(( (before - lowest) / 1024 ))" "$(( before / 1024 ))" \
      "$(( (lowest - model_kib) / 1024 ))"
    printf 'pss_anon_mib=%s pss_file_mib=%s pss_shmem_mib=%s device_mib=%s\n' \
      "$anon" "$file" "$shmem" "$device"
    printf 'END_CASE failed=%s guard=%s timed_out=%s ready_s=%s active_s=%s wall_s=%s\n' \
      "$failed" "$guard" "$timed_out" \
      "$(awk -v a="$case_start" -v b="$t_ready" 'BEGIN { printf "%.1f", (b > 0 ? (b - a) / 1e9 : 0) }')" \
      "$(awk -v a="$t_go" -v b="$t_done" 'BEGIN { printf "%.2f", (b > 0 ? (b - a) / 1e9 : 0) }')" \
      "$(awk -v a="$case_start" -v b="$(date +%s%N)" 'BEGIN { printf "%.1f", (b - a) / 1e9 }')"
  } >>"$raw_log"
  rm -f "$work"/agent.* "$work/go.active" "$work/go.idle"
}

for run in $(seq 1 "$repetitions"); do
  order=()
  for count in $counts; do
    if (( run % 2 == 1 )); then order+=("$count"); else order=("$count" "${order[@]}"); fi
  done
  if (( run % 2 == 1 )); then first=$mig_a other=$mig_b; else first=$mig_b other=$mig_a; fi
  rm -f "$huge_dir/kv.0" "$work/state.huge"
  sync
  sudo -n sh -c 'echo 1 >/proc/sys/vm/drop_caches'
  cat "$model" >/dev/null
  env CUDA_VISIBLE_DEVICES="$first" GGML_CUDA_HOST_PTR=1 \
    LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_GROW="$grow" \
    "$publisher" parent "$model" "$context" "$prefix" "$work/state.huge" "" 0 \
    >"$work/publish.out" 2>"$work/publish.err" || true
  # The published rows are not written again by anyone.
  chmod 0400 "$huge_dir/kv.0"
  tokens=$(field_of prefix_tokens "$work/publish.out")
  {
    printf 'BEGIN_KVRESIDENT run=%s context=%s publisher_mig=%s prefix_tokens=%s ' \
      "$run" "$context" "${first:4:8}" "${tokens:-0}"
    printf 'file_used_mib=%s mem_free_mib=%s kernel_reclaimable_mib=%s\n' \
      "$(( $(du -k "$huge_dir/kv.0" | cut -f1) / 1024 ))" \
      "$(( $(meminfo_kib MemFree) / 1024 ))" \
      "$(( $(meminfo_kib KReclaimable) / 1024 ))"
  } >>"$raw_log"
  for count in "${order[@]}"; do
    run_case "$count" "$tokens" "$first" "$other"
  done
  printf 'END_KVRESIDENT\n' >>"$raw_log"
  rm -f "$huge_dir/kv.0" "$work/state.huge"
done

awk -f "$script_dir/summarize_engine_kvresident.awk" "$raw_log" \
  >"$result_dir/engine_kvresident_summary.csv"
(
  cd "$script_dir"
  sha256sum kv_extents2.patch kv_agent.cpp kv_fork.cpp \
    summarize_engine_kvresident.awk run_engine_kvresident.sh \
    "$(realpath --relative-to=. "$prefix_file")" \
    "$(realpath --relative-to=. "$tasks_file")"
) >"$result_dir/source_hashes.txt"
printf 'engine_kvresident_result_dir=%s\n' "$result_dir"
