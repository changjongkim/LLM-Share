#!/usr/bin/env bash
# Agents on one prefix, at scale and with a chosen workload. A publisher
# computes the key-value cache of a prefix and exits; N agent processes,
# every second one in the other MIG instance, continue it with their own
# task. Two ways to obtain the prefix:
#
#   restore      the agent copies it from the engine's state file
#   extent_lazy  the agent maps the rows of the prefix read-only and keeps
#                the rows it writes in private memory that follows use
#
# The prefix is PARAGRAPHS copies of one paragraph, or the text of
# PREFIX_FILE; the task of agent i is "summarize ... in i+2 sentences", or
# line i+1 of TASKS_FILE. The log has the format of run_engine_kvshare.sh
# and is summarized by the same script. The weights are read in place.
#
# A case is skipped, and logged as skipped, when starting it would leave
# less than RESERVE_GIB of available memory by the estimate of
# ESTIMATE_MIB_PER_AGENT for restore.
#
# Kill gates, fixed before the campaign:
#   S1  no agent fails in any case that is not skipped;
#   S2  every agent on extents writes the text of the agent that restores;
#   S3  the memory of the agents on extents grows by less per added agent
#       than that of the agents that restore, at every step of the count;
#   S4  the summed generation speed on extents is at least 0.92 of that of
#       restore at every count (half of the agents run in the 6-SM
#       instance, where a cache in host memory is known to cost up to 5.8%).
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
agent_counts=${AGENT_COUNTS:-"8 16 32"}
paragraphs=${PARAGRAPHS:-320}
prefix_file=${PREFIX_FILE:-}
tasks_file=${TASKS_FILE:-}
context=${CONTEXT:-}
n_gen=${N_GEN:-64}
grow=${GROW_ROWS:-256}
limit_s=${CASE_LIMIT_S:-1200}
reserve_gib=${RESERVE_GIB:-24}
estimate_mib=${ESTIMATE_MIB_PER_AGENT:-2600}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
engine=${KV_ENGINE:-"$script_dir/llama.cpp-kv"}
driver=${DRIVER:-"$script_dir/kv_fork"}
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-kvscale"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
if [[ ! -x "$driver" || ! -r "$model" ]]; then
  echo "missing driver or model; run make and see README.md" >&2
  exit 2
fi
for file in "$prefix_file" "$tasks_file"; do
  if [[ -n "$file" && ! -r "$file" ]]; then echo "cannot read $file" >&2; exit 2; fi
done
if ! [[ "$paragraphs" =~ ^[1-9][0-9]*$ ]]; then
  echo "PARAGRAPHS must be one positive integer (it also labels a PREFIX_FILE)" >&2
  exit 2
fi
if ! sudo -n true 2>/dev/null; then
  echo "passwordless sudo is required for the tmpfs mount" >&2
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

paragraph="The scheduler keeps a run queue for every processor and moves a task between queues when the load is uneven. A page fault enters the kernel, which finds the mapping, allocates a frame, fills it, and updates the page table before the task resumes. "
prefix="$work/prefix"
if [[ -n "$prefix_file" ]]; then
  cp "$prefix_file" "$prefix"
else
  : >"$prefix"
  for _ in $(seq 1 "$paragraphs"); do printf '%s' "$paragraph" >>"$prefix"; done
fi
if [[ -z "$context" ]]; then
  # The power of two that leaves room after the prefix: a paragraph is 52
  # tokens, and other text is taken as one token per three bytes.
  if [[ -n "$prefix_file" ]]; then
    need=$(( $(stat -c %s "$prefix") / 3 + 1024 ))
  else
    need=$(( paragraphs * 52 + 512 ))
  fi
  context=4096
  while (( context < need )); do context=$(( context * 2 )); done
fi
suffix_of() {  # agent index
  if [[ -n "$tasks_file" ]]; then
    local lines line
    lines=$(wc -l <"$tasks_file")
    line=$(sed -n "$(( $1 % lines + 1 ))p" "$tasks_file")
    printf '\n\nTask for agent %s: %s\n' "$1" "$line"
  else
    printf '\n\nTask for agent %s: summarize the text above in %s sentences.\n' \
      "$1" "$(( $1 + 2 ))"
  fi
}
field_of() {  # name file: value of name=VALUE on the first line that has it
  sed -n "s/^.* $1=\\([^ ]*\\).*\$/\\1/p" "$2" | head -n 1
}

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\nagent_counts=%s\n' \
    "$mig_a" "$mig_b" "$repetitions" "$agent_counts"
  printf 'paragraphs=%s\ncontext=%s\nn_gen=%s\ngrow_rows=%s\n' "$paragraphs" \
    "$context" "$n_gen" "$grow"
  printf 'prefix_file=%s\nprefix_sha256=%s\ntasks_file=%s\n' \
    "${prefix_file:-none}" "$(sha256sum "$prefix" | cut -d' ' -f1)" \
    "${tasks_file:-none}"
  printf 'reserve_gib=%s\nestimate_mib_per_agent=%s\n' "$reserve_gib" "$estimate_mib"
  printf 'model=%s\n' "$(basename "$model")"
  printf 'engine_commit=%s\n' "$(git -C "$engine" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$driver" "$engine/build/bin/libllama.so" \
    "$engine/build/bin/libggml-cuda.so"
} >"$result_dir/metadata.txt"

publish() {  # label mig state-file env...
  local label=$1 mig=$2 state=$3
  shift 3
  rm -f "$state"
  env CUDA_VISIBLE_DEVICES="$mig" GGML_CUDA_HOST_PTR=1 "$@" \
    "$driver" parent "$model" "$context" "$prefix" "$state" "" 0 \
    >"$work/publish.out" 2>"$work/publish.err" || true
  local line
  line=$(grep '^PUBLISHED ' "$work/publish.out" || true)
  printf 'PUBLISH store=%s %s' "$label" "${line#PUBLISHED }"
}

run_case() {  # mode agents prefix-tokens first-mig other-mig
  local mode=$1 agents=$2 tokens=$3 first=$4 other=$5
  local state envs=() store="" index mig pids=() migs=() starts=() ends=() failed=0
  if [[ "$mode" == restore ]]; then
    state="$work/state.device"
  else
    state="$work/state.huge"
    envs=(LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_PREFIX="$tokens" LLAMA_KV_GROW="$grow")
    store="$huge_dir/kv.0"
  fi
  local before lowest sample pss=0 rss=0
  before=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
  if (( before / 1024 - agents * estimate_mib < reserve_gib * 1024 )); then
    printf 'SKIPPED mode=%s agents=%s mem_available_mib=%s\n' "$mode" "$agents" \
      "$(( before / 1024 ))" >>"$raw_log"
    return 0
  fi
  lowest=$before
  local case_start
  case_start=$(date +%s%N)
  for index in $(seq 0 $((agents - 1))); do
    if (( index % 2 == 0 )); then mig=$first; else mig=$other; fi
    migs+=("$mig")
    starts+=("$(date +%s%N)")
    env CUDA_VISIBLE_DEVICES="$mig" GGML_CUDA_HOST_PTR=1 "${envs[@]}" \
      "$driver" child "$model" "$context" "$state" "$(suffix_of "$index")" \
      "$n_gen" >"$work/agent.$index" 2>"$work/agent.$index.err" &
    pids+=($!)
    ends+=(0)
  done
  live_pids=("${pids[@]}")
  # The lowest available memory while the agents run, the pages of the
  # publisher's cache file that the agents map, and when each agent leaves.
  local alive=$agents tick=0 smaps=()
  for index in "${pids[@]}"; do smaps+=("/proc/$index/smaps"); done
  while (( alive > 0 )); do
    alive=0
    for index in "${!pids[@]}"; do
      if (( ends[index] == 0 )); then
        if kill -0 "${pids[$index]}" 2>/dev/null; then
          alive=$((alive + 1))
        else
          ends[index]=$(date +%s%N)
        fi
      fi
    done
    if (( tick % 5 == 0 )); then
      sample=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
      (( sample < lowest )) && lowest=$sample
    fi
    if [[ -n "$store" ]] && (( tick % 25 == 5 )); then
      # An agent may leave between the liveness test and the read.
      # shellcheck disable=SC2002
      sample=$(cat "${smaps[@]}" 2>/dev/null |
        awk -v file="$store" '
          /^[0-9a-f]+-[0-9a-f]+ / { inside = ($6 == file) }
          inside && /^Pss:/ { pss += $2 }
          inside && /^Rss:/ { rss += $2 }
          END { printf "%d %d", pss / 1024, rss / 1024 }' || true)
      if [[ -n "$sample" ]]; then
        (( ${sample% *} > pss )) && pss=${sample% *}
        (( ${sample#* } > rss )) && rss=${sample#* }
      fi
    fi
    if (( ($(date +%s%N) - case_start) / 1000000000 > limit_s )); then
      kill -9 "${pids[@]}" 2>/dev/null || true
    fi
    tick=$((tick + 1))
    sleep 0.02
  done
  live_pids=()
  {
    printf 'CASE mode=%s agents=%s\n' "$mode" "$agents"
    for index in "${!pids[@]}"; do
      local code=0 text result
      wait "${pids[$index]}" || code=$?
      result=$(grep '^RESULT ' "$work/agent.$index" || true)
      text=$({ grep '^TEXT ' "$work/agent.$index" || true; } | sha256sum | cut -c1-16)
      if (( code != 0 )) || [[ -z "$result" ]]; then
        failed=$((failed + 1))
        text=none
        sed 's/^/STDERR /' "$work/agent.$index.err" | tail -n 5
      fi
      printf 'AGENT index=%s mig=%s exit=%s wall_ms=%s text=%s %s\n' "$index" \
        "${migs[$index]:4:8}" "$code" \
        "$(( (ends[index] - starts[index]) / 1000000 ))" "$text" \
        "${result#RESULT }"
    done
    printf 'MEMORY mem_available_drop_mib=%s prefix_pss_mib=%s prefix_rss_mib=%s\n' \
      "$(( (before - lowest) / 1024 ))" "$pss" "$rss"
    printf 'END_CASE failed_agents=%s\n' "$failed"
  } >>"$raw_log"
}

for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then
    first=$mig_a other=$mig_b modes=(restore extent_lazy)
  else
    first=$mig_b other=$mig_a modes=(extent_lazy restore)
  fi
  rm -f "$huge_dir/kv.0"
  {
    printf 'BEGIN_KVSHARE run=%s paragraphs=%s context=%s publisher_mig=%s\n' \
      "$run" "$paragraphs" "$context" "${first:4:8}"
    publish device "$first" "$work/state.device"
    printf '\n'
    publish huge "$first" "$work/state.huge" \
      LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_GROW="$grow"
    printf ' file_used_kib=%s\n' "$(du -k "$huge_dir/kv.0" | cut -f1)"
  } >>"$raw_log"
  # The published rows are not written again by anyone.
  chmod 0400 "$huge_dir/kv.0"
  tokens=$(field_of prefix_tokens "$work/publish.out")
  for agents in $agent_counts; do
    for mode in "${modes[@]}"; do
      run_case "$mode" "$agents" "$tokens" "$first" "$other"
    done
  done
  printf 'END_KVSHARE\n' >>"$raw_log"
  rm -f "$huge_dir/kv.0" "$work"/state.*
done

awk -f "$script_dir/summarize_engine_kvshare.awk" "$raw_log" \
  >"$result_dir/engine_kvscale_summary.csv"
awk -v table=publish -f "$script_dir/summarize_engine_kvshare.awk" "$raw_log" \
  >"$result_dir/engine_kvscale_publish.csv"
(
  cd "$script_dir"
  sha256sum kv_extents.patch kv_fork.cpp summarize_engine_kvshare.awk \
    run_engine_kvscale.sh
) >"$result_dir/source_hashes.txt"
printf 'engine_kvscale_result_dir=%s\n' "$result_dir"
