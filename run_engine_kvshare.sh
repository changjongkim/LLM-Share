#!/usr/bin/env bash
# Agents that start from one computed prompt prefix. A publisher computes the
# key-value cache of the prefix once; N agent processes, half of them in the
# other MIG instance, then continue it with their own suffix. Compares the
# ways an agent can obtain the prefix:
#
#   recompute          the agent computes the prefix itself (device memory)
#   restore            the agent copies it from the engine's state file
#   cow, cow_small     the agent maps the publisher's cache file writable and
#                      private, after a CPU read of the prefix (copy-on-write)
#   extent             the agent maps the rows of the prefix read-only and
#                      keeps the rows it writes in private memory
#   extent_lazy        the same; private memory follows the rows in use
#   extent_lazy_small  the same over a cache file with 4 KiB pages
#
# The cache file of the publisher is on a tmpfs with 2 MiB pages, or with
# 4 KiB pages for the modes named *_small. The weights are read in place in
# every mode. Kill gates, fixed before the campaign: every agent of every
# mode must produce the text of the agent that recomputes; the agents of an
# extent mode must hold one copy of the prefix between them; extent_lazy must
# keep 97% of the generation speed of restore; attaching must not take longer
# than restoring; publishing must not take longer than saving the state.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
agent_counts=${AGENT_COUNTS:-"1 4 8"}
paragraphs=${PARAGRAPHS:-"80 320"}
n_gen=${N_GEN:-64}
grow=${GROW_ROWS:-256}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
engine=${KV_ENGINE:-"$script_dir/llama.cpp-kv"}
driver=${DRIVER:-"$script_dir/kv_fork"}
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-kvshare"}
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
if ! sudo -n true 2>/dev/null; then
  echo "passwordless sudo is required for the tmpfs mounts" >&2
  exit 2
fi
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi

huge_dir="$mount_root/kv_huge"
small_dir="$mount_root/kv_small"
mounted_huge=0
mounted_small=0
work=
cleanup() {
  if (( mounted_huge )); then sudo -n umount "$huge_dir" || true; fi
  if (( mounted_small )); then sudo -n umount "$small_dir" || true; fi
  rmdir "$huge_dir" "$small_dir" "$mount_root" 2>/dev/null || true
  if [[ -n "$work" ]]; then rm -rf "$work"; fi
}
trap cleanup EXIT
mkdir -p "$huge_dir" "$small_dir" "$result_dir"
sudo -n mount -t tmpfs -o "huge=always,size=4096m,uid=$(id -u),gid=$(id -g)" \
  tmpfs "$huge_dir"
mounted_huge=1
sudo -n mount -t tmpfs -o "huge=never,size=4096m,uid=$(id -u),gid=$(id -g)" \
  tmpfs "$small_dir"
mounted_small=1
work=$(mktemp -d "$result_dir/work.XXXXXX")

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\nagent_counts=%s\n' \
    "$mig_a" "$mig_b" "$repetitions" "$agent_counts"
  printf 'paragraphs=%s\nn_gen=%s\ngrow_rows=%s\n' "$paragraphs" "$n_gen" "$grow"
  printf 'model=%s\n' "$(basename "$model")"
  printf 'engine_commit=%s\n' "$(git -C "$engine" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$driver" "$engine/build/bin/libllama.so" \
    "$engine/build/bin/libggml-cuda.so"
} >"$result_dir/metadata.txt"

paragraph="The scheduler keeps a run queue for every processor and moves a task between queues when the load is uneven. A page fault enters the kernel, which finds the mapping, allocates a frame, fills it, and updates the page table before the task resumes. "
suffix_of() {  # agent index
  printf '\n\nTask for agent %s: summarize the text above in %s sentences.\n' \
    "$1" "$(( $1 + 2 ))"
}
# The context is the power of two that leaves room after the prefix.
context_of() {  # paragraphs
  local need=$(( $1 * 52 + 512 )) size=4096
  while (( size < need )); do size=$(( size * 2 )); done
  printf '%s' "$size"
}
field_of() {  # name file: value of name=VALUE on the first line that has it
  sed -n "s/^.* $1=\\([^ ]*\\).*\$/\\1/p" "$2" | head -n 1
}

publish() {  # label mig context prefix-file state-file env...
  local label=$1 mig=$2 context=$3 prefix=$4 state=$5
  shift 5
  rm -f "$state"
  env CUDA_VISIBLE_DEVICES="$mig" GGML_CUDA_HOST_PTR=1 "$@" \
    "$driver" parent "$model" "$context" "$prefix" "$state" "" 0 \
    >"$work/publish.out" 2>"$work/publish.err" || true
  local line
  line=$(grep '^PUBLISHED ' "$work/publish.out" || true)
  printf 'PUBLISH store=%s %s' "$label" "${line#PUBLISHED }"
}

# Environment and state file of an agent in a mode; prints "STATE ENV...".
mode_setup() {  # mode prefix-tokens
  case $1 in
    recompute) echo "-" ;;
    restore) echo "$work/state.device" ;;
    cow) echo "$work/state.huge LLAMA_KV_HOST=$huge_dir/kv LLAMA_KV_PREFIX=$2 LLAMA_KV_COW=1" ;;
    cow_small) echo "$work/state.small LLAMA_KV_HOST=$small_dir/kv LLAMA_KV_PREFIX=$2 LLAMA_KV_COW=1" ;;
    extent) echo "$work/state.huge LLAMA_KV_HOST=$huge_dir/kv LLAMA_KV_PREFIX=$2" ;;
    extent_lazy) echo "$work/state.huge LLAMA_KV_HOST=$huge_dir/kv LLAMA_KV_PREFIX=$2 LLAMA_KV_GROW=$grow" ;;
    extent_lazy_small) echo "$work/state.small LLAMA_KV_HOST=$small_dir/kv LLAMA_KV_PREFIX=$2 LLAMA_KV_GROW=$grow" ;;
  esac
}

run_case() {  # mode agents context prefix-file prefix-tokens first-mig other-mig
  local mode=$1 agents=$2 context=$3 prefix=$4 tokens=$5 first=$6 other=$7
  local state extra index mig pids=() migs=() starts=() ends=() failed=0
  read -r state extra <<<"$(mode_setup "$mode" "$tokens")"
  local envs=()
  if [[ -n "$extra" ]]; then read -r -a envs <<<"$extra"; fi
  local store=
  case $mode in
    cow | extent | extent_lazy) store="$huge_dir/kv.0" ;;
    cow_small | extent_lazy_small) store="$small_dir/kv.0" ;;
  esac
  local before lowest sample pss=0 rss=0
  before=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
  lowest=$before
  for index in $(seq 0 $((agents - 1))); do
    if (( index % 2 == 0 )); then mig=$first; else mig=$other; fi
    migs+=("$mig")
    starts+=("$(date +%s%N)")
    if [[ "$mode" == recompute ]]; then
      env CUDA_VISIBLE_DEVICES="$mig" GGML_CUDA_HOST_PTR=1 \
        "$driver" alone "$model" "$context" "$prefix" "$(suffix_of "$index")" \
        "$n_gen" >"$work/agent.$index" 2>"$work/agent.$index.err" &
    else
      env CUDA_VISIBLE_DEVICES="$mig" GGML_CUDA_HOST_PTR=1 "${envs[@]}" \
        "$driver" child "$model" "$context" "$state" "$(suffix_of "$index")" \
        "$n_gen" >"$work/agent.$index" 2>"$work/agent.$index.err" &
    fi
    pids+=($!)
    ends+=(0)
  done
  # The lowest available memory while the agents run; the pages of the
  # publisher's cache file that the agents map (resident, and proportional
  # share); and when each agent leaves.
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
    if [[ -n "$store" ]] && (( tick % 10 == 5 )); then
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
    tick=$((tick + 1))
    sleep 0.02
  done
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

forward=(recompute restore cow cow_small extent extent_lazy extent_lazy_small)
backward=(extent_lazy_small extent_lazy extent cow_small cow restore recompute)
for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then
    first=$mig_a other=$mig_b modes=("${forward[@]}")
  else
    first=$mig_b other=$mig_a modes=("${backward[@]}")
  fi
  for count in $paragraphs; do
    context=$(context_of "$count")
    prefix="$work/prefix.$count"
    : >"$prefix"
    for _ in $(seq 1 "$count"); do printf '%s' "$paragraph" >>"$prefix"; done
    rm -f "$huge_dir/kv.0" "$small_dir/kv.0"
    {
      printf 'BEGIN_KVSHARE run=%s paragraphs=%s context=%s publisher_mig=%s\n' \
        "$run" "$count" "$context" "${first:4:8}"
      publish device "$first" "$context" "$prefix" "$work/state.device"
      printf '\n'
      publish huge "$first" "$context" "$prefix" "$work/state.huge" \
        LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_GROW="$grow"
      printf ' file_used_kib=%s\n' "$(du -k "$huge_dir/kv.0" | cut -f1)"
      publish small "$first" "$context" "$prefix" "$work/state.small" \
        LLAMA_KV_HOST="$small_dir/kv" LLAMA_KV_GROW="$grow"
      printf ' file_used_kib=%s\n' "$(du -k "$small_dir/kv.0" | cut -f1)"
    } >>"$raw_log"
    # The published rows are not written again by anyone.
    chmod 0400 "$huge_dir/kv.0" "$small_dir/kv.0"
    tokens=$(field_of prefix_tokens "$work/publish.out")
    for agents in $agent_counts; do
      for mode in "${modes[@]}"; do
        run_case "$mode" "$agents" "$context" "$prefix" "$tokens" "$first" "$other"
      done
    done
    printf 'END_KVSHARE\n' >>"$raw_log"
    rm -f "$huge_dir/kv.0" "$small_dir/kv.0" "$work"/state.*
  done
done

awk -f "$script_dir/summarize_engine_kvshare.awk" "$raw_log" \
  >"$result_dir/engine_kvshare_summary.csv"
(
  cd "$script_dir"
  sha256sum kv_extents.patch kv_fork.cpp summarize_engine_kvshare.awk \
    run_engine_kvshare.sh
) >"$result_dir/source_hashes.txt"
printf 'engine_kvshare_result_dir=%s\n' "$result_dir"
