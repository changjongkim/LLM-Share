#!/usr/bin/env bash
# The memory of the whole serving stack, and how many agents fit. A publisher
# computes the key-value cache of a prefix and exits; N agent processes, every
# second one in the other MIG instance, continue it with their own task. Four
# stacks, which differ in what an agent shares with the others:
#
#   none     nothing: the engine at the upstream commit, without any patch of
#            this repository; every agent copies the weights to device memory
#            and copies the prefix from the engine's state file
#   weights  the weights are read in place; the prefix is copied
#   kv       the weights are copied (read from the model file instead of
#            mapped); the prefix is mapped read-only and the rows an agent
#            writes are private memory that follows use
#   both     the weights are read in place and the prefix is mapped
#
# The prefix is the text of PREFIX_FILE; the task of agent i is line i+1 of
# TASKS_FILE. Every stack is run with its own list of agent counts, because
# the stacks that copy do not fit at the larger counts: a case is skipped,
# and logged as skipped, when starting it would leave less than RESERVE_GIB
# of available memory by the estimate of the stack.
#
# Before every repetition the page cache is dropped and the model file is
# read again, so that no case waits for the reclaim of unrelated cache. The
# pages that the GPU driver keeps after a process has freed its device memory
# (KReclaimable) are left as they are; they count as available.
#
# The memory of a case is the largest drop of MemAvailable while its agents
# run. The files that the agents share (the model file when the weights are
# read in place, and the publisher's cache file) are created before the case
# and are reported apart. While the agents run, the sums of Pss_Anon,
# Pss_File and Pss_Shmem over the agents and the device memory that
# nvidia-smi reports for them are sampled every SAMPLE_MS.
#
# Kill gates, fixed before the campaign:
#   K1  no agent fails in any case that is not skipped;
#   K2  within a repetition, the agents with the same index write one text
#       in every stack and at every count in which the index occurs;
#   K3  at every count that two stacks have in common, `both` holds less
#       memory than `weights` and than `kv`, and each of these less than
#       `none`;
#   K4  the memory per added agent of `both` (least-squares slope over the
#       counts, per repetition) is at most 0.20 of that of `none`;
#   K5  the summed generation speed of `both` is at least 0.92 of that of
#       `none` at every count they have in common (half of the agents run in
#       the 6-SM instance, where a cache in host memory is known to cost up
#       to 5.8%).
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
none_counts=${NONE_COUNTS:-"1 2 4 8 12"}
kv_counts=${KV_COUNTS:-"1 2 4 8 12"}
weights_counts=${WEIGHTS_COUNTS:-"1 2 4 8 12 16 32"}
both_counts=${BOTH_COUNTS:-"1 2 4 8 12 16 32 64"}
prefix_file=${PREFIX_FILE:-"$script_dir/workloads/agent_prefix.txt"}
tasks_file=${TASKS_FILE:-"$script_dir/workloads/agent_tasks.txt"}
context=${CONTEXT:-16384}
n_gen=${N_GEN:-64}
grow=${GROW_ROWS:-256}
limit_s=${CASE_LIMIT_S:-1200}
sample_ms=${SAMPLE_MS:-4000}
reserve_gib=${RESERVE_GIB:-20}
# Estimates of the memory of one agent, in MiB, for the guard only.
estimate_none=${ESTIMATE_NONE_MIB:-6600}
estimate_kv=${ESTIMATE_KV_MIB:-5600}
estimate_weights=${ESTIMATE_WEIGHTS_MIB:-2000}
estimate_both=${ESTIMATE_BOTH_MIB:-900}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
engine=${KV_ENGINE:-"$script_dir/llama.cpp-kv"}
stock_engine=${STOCK_ENGINE:-"$script_dir/llama.cpp-stock"}
driver=${DRIVER:-"$script_dir/kv_fork"}
stock_driver=${STOCK_DRIVER:-"$script_dir/kv_fork_stock"}
copy_driver=${COPY_DRIVER:-"$script_dir/kv_fork_nommap"}
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-kvstack"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
if [[ ! -x "$driver" || ! -x "$stock_driver" || ! -x "$copy_driver" ||
      ! -r "$model" ]]; then
  echo "missing driver or model; run make and see README.md" >&2
  exit 2
fi
for file in "$prefix_file" "$tasks_file"; do
  if [[ ! -r "$file" ]]; then echo "cannot read $file" >&2; exit 2; fi
done
if [[ -n "$(git -C "$stock_engine" status --porcelain --untracked-files=no)" ]]; then
  echo "the engine of the stack that shares nothing is not unmodified" >&2
  exit 2
fi
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

suffix_of() {  # agent index
  printf '\n\nTask for agent %s: %s\n' "$1" \
    "$(sed -n "$(( $1 % task_lines + 1 ))p" "$tasks_file")"
}
field_of() {  # name file: value of name=VALUE on the first line that has it
  sed -n "s/^.* $1=\\([^ ]*\\).*\$/\\1/p" "$2" | head -n 1
}
meminfo_kib() {  # name
  awk -v name="$1:" '$1 == name { print $2 }' /proc/meminfo
}

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\n' "$mig_a" "$mig_b" "$repetitions"
  printf 'none_counts=%s\nkv_counts=%s\nweights_counts=%s\nboth_counts=%s\n' \
    "$none_counts" "$kv_counts" "$weights_counts" "$both_counts"
  printf 'context=%s\nn_gen=%s\ngrow_rows=%s\nsample_ms=%s\n' "$context" \
    "$n_gen" "$grow" "$sample_ms"
  printf 'prefix_file=%s\nprefix_sha256=%s\ntasks_file=%s\ntasks_sha256=%s\n' \
    "$(basename "$prefix_file")" "$(sha256sum "$prefix" | cut -d' ' -f1)" \
    "$(basename "$tasks_file")" "$(sha256sum "$tasks_file" | cut -d' ' -f1)"
  printf 'reserve_gib=%s\nestimate_mib=none:%s,kv:%s,weights:%s,both:%s\n' \
    "$reserve_gib" "$estimate_none" "$estimate_kv" "$estimate_weights" \
    "$estimate_both"
  printf 'model=%s\nmodel_bytes=%s\n' "$(basename "$model")" "$(stat -c %s "$model")"
  printf 'mem_total_kib=%s\n' "$(meminfo_kib MemTotal)"
  printf 'engine_commit=%s\nstock_engine_commit=%s\n' \
    "$(git -C "$engine" rev-parse HEAD)" "$(git -C "$stock_engine" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$driver" "$copy_driver" "$engine/build/bin/libllama.so" \
    "$engine/build/bin/libggml-cuda.so" "$stock_driver" \
    "$stock_engine/build/bin/libllama.so" \
    "$stock_engine/build/bin/libggml-cuda.so"
} >"$result_dir/metadata.txt"

publish() {  # label mig driver state-file env...
  local label=$1 mig=$2 program=$3 state=$4
  shift 4
  rm -f "$state"
  env CUDA_VISIBLE_DEVICES="$mig" "$@" \
    "$program" parent "$model" "$context" "$prefix" "$state" "" 0 \
    >"$work/publish.out" 2>"$work/publish.err" || true
  local line
  line=$(grep '^PUBLISHED ' "$work/publish.out" || true)
  printf 'PUBLISH store=%s %s' "$label" "${line#PUBLISHED }"
}

# The sums over the agents that are alive: three lines of smaps_rollup, and
# the device memory of the processes by nvidia-smi.
sample_agents() {  # pid...
  local rollups=() pid
  for pid in "$@"; do rollups+=("/proc/$pid/smaps_rollup"); done
  # An agent may leave between the liveness test and the read.
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

run_case() {  # mode agents prefix-tokens first-mig other-mig
  local mode=$1 agents=$2 tokens=$3 first=$4 other=$5
  local state program envs=() estimate index mig
  local pids=() migs=() starts=() ends=() failed=0
  case "$mode" in
    none)
      program=$stock_driver state="$work/state.stock" estimate=$estimate_none ;;
    weights)
      program=$driver state="$work/state.device" estimate=$estimate_weights
      envs=(GGML_CUDA_HOST_PTR=1) ;;
    kv)
      # The device accepts host memory, which the mapped prefix needs; the
      # program does not map the model file, so the weights are copied.
      program=$copy_driver state="$work/state.huge" estimate=$estimate_kv
      envs=(GGML_CUDA_HOST_PTR=1 LLAMA_KV_HOST="$huge_dir/kv"
            LLAMA_KV_PREFIX="$tokens" LLAMA_KV_GROW="$grow") ;;
    both)
      program=$driver state="$work/state.huge" estimate=$estimate_both
      envs=(GGML_CUDA_HOST_PTR=1 LLAMA_KV_HOST="$huge_dir/kv"
            LLAMA_KV_PREFIX="$tokens" LLAMA_KV_GROW="$grow") ;;
    *) echo "unknown mode $mode" >&2; exit 2 ;;
  esac
  local before lowest sample free_before
  before=$(meminfo_kib MemAvailable)
  free_before=$(meminfo_kib MemFree)
  if (( before / 1024 - agents * estimate < reserve_gib * 1024 )); then
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
    env CUDA_VISIBLE_DEVICES="$mig" "${envs[@]}" \
      "$program" child "$model" "$context" "$state" "$(suffix_of "$index")" \
      "$n_gen" >"$work/agent.$index" 2>"$work/agent.$index.err" &
    pids+=($!)
    ends+=(0)
  done
  live_pids=("${pids[@]}")
  local alive=$agents tick=0 elapsed_ms next_sample_ms=2500 samples=0
  local anon=0 file=0 shmem=0 device=0 parts=()
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
      sample=$(meminfo_kib MemAvailable)
      (( sample < lowest )) && lowest=$sample
    fi
    elapsed_ms=$(( ($(date +%s%N) - case_start) / 1000000 ))
    if (( alive == agents && elapsed_ms >= next_sample_ms )); then
      # Only while every agent is alive, so that the sums are of all agents.
      read -r -a parts <<<"$(sample_agents "${pids[@]}")"
      if (( ${#parts[@]} == 4 )); then
        samples=$((samples + 1))
        (( parts[0] > anon )) && anon=${parts[0]}
        (( parts[1] > file )) && file=${parts[1]}
        (( parts[2] > shmem )) && shmem=${parts[2]}
        (( parts[3] > device )) && device=${parts[3]}
      fi
      next_sample_ms=$(( elapsed_ms + sample_ms ))
    fi
    if (( elapsed_ms / 1000 > limit_s )); then
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
    printf 'MEMORY mem_available_drop_mib=%s mem_available_before_mib=%s ' \
      "$(( (before - lowest) / 1024 ))" "$(( before / 1024 ))"
    printf 'mem_free_before_mib=%s ' "$(( free_before / 1024 ))"
    printf 'pss_anon_mib=%s pss_file_mib=%s pss_shmem_mib=%s device_mib=%s samples=%s\n' \
      "$anon" "$file" "$shmem" "$device" "$samples"
    printf 'END_CASE failed_agents=%s wall_ms=%s\n' "$failed" \
      "$(( ($(date +%s%N) - case_start) / 1000000 ))"
  } >>"$raw_log"
}

counts_of() {  # mode
  case "$1" in
    none) printf '%s' "$none_counts" ;;
    kv) printf '%s' "$kv_counts" ;;
    weights) printf '%s' "$weights_counts" ;;
    both) printf '%s' "$both_counts" ;;
  esac
}

for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then
    first=$mig_a other=$mig_b modes=(none kv weights both)
  else
    first=$mig_b other=$mig_a modes=(both weights kv none)
  fi
  rm -f "$huge_dir/kv.0"
  sync
  sudo -n sh -c 'echo 1 >/proc/sys/vm/drop_caches'
  cat "$model" >/dev/null
  {
    printf 'BEGIN_KVSTACK run=%s context=%s publisher_mig=%s mem_free_mib=%s ' \
      "$run" "$context" "${first:4:8}" "$(( $(meminfo_kib MemFree) / 1024 ))"
    printf 'kernel_reclaimable_mib=%s\n' "$(( $(meminfo_kib KReclaimable) / 1024 ))"
    publish stock "$first" "$stock_driver" "$work/state.stock"
    printf '\n'
    publish device "$first" "$driver" "$work/state.device" GGML_CUDA_HOST_PTR=1
    printf '\n'
    publish huge "$first" "$driver" "$work/state.huge" GGML_CUDA_HOST_PTR=1 \
      LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_GROW="$grow"
    printf ' file_used_kib=%s\n' "$(du -k "$huge_dir/kv.0" | cut -f1)"
  } >>"$raw_log"
  # The published rows are not written again by anyone.
  chmod 0400 "$huge_dir/kv.0"
  tokens=$(field_of prefix_tokens "$work/publish.out")
  for mode in "${modes[@]}"; do
    for agents in $(counts_of "$mode"); do
      run_case "$mode" "$agents" "$tokens" "$first" "$other"
    done
  done
  printf 'END_KVSTACK\n' >>"$raw_log"
  rm -f "$huge_dir/kv.0" "$work"/state.*
done

awk -f "$script_dir/summarize_engine_kvstack.awk" "$raw_log" \
  >"$result_dir/engine_kvstack_summary.csv"
awk -v table=fit -v model_bytes="$(stat -c %s "$model")" \
  -v mem_total_kib="$(meminfo_kib MemTotal)" \
  -f "$script_dir/summarize_engine_kvstack.awk" "$raw_log" \
  >"$result_dir/engine_kvstack_fit.csv"
awk -v table=texts -f "$script_dir/summarize_engine_kvstack.awk" "$raw_log" \
  >"$result_dir/engine_kvstack_texts.csv"
(
  cd "$script_dir"
  sha256sum kv_extents.patch kv_fork.cpp kv_fork_nommap.cpp \
    summarize_engine_kvstack.awk \
    run_engine_kvstack.sh "$prefix_file" "$tasks_file"
) >"$result_dir/source_hashes.txt"
printf 'engine_kvstack_result_dir=%s\n' "$result_dir"
