#!/usr/bin/env bash
# Eight agents on one long prefix when the agents may share a process: the
# engine's own sharing of a prefix between the sequences of one process, and
# its combination with extents between two such processes.
#
#   one_server           one process in the 12-SM instance, eight sequences;
#                        it computes the prefix and shares it between its
#                        sequences (cache in device memory)
#   two_servers_compute  one process in each MIG instance, four sequences
#                        each; each computes the prefix itself
#   two_servers_copy     the server in the 12-SM instance computes the prefix
#                        and saves the engine's state file; the server in the
#                        6-SM instance copies it
#   two_servers_extent   the server in the 12-SM instance keeps its cache in a
#                        file on a tmpfs with 2 MiB pages and publishes the
#                        prefix; the server in the 6-SM instance maps it
#
# The weights are read in place in every configuration. The second server
# starts when the first has published; the first keeps generating.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
agents=${AGENTS:-8}
paragraphs=${PARAGRAPHS:-320}
n_gen=${N_GEN:-64}
grow=${GROW_ROWS:-256}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
engine=${KV_ENGINE:-"$script_dir/llama.cpp-kv"}
driver=${DRIVER:-"$script_dir/kv_batch"}
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-kvbatch"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
if (( agents % 2 != 0 )); then
  echo "AGENTS must be even" >&2
  exit 2
fi
if [[ ! -x "$driver" || ! -r "$model" ]]; then
  echo "missing driver or model; run make and see README.md" >&2
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
pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
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

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\nagents=%s\n' \
    "$mig_a" "$mig_b" "$repetitions" "$agents"
  printf 'paragraphs=%s\nn_gen=%s\ngrow_rows=%s\n' "$paragraphs" "$n_gen" "$grow"
  printf 'model=%s\n' "$(basename "$model")"
  printf 'engine_commit=%s\n' "$(git -C "$engine" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$driver" "$engine/build/bin/libllama.so" \
    "$engine/build/bin/libggml-cuda.so"
} >"$result_dir/metadata.txt"

paragraph="The scheduler keeps a run queue for every processor and moves a task between queues when the load is uneven. A page fault enters the kernel, which finds the mapping, allocates a frame, fills it, and updates the page table before the task resumes. "
half=$(( agents / 2 ))
context=4096
while (( context < paragraphs * 52 + 512 + agents * (n_gen + 32) )); do
  context=$(( context * 2 ))
done
prefix="$work/prefix"
: >"$prefix"
for _ in $(seq 1 "$paragraphs"); do printf '%s' "$paragraph" >>"$prefix"; done

# Starts a server in the background; its output goes to $work/server.NAME.
server() {  # name mig env... -- role-arguments...
  local name=$1 mig=$2
  shift 2
  local envs=()
  while [[ "$1" != -- ]]; do envs+=("$1"); shift; done
  shift
  env CUDA_VISIBLE_DEVICES="$mig" GGML_CUDA_HOST_PTR=1 "${envs[@]}" \
    "$driver" "$@" >"$work/server.$name" 2>"$work/server.$name.err" &
  pids+=($!)
}

run_case() {  # configuration
  local config=$1 before lowest sample start second_start=0 tokens failed=0
  pids=()
  rm -f "$work"/server.* "$work"/state.* "$huge_dir/kv.0"
  before=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
  lowest=$before
  start=$(date +%s%N)
  case $config in
    one_server)
      server a "$mig_a" -- alone "$model" "$context" "$prefix" "$agents" 0 "$n_gen" ;;
    two_servers_compute)
      server a "$mig_a" -- alone "$model" "$context" "$prefix" "$half" 0 "$n_gen"
      server b "$mig_b" -- alone "$model" "$context" "$prefix" "$half" "$half" "$n_gen" ;;
    two_servers_copy)
      server a "$mig_a" -- parent "$model" "$context" "$prefix" "$work/state.device" \
        "$half" 0 "$n_gen" ;;
    two_servers_extent)
      server a "$mig_a" LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_GROW="$grow" -- \
        parent "$model" "$context" "$prefix" "$work/state.huge" "$half" 0 "$n_gen" ;;
  esac
  local waiting=0
  if [[ "$config" == two_servers_copy || "$config" == two_servers_extent ]]; then
    waiting=1
  fi
  while kill -0 "${pids[@]}" 2>/dev/null; do
    if (( waiting )) && grep -q '^PUBLISHED ' "$work/server.a" 2>/dev/null; then
      waiting=0
      tokens=$(sed -n 's/^PUBLISHED prefix_tokens=\([0-9]*\).*/\1/p' "$work/server.a")
      second_start=$(( ($(date +%s%N) - start) / 1000000 ))
      if [[ "$config" == two_servers_copy ]]; then
        server b "$mig_b" -- child "$model" "$context" "$work/state.device" \
          "$half" "$half" "$n_gen"
      else
        server b "$mig_b" LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_PREFIX="$tokens" \
          LLAMA_KV_GROW="$grow" -- child "$model" "$context" "$work/state.huge" \
          "$half" "$half" "$n_gen"
      fi
    fi
    sample=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
    (( sample < lowest )) && lowest=$sample
    sleep 0.05
  done
  {
    printf 'CASE config=%s\n' "$config"
    local name index=0 code result published
    for name in a b; do
      [[ -e "$work/server.$name" ]] || continue
      code=0
      wait "${pids[$index]}" || code=$?
      index=$((index + 1))
      result=$(grep '^RESULT ' "$work/server.$name" || true)
      published=$(grep '^PUBLISHED ' "$work/server.$name" || true)
      if (( code != 0 )) || [[ -z "$result" ]]; then
        failed=$((failed + 1))
        sed 's/^/STDERR /' "$work/server.$name.err" | tail -n 5
      fi
      local started=0
      if [[ "$name" == b && "$config" != two_servers_compute ]]; then started=$second_start; fi
      printf 'SERVER name=%s exit=%s started_ms=%s %s\n' "$name" "$code" "$started" \
        "${result#RESULT }"
      if [[ -n "$published" ]]; then printf 'PUBLISH %s\n' "${published#PUBLISHED }"; fi
      { grep '^TEXT ' "$work/server.$name" || true; } |
        while read -r _ agent text; do
          printf 'AGENT index=%s text=%s\n' "$agent" \
            "$(printf '%s' "$text" | sha256sum | cut -c1-16)"
        done
    done
    printf 'MEMORY mem_available_drop_mib=%s\n' "$(( (before - lowest) / 1024 ))"
    printf 'END_CASE failed=%s\n' "$failed"
  } >>"$raw_log"
  pids=()
  rm -f "$work"/server.* "$work"/state.* "$huge_dir/kv.0"
}

forward=(one_server two_servers_compute two_servers_copy two_servers_extent)
backward=(two_servers_extent two_servers_copy two_servers_compute one_server)
for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then configs=("${forward[@]}"); else configs=("${backward[@]}"); fi
  printf 'BEGIN_KVBATCH run=%s paragraphs=%s context=%s agents=%s\n' "$run" \
    "$paragraphs" "$context" "$agents" >>"$raw_log"
  for config in "${configs[@]}"; do
    run_case "$config"
    # Let the page cache of a state file settle before the next baseline.
    sleep 1
  done
  printf 'END_KVBATCH\n' >>"$raw_log"
done

awk -f "$script_dir/summarize_engine_kvbatch.awk" "$raw_log" \
  >"$result_dir/engine_kvbatch_summary.csv"
(
  cd "$script_dir"
  sha256sum kv_extents.patch kv_batch.cpp summarize_engine_kvbatch.awk \
    run_engine_kvbatch.sh
) >"$result_dir/source_hashes.txt"
printf 'engine_kvbatch_result_dir=%s\n' "$result_dir"
