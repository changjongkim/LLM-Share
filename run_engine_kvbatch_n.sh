#!/usr/bin/env bash
# Eight agents on one long prefix split over N batching servers on the whole
# GPU (time slicing), N in SERVERS: the engine's own sharing of a prefix
# between the sequences of one process, and its combination with extents or a
# copy between the servers. AGENTS/N sequences per server.
#
#   1:none     one server with all sequences (the in-process reference)
#   N:copy     server 0 computes the prefix and saves the engine's state file;
#              servers 1..N-1 copy it
#   N:extent   server 0 keeps its cache in a file on a tmpfs with 2 MiB pages
#              and publishes the prefix; servers 1..N-1 map it read-only
#   1:extent   (ONE_EXTENT=1) one server with all sequences on the cache of
#              N:extent, the reference that separates the populate-ahead
#              cache of server 0 from the sharing between servers
#
# The weights are read in place in every configuration. The other servers
# start when server 0 has published; server 0 keeps generating. N = AGENTS
# gives one sequence per server, i.e. separate agent processes.
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
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-kvbatch-n"}
servers_list=${SERVERS:-"2 4 8"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
for n in $servers_list; do
  if (( n < 2 || agents % n != 0 )); then echo "every N in SERVERS must be 2 or more and divide AGENTS" >&2; exit 2; fi
done
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
  printf 'paragraphs=%s\nn_gen=%s\ngrow_rows=%s\nservers=%s\n' "$paragraphs" "$n_gen" "$grow" "$servers_list"
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
server() {  # name env... -- role-arguments...
  local name=$1
  shift 1
  local envs=()
  while [[ "$1" != -- ]]; do envs+=("$1"); shift; done
  shift
  env CUDA_VISIBLE_DEVICES="$mig_a" GGML_CUDA_HOST_PTR=1 "${envs[@]}" \
    "$driver" "$@" >"$work/server.$name" 2>"$work/server.$name.err" &
  pids+=($!)
  names+=("$name")
}

run_case() {  # N:mode
  local n=${1%%:*} mode=${1##*:} before lowest sample start second_start=0 tokens failed=0
  local per=$(( agents / n )) i
  pids=(); names=()
  rm -f "$work"/server.* "$work"/state.* "$huge_dir/kv.0"
  before=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
  lowest=$before
  start=$(date +%s%N)
  case $mode in
    none)   server s0 -- alone "$model" "$context" "$prefix" "$agents" 0 "$n_gen" ;;
    copy)   server s0 -- parent "$model" "$context" "$prefix" "$work/state.device" "$per" 0 "$n_gen" ;;
    extent) server s0 LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_GROW="$grow" -- \
              parent "$model" "$context" "$prefix" "$work/state.huge" "$per" 0 "$n_gen" ;;
  esac
  local waiting=0
  [[ "$mode" != none ]] && waiting=1
  while kill -0 "${pids[@]}" 2>/dev/null; do
    if (( waiting )) && grep -q '^PUBLISHED ' "$work/server.s0" 2>/dev/null; then
      waiting=0
      tokens=$(sed -n 's/^PUBLISHED prefix_tokens=\([0-9]*\).*/\1/p' "$work/server.s0")
      second_start=$(( ($(date +%s%N) - start) / 1000000 ))
      for i in $(seq 1 $(( n - 1 ))); do
        if [[ "$mode" == copy ]]; then
          server "s$i" -- child "$model" "$context" "$work/state.device" "$per" $(( i * per )) "$n_gen"
        else
          server "s$i" LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_PREFIX="$tokens" LLAMA_KV_GROW="$grow" -- \
            child "$model" "$context" "$work/state.huge" "$per" $(( i * per )) "$n_gen"
        fi
      done
    fi
    sample=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
    (( sample < lowest )) && lowest=$sample
    sleep 0.05
  done
  {
    printf 'CASE config=%s servers=%s mode=%s\n' "$1" "$n" "$mode"
    local index code result published started name
    for index in "${!names[@]}"; do
      name=${names[$index]}
      code=0
      wait "${pids[$index]}" || code=$?
      result=$(grep '^RESULT ' "$work/server.$name" || true)
      published=$(grep '^PUBLISHED ' "$work/server.$name" || true)
      if (( code != 0 )) || [[ -z "$result" ]]; then
        failed=$((failed + 1))
        sed 's/^/STDERR /' "$work/server.$name.err" | tail -n 5
      fi
      started=0
      [[ "$name" != s0 ]] && started=$second_start
      printf 'SERVER name=%s exit=%s started_ms=%s %s\n' "$name" "$code" "$started" "${result#RESULT }"
      if [[ -n "$published" ]]; then printf 'PUBLISH %s\n' "${published#PUBLISHED }"; fi
      { grep '^TEXT ' "$work/server.$name" || true; } |
        while read -r _ agent text; do
          printf 'AGENT index=%s text=%s\n' "$agent" "$(printf '%s' "$text" | sha256sum | cut -c1-16)"
        done
    done
    printf 'MEMORY mem_available_drop_mib=%s\n' "$(( (before - lowest) / 1024 ))"
    printf 'END_CASE failed=%s\n' "$failed"
  } >>"$raw_log"
  pids=(); names=()
  rm -f "$work"/server.* "$work"/state.* "$huge_dir/kv.0"
}

forward=("1:none")
[[ "${ONE_EXTENT:-0}" == 1 ]] && forward+=("1:extent")
for n in $servers_list; do forward+=("$n:copy" "$n:extent"); done
backward=(); for c in "${forward[@]}"; do backward=("$c" "${backward[@]}"); done
for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then configs=("${forward[@]}"); else configs=("${backward[@]}"); fi
  printf 'BEGIN_KVBATCH run=%s paragraphs=%s context=%s agents=%s\n' "$run" \
    "$paragraphs" "$context" "$agents" >>"$raw_log"
  for config in "${configs[@]}"; do
    run_case "$config"
    sleep 1
  done
  printf 'END_KVBATCH\n' >>"$raw_log"
done

awk -f "$script_dir/summarize_engine_kvbatch_n.awk" "$raw_log" >"$result_dir/engine_kvbatch_n_summary.csv"
(
  cd "$script_dir"
  sha256sum kv_extents.patch kv_batch.cpp run_engine_kvbatch_n.sh summarize_engine_kvbatch_n.awk
) >"$result_dir/source_hashes.txt"
printf 'kvbatch_n_result_dir=%s\n' "$result_dir"
