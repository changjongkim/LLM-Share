#!/usr/bin/env bash
# Where does the generation speed of an agent go when its key-value cache is
# host memory? One or two processes in one MIG instance (two are
# time-sliced) generate a long continuation of the same prefix with the
# cache in device memory or in host memory of several kinds:
#
#   device        the agent computes the prefix; cache in device memory
#   anon          the same; cache in private host memory (2 MiB pages)
#   anon_lazy     the same; private host memory that follows use (4 KiB pages)
#   restore       the agent copies the prefix from the state file; device memory
#   extent        the prefix is a read-only mapping of the publisher's file
#                 (2 MiB pages), the tail private host memory (2 MiB pages)
#   extent_lazy   the same with a tail that follows use (4 KiB pages)
#
# Every mode is compared with `device` of the same repetition. The weights
# are read in place in every mode.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
repetitions=${REPETITIONS:-6}
process_counts=${PROCESS_COUNTS:-"1 2"}
paragraphs=${PARAGRAPHS:-80}
n_gen=${N_GEN:-256}
grow=${GROW_ROWS:-256}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
engine=${KV_ENGINE:-"$script_dir/llama.cpp-kv"}
driver=${DRIVER:-"$script_dir/kv_fork"}
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-kvspeed"}
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
cleanup() {
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
  printf 'mig=%s\nrepetitions=%s\nprocess_counts=%s\n' "$mig" "$repetitions" \
    "$process_counts"
  printf 'paragraphs=%s\nn_gen=%s\ngrow_rows=%s\n' "$paragraphs" "$n_gen" "$grow"
  printf 'model=%s\n' "$(basename "$model")"
  printf 'engine_commit=%s\n' "$(git -C "$engine" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$driver" "$engine/build/bin/libllama.so" \
    "$engine/build/bin/libggml-cuda.so"
} >"$result_dir/metadata.txt"

paragraph="The scheduler keeps a run queue for every processor and moves a task between queues when the load is uneven. A page fault enters the kernel, which finds the mapping, allocates a frame, fills it, and updates the page table before the task resumes. "
suffix_of() {  # process index
  printf '\n\nTask for agent %s: summarize the text above in %s sentences.\n' \
    "$1" "$(( $1 + 2 ))"
}
context=4096
while (( context < paragraphs * 52 + 512 + n_gen )); do context=$(( context * 2 )); done
prefix="$work/prefix"
: >"$prefix"
for _ in $(seq 1 "$paragraphs"); do printf '%s' "$paragraph" >>"$prefix"; done

publish() {  # state-file env...
  local state=$1
  shift
  rm -f "$state"
  env CUDA_VISIBLE_DEVICES="$mig" GGML_CUDA_HOST_PTR=1 "$@" \
    "$driver" parent "$model" "$context" "$prefix" "$state" "" 0 \
    >"$work/publish.out" 2>"$work/publish.err" || true
  sed -n 's/^PUBLISHED prefix_tokens=\([0-9]*\).*/\1/p' "$work/publish.out"
}

run_case() {  # mode processes prefix-tokens
  local mode=$1 processes=$2 tokens=$3 index pids=() failed=0
  local role=alone source="$prefix" envs=()
  case $mode in
    device) ;;
    anon) envs=(LLAMA_KV_HOST=anon) ;;
    anon_lazy) envs=(LLAMA_KV_HOST=anon LLAMA_KV_GROW="$grow") ;;
    restore) role=child source="$work/state.device" ;;
    extent)
      role=child source="$work/state.huge"
      envs=(LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_PREFIX="$tokens") ;;
    extent_lazy)
      role=child source="$work/state.huge"
      envs=(LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_PREFIX="$tokens"
        LLAMA_KV_GROW="$grow") ;;
  esac
  for index in $(seq 0 $((processes - 1))); do
    env CUDA_VISIBLE_DEVICES="$mig" GGML_CUDA_HOST_PTR=1 "${envs[@]}" \
      "$driver" "$role" "$model" "$context" "$source" "$(suffix_of "$index")" \
      "$n_gen" >"$work/agent.$index" 2>"$work/agent.$index.err" &
    pids+=($!)
  done
  {
    printf 'CASE mode=%s processes=%s\n' "$mode" "$processes"
    for index in "${!pids[@]}"; do
      local code=0 result
      wait "${pids[$index]}" || code=$?
      result=$(grep '^RESULT ' "$work/agent.$index" || true)
      if (( code != 0 )) || [[ -z "$result" ]]; then
        failed=$((failed + 1))
        sed 's/^/STDERR /' "$work/agent.$index.err" | tail -n 5
      fi
      printf 'AGENT index=%s exit=%s %s\n' "$index" "$code" "${result#RESULT }"
    done
    printf 'END_CASE failed_agents=%s\n' "$failed"
  } >>"$raw_log"
}

forward=(device anon anon_lazy restore extent extent_lazy)
backward=(extent_lazy extent restore anon_lazy anon device)
for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then modes=("${forward[@]}"); else modes=("${backward[@]}"); fi
  rm -f "$huge_dir/kv.0"
  tokens=$(publish "$work/state.device")
  tokens=$(publish "$work/state.huge" LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_GROW="$grow")
  chmod 0400 "$huge_dir/kv.0"
  printf 'BEGIN_KVSPEED run=%s prefix_tokens=%s context=%s\n' "$run" \
    "${tokens:-0}" "$context" >>"$raw_log"
  for processes in $process_counts; do
    for mode in "${modes[@]}"; do
      run_case "$mode" "$processes" "${tokens:-0}"
    done
  done
  printf 'END_KVSPEED\n' >>"$raw_log"
  rm -f "$huge_dir/kv.0" "$work"/state.*
done

awk -f "$script_dir/summarize_engine_kvspeed.awk" "$raw_log" \
  >"$result_dir/engine_kvspeed_summary.csv"
(
  cd "$script_dir"
  sha256sum kv_extents.patch kv_fork.cpp summarize_engine_kvspeed.awk \
    run_engine_kvspeed.sh
) >"$result_dir/source_hashes.txt"
printf 'engine_kvspeed_result_dir=%s\n' "$result_dir"
