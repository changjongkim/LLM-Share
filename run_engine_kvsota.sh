#!/usr/bin/env bash
# Baselines in one engine. A parent computes a prefix, hands its key-value
# cache to N children and keeps generating while they generate. Every way to
# hand the cache over that the related systems use is run in the same engine
# build (llama.cpp-sota) and through the same two programs:
#
#   copy    the state file carries the rows and each child copies them into
#           its own device memory, which holds the whole context (the
#           unmodified engine; cache stores such as LMCache copy the same
#           way from a host copy)
#   demand  the same copy into device memory that is backed on demand as
#           rows are used, through CUDA's virtual memory interface (the
#           mechanism of vAttention); nothing is shared
#   device  the parent's cache is device memory in 2 MiB allocations; a
#           child imports the allocations of the prefix read-only and grows
#           its own behind them (sharing in device memory, as Omni-Flow and
#           the VMM sharing of Liu et al. do within one GPU)
#   cow     the parent's cache is a file on a tmpfs with 2 MiB pages; a
#           child maps it private and writable after a CPU read of the
#           prefix (copy-on-write of the kernel)
#   extent  the same file; a child maps the rows of the prefix read-only
#           and keeps its own rows in private memory that follows use
#
# and two placements of the children:
#
#   same    every child in the MIG instance of the parent (12 SMs)
#   cross   every second child in the other MIG instance (6 SMs)
#
# Kill gates, fixed before the campaign:
#   B1  no process fails, except the children of `device` in the other MIG
#       instance, whose import is expected to be refused;
#   B2  every child that completes writes the text of the child that
#       received a copy in the same placement and repetition;
#   B3  in `same`, the children need less memory with extent than with
#       device, less with device than with demand, and less with demand than
#       with copy; extent needs no more than cow;
#   B4  a child attaches faster with extent than with every other mode, and
#       the parent pauses for less with extent than with copy, demand and
#       device;
#   B5  in `cross`, extent and cow complete with every child, and device
#       completes only with the children of the parent's instance.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
child_counts=${CHILD_COUNTS:-"8"}
placements=${PLACEMENTS:-"same cross"}
paragraphs=${PARAGRAPHS:-320}
n_gen=${N_GEN:-64}
grow=${GROW_ROWS:-256}
limit_s=${CASE_LIMIT_S:-900}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
engine=${SOTA_ENGINE:-"$script_dir/llama.cpp-sota"}
parent_driver=${PARENT_DRIVER:-"$script_dir/kv_spawn_sota"}
child_driver=${CHILD_DRIVER:-"$script_dir/kv_fork_sota"}
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-kvsota"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
if [[ ! -x "$parent_driver" || ! -x "$child_driver" || ! -r "$model" ]]; then
  echo "missing drivers or model; run make kv_spawn_sota kv_fork_sota" >&2
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
parent_pid=
cleanup() {
  if [[ -n "$parent_pid" ]]; then kill -9 "$parent_pid" 2>/dev/null || true; fi
  pkill -9 -P $$ 2>/dev/null || true
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
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\nchild_counts=%s\nplacements=%s\n' \
    "$mig_a" "$mig_b" "$repetitions" "$child_counts" "$placements"
  printf 'paragraphs=%s\nn_gen=%s\ngrow_rows=%s\ncase_limit_s=%s\n' "$paragraphs" \
    "$n_gen" "$grow" "$limit_s"
  printf 'model=%s\n' "$(basename "$model")"
  printf 'engine_commit=%s\n' "$(git -C "$engine" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$parent_driver" "$child_driver" "$engine/build/bin/libllama.so" \
    "$engine/build/bin/libggml-cuda.so"
} >"$result_dir/metadata.txt"

paragraph="The scheduler keeps a run queue for every processor and moves a task between queues when the load is uneven. A page fault enters the kernel, which finds the mapping, allocates a frame, fills it, and updates the page table before the task resumes. "
context=4096
while (( context < paragraphs * 52 + 512 )); do context=$(( context * 2 )); done
prefix="$work/prefix"
: >"$prefix"
for _ in $(seq 1 "$paragraphs"); do printf '%s' "$paragraph" >>"$prefix"; done
text_of() {  # output file
  { grep '^TEXT ' "$1" 2>/dev/null || true; } | sha256sum | cut -c1-16
}

run_case() {  # placement mode children
  local placement=$1 mode=$2 count=$3 index envs=() migs=() state="$work/state"
  case $mode in
    demand) envs=(LLAMA_KV_VMM=2 LLAMA_KV_GROW="$grow") ;;
    device) envs=(LLAMA_KV_VMM=1 LLAMA_KV_GROW="$grow") ;;
    cow) envs=(LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_GROW="$grow" LLAMA_KV_COW=1) ;;
    extent) envs=(LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_GROW="$grow") ;;
  esac
  for index in $(seq 0 $((count - 1))); do
    if [[ "$placement" == cross ]] && (( index % 2 == 1 )); then
      migs+=("$mig_b")
    else
      migs+=("$mig_a")
    fi
  done
  rm -rf "$state" "$work/go" "$huge_dir/kv.0" "$work/out"
  mkdir -p "$work/out"
  env CUDA_VISIBLE_DEVICES="$mig_a" GGML_CUDA_HOST_PTR=1 "${envs[@]}" \
    "$parent_driver" "$model" "$context" "$prefix" "$state" "$n_gen" "$work/go" \
    "$work/out" "$child_driver" "${migs[@]}" >"$work/parent" 2>"$work/parent.err" &
  parent_pid=$!
  until grep -q '^PUBLISHED ' "$work/parent" 2>/dev/null; do
    kill -0 "$parent_pid" 2>/dev/null || break
    sleep 0.02
  done
  local before lowest sample failed=0 timed_out=0 started
  before=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
  lowest=$before
  started=$(date +%s)
  touch "$work/go"
  while kill -0 "$parent_pid" 2>/dev/null; do
    sample=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
    (( sample < lowest )) && lowest=$sample
    if (( $(date +%s) - started > limit_s )); then
      timed_out=1
      pkill -9 -P "$parent_pid" 2>/dev/null || true
      kill -9 "$parent_pid" 2>/dev/null || true
    fi
    sleep 0.05
  done
  {
    printf 'CASE placement=%s mode=%s children=%s\n' "$placement" "$mode" "$count"
    local code=0 result published
    wait "$parent_pid" || code=$?
    parent_pid=
    result=$(grep '^RESULT ' "$work/parent" || true)
    published=$(grep '^PUBLISHED ' "$work/parent" || true)
    if (( code != 0 )) || [[ -z "$result" ]]; then
      failed=$((failed + 1))
      sed 's/^/STDERR /' "$work/parent.err" | tail -n 5
    fi
    # shellcheck disable=SC2001
    printf 'PARENT exit=%s text=%s %s %s\n' "$code" "$(text_of "$work/parent")" \
      "${published#PUBLISHED }" \
      "$(sed 's/^RESULT role=parent prefix_tokens=[0-9]* //' <<<"$result")"
    for index in $(seq 0 $((count - 1))); do
      code=$(sed -n "s/^CHILD index=$index exit=\\([0-9]*\\)\$/\\1/p" "$work/parent")
      result=$(grep '^RESULT ' "$work/out/child.$index" 2>/dev/null || true)
      local own=1
      [[ "${migs[$index]}" == "$mig_a" ]] || own=0
      if [[ "${code:-1}" != 0 || -z "$result" ]]; then
        failed=$((failed + 1))
        sed 's/^/STDERR /' "$work/out/child.$index.err" 2>/dev/null | tail -n 2
      fi
      printf 'CHILD index=%s own_instance=%s exit=%s text=%s %s\n' "$index" "$own" \
        "${code:-none}" "$(text_of "$work/out/child.$index")" "${result#RESULT }"
    done
    printf 'MEMORY mem_available_drop_mib=%s\n' "$(( (before - lowest) / 1024 ))"
    printf 'END_CASE failed=%s timed_out=%s\n' "$failed" "$timed_out"
  } >>"$raw_log"
  rm -rf "$state" "$work/go" "$huge_dir/kv.0" "$work/out"
}

forward=(copy demand device cow extent)
backward=(extent cow device demand copy)
for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then modes=("${forward[@]}"); else modes=("${backward[@]}"); fi
  printf 'BEGIN_KVSOTA run=%s paragraphs=%s context=%s\n' "$run" "$paragraphs" \
    "$context" >>"$raw_log"
  for placement in $placements; do
    for count in $child_counts; do
      for mode in "${modes[@]}"; do
        run_case "$placement" "$mode" "$count"
      done
    done
  done
  printf 'END_KVSOTA\n' >>"$raw_log"
done

awk -f "$script_dir/summarize_engine_kvsota.awk" "$raw_log" \
  >"$result_dir/engine_kvsota_summary.csv"
(
  cd "$script_dir"
  sha256sum kv_sota.patch kv_spawn.cpp kv_fork.cpp summarize_engine_kvsota.awk \
    run_engine_kvsota.sh
) >"$result_dir/source_hashes.txt"
printf 'engine_kvsota_result_dir=%s\n' "$result_dir"
