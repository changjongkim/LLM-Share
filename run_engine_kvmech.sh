#!/usr/bin/env bash
# Mechanisms in one engine, with the device-memory baseline tuned and with
# the steps of the extent mechanism left out one at a time. A parent computes
# a prefix, hands its key-value cache to N children and keeps generating
# while they generate. All modes run in the same engine build
# (llama.cpp-tuned) and through the same two programs.
#
# A case is a cell PLACEMENT:CHILDREN:PARAGRAPHS (the prefix is PARAGRAPHS
# copies of one paragraph of 51 tokens) and a mode. CELLS lists the cells;
# without it they follow from PLACEMENTS, CHILD_COUNTS and PARAGRAPHS when
# one of these is set, and from the mode set otherwise.
#
# MODE_SET=baselines (8 children in both placements on the long prefix, 1
# and 4 children and the short prefix in `same`):
#
#   copy          the state file carries the rows and each child copies them
#                 into its own device memory, which holds the whole context
#   demand        the same copy into device memory that is backed on demand
#                 through CUDA's virtual memory interface; nothing is shared
#   device        the parent's cache is device memory in 2 MiB allocations; a
#                 child imports the allocations of the prefix read-only, one
#                 access setting and one wait for the device per allocation
#   device_tuned  the same, with one access setting per tensor, the
#                 allocations that the prefix ends in copied with one wait
#                 for the device, a child's own memory in one allocation per
#                 tensor and step, and its rows zeroed with one wait
#                 (LLAMA_KV_VMM_BATCH=1)
#   device_merged the same, and the parent copies the rows it publishes into
#                 one allocation per tensor, so that a child imports one
#                 handle per tensor instead of one per 2 MiB
#                 (LLAMA_KV_VMM_BATCH=2). An allocation cannot be extended or
#                 joined with another, and a mapping covers a whole
#                 allocation, so one per tensor is the fewest; 2 MiB is the
#                 smallest allocation, so a child still copies the allocation
#                 that the prefix ends in.
#   cow           the parent's cache is a file on a tmpfs with 2 MiB pages; a
#                 child maps it private and writable after a CPU read
#   extent        the same file; a child maps the rows of the prefix
#                 read-only and keeps its own rows in private memory that
#                 follows use
#
# MODE_SET=ablation (8 children, placement cross, the long prefix): copy and
# extent as above, and extent with one step changed:
#
#   no_read           no CPU read of the mapped rows (LLAMA_KV_TOUCH=0)
#   no_populate       a child puts no memory behind its own rows before the
#                     device writes them (LLAMA_KV_POPULATE=0)
#   writable          the prefix is mapped writable and private (cow)
#   writable_no_read  the same without the CPU read (LLAMA_KV_COW=2)
#   grow_1024, grow_4096   the child's memory follows use in larger steps
#   grow_all          the child's memory for the whole context at the start
#   small_pages       the file is on a tmpfs with 4 KiB pages
#
# MODE_SET=locality (4 children with the parent in one instance, the long
# prefix; PRODUCER_MIG chooses the instance): where the loss of generation
# speed with a cache in host memory comes from. copy, extent, grow_all and
# small_pages as above, and a private cache in host memory that receives a
# copy of the rows:
#
#   host_copy       in 4 KiB pages that follow use
#   host_copy_huge  in 2 MiB pages, all at the start
#
# The ratios of the generation speed answer, in this order: host memory
# against device memory (host_copy_huge to copy), the page size of private
# memory (host_copy to host_copy_huge), sharing (extent to host_copy), the
# page size of an agent's own rows (grow_all to extent) and the page size of
# the mapped prefix (small_pages to extent).
#
# Placements of the children:
#
#   same    every child in the MIG instance of the parent (12 SMs)
#   cross   every second child in the other MIG instance (6 SMs)
#
# Kill gates of the baselines, fixed before the campaign:
#   B1  no process fails, except the children of the three device modes in
#       the other MIG instance, whose import is expected to be refused;
#   B2  every child that completes writes the text of the child that
#       received a copy in the same placement and repetition;
#   B3  in `same`, the children need less memory with extent than with
#       each of the three device modes, less with each device mode than with
#       demand, and less with demand than with copy; extent needs no more
#       than cow;
#   B4  a child attaches faster with extent than with every other mode, the
#       tuned device modes included, and the parent pauses for less with
#       extent than with copy, demand and the three device modes;
#   B5  in `cross`, extent and cow complete with every child, and the three
#       device modes complete only with the children of the parent's
#       instance;
#   B6  in `same`, a child attaches with device_merged in less than half the
#       time of device.
#
# Kill gates of the locality set, fixed before the campaign:
#   L1  no process fails;
#   L2  every child writes the text of the child that received a copy.
#
# Kill gates of the ablation, fixed before the campaign:
#   A1  no process fails in copy, extent, writable, grow_1024, grow_4096,
#       grow_all and small_pages; failures in no_read, no_populate and
#       writable_no_read are counted and reported;
#   A2  every child that completes in the modes of A1 writes the text of the
#       child that received a copy; the other modes are counted and reported;
#   A3  the children need no less memory with each larger step: extent,
#       grow_1024, grow_4096, grow_all;
#   A4  a child attaches faster with extent than with small_pages.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
mode_set=${MODE_SET:-baselines}
case "$mode_set" in
  baselines)
    forward=(copy demand device device_tuned device_merged cow extent)
    default_cells="same:8:320 cross:8:320 same:4:320 same:1:320 same:8:80" ;;
  ablation)
    forward=(copy extent no_read no_populate writable writable_no_read
             grow_1024 grow_4096 grow_all small_pages)
    default_cells="cross:8:320" ;;
  locality)
    forward=(copy host_copy_huge host_copy extent grow_all small_pages)
    default_cells="same:4:320" ;;
  *) echo "MODE_SET must be baselines, ablation or locality" >&2; exit 2 ;;
esac
if [[ -n "${MODES:-}" ]]; then read -r -a forward <<<"$MODES"; fi
if [[ -n "${CELLS:-}" ]]; then
  cells=$CELLS
elif [[ -n "${PLACEMENTS:-}${CHILD_COUNTS:-}${PARAGRAPHS:-}" ]]; then
  cells=
  for placement in ${PLACEMENTS:-same cross}; do
    for count in ${CHILD_COUNTS:-8}; do
      cells+="$placement:$count:${PARAGRAPHS:-320} "
    done
  done
else
  cells=$default_cells
fi
for cell in $cells; do
  if ! [[ "$cell" =~ ^(same|cross):[1-9][0-9]*:[1-9][0-9]*$ ]]; then
    echo "a cell is PLACEMENT:CHILDREN:PARAGRAPHS, not $cell" >&2
    exit 2
  fi
done
n_gen=${N_GEN:-64}
grow=${GROW_ROWS:-256}
limit_s=${CASE_LIMIT_S:-900}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
engine=${TUNED_ENGINE:-"$script_dir/llama.cpp-tuned"}
parent_driver=${PARENT_DRIVER:-"$script_dir/kv_spawn_tuned"}
child_driver=${CHILD_DRIVER:-"$script_dir/kv_fork_tuned"}
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-kvmech-$mode_set"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
if [[ ! -x "$parent_driver" || ! -x "$child_driver" || ! -r "$model" ]]; then
  echo "missing drivers or model; run make kv_spawn_tuned kv_fork_tuned" >&2
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
small_dir="$mount_root/kv_small"
mounted_huge=0
mounted_small=0
work=
parent_pid=
cleanup() {
  if [[ -n "$parent_pid" ]]; then kill -9 "$parent_pid" 2>/dev/null || true; fi
  pkill -9 -P $$ 2>/dev/null || true
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
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\ncells=%s\n' \
    "$mig_a" "$mig_b" "$repetitions" "$cells"
  printf 'mode_set=%s\nmodes=%s\n' "$mode_set" "${forward[*]}"
  printf 'n_gen=%s\ngrow_rows=%s\ncase_limit_s=%s\n' \
    "$n_gen" "$grow" "$limit_s"
  printf 'model=%s\n' "$(basename "$model")"
  printf 'engine_commit=%s\n' "$(git -C "$engine" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$parent_driver" "$child_driver" "$engine/build/bin/libllama.so" \
    "$engine/build/bin/libggml-cuda.so"
} >"$result_dir/metadata.txt"

paragraph="The scheduler keeps a run queue for every processor and moves a task between queues when the load is uneven. A page fault enters the kernel, which finds the mapping, allocates a frame, fills it, and updates the page table before the task resumes. "
context=
prefix=
paragraphs=
use_prefix() {  # paragraphs: sets the prefix file and the context of the cases
  paragraphs=$1
  context=4096
  while (( context < paragraphs * 52 + 512 )); do context=$(( context * 2 )); done
  prefix="$work/prefix.$paragraphs"
  if [[ ! -e "$prefix" ]]; then
    for _ in $(seq 1 "$paragraphs"); do printf '%s' "$paragraph"; done >"$prefix"
  fi
}
text_of() {  # output file
  { grep '^TEXT ' "$1" 2>/dev/null || true; } | sha256sum | cut -c1-16
}

run_case() {  # placement mode children
  local placement=$1 mode=$2 count=$3 index envs=() migs=() state="$work/state"
  local file="$huge_dir/kv"
  case $mode in
    copy) ;;
    demand) envs=(LLAMA_KV_VMM=2 LLAMA_KV_GROW="$grow") ;;
    device) envs=(LLAMA_KV_VMM=1 LLAMA_KV_GROW="$grow") ;;
    device_tuned) envs=(LLAMA_KV_VMM=1 LLAMA_KV_VMM_BATCH=1 LLAMA_KV_GROW="$grow") ;;
    device_merged) envs=(LLAMA_KV_VMM=1 LLAMA_KV_VMM_BATCH=2 LLAMA_KV_GROW="$grow") ;;
    cow | writable) envs=(LLAMA_KV_HOST="$file" LLAMA_KV_GROW="$grow" LLAMA_KV_COW=1) ;;
    writable_no_read) envs=(LLAMA_KV_HOST="$file" LLAMA_KV_GROW="$grow" LLAMA_KV_COW=2) ;;
    extent) envs=(LLAMA_KV_HOST="$file" LLAMA_KV_GROW="$grow") ;;
    no_read) envs=(LLAMA_KV_HOST="$file" LLAMA_KV_GROW="$grow" LLAMA_KV_TOUCH=0) ;;
    no_populate) envs=(LLAMA_KV_HOST="$file" LLAMA_KV_GROW="$grow" LLAMA_KV_POPULATE=0) ;;
    grow_1024) envs=(LLAMA_KV_HOST="$file" LLAMA_KV_GROW=1024) ;;
    grow_4096) envs=(LLAMA_KV_HOST="$file" LLAMA_KV_GROW=4096) ;;
    grow_all) envs=(LLAMA_KV_HOST="$file") ;;
    small_pages) file="$small_dir/kv"; envs=(LLAMA_KV_HOST="$file" LLAMA_KV_GROW="$grow") ;;
    host_copy) envs=(LLAMA_KV_HOST=anon LLAMA_KV_GROW="$grow") ;;
    host_copy_huge) envs=(LLAMA_KV_HOST=anon) ;;
    *) echo "unknown mode $mode" >&2; exit 2 ;;
  esac
  for index in $(seq 0 $((count - 1))); do
    if [[ "$placement" == cross ]] && (( index % 2 == 1 )); then
      migs+=("$mig_b")
    else
      migs+=("$mig_a")
    fi
  done
  rm -rf "$state" "$work/go" "$huge_dir/kv.0" "$small_dir/kv.0" "$work/out"
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
    printf 'CASE placement=%s mode=%s children=%s paragraphs=%s context=%s\n' \
      "$placement" "$mode" "$count" "$paragraphs" "$context"
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
  rm -rf "$state" "$work/go" "$huge_dir/kv.0" "$small_dir/kv.0" "$work/out"
}

backward=()
for mode in "${forward[@]}"; do backward=("$mode" "${backward[@]}"); done
for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then modes=("${forward[@]}"); else modes=("${backward[@]}"); fi
  printf 'BEGIN_KVMECH run=%s\n' "$run" >>"$raw_log"
  for cell in $cells; do
    IFS=: read -r placement count size <<<"$cell"
    use_prefix "$size"
    for mode in "${modes[@]}"; do
      run_case "$placement" "$mode" "$count"
    done
  done
  printf 'END_KVMECH\n' >>"$raw_log"
done

awk -v order="${forward[*]}" -f "$script_dir/summarize_engine_kvmech.awk" "$raw_log" \
  >"$result_dir/engine_kvmech_summary.csv"
(
  cd "$script_dir"
  sha256sum kv_tuned.patch kv_spawn.cpp kv_fork.cpp summarize_engine_kvmech.awk \
    run_engine_kvmech.sh
) >"$result_dir/source_hashes.txt"
printf 'engine_kvmech_result_dir=%s\n' "$result_dir"
