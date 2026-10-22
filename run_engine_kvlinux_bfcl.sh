#!/usr/bin/env bash
# The sharing mechanisms of the kernel against STATOR, in one engine. A parent
# computes a prefix and publishes it, and N children continue it with a task
# of their own, every second child in the other MIG instance, while the parent
# generates its own continuation. The weights are read in place in every mode,
# so the modes differ only in how a child obtains the rows of the prefix:
#
#   copy             the engine's state file, restored into the device memory
#                    of each child (no mechanism of the kernel)
#   host_copy        the same rows copied into private host memory that
#                    follows use (the memory that KSM could merge)
#   ro_plain         the cache file mapped read-only, as plain mmap does: no
#                    CPU read of the mapped rows, and the child's own rows in
#                    memory that the kernel provides on the GPU's first write
#   shared_rw        the same with the prefix mapped shared and writable, as
#                    plain shared memory does (LLAMA_KV_SHARED=1, kv_linux.patch)
#   cow_plain        the cache file as a private writable mapping without the
#                    CPU read: the copy-on-write of the kernel (CoW Map)
#   extent           STATOR: the prefix mapped read-only and read once from
#                    the CPU, the child's own rows in a private extent that
#                    the CPU populates ahead of the GPU
#   shared_rw_write  shared_rw, and after the publish a sharer with a defect
#                    in the parent's MIG instance writes the first word of
#                    every 2 MiB block of the cache file through the GPU,
#                    mapped as shared_rw maps it
#   extent_write     extent with the same sharer, mapped as extent maps it
#
# For every case the runner also reads the page faults that the driver's
# bottom-half thread ("UVM GPU<n> BH") takes on behalf of the GPU, and the CPU
# time of that thread, from the publish to the exit of the last process.
#
# MODE_SET=mechanisms (default): the eight modes, cell cross:8:320.
# MODE_SET=scaling: copy, ro_plain, cow_plain and extent, cells cross:N:320
#   for N = 1, 4, 8, 16 and 32.
#
# Kill gates, fixed before the campaign:
#   K1  no process fails in copy, host_copy, ro_plain, shared_rw, cow_plain
#       and extent;
#   K2  every child of these modes writes the text of the child that
#       received a copy in the same cell, repetition and index;
#   K3  at 8 children, the GPU takes fewer page faults with extent than with
#       each of ro_plain, shared_rw and cow_plain;
#   K4  at 8 children, cow_plain needs at least 1.5 times the memory of
#       extent, and ro_plain and shared_rw need no more than 1.1 times;
#   K5  in shared_rw_write the write succeeds and changes the file, and in
#       extent_write the write is refused and the file is unchanged, in every
#       case;
#   K6  in extent_write every child writes the text of copy.
# Reported, not gated: the texts of shared_rw_write, the first token and the
# fault-thread time by the number of children.
# Copy of run_engine_kvlinux.sh (review of 2026-10-22): the prefix of a cell of
# PARAGRAPHS paragraphs is the BFCL prefix of the same number of tokens
# (PARAGRAPHS * 51 + 1, workloads/bfcl_len, make_workload_bfcl_tokens.py), and
# the parent passes the request of line i of its tasks to child i
# (kv_spawn_linux_tasks, STATOR_TASKS_FILE). Everything else is unchanged.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
mode_set=${MODE_SET:-mechanisms}
case "$mode_set" in
  mechanisms)
    forward=(copy host_copy ro_plain shared_rw cow_plain extent shared_rw_write extent_write)
    default_cells="cross:8:320" ;;
  scaling)
    forward=(copy ro_plain cow_plain extent)
    default_cells="cross:1:320 cross:4:320 cross:8:320 cross:16:320 cross:32:320" ;;
  *) echo "MODE_SET must be mechanisms or scaling" >&2; exit 2 ;;
esac
if [[ -n "${MODES:-}" ]]; then read -r -a forward <<<"$MODES"; fi
cells=${CELLS:-$default_cells}
for cell in $cells; do
  if ! [[ "$cell" =~ ^(same|cross):[1-9][0-9]*:[1-9][0-9]*$ ]]; then
    echo "a cell is PLACEMENT:CHILDREN:PARAGRAPHS, not $cell" >&2
    exit 2
  fi
done
n_gen=${N_GEN:-64}
grow=${GROW_ROWS:-256}
limit_s=${CASE_LIMIT_S:-900}
floor_mib=${FLOOR_MIB:-16384}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
engine=${TUNED_ENGINE:-"$script_dir/llama.cpp-linux"}
parent_driver=${PARENT_DRIVER:-"$script_dir/kv_spawn_linux_tasks"}
prefix_dir=${PREFIX_DIR:-"$script_dir/workloads/bfcl_len"}
child_driver=${CHILD_DRIVER:-"$script_dir/kv_fork_linux"}
probe=${PROBE:-"$script_dir/linux_share_probe"}
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-kvlinux-$mode_set"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
if [[ ! -x "$parent_driver" || ! -x "$child_driver" || ! -x "$probe" || ! -r "$model" ]]; then
  echo "missing drivers, probe or model; run make kv_spawn_linux kv_fork_linux linux_share_probe" >&2
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

# The page faults and the CPU time (clock ticks) of the driver's bottom-half
# threads, which serve the faults of the GPU.
fault_counters() {
  local pages=0 ticks=0 stat line rest
  for stat in /proc/[0-9]*/stat; do
    { read -r line <"$stat"; } 2>/dev/null || continue
    [[ "$line" == *"(UVM GPU"*" BH)"* ]] || continue
    rest=${line##*) }
    # shellcheck disable=SC2086
    set -- $rest
    pages=$(( pages + $8 + ${10} ))
    ticks=$(( ticks + ${12} + ${13} ))
  done
  printf '%s %s' "$pages" "$ticks"
}
tick_ms=$(( 1000 / $(getconf CLK_TCK) ))

huge_dir="$mount_root/kv_huge"
mounted_huge=0
work=
parent_pid=
anchor_pid=
cleanup() {
  if [[ -n "$parent_pid" ]]; then kill -9 "$parent_pid" 2>/dev/null || true; fi
  if [[ -n "$anchor_pid" ]]; then kill "$anchor_pid" 2>/dev/null || true; fi
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

# The driver creates the threads that serve GPU faults while a process uses
# the GPU and removes them after the last one; a process that holds a context
# for the whole campaign keeps them, and their counters, between the cases.
CUDA_VISIBLE_DEVICES="$mig_a" "$probe" hold 1000000 >"$work/anchor" 2>&1 &
anchor_pid=$!
for _ in $(seq 1 100); do grep -qs 'UVM GPU' /proc/[0-9]*/comm && break; sleep 0.1; done
if ! grep -qs 'UVM GPU' /proc/[0-9]*/comm; then
  echo "no bottom-half thread of the driver found" >&2
  exit 2
fi

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\ncells=%s\n' \
    "$mig_a" "$mig_b" "$repetitions" "$cells"
  printf 'mode_set=%s\nmodes=%s\n' "$mode_set" "${forward[*]}"
  printf 'n_gen=%s\ngrow_rows=%s\ncase_limit_s=%s\nfloor_mib=%s\n' \
    "$n_gen" "$grow" "$limit_s" "$floor_mib"
  printf 'model=%s\n' "$(basename "$model")"
  printf 'prefix_dir=%s\n' "$prefix_dir"
  sha256sum "$prefix_dir"/bfcl_prefix_*.txt "$prefix_dir"/bfcl_tasks_*.txt
  printf 'engine_commit=%s\n' "$(git -C "$engine" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$parent_driver" "$child_driver" "$probe" "$engine/build/bin/libllama.so" \
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
  local tokens=$(( paragraphs * 51 + 1 ))
  prefix="$work/prefix.$paragraphs"
  export STATOR_TASKS_FILE="$prefix_dir/bfcl_tasks_$tokens.txt"
  if [[ ! -r "$prefix_dir/bfcl_prefix_$tokens.txt" || ! -r "$STATOR_TASKS_FILE" ]]; then
    echo "no BFCL prefix of $tokens tokens in $prefix_dir" >&2
    exit 2
  fi
  cp "$prefix_dir/bfcl_prefix_$tokens.txt" "$prefix"
}
text_of() {  # output file
  { grep '^TEXT ' "$1" 2>/dev/null || true; } | sha256sum | cut -c1-16
}

run_case() {  # placement mode children
  local placement=$1 mode=$2 count=$3 index envs=() migs=() state="$work/state"
  local file="$huge_dir/kv" writer=
  local plain=(LLAMA_KV_HOST="$file" LLAMA_KV_GROW="$grow" LLAMA_KV_TOUCH=0 LLAMA_KV_POPULATE=0)
  case $mode in
    copy) ;;
    host_copy) envs=(LLAMA_KV_HOST=anon LLAMA_KV_GROW="$grow") ;;
    ro_plain) envs=("${plain[@]}") ;;
    shared_rw) envs=("${plain[@]}" LLAMA_KV_SHARED=1) ;;
    shared_rw_write) envs=("${plain[@]}" LLAMA_KV_SHARED=1); writer=shared ;;
    cow_plain) envs=(LLAMA_KV_HOST="$file" LLAMA_KV_GROW="$grow" LLAMA_KV_COW=2) ;;
    extent) envs=(LLAMA_KV_HOST="$file" LLAMA_KV_GROW="$grow") ;;
    extent_write) envs=(LLAMA_KV_HOST="$file" LLAMA_KV_GROW="$grow"); writer=readonly ;;
    *) echo "unknown mode $mode" >&2; exit 2 ;;
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
  local writer_line=
  if [[ -n "$writer" ]] && kill -0 "$parent_pid" 2>/dev/null; then
    writer_line=$(CUDA_VISIBLE_DEVICES="$mig_a" "$probe" write_file "$huge_dir/kv.0" "$writer" \
      2>&1 | grep '^WRITER ' || true)
    [[ -n "$writer_line" ]] || writer_line="WRITER mapping=$writer gpu_write=none blocks_changed=0"
  fi
  local before lowest sample failed=0 timed_out=0 guard=0 started faults_before faults_after
  faults_before=$(fault_counters)
  before=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
  lowest=$before
  started=$(date +%s)
  touch "$work/go"
  while kill -0 "$parent_pid" 2>/dev/null; do
    sample=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
    (( sample < lowest )) && lowest=$sample
    if (( sample < floor_mib * 1024 )); then
      guard=1
      pkill -9 -P "$parent_pid" 2>/dev/null || true
      kill -9 "$parent_pid" 2>/dev/null || true
    elif (( $(date +%s) - started > limit_s )); then
      timed_out=1
      pkill -9 -P "$parent_pid" 2>/dev/null || true
      kill -9 "$parent_pid" 2>/dev/null || true
    fi
    sleep 0.05
  done
  faults_after=$(fault_counters)
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
    [[ -z "$writer_line" ]] || printf '%s\n' "$writer_line"
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
    read -r pages_before ticks_before <<<"$faults_before"
    read -r pages_after ticks_after <<<"$faults_after"
    printf 'KERNEL gpu_fault_pages=%s fault_thread_ms=%s\n' \
      "$(( pages_after - pages_before ))" "$(( (ticks_after - ticks_before) * tick_ms ))"
    printf 'MEMORY mem_available_drop_mib=%s\n' "$(( (before - lowest) / 1024 ))"
    printf 'END_CASE failed=%s timed_out=%s guard=%s\n' "$failed" "$timed_out" "$guard"
  } >>"$raw_log"
  rm -rf "$state" "$work/go" "$huge_dir/kv.0" "$work/out"
}

backward=()
for mode in "${forward[@]}"; do backward=("$mode" "${backward[@]}"); done
for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then modes=("${forward[@]}"); else modes=("${backward[@]}"); fi
  printf 'BEGIN_KVLINUX run=%s\n' "$run" >>"$raw_log"
  for cell in $cells; do
    IFS=: read -r placement count size <<<"$cell"
    use_prefix "$size"
    for mode in "${modes[@]}"; do
      run_case "$placement" "$mode" "$count"
    done
  done
  printf 'END_KVLINUX\n' >>"$raw_log"
done

awk -v order="${forward[*]}" -f "$script_dir/summarize_engine_kvlinux.awk" "$raw_log" \
  >"$result_dir/engine_kvlinux_summary.csv"
(
  cd "$script_dir"
  sha256sum kv_tuned2.patch kv_linux.patch kv_spawn.cpp kv_fork.cpp linux_share_probe.cu \
    summarize_engine_kvlinux.awk run_engine_kvlinux.sh
) >"$result_dir/source_hashes.txt"
printf 'engine_kvlinux_result_dir=%s\n' "$result_dir"
