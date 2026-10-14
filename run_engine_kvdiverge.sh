#!/usr/bin/env bash
# Why children of STATOR (Map Only) write a text different from that of a copy
# in 2 of 96 cases with 16 children and 10 of 192 with 32 (campaign
# 20261014-engine-kvlinux-scaling-full-v1), while every child of STATOR and of
# the other mechanisms writes it. This runner is run_engine_kvlinux.sh with the
# texts of every child kept in texts/ and with the two steps that Map Only
# leaves out removed one at a time:
#
#   copy         the engine's state file restored into each child
#   extent       STATOR
#   ro_plain     STATOR without the CPU read and without populate-ahead
#   no_read      STATOR without the CPU read of the mapped prefix
#   no_populate  STATOR without populate-ahead of the private extent
#
# Nothing is gated; the record reports, per mode, the children whose text
# differs from the child of copy with the same index in the same repetition,
# and where the two texts first differ.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-4}
mode_set=diverge
forward=(copy extent ro_plain no_read no_populate)
default_cells="same:32:320"
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
parent_driver=${PARENT_DRIVER:-"$script_dir/kv_spawn_linux"}
child_driver=${CHILD_DRIVER:-"$script_dir/kv_fork_linux"}
probe=${PROBE:-"$script_dir/linux_share_probe"}
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-kvdiverge"}
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
mkdir -p "$huge_dir" "$result_dir/texts"
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
    no_read) envs=(LLAMA_KV_HOST="$file" LLAMA_KV_GROW="$grow" LLAMA_KV_TOUCH=0) ;;
    no_populate) envs=(LLAMA_KV_HOST="$file" LLAMA_KV_GROW="$grow" LLAMA_KV_POPULATE=0) ;;
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
      { grep '^TEXT ' "$work/out/child.$index" 2>/dev/null || true; } \
        >"$result_dir/texts/$run.$mode.$count.$index.txt"
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
    summarize_engine_kvlinux.awk run_engine_kvdiverge.sh
) >"$result_dir/source_hashes.txt"
printf 'engine_kvdiverge_result_dir=%s\n' "$result_dir"
