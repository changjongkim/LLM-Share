#!/usr/bin/env bash
# Kernel samepage merging (KSM) as the kernel's mechanism for identical pages
# of different processes. A parent computes a prefix and publishes it, and N
# children continue it, every second one in the other MIG instance, and
# generate N_GEN tokens each, long enough for ksmd to merge their copies. The
# runner samples the available memory and the pages that KSM shares every
# SAMPLE_S seconds from the start of the children.
#
#   host_copy      the prefix copied into private host memory of each child,
#                  the weights read in place; KSM off
#   host_copy_ksm  the same, with every process asking KSM to merge all its
#                  anonymous memory (ksm_merge_shim.so, prctl
#                  PR_SET_MEMORY_MERGE, as systemd's MemoryKSM=) and ksmd
#                  scanning KSM_PAGES pages every KSM_SLEEP_MS milliseconds
#   device_ksm     the unmodified engine's placement: weights and cache in
#                  device memory of each process (no GGML_CUDA_HOST_PTR, the
#                  prefix restored from a state file), with KSM as above
#   extent         STATOR; KSM off
#
# The settings of ksmd are restored on exit; `run=2` before each case
# unmerges what a previous case left.
#
# Kill gates, fixed before the campaign:
#   S1  no process fails, and every child writes the text of the child of
#       host_copy in the same repetition and index;
#   S2  the highest memory of the children with host_copy_ksm is at least
#       0.9 times that with host_copy (KSM merges after the copies exist);
#   S3  with device_ksm, the pages that KSM shares at the end of a case are
#       less than 1% of the memory of the children.
# Reported, not gated: the memory at the end of a case, the time until KSM
# has merged half and 90% of what it merges, the CPU time of ksmd, the
# generation speed relative to host_copy, and the GPU page faults.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
read -r -a forward <<<"${MODES:-host_copy host_copy_ksm device_ksm extent}"
children=${CHILDREN:-8}
paragraphs=${PARAGRAPHS:-320}
n_gen=${N_GEN:-1024}
grow=${GROW_ROWS:-256}
sample_s=${SAMPLE_S:-0.5}
ksm_pages=${KSM_PAGES:-20000}
ksm_sleep_ms=${KSM_SLEEP_MS:-1}
limit_s=${CASE_LIMIT_S:-1500}
floor_mib=${FLOOR_MIB:-16384}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
engine=${TUNED_ENGINE:-"$script_dir/llama.cpp-linux"}
parent_driver=${PARENT_DRIVER:-"$script_dir/kv_spawn_linux"}
child_driver=${CHILD_DRIVER:-"$script_dir/kv_fork_linux"}
shim=${KSM_SHIM:-"$script_dir/ksm_merge_shim.so"}
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-kvksm"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"
ksm=/sys/kernel/mm/ksm

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
if [[ ! -x "$parent_driver" || ! -x "$child_driver" || ! -r "$shim" || ! -r "$model" ]]; then
  echo "missing drivers, shim or model; run make kv_spawn_linux kv_fork_linux ksm_merge_shim.so" >&2
  exit 2
fi
if ! sudo -n true 2>/dev/null || [[ ! -e "$ksm/run" ]]; then
  echo "passwordless sudo and KSM are required" >&2
  exit 2
fi
if [[ "$(cat "$ksm/pages_sharing")" != 0 ]]; then
  echo "KSM already shares pages of other processes" >&2
  exit 2
fi
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi

ksm_run_before=$(cat "$ksm/run")
ksm_pages_before=$(cat "$ksm/pages_to_scan")
ksm_sleep_before=$(cat "$ksm/sleep_millisecs")
ksm_set() {  # name value
  echo "$2" | sudo -n tee "$ksm/$1" >/dev/null
}
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
ksmd_ticks() {
  local pid line rest
  pid=$(pgrep -x ksmd | head -n 1)
  [[ -n "$pid" ]] || { printf '0'; return; }
  read -r line <"/proc/$pid/stat"
  rest=${line##*) }
  # shellcheck disable=SC2086
  set -- $rest
  printf '%s' "$(( ${12} + ${13} ))"
}
tick_ms=$(( 1000 / $(getconf CLK_TCK) ))

huge_dir="$mount_root/kv_huge"
mounted_huge=0
work=
parent_pid=
cleanup() {
  if [[ -n "$parent_pid" ]]; then kill -9 "$parent_pid" 2>/dev/null || true; fi
  pkill -9 -P $$ 2>/dev/null || true
  ksm_set run 2 || true
  ksm_set pages_to_scan "$ksm_pages_before" || true
  ksm_set sleep_millisecs "$ksm_sleep_before" || true
  ksm_set run "$ksm_run_before" || true
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
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\nmodes=%s\n' \
    "$mig_a" "$mig_b" "$repetitions" "${forward[*]}"
  printf 'children=%s\nparagraphs=%s\nn_gen=%s\ngrow_rows=%s\nsample_s=%s\n' \
    "$children" "$paragraphs" "$n_gen" "$grow" "$sample_s"
  printf 'ksm_pages_to_scan=%s\nksm_sleep_millisecs=%s\ncase_limit_s=%s\nfloor_mib=%s\n' \
    "$ksm_pages" "$ksm_sleep_ms" "$limit_s" "$floor_mib"
  printf 'ksm_before=run:%s,pages_to_scan:%s,sleep_millisecs:%s\n' \
    "$ksm_run_before" "$ksm_pages_before" "$ksm_sleep_before"
  printf 'model=%s\n' "$(basename "$model")"
  printf 'engine_commit=%s\n' "$(git -C "$engine" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$parent_driver" "$child_driver" "$shim" "$engine/build/bin/libllama.so" \
    "$engine/build/bin/libggml-cuda.so"
} >"$result_dir/metadata.txt"

paragraph="The scheduler keeps a run queue for every processor and moves a task between queues when the load is uneven. A page fault enters the kernel, which finds the mapping, allocates a frame, fills it, and updates the page table before the task resumes. "
context=4096
while (( context < paragraphs * 52 + 512 )); do context=$(( context * 2 )); done
prefix="$work/prefix"
for _ in $(seq 1 "$paragraphs"); do printf '%s' "$paragraph"; done >"$prefix"
text_of() {  # output file
  { grep '^TEXT ' "$1" 2>/dev/null || true; } | sha256sum | cut -c1-16
}

run_case() {  # mode
  local mode=$1 index envs=() migs=() state="$work/state" weights=(GGML_CUDA_HOST_PTR=1)
  local file="$huge_dir/kv" merge=0
  case $mode in
    host_copy) envs=(LLAMA_KV_HOST=anon LLAMA_KV_GROW="$grow") ;;
    host_copy_ksm) envs=(LLAMA_KV_HOST=anon LLAMA_KV_GROW="$grow" LD_PRELOAD="$shim"); merge=1 ;;
    device_ksm) envs=(LD_PRELOAD="$shim"); weights=(); merge=1 ;;
    extent) envs=(LLAMA_KV_HOST="$file" LLAMA_KV_GROW="$grow") ;;
    *) echo "unknown mode $mode" >&2; exit 2 ;;
  esac
  for index in $(seq 0 $((children - 1))); do
    if (( index % 2 == 1 )); then migs+=("$mig_b"); else migs+=("$mig_a"); fi
  done
  ksm_set run 2
  while [[ "$(cat "$ksm/pages_sharing")" != 0 ]]; do sleep 0.2; done
  ksm_set pages_to_scan "$ksm_pages"
  ksm_set sleep_millisecs "$ksm_sleep_ms"
  ksm_set run 0
  rm -rf "$state" "$work/go" "$huge_dir/kv.0" "$work/out" "$work/samples"
  mkdir -p "$work/out"
  env CUDA_VISIBLE_DEVICES="$mig_a" "${weights[@]}" "${envs[@]}" \
    "$parent_driver" "$model" "$context" "$prefix" "$state" "$n_gen" "$work/go" \
    "$work/out" "$child_driver" "${migs[@]}" >"$work/parent" 2>"$work/parent.err" &
  parent_pid=$!
  until grep -q '^PUBLISHED ' "$work/parent" 2>/dev/null; do
    kill -0 "$parent_pid" 2>/dev/null || break
    sleep 0.02
  done
  local before lowest highest_merge=0 sample failed=0 timed_out=0 guard=0 started now
  local faults_before faults_after ksmd_before ksmd_after shared sharing
  faults_before=$(fault_counters)
  ksmd_before=$(ksmd_ticks)
  before=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
  lowest=$before
  started=$(date +%s.%N)
  (( merge )) && ksm_set run 1
  touch "$work/go"
  while kill -0 "$parent_pid" 2>/dev/null; do
    sample=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
    shared=$(cat "$ksm/pages_shared"); sharing=$(cat "$ksm/pages_sharing")
    now=$(awk -v a="$started" -v b="$(date +%s.%N)" 'BEGIN { printf "%.2f", b - a }')
    printf 'SAMPLE t=%s drop_mib=%s pages_shared=%s pages_sharing=%s\n' \
      "$now" "$(( (before - sample) / 1024 ))" "$shared" "$sharing" >>"$work/samples"
    (( sample < lowest )) && lowest=$sample
    if (( sample < floor_mib * 1024 )); then
      guard=1
      pkill -9 -P "$parent_pid" 2>/dev/null || true
      kill -9 "$parent_pid" 2>/dev/null || true
    elif awk -v n="$now" -v l="$limit_s" 'BEGIN { exit !(n > l) }'; then
      timed_out=1
      pkill -9 -P "$parent_pid" 2>/dev/null || true
      kill -9 "$parent_pid" 2>/dev/null || true
    fi
    sleep "$sample_s"
  done
  ksm_set run 0
  faults_after=$(fault_counters)
  ksmd_after=$(ksmd_ticks)
  {
    printf 'CASE mode=%s children=%s paragraphs=%s context=%s\n' \
      "$mode" "$children" "$paragraphs" "$context"
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
    for index in $(seq 0 $((children - 1))); do
      code=$(sed -n "s/^CHILD index=$index exit=\\([0-9]*\\)\$/\\1/p" "$work/parent")
      result=$(grep '^RESULT ' "$work/out/child.$index" 2>/dev/null || true)
      if [[ "${code:-1}" != 0 || -z "$result" ]]; then
        failed=$((failed + 1))
        sed 's/^/STDERR /' "$work/out/child.$index.err" 2>/dev/null | tail -n 2
      fi
      printf 'CHILD index=%s exit=%s text=%s %s\n' "$index" "${code:-none}" \
        "$(text_of "$work/out/child.$index")" "${result#RESULT }"
    done
    cat "$work/samples"
    read -r pages_before ticks_before <<<"$faults_before"
    read -r pages_after ticks_after <<<"$faults_after"
    printf 'KERNEL gpu_fault_pages=%s fault_thread_ms=%s ksmd_ms=%s\n' \
      "$(( pages_after - pages_before ))" "$(( (ticks_after - ticks_before) * tick_ms ))" \
      "$(( (ksmd_after - ksmd_before) * tick_ms ))"
    printf 'MEMORY mem_available_drop_mib=%s\n' "$(( (before - lowest) / 1024 ))"
    printf 'END_CASE failed=%s timed_out=%s guard=%s\n' "$failed" "$timed_out" "$guard"
  } >>"$raw_log"
  rm -rf "$state" "$work/go" "$huge_dir/kv.0" "$work/out" "$work/samples"
}

backward=()
for mode in "${forward[@]}"; do backward=("$mode" "${backward[@]}"); done
for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then modes=("${forward[@]}"); else modes=("${backward[@]}"); fi
  printf 'BEGIN_KVKSM run=%s\n' "$run" >>"$raw_log"
  for mode in "${modes[@]}"; do run_case "$mode"; done
  printf 'END_KVKSM\n' >>"$raw_log"
done

awk -v order="${forward[*]}" -f "$script_dir/summarize_engine_kvksm.awk" "$raw_log" \
  >"$result_dir/engine_kvksm_summary.csv"
(
  cd "$script_dir"
  sha256sum kv_tuned2.patch kv_linux.patch kv_spawn.cpp kv_fork.cpp ksm_merge_shim.c \
    summarize_engine_kvksm.awk run_engine_kvksm.sh
) >"$result_dir/source_hashes.txt"
printf 'engine_kvksm_result_dir=%s\n' "$result_dir"
