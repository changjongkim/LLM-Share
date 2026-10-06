#!/usr/bin/env bash
# The device-memory counterpart of extents, in the engine. A parent computes
# a prefix, publishes it, starts four children in its own MIG instance, and
# generates its own continuation while they generate theirs. Three ways to
# hand the cache over:
#
#   copy    the engine's state file holds the rows; each child copies them
#           into its own device memory
#   extent  host memory: the parent's cache is a file on a tmpfs with 2 MiB
#           pages; a child maps the rows of the prefix read-only and keeps a
#           private tail that follows use
#   vmm     device memory: the parent's cache consists of 2 MiB allocations
#           made with CUDA's virtual memory interface; a child maps the
#           allocations that the prefix fills read-only, copies the one it
#           ends in, and maps allocations of its own behind them as it grows
#
# All three run in one engine build (llama.cpp-vmm) and through the same two
# programs. A fourth and fifth case start one child in the other MIG
# instance, with extent and with vmm: the host mapping crosses the instance
# boundary, and the import of device memory is expected to be refused.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
children=${CHILDREN:-4}
paragraphs=${PARAGRAPHS:-"80 320"}
n_gen=${N_GEN:-64}
grow=${GROW_ROWS:-256}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
engine=${VMM_ENGINE:-"$script_dir/llama.cpp-vmm"}
parent_driver=${PARENT_DRIVER:-"$script_dir/kv_spawn"}
child_driver=${CHILD_DRIVER:-"$script_dir/kv_fork_vmm"}
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-kvvmm"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
if [[ ! -x "$parent_driver" || ! -x "$child_driver" || ! -r "$model" ]]; then
  echo "missing drivers or model; run make and see README.md" >&2
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
  if [[ -n "$parent_pid" ]]; then kill "$parent_pid" 2>/dev/null || true; fi
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
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\nchildren=%s\n' \
    "$mig_a" "$mig_b" "$repetitions" "$children"
  printf 'paragraphs=%s\nn_gen=%s\ngrow_rows=%s\n' "$paragraphs" "$n_gen" "$grow"
  printf 'model=%s\n' "$(basename "$model")"
  printf 'engine_commit=%s\n' "$(git -C "$engine" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$parent_driver" "$child_driver" "$engine/build/bin/libllama.so" \
    "$engine/build/bin/libggml-cuda.so"
} >"$result_dir/metadata.txt"

paragraph="The scheduler keeps a run queue for every processor and moves a task between queues when the load is uneven. A page fault enters the kernel, which finds the mapping, allocates a frame, fills it, and updates the page table before the task resumes. "
text_of() {  # output file
  { grep '^TEXT ' "$1" || true; } | sha256sum | cut -c1-16
}

run_case() {  # mode context prefix-file parent-mig child-mig child-count
  local mode=$1 context=$2 prefix=$3 parent_mig=$4 child_mig=$5 count=$6
  local envs=() state="$work/state" index migs=()
  case $mode in
    extent | extent_cross)
      envs=(LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_GROW="$grow") ;;
    vmm | vmm_cross)
      envs=(LLAMA_KV_VMM=1 LLAMA_KV_GROW="$grow") ;;
  esac
  for index in $(seq 1 "$count"); do migs+=("$child_mig"); done
  rm -rf "$state" "$work/go" "$huge_dir/kv.0" "$work/out"
  mkdir -p "$work/out"
  env CUDA_VISIBLE_DEVICES="$parent_mig" GGML_CUDA_HOST_PTR=1 "${envs[@]}" \
    "$parent_driver" "$model" "$context" "$prefix" "$state" "$n_gen" "$work/go" \
    "$work/out" "$child_driver" "${migs[@]}" >"$work/parent" 2>"$work/parent.err" &
  parent_pid=$!
  until grep -q '^PUBLISHED ' "$work/parent" 2>/dev/null; do
    kill -0 "$parent_pid" 2>/dev/null || break
    sleep 0.02
  done
  local before lowest sample failed=0
  before=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
  lowest=$before
  touch "$work/go"
  while kill -0 "$parent_pid" 2>/dev/null; do
    sample=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
    (( sample < lowest )) && lowest=$sample
    sleep 0.05
  done
  {
    printf 'CASE mode=%s children=%s child_mig=%s\n' "$mode" "$count" "${child_mig:4:8}"
    local code=0 result published
    wait "$parent_pid" || code=$?
    parent_pid=
    result=$(grep '^RESULT ' "$work/parent" || true)
    published=$(grep '^PUBLISHED ' "$work/parent" || true)
    if (( code != 0 )) || [[ -z "$result" ]]; then
      failed=$((failed + 1))
      sed 's/^/STDERR /' "$work/parent.err" | tail -n 5
    fi
    printf 'PARENT exit=%s text=%s %s %s\n' "$code" "$(text_of "$work/parent")" \
      "${published#PUBLISHED }" \
      "$(sed 's/^RESULT role=parent prefix_tokens=[0-9]* //' <<<"$result")"
    for index in $(seq 0 $((count - 1))); do
      code=$(sed -n "s/^CHILD index=$index exit=\\([0-9]*\\)\$/\\1/p" "$work/parent")
      result=$(grep '^RESULT ' "$work/out/child.$index" 2>/dev/null || true)
      if [[ "${code:-1}" != 0 || -z "$result" ]]; then
        failed=$((failed + 1))
        sed 's/^/STDERR /' "$work/out/child.$index.err" 2>/dev/null | tail -n 3
      fi
      printf 'CHILD index=%s exit=%s text=%s %s\n' "$index" "${code:-none}" \
        "$(text_of "$work/out/child.$index")" "${result#RESULT }"
    done
    printf 'MEMORY mem_available_drop_mib=%s\n' "$(( (before - lowest) / 1024 ))"
    printf 'END_CASE failed=%s\n' "$failed"
  } >>"$raw_log"
  rm -rf "$state" "$work/go" "$huge_dir/kv.0" "$work/out"
}

for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then
    first=$mig_a other=$mig_b modes=(copy extent vmm extent_cross vmm_cross)
  else
    first=$mig_b other=$mig_a modes=(vmm_cross extent_cross vmm extent copy)
  fi
  for count in $paragraphs; do
    context=4096
    while (( context < count * 52 + 512 )); do context=$(( context * 2 )); done
    prefix="$work/prefix.$count"
    : >"$prefix"
    for _ in $(seq 1 "$count"); do printf '%s' "$paragraph" >>"$prefix"; done
    printf 'BEGIN_KVVMM run=%s paragraphs=%s context=%s parent_mig=%s\n' "$run" \
      "$count" "$context" "${first:4:8}" >>"$raw_log"
    for mode in "${modes[@]}"; do
      case $mode in
        *_cross) run_case "$mode" "$context" "$prefix" "$first" "$other" 1 ;;
        *) run_case "$mode" "$context" "$prefix" "$first" "$first" "$children" ;;
      esac
    done
    printf 'END_KVVMM\n' >>"$raw_log"
  done
done

awk -f "$script_dir/summarize_engine_kvvmm.awk" "$raw_log" \
  >"$result_dir/engine_kvvmm_summary.csv"
(
  cd "$script_dir"
  sha256sum kv_vmm.patch kv_spawn.cpp kv_fork.cpp summarize_engine_kvvmm.awk \
    run_engine_kvvmm.sh
) >"$result_dir/source_hashes.txt"
printf 'engine_kvvmm_result_dir=%s\n' "$result_dir"
