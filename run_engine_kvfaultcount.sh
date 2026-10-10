#!/usr/bin/env bash
# The pages that the kernel faults in on behalf of the GPU, by the way an agent
# obtains the prefix. The driver services a GPU fault in its bottom-half kernel
# thread ("UVM GPU<n> BH"), which calls the fault handler of the kernel for the
# pages of the faulting block; the kernel counts these calls as page faults of
# that thread (/proc/<pid>/stat, minor and major). No other work of the thread
# faults: the counter stands still while the GPU computes on device memory.
#
# A publisher computes the prefix and exits. Then AGENTS agents, every second
# one in the other MIG instance, continue it with a task of their own, and the
# counter is read before they start and after the last one has left.
#
#   copy          the prefix is restored from a state file into device memory
#   extent        the prefix is mapped read-only and read once from the CPU;
#                 the private extent is populated ahead of the GPU
#   no_read       the same without the CPU read (LLAMA_KV_TOUCH=0)
#   no_populate   the same without populating ahead (LLAMA_KV_POPULATE=0)
#   cow           the cache file as a writable private mapping, after a CPU read
#   cow_noread    the same as the kernel provides it, without the CPU read
#
# In addition, a publisher that keeps generating after it has published (a
# fork) is run alone, with the counter read while it computes the prefix and
# after it has published: `fork_copy` saves a state file with the rows,
# `fork_extent` publishes its cache file, and `fork_refill` does the same with
# kv_refill_shim.c preloaded, a prototype that puts the entries of the 2 MiB
# block in which each tensor of the prefix ends back from the CPU after the
# protection change (the kernel drops the huge mapping of a block that the
# change divides).
#
# Gates, fixed before the campaign:
#   F1  no process fails;
#   F2  the counter does not move in copy, in extent, and while a publisher
#       computes its prefix;
#   F3  it moves in no_read, no_populate, cow and cow_noread;
#   F4  the agents of every mode write the texts of the agents of copy.
# The count after a publisher has published is reported, not gated.
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
driver=${DRIVER:-"$script_dir/kv_fork_tuned"}
engine=${TUNED_ENGINE:-"$script_dir/llama.cpp-tuned"}
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-kvfaultcount"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if [[ ! -x "$driver" || ! -r "$model" ]]; then
  echo "missing driver or model" >&2
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

# The page faults of the bottom-half threads of the driver.
fault_pages() {
  local total=0 stat rest pid
  for stat in /proc/[0-9]*/stat; do
    { read -r line <"$stat"; } 2>/dev/null || continue
    [[ "$line" == *"(UVM GPU"*" BH)"* ]] || continue
    rest=${line##*) }
    # shellcheck disable=SC2086
    set -- $rest
    total=$(( total + $8 + ${10} ))
  done
  printf '%s' "$total"
}
if [[ "$(fault_pages)" == 0 ]] && ! grep -qs 'UVM GPU' /proc/[0-9]*/comm; then
  echo "no bottom-half thread of the driver found" >&2
  exit 2
fi

huge_dir="$mount_root/kv_huge"
mounted=0
cleanup() {
  if (( mounted )); then sudo -n umount "$huge_dir" || true; fi
  rmdir "$huge_dir" "$mount_root" 2>/dev/null || true
}
trap cleanup EXIT
mkdir -p "$huge_dir" "$result_dir"
sudo -n mount -t tmpfs -o "huge=always,size=4096m,uid=$(id -u),gid=$(id -g)" tmpfs "$huge_dir"
mounted=1
work=$(mktemp -d "$result_dir/work.XXXXXX")
cc -O2 -shared -fPIC -o "$work/refill.so" "$script_dir/kv_refill_shim.c" -ldl

context=4096
while (( context < paragraphs * 52 + 512 )); do context=$(( context * 2 )); done
paragraph="The scheduler keeps a run queue for every processor and moves a task between queues when the load is uneven. A page fault enters the kernel, which finds the mapping, allocates a frame, fills it, and updates the page table before the task resumes. "
prefix="$work/prefix"
for _ in $(seq 1 "$paragraphs"); do printf '%s' "$paragraph"; done >"$prefix"
suffix_of() { printf '\n\nTask for agent %s: list three checks before a release.\n' "$1"; }

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\nagents=%s\nparagraphs=%s\n' \
    "$mig_a" "$mig_b" "$repetitions" "$agents" "$paragraphs"
  printf 'context=%s\nn_gen=%s\ngrow_rows=%s\n' "$context" "$n_gen" "$grow"
  printf 'model=%s\n' "$(basename "$model")"
  printf 'engine_commit=%s\n' "$(git -C "$engine" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$driver" "$engine/build/bin/libllama.so" "$engine/build/bin/libggml-cuda.so"
} >"$result_dir/metadata.txt"

publish() {  # store: writes the state of the prefix and leaves
  local store=$1 envs=(CUDA_VISIBLE_DEVICES="$mig_a" GGML_CUDA_HOST_PTR=1)
  if [[ "$store" == file ]]; then
    envs+=(LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_GROW="$grow")
  fi
  local before after
  before=$(fault_pages)
  env "${envs[@]}" "$driver" parent "$model" "$context" "$prefix" "$work/state.$store" "" 0 \
    >"$work/publish.$store" 2>"$work/publish.$store.err" || true
  after=$(fault_pages)
  printf 'PUBLISH store=%s fault_pages=%s %s\n' "$store" "$(( after - before ))" \
    "$(sed -n 's/^PUBLISHED //p' "$work/publish.$store")" >>"$raw_log"
}

run_agents() {  # mode
  local mode=$1 store=file envs=(GGML_CUDA_HOST_PTR=1) index mig pids=() failed=0
  local tokens
  tokens=$(sed -n 's/^.*prefix_tokens=\([0-9]*\).*$/\1/p' "$work/publish.file" | head -n 1)
  case $mode in
    copy) store=rows ;;
    extent) envs+=(LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_PREFIX="$tokens" LLAMA_KV_GROW="$grow") ;;
    no_read) envs+=(LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_PREFIX="$tokens" LLAMA_KV_GROW="$grow" LLAMA_KV_TOUCH=0) ;;
    no_populate) envs+=(LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_PREFIX="$tokens" LLAMA_KV_GROW="$grow" LLAMA_KV_POPULATE=0) ;;
    cow) envs+=(LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_PREFIX="$tokens" LLAMA_KV_GROW="$grow" LLAMA_KV_COW=1) ;;
    cow_noread) envs+=(LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_PREFIX="$tokens" LLAMA_KV_GROW="$grow" LLAMA_KV_COW=2) ;;
    *) echo "unknown mode $mode" >&2; exit 2 ;;
  esac
  local before after
  before=$(fault_pages)
  for index in $(seq 0 $((agents - 1))); do
    if (( index % 2 == 0 )); then mig=$mig_a; else mig=$mig_b; fi
    env CUDA_VISIBLE_DEVICES="$mig" "${envs[@]}" "$driver" child "$model" "$context" \
      "$work/state.$store" "$(suffix_of "$index")" "$n_gen" \
      >"$work/agent.$index" 2>"$work/agent.$index.err" &
    pids+=($!)
  done
  {
    printf 'CASE mode=%s agents=%s\n' "$mode" "$agents"
    for index in "${!pids[@]}"; do
      local code=0 result text
      wait "${pids[$index]}" || code=$?
      result=$(grep '^RESULT ' "$work/agent.$index" || true)
      text=$({ grep '^TEXT ' "$work/agent.$index" || true; } | sha256sum | cut -c1-16)
      if (( code != 0 )) || [[ -z "$result" ]]; then
        failed=$((failed + 1))
        text=none
        sed 's/^/STDERR /' "$work/agent.$index.err" | tail -n 3
      fi
      printf 'AGENT index=%s exit=%s text=%s %s\n' "$index" "$code" "$text" "${result#RESULT }"
    done
    after=$(fault_pages)
    printf 'END_CASE failed=%s fault_pages=%s\n' "$failed" "$(( after - before ))"
  } >>"$raw_log"
}

run_fork() {  # mode: a publisher that keeps generating after it has published
  local mode=$1 envs=(CUDA_VISIBLE_DEVICES="$mig_a" GGML_CUDA_HOST_PTR=1) pre=()
  if [[ "$mode" != fork_copy ]]; then
    envs+=(LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_GROW="$grow")
  fi
  if [[ "$mode" == fork_refill ]]; then pre=(LD_PRELOAD="$work/refill.so"); fi
  rm -f "$huge_dir/kv.0" "$work/state.fork" "$work/go"
  local start mid end code=0 pid
  start=$(fault_pages)
  env "${envs[@]}" "${pre[@]}" "$driver" parent "$model" "$context" "$prefix" "$work/state.fork" \
    "$(suffix_of parent)" "$n_gen" "$work/go" >"$work/fork" 2>"$work/fork.err" &
  pid=$!
  until grep -q '^PUBLISHED ' "$work/fork" 2>/dev/null; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.02
  done
  mid=$(fault_pages)
  touch "$work/go"
  wait "$pid" || code=$?
  end=$(fault_pages)
  printf 'FORK mode=%s exit=%s fault_pages_before_publish=%s fault_pages_after_publish=%s %s %s\n' \
    "$mode" "$code" "$(( mid - start ))" "$(( end - mid ))" \
    "$(sed -n 's/^PUBLISHED //p' "$work/fork")" \
    "$(sed -n 's/^RESULT role=parent prefix_tokens=[0-9]* //p' "$work/fork")" >>"$raw_log"
}

forward=(copy extent no_read no_populate cow cow_noread)
backward=()
for mode in "${forward[@]}"; do backward=("$mode" "${backward[@]}"); done
: >"$raw_log"
for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then modes=("${forward[@]}"); else modes=("${backward[@]}"); fi
  printf 'BEGIN_KVFAULTCOUNT run=%s\n' "$run" >>"$raw_log"
  run_fork fork_copy
  run_fork fork_extent
  run_fork fork_refill
  rm -f "$huge_dir/kv.0"
  publish rows
  publish file
  chmod 0400 "$huge_dir/kv.0"
  for mode in "${modes[@]}"; do
    run_agents "$mode"
  done
  printf 'END_KVFAULTCOUNT\n' >>"$raw_log"
  rm -f "$huge_dir/kv.0" "$work"/state.*
done

awk -f "$script_dir/summarize_engine_kvfaultcount.awk" "$raw_log" \
  >"$result_dir/engine_kvfaultcount_summary.csv"
(
  cd "$script_dir"
  sha256sum kv_tuned.patch kv_fork.cpp kv_refill_shim.c summarize_engine_kvfaultcount.awk \
    run_engine_kvfaultcount.sh
) >"$result_dir/source_hashes.txt"
rm -rf "$work"
printf 'engine_kvfaultcount_result_dir=%s\n' "$result_dir"
