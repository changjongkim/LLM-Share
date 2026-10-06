#!/usr/bin/env bash
# Is the key-value cache of a prefix the same bits wherever it is computed?
# Computes the cache of one prefix repeatedly in each MIG instance, into a
# file, and compares the files: between repetitions in one instance, and
# between the two instances. The answer decides what the text of an agent
# that attaches to a prefix computed in the other instance may be compared
# with.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
paragraphs=${PARAGRAPHS:-"5 20 80"}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
engine=${KV_ENGINE:-"$script_dir/llama.cpp-kv"}
driver=${DRIVER:-"$script_dir/kv_fork"}
store_root=${STORE_ROOT:-/dev/shm}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-kvdet"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if [[ ! -x "$driver" || ! -r "$model" ]]; then
  echo "missing driver or model; run make and see README.md" >&2
  exit 2
fi
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi
mkdir -p "$result_dir"
work=$(mktemp -d "$result_dir/work.XXXXXX")
store=$(mktemp -d "$store_root/llm_share_kvdet.XXXXXX")
trap 'rm -rf "$work" "$store"' EXIT

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\nparagraphs=%s\n' \
    "$mig_a" "$mig_b" "$repetitions" "$paragraphs"
  printf 'model=%s\n' "$(basename "$model")"
  printf 'engine_commit=%s\n' "$(git -C "$engine" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$driver" "$engine/build/bin/libllama.so" \
    "$engine/build/bin/libggml-cuda.so"
} >"$result_dir/metadata.txt"

paragraph="The scheduler keeps a run queue for every processor and moves a task between queues when the load is uneven. A page fault enters the kernel, which finds the mapping, allocates a frame, fills it, and updates the page table before the task resumes. "
compute() {  # mig context prefix-file name
  rm -f "$store/$4.0"
  env CUDA_VISIBLE_DEVICES="$1" GGML_CUDA_HOST_PTR=1 LLAMA_KV_HOST="$store/$4" \
    "$driver" parent "$model" "$2" "$3" "$work/state" "" 0 \
    >"$work/out" 2>"$work/err" || true
  sed -n 's/^PUBLISHED \(prefix_tokens=[0-9]*\).*/\1/p' "$work/out"
}

for count in $paragraphs; do
  context=4096
  while (( context < count * 52 + 512 )); do context=$(( context * 2 )); done
  # One key or value tensor: a row of 1,024 bytes for every cell.
  tensor_bytes=$(( context * 1024 ))
  prefix="$work/prefix.$count"
  : >"$prefix"
  for _ in $(seq 1 "$count"); do printf '%s' "$paragraph" >>"$prefix"; done
  for run in $(seq 1 "$repetitions"); do
    {
      printf 'BEGIN_KVDET run=%s paragraphs=%s context=%s\n' "$run" "$count" "$context"
      tokens=$(compute "$mig_a" "$context" "$prefix" a)
      printf 'CACHE instance=a %s sha256=%s\n' "${tokens:-failed}" \
        "$(sha256sum "$store/a.0" | cut -c1-16)"
      tokens=$(compute "$mig_b" "$context" "$prefix" b)
      printf 'CACHE instance=b %s sha256=%s\n' "${tokens:-failed}" \
        "$(sha256sum "$store/b.0" | cut -c1-16)"
      printf 'ACROSS %s\n' \
        "$(python3 "$script_dir/kv_file_diff.py" "$store/a.0" "$store/b.0" "$tensor_bytes")"
      printf 'END_KVDET\n'
    } >>"$raw_log"
  done
done

awk -f "$script_dir/summarize_engine_kvdet.awk" "$raw_log" \
  >"$result_dir/engine_kvdet_summary.csv"
(
  cd "$script_dir"
  sha256sum kv_extents.patch kv_fork.cpp kv_file_diff.py \
    summarize_engine_kvdet.awk run_engine_kvdet.sh
) >"$result_dir/source_hashes.txt"
printf 'engine_kvdet_result_dir=%s\n' "$result_dir"
