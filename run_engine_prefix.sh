#!/usr/bin/env bash
# What sharing a prompt prefix between serving processes costs today. An
# agent that starts with a long shared prefix (a system prompt, tool
# descriptions) either recomputes its key-value cache or restores it from the
# engine's prompt-cache file, which copies it. Measures both, with the
# weights read in place, as the baseline for sharing the prefix state itself.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
paragraphs=${PARAGRAPHS:-"20 80"}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
bin_dir=${BIN_DIR:-"$script_dir/llama.cpp/build/bin"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-prefix"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi
mkdir -p "$result_dir"
work=$(mktemp -d "$result_dir/work.XXXXXX")
{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\nparagraphs=%s\n' \
    "$mig_a" "$mig_b" "$repetitions" "$paragraphs"
  printf 'model=%s\n' "$(basename "$model")"
  printf 'engine_commit=%s\n' "$(git -C "$script_dir/llama.cpp" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$bin_dir/llama-completion"
} >"$result_dir/metadata.txt"

paragraph="The scheduler keeps a run queue for every processor and moves a task between queues when the load is uneven. A page fault enters the kernel, which finds the mapping, allocates a frame, fills it, and updates the page table before the task resumes. "
migs=("$mig_a" "$mig_b")

# One generation of a single token; prints the wall time in milliseconds.
generate() {  # mig prompt-file extra...
  local mig=$1 prompt=$2 start end
  shift 2
  start=$(date +%s%N)
  env CUDA_VISIBLE_DEVICES="$mig" GGML_CUDA_HOST_PTR=1 \
    "$bin_dir/llama-completion" -m "$model" -ngl 99 -n 1 -c 8192 --temp 0 \
    --seed 1 -no-cnv -f "$prompt" "$@" >"$work/text" 2>"$work/perf"
  end=$(date +%s%N)
  printf '%s' "$(( (end - start) / 1000000 ))"
}

for run in $(seq 1 "$repetitions"); do
  mig=${migs[$((run % 2))]}
  for count in $paragraphs; do
    prompt="$work/prompt.$count"
    : >"$prompt"
    for _ in $(seq 1 "$count"); do printf '%s' "$paragraph" >>"$prompt"; done
    cache="$work/cache.$count"
    rm -f "$cache"
    {
      printf 'BEGIN_ENGINE_PREFIX run=%s mig=%s paragraphs=%s\n' "$run" \
        "${mig:4:8}" "$count"
      wall=$(generate "$mig" "$prompt")
      printf 'CASE mode=recompute wall_ms=%s\n' "$wall"
      grep -E 'load time|prompt eval time' "$work/perf" |
        sed 's/^.*common_perf_print: */PERF recompute /'
      wall=$(generate "$mig" "$prompt" --prompt-cache "$cache" --prompt-cache-all)
      printf 'CASE mode=save wall_ms=%s cache_bytes=%s\n' "$wall" \
        "$(stat -c %s "$cache" 2>/dev/null || echo 0)"
      wall=$(generate "$mig" "$prompt" --prompt-cache "$cache" --prompt-cache-ro)
      printf 'CASE mode=restore wall_ms=%s\n' "$wall"
      grep -E 'load time|prompt eval time' "$work/perf" |
        sed 's/^.*common_perf_print: */PERF restore /'
      printf 'END_ENGINE_PREFIX\n'
    } >>"$raw_log"
  done
done
rm -rf "$work"

awk -f "$script_dir/summarize_engine_prefix.awk" "$raw_log" \
  >"$result_dir/engine_prefix_summary.csv"
(
  cd "$script_dir"
  sha256sum inplace_weights.patch summarize_engine_prefix.awk run_engine_prefix.sh
) >"$result_dir/source_hashes.txt"
printf 'engine_prefix_result_dir=%s\n' "$result_dir"
