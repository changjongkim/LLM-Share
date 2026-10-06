#!/usr/bin/env bash
# The alternative to N serving processes: one process that decodes N
# sequences in a batch. It needs one copy of the weights by construction and
# gives no isolation between the sequences. Measured with the engine's own
# batched benchmark, with weights in device memory and read in place.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
n_prompt=${N_PROMPT:-512}
n_gen=${N_GEN:-128}
parallel=${PARALLEL:-"1,2,4,8"}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
bin_dir=${BIN_DIR:-"$script_dir/llama.cpp/build/bin"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-batched"}
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
[[ -x "$bin_dir/llama-batched-bench" ]] || { echo "missing llama-batched-bench" >&2; exit 2; }

mkdir -p "$result_dir"
{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\n' "$mig_a" "$mig_b" "$repetitions"
  printf 'n_prompt=%s\nn_gen=%s\nparallel=%s\n' "$n_prompt" "$n_gen" "$parallel"
  printf 'model=%s\n' "$(basename "$model")"
  printf 'engine_commit=%s\n' "$(git -C "$script_dir/llama.cpp" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$bin_dir/llama-batched-bench"
} >"$result_dir/metadata.txt"

migs=("$mig_a" "$mig_b")
for run in $(seq 1 "$repetitions"); do
  mig=${migs[$((run % 2))]}
  if (( run % 4 < 2 )); then modes=(copy inplace); else modes=(inplace copy); fi
  for mode in "${modes[@]}"; do
    extra=()
    if [[ "$mode" == inplace ]]; then extra=(GGML_CUDA_HOST_PTR=1); fi
    status=0
    {
      printf 'BEGIN_ENGINE_BATCHED run=%s mig=%s mode=%s\n' "$run" "${mig:4:8}" "$mode"
      env CUDA_VISIBLE_DEVICES="$mig" "${extra[@]}" \
        "$bin_dir/llama-batched-bench" -m "$model" -ngl 99 -c 8192 -b 2048 \
        -ub 512 -npp "$n_prompt" -ntg "$n_gen" -npl "$parallel" 2>/dev/null |
        grep -E '^\| +[0-9]' | sed 's/^/ROW /' || status=$?
      printf 'END_ENGINE_BATCHED status=%s\n' "$status"
    } >>"$raw_log"
  done
done

awk -f "$script_dir/summarize_engine_batched.awk" "$raw_log" \
  >"$result_dir/engine_batched_summary.csv"
(
  cd "$script_dir"
  sha256sum inplace_weights.patch summarize_engine_batched.awk run_engine_batched.sh
) >"$result_dir/source_hashes.txt"
printf 'engine_batched_result_dir=%s\n' "$result_dir"
