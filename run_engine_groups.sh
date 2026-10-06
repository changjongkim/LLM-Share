#!/usr/bin/env bash
# The configuration the measurements point to: one batching server process
# per MIG instance, so that sequences are batched inside a group and a fault
# of one group cannot reach the other, with the two servers reading one copy
# of the weights in place. Measures the generation rate of both servers at
# the same time and the memory they hold, against two servers with device
# copies.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
sequences=${SEQUENCES:-"4 8"}
n_prompt=${N_PROMPT:-512}
n_gen=${N_GEN:-128}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
bin_dir=${BIN_DIR:-"$script_dir/llama.cpp/build/bin"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-groups"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]]; then
  echo "REPETITIONS must be a positive integer" >&2
  exit 2
fi
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi
[[ -x "$bin_dir/llama-batched-bench" ]] || { echo "missing llama-batched-bench" >&2; exit 2; }

mkdir -p "$result_dir"
work=$(mktemp -d "$result_dir/work.XXXXXX")
{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\nsequences=%s\n' \
    "$mig_a" "$mig_b" "$repetitions" "$sequences"
  printf 'n_prompt=%s\nn_gen=%s\nmodel=%s\n' "$n_prompt" "$n_gen" "$(basename "$model")"
  printf 'engine_commit=%s\n' "$(git -C "$script_dir/llama.cpp" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$bin_dir/llama-batched-bench"
} >"$result_dir/metadata.txt"

run_pair() {  # run mode sequences
  local run=$1 mode=$2 count=$3 extra=() pids=() index=0 mig
  if [[ "$mode" == inplace ]]; then extra=(GGML_CUDA_HOST_PTR=1); fi
  local before lowest sample mapped=0
  before=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
  lowest=$before
  for mig in "$mig_a" "$mig_b"; do
    env CUDA_VISIBLE_DEVICES="$mig" "${extra[@]}" \
      "$bin_dir/llama-batched-bench" -m "$model" -ngl 99 -c 8192 -b 2048 \
      -ub 512 -npp "$n_prompt" -ntg "$n_gen" -npl "$count" \
      >"$work/server.$index" 2>/dev/null &
    pids+=($!)
    index=$((index + 1))
  done
  local smaps=()
  for index in "${pids[@]}"; do smaps+=("/proc/$index/smaps"); done
  while kill -0 "${pids[@]}" 2>/dev/null; do
    sample=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
    (( sample < lowest )) && lowest=$sample
    # shellcheck disable=SC2002
    sample=$(cat "${smaps[@]}" 2>/dev/null |
      awk -v file="$model" '
        /^[0-9a-f]+-[0-9a-f]+ / { inside = ($6 == file) }
        inside && /^Pss:/ { pss += $2 }
        END { printf "%d", pss / 1024 }' || true)
    (( ${sample:-0} > mapped )) && mapped=$sample
    sleep 0.5
  done
  local failed=0
  for index in "${!pids[@]}"; do
    wait "${pids[$index]}" || failed=$((failed + 1))
  done
  {
    printf 'BEGIN_ENGINE_GROUPS run=%s mode=%s sequences=%s\n' "$run" "$mode" "$count"
    grep -E '^\| +[0-9]' "$work/server.0" | sed 's/^/ROW instance=a /'
    grep -E '^\| +[0-9]' "$work/server.1" | sed 's/^/ROW instance=b /'
    printf 'MEMORY mem_available_drop_mib=%s model_mapped_mib=%s\n' \
      "$(( (before - lowest) / 1024 ))" "$mapped"
    printf 'END_ENGINE_GROUPS failed_servers=%s\n' "$failed"
  } >>"$raw_log"
}

for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then modes=(copy inplace); else modes=(inplace copy); fi
  for count in $sequences; do
    for mode in "${modes[@]}"; do
      run_pair "$run" "$mode" "$count"
    done
  done
done
rm -rf "$work"

awk -f "$script_dir/summarize_engine_groups.awk" "$raw_log" \
  >"$result_dir/engine_groups_summary.csv"
(
  cd "$script_dir"
  sha256sum inplace_weights.patch summarize_engine_groups.awk run_engine_groups.sh
) >"$result_dir/source_hashes.txt"
printf 'engine_groups_result_dir=%s\n' "$result_dir"
