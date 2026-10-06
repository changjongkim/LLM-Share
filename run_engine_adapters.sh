#!/usr/bin/env bash
# Four serving processes, each with its own LoRA adapter over one base
# model, two per MIG instance, at the same time. With weights copied to
# device memory every process holds the base; with weights read in place
# all four map one copy and keep only their adapter private. Checks that
# every adapter produces its own text and that the text does not depend on
# where the base lives.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
bin_dir=${BIN_DIR:-"$script_dir/llama.cpp/build/bin"}
adapter_dir=${ADAPTER_DIR:-"$script_dir/adapters"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-adapters"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"
prompt="Write a detailed technical essay about how operating systems manage memory."

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]]; then
  echo "REPETITIONS must be a positive integer" >&2
  exit 2
fi
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi
for seed in 1 2 3 4; do
  [[ -f "$adapter_dir/lora_$seed.gguf" ]] || {
    echo "missing adapter lora_$seed.gguf; run make_lora.py first" >&2
    exit 2
  }
done

mkdir -p "$result_dir"
work=$(mktemp -d "$result_dir/work.XXXXXX")
{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\n' "$mig_a" "$mig_b" "$repetitions"
  printf 'model=%s\nmodel_bytes=%s\n' "$(basename "$model")" "$(stat -c %s "$model")"
  printf 'engine_commit=%s\n' "$(git -C "$script_dir/llama.cpp" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$bin_dir/llama-completion" "$adapter_dir"/lora_[1-4].gguf
} >"$result_dir/metadata.txt"

# Starts one process per adapter (0 = no adapter) and records their texts
# and the memory they hold together.
run_group() {  # run mode
  local run=$1 mode=$2 adapter mig pids=() extra=() lora=()
  if [[ "$mode" == inplace ]]; then extra=(GGML_CUDA_HOST_PTR=1); fi
  local before lowest sample mapped=0
  before=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
  lowest=$before
  for adapter in 0 1 2 3 4; do
    if (( adapter % 2 == 0 )); then mig=$mig_a; else mig=$mig_b; fi
    lora=()
    if (( adapter > 0 )); then lora=(--lora "$adapter_dir/lora_$adapter.gguf"); fi
    env CUDA_VISIBLE_DEVICES="$mig" "${extra[@]}" "$bin_dir/llama-completion" \
      -m "$model" "${lora[@]}" -ngl 99 -n 128 -c 4096 --temp 0 --seed 1 \
      -no-cnv --ignore-eos -p "$prompt" >"$work/text.$adapter" \
      2>"$work/perf.$adapter" &
    pids+=($!)
  done
  local smaps=()
  for adapter in "${pids[@]}"; do smaps+=("/proc/$adapter/smaps"); done
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
  for adapter in "${!pids[@]}"; do
    wait "${pids[$adapter]}" || failed=$((failed + 1))
  done
  {
    printf 'BEGIN_ENGINE_ADAPTERS run=%s mode=%s\n' "$run" "$mode"
    for adapter in 0 1 2 3 4; do
      printf 'TEXT adapter=%s bytes=%s sha256=%s\n' "$adapter" \
        "$(stat -c %s "$work/text.$adapter")" \
        "$(sha256sum "$work/text.$adapter" | cut -d' ' -f1)"
    done
    printf 'MEMORY mem_available_drop_mib=%s model_mapped_mib=%s\n' \
      "$(( (before - lowest) / 1024 ))" "$mapped"
    printf 'END_ENGINE_ADAPTERS failed_processes=%s\n' "$failed"
  } >>"$raw_log"
}

for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then modes=(copy inplace); else modes=(inplace copy); fi
  for mode in "${modes[@]}"; do
    run_group "$run" "$mode"
  done
done
rm -rf "$work"

awk -f "$script_dir/summarize_engine_adapters.awk" "$raw_log" \
  >"$result_dir/engine_adapters_summary.csv"
(
  cd "$script_dir"
  sha256sum inplace_weights.patch make_lora.py summarize_engine_adapters.awk \
    run_engine_adapters.sh
) >"$result_dir/source_hashes.txt"
printf 'engine_adapters_result_dir=%s\n' "$result_dir"
