#!/usr/bin/env bash
# One serving process: the engine with weights copied to device memory (the
# upstream behaviour on an integrated GPU) against weights read in place from
# the mapped model file. Measures tokens per second, load time, where the
# weights live, and whether the generated text is identical.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
bin_dir=${BIN_DIR:-"$script_dir/llama.cpp/build/bin"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-single"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"
prompt="Write a detailed technical essay about how operating systems manage memory."

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi
for file in "$model" "$bin_dir/llama-bench" "$bin_dir/llama-completion"; do
  [[ -e "$file" ]] || { echo "missing $file" >&2; exit 2; }
done

mkdir -p "$result_dir"
{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\n' "$mig_a" "$mig_b" "$repetitions"
  printf 'model=%s\nmodel_bytes=%s\n' "$(basename "$model")" "$(stat -c %s "$model")"
  printf 'model_sha256=%s\n' "$(sha256sum "$model" | cut -d' ' -f1)"
  printf 'engine_commit=%s\n' "$(git -C "$script_dir/llama.cpp" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$bin_dir/llama-bench" "$bin_dir/llama-completion"
} >"$result_dir/metadata.txt"

# Sets the engine environment of a weights mode.
#   copy            upstream: weights are read into device memory
#   inplace         the mapped file is the weight buffer; the CPU reads every
#                   page once at load
#   inplace_noread  the same without the CPU read
engine_env() {
  case $1 in
    copy) printf '' ;;
    inplace) printf 'GGML_CUDA_HOST_PTR=1' ;;
    inplace_noread) printf 'GGML_CUDA_HOST_PTR=1 GGML_CUDA_HOST_PTR_PREREAD=0' ;;
  esac
}

modes=(copy inplace inplace_noread)
migs=("$mig_a" "$mig_b")
for run in $(seq 1 "$repetitions"); do
  mig=${migs[$((run % 2))]}
  for offset in 0 1 2; do
    mode=${modes[$(((run + offset) % 3))]}
    read -r -a extra <<<"$(engine_env "$mode")"
    printf 'BEGIN_ENGINE_SINGLE run=%s mig=%s mode=%s\n' "$run" "${mig:4:8}" \
      "$mode" >>"$raw_log"

    # Throughput.
    env CUDA_VISIBLE_DEVICES="$mig" "${extra[@]}" "$bin_dir/llama-bench" \
      -m "$model" -ngl 99 -p 512 -n 128 -r 3 -o jsonl 2>/dev/null |
      sed 's/^/BENCH /' >>"$raw_log"

    # Load time, memory while generating, and the generated text.
    available_before=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
    env CUDA_VISIBLE_DEVICES="$mig" "${extra[@]}" "$bin_dir/llama-completion" \
      -m "$model" -ngl 99 -n 256 -c 4096 --temp 0 --seed 1 -no-cnv --ignore-eos \
      -p "$prompt" >"$result_dir/text.tmp" 2>"$result_dir/perf.tmp" &
    pid=$!
    sleep 5
    available_during=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
    mapped=$(awk -v file="$model" '
      /^[0-9a-f]+-[0-9a-f]+ / { inside = ($6 == file) }
      inside && /^Rss:/ { rss += $2 }
      inside && /^Pss:/ { pss += $2 }
      END { printf "model_rss_mib=%.0f model_pss_mib=%.0f", rss / 1024, pss / 1024 }
    ' "/proc/$pid/smaps" 2>/dev/null || true)
    status=0
    wait "$pid" || status=$?
    {
      printf 'MEMORY mem_available_drop_mib=%s %s\n' \
        "$(( (available_before - available_during) / 1024 ))" "$mapped"
      grep -E 'load time|prompt eval time|  eval time' "$result_dir/perf.tmp" |
        sed 's/^.*common_perf_print: */PERF /'
      printf 'OUTPUT bytes=%s sha256=%s\n' "$(stat -c %s "$result_dir/text.tmp")" \
        "$(sha256sum "$result_dir/text.tmp" | cut -d' ' -f1)"
      printf 'END_ENGINE_SINGLE status=%s\n' "$status"
    } >>"$raw_log"
  done
done
rm -f "$result_dir/text.tmp" "$result_dir/perf.tmp"

awk -f "$script_dir/summarize_engine_single.awk" "$raw_log" \
  >"$result_dir/engine_single_summary.csv"
(
  cd "$script_dir"
  sha256sum inplace_weights.patch summarize_engine_single.awk run_engine_single.sh
) >"$result_dir/source_hashes.txt"
printf 'engine_single_result_dir=%s\n' "$result_dir"
