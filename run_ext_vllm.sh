#!/usr/bin/env bash
# A serving system that shares a prefix inside one process as a baseline:
# vLLM, which serves all agents as sequences of one engine and reuses the
# cache blocks of a common prefix. One server runs in the 12-SM MIG
# instance; eight agents ask for the continuation of the same prefix, each
# with its own task, in two rounds: the first with an empty server (cold),
# the second with the same prompts again (warm).
#
# vLLM does not read the GGUF file of the other campaigns. The server loads
# the same model, Qwen2.5-7B-Instruct, from the 4-bit AWQ weights that its
# authors publish, so that the size of the weights is comparable. vLLM
# reserves its cache at the start; the reservation is KV_CACHE_GIB.
# Memory is the drop of MemAvailable from before the server starts.
#
# What it took to run vLLM 0.20.0 on this platform (ext/FEASIBILITY.md has
# the installation): the modules xgrammar, compressed-tensors and triton
# added to its environment; the device selected by index, since vLLM parses
# CUDA_VISIBLE_DEVICES as integers and index 0 is the 12-SM instance (the
# 6-SM instance cannot be selected this way); the CUDA headers named to
# Triton (TRITON_CUDART_PATH); and compilation with Inductor switched off
# (-O0).
#
# Validity criteria, fixed before the campaign:
#   X1  the server starts in every repetition and every request completes;
#   X2  cold and warm first token, throughput and memory are reported, and
#       whether the warm round reuses the prefix is read from the prompt
#       tokens that the server reports as cached; it is a result, not a gate.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repetitions=${REPETITIONS:-6}
agents=${AGENTS:-8}
prefix_file=${PREFIX_FILE:-"$script_dir/workloads/agent_prefix.txt"}
tasks_file=${TASKS_FILE:-"$script_dir/workloads/agent_tasks.txt"}
context=${CONTEXT:-16384}
n_gen=${N_GEN:-64}
kv_cache_gib=${KV_CACHE_GIB:-8}
port=${PORT:-8200}
start_limit_s=${START_LIMIT_S:-900}
venv=${VLLM_VENV:-"$script_dir/ext/venv-vllm"}
model_dir=${VLLM_MODEL:-"$(ls -d "$script_dir"/ext/hf/hub/models--Qwen--Qwen2.5-7B-Instruct-AWQ/snapshots/* | head -n 1)"}
extra_args=${VLLM_ARGS:-"-O0 --enable-prompt-tokens-details"}
device_index=${VLLM_DEVICE_INDEX:-0}
cuda_include=${CUDA_INCLUDE:-/usr/local/cuda-13.0/include}
client="$script_dir/openai_client.py"
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-ext-vllm"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
if [[ ! -x "$venv/bin/vllm" || ! -d "$model_dir" ]]; then
  echo "missing the vLLM environment or the model; see ext/FEASIBILITY.md" >&2
  exit 2
fi
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi

server_pid=
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill -TERM "$server_pid" 2>/dev/null || true
    sleep 2
    pkill -KILL -P "$server_pid" 2>/dev/null || true
    kill -KILL "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
mkdir -p "$result_dir"
# shellcheck disable=SC1091
source "$script_dir/ext/env-torch.sh" "$venv"

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'repetitions=%s\nagents=%s\ncontext=%s\nn_gen=%s\nkv_cache_gib=%s\n' \
    "$repetitions" "$agents" "$context" "$n_gen" "$kv_cache_gib"
  printf 'model_dir=%s\nmodel_bytes=%s\n' "$(basename "$(dirname "$(dirname "$model_dir")")")" \
    "$(du -sbL "$model_dir" | cut -f1)"
  printf 'vllm=%s\ntorch=%s\n' \
    "$("$venv/bin/python" -c 'import importlib.metadata as m; print(m.version("vllm"))')" \
    "$("$venv/bin/python" -c 'import importlib.metadata as m; print(m.version("torch"))')"
  printf 'extra_args=%s\ndevice_index=%s\n' "$extra_args" "$device_index"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
} >"$result_dir/metadata.txt"

meminfo_kib() { awk -v name="$1:" '$1 == name { print $2 }' /proc/meminfo; }

for run in $(seq 1 "$repetitions"); do
  before=$(meminfo_kib MemAvailable)
  started=$(date +%s%N)
  # shellcheck disable=SC2086
  env CUDA_VISIBLE_DEVICES="$device_index" HF_HUB_OFFLINE=1 HF_HOME="$script_dir/ext/hf" \
    TRITON_CUDART_PATH="$cuda_include" TRITON_CUDACRT_PATH="$cuda_include" \
    "$venv/bin/vllm" serve "$model_dir" --port "$port" --served-model-name stator-baseline \
    --max-model-len "$context" --max-num-seqs "$agents" \
    --kv-cache-memory-bytes "$(( kv_cache_gib * 1073741824 ))" $extra_args \
    >"$result_dir/server.$run.log" 2>&1 &
  server_pid=$!
  ready=1
  python3 "$client" wait "$port" "$start_limit_s" || ready=0
  startup_ms=$(( ($(date +%s%N) - started) / 1000000 ))
  ready_drop=$(( (before - $(meminfo_kib MemAvailable)) / 1024 ))
  lowest=$(meminfo_kib MemAvailable)
  {
    printf 'BEGIN_VLLM run=%s\n' "$run"
    printf 'SERVER ready=%s alive=%s startup_ms=%s ready_drop_mib=%s\n' "$ready" \
      "$(kill -0 "$server_pid" 2>/dev/null && echo 1 || echo 0)" "$startup_ms" "$ready_drop"
  } >>"$raw_log"
  if (( ready )); then
    for round in cold warm; do
      python3 "$client" agents stator-baseline "$prefix_file" "$tasks_file" "$n_gen" "$port" \
        "$agents" >"$result_dir/round.out" &
      client_pid=$!
      while kill -0 "$client_pid" 2>/dev/null; do
        sample=$(meminfo_kib MemAvailable)
        (( sample < lowest )) && lowest=$sample
        sleep 0.05
      done
      wait "$client_pid" || true
      {
        printf 'ROUND name=%s\n' "$round"
        cat "$result_dir/round.out"
        printf 'MEMORY round=%s peak_drop_mib=%s\n' "$round" "$(( (before - lowest) / 1024 ))"
      } >>"$raw_log"
    done
    rm -f "$result_dir/round.out"
  else
    tail -n 20 "$result_dir/server.$run.log" | sed 's/^/SERVER_LOG /' >>"$raw_log"
  fi
  cleanup
  wait "$server_pid" 2>/dev/null || true
  server_pid=
  printf 'END_VLLM\n' >>"$raw_log"
  sleep 3
done

awk -f "$script_dir/summarize_ext_vllm.awk" "$raw_log" >"$result_dir/ext_vllm_summary.csv"
(
  cd "$script_dir"
  sha256sum openai_client.py summarize_ext_vllm.awk run_ext_vllm.sh
) >"$result_dir/source_hashes.txt"
printf 'ext_vllm_result_dir=%s\n' "$result_dir"
