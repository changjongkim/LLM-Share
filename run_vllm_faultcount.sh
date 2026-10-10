#!/usr/bin/env bash
# The pages that the kernel faults in on behalf of the GPU while two vLLM
# servers hand a prefix over, with and without the plugin vllm_stator.
# run_engine_kvfaultcount.sh describes the counter: the driver services a GPU
# fault in its bottom-half kernel thread, and the kernel counts the calls of
# its fault handler as page faults of that thread.
#
#   vllm     two unmodified servers; weights and cache are device memory
#   stator   the plugin: the first server publishes its weights and the cache
#            blocks of the prefix as host mappings, the second maps both
#
# A case: server 1 starts in the 12-SM instance and answers one request with
# the prefix alone. When that request has finished, the plugin publishes the
# blocks of the prefix: it changes their protection to read-only, in one range
# whose two ends lie inside 2 MiB blocks of the cache file. Server 1 then
# answers two agents, one after the other. Server 2 starts in the 8-SM
# instance, where the plugin maps the published blocks, and answers two
# agents, one after the other. The counter is read around every step.
#
# Fixed before the campaign:
#   C1  every server starts and every request completes;
#   C2  the counter does not move in any step of `vllm`.
# The pages of every step of `stator` are reported, not gated: the question
# of the campaign is how many pages the publisher faults on after the
# protection change, and whether the count returns to zero afterwards.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repetitions=${REPETITIONS:-3}
modes=${MODES:-"vllm stator"}
agents=${AGENTS:-8}
prefix_file=${PREFIX_FILE:-"$script_dir/workloads/agent_prefix.txt"}
tasks_file=${TASKS_FILE:-"$script_dir/workloads/agent_tasks.txt"}
context=${CONTEXT:-16384}
n_gen=${N_GEN:-64}
kv_cache_gib=${KV_CACHE_GIB:-2}
port=${PORT:-8210}
start_limit_s=${START_LIMIT_S:-900}
first_mig=${FIRST_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
second_mig=${SECOND_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
venv=${VLLM_VENV:-"$script_dir/ext/venv-vllm"}
model_dir=${VLLM_MODEL:-"$(ls -d "$script_dir"/ext/hf/hub/models--Qwen--Qwen2.5-7B-Instruct-AWQ/snapshots/* | head -n 1)"}
extra_args=${VLLM_ARGS:-"-O0 --enable-prompt-tokens-details --gpu-memory-utilization 0.4"}
cuda_include=${CUDA_INCLUDE:-/usr/local/cuda-13.0/include}
model_link=${VLLM_MODEL_LINK:-"$script_dir/ext/hf/qwen-awq"}
mount_dir=${SHARE_MOUNT:-"$script_dir/vllm_share_mount"}
mount_mib=${SHARE_MOUNT_MIB:-$(( kv_cache_gib * 1024 + 8192 ))}
client="$script_dir/vllm_share_client.py"
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-vllm-faultcount"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"
connector='{"kv_connector":"StatorConnector","kv_connector_module_path":"stator_vllm.connector","kv_role":"kv_both"}'

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]]; then
  echo "REPETITIONS must be a positive integer" >&2
  exit 2
fi
if [[ ! -x "$venv/bin/vllm" || ! -d "$model_dir" ]]; then
  echo "missing the vLLM environment or the model; see ext/FEASIBILITY.md" >&2
  exit 2
fi
if ! "$venv/bin/python" -c 'import stator_vllm' 2>/dev/null; then
  echo "the plugin is not installed: $venv/bin/pip install --no-deps -e vllm_stator" >&2
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
  local total=0 stat rest line
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

ln -sfn "$model_dir" "$model_link"

pids=()
stop_servers() {
  local pid
  for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for _ in $(seq 1 20); do
    local alive=0
    for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
    (( alive )) || break
    sleep 0.5
  done
  for pid in "${pids[@]}"; do
    pkill -KILL -P "$pid" 2>/dev/null || true
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
  pids=()
}
cleanup() {
  stop_servers
  if mountpoint -q "$mount_dir"; then sudo -n umount "$mount_dir" || true; fi
  rmdir "$mount_dir" 2>/dev/null || true
}
trap cleanup EXIT
mkdir -p "$result_dir"
# shellcheck disable=SC1091
source "$script_dir/ext/env-torch.sh" "$venv"
mkdir -p "$mount_dir"
mountpoint -q "$mount_dir" || sudo -n mount -t tmpfs \
  -o "huge=always,size=${mount_mib}m,uid=$(id -u),gid=$(id -g)" tmpfs "$mount_dir"

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'repetitions=%s\nmodes=%s\nagents=%s\ncontext=%s\nn_gen=%s\nkv_cache_gib=%s\n' \
    "$repetitions" "$modes" "$agents" "$context" "$n_gen" "$kv_cache_gib"
  printf 'first_mig=%s\nsecond_mig=%s\n' "$first_mig" "$second_mig"
  printf 'model_dir=%s\n' "$(basename "$(dirname "$(dirname "$model_dir")")")"
  for package in vllm torch stator-vllm; do
    printf '%s=%s\n' "$package" \
      "$("$venv/bin/python" -c "import importlib.metadata as m; print(m.version('$package'))")"
  done
  printf 'extra_args=%s\n' "$extra_args"
  printf 'shmem_huge=%s\n' "$(cat /sys/kernel/mm/transparent_hugepage/shmem_enabled)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
} >"$result_dir/metadata.txt"

ports_free() {
  local busy
  for _ in $(seq 1 60); do
    busy=$(ss -ltn 2>/dev/null | grep -cE ":($port|$(( port + 1 ))) " || true)
    (( busy == 0 )) && return 0
    sleep 0.5
  done
  return 1
}

# serve DEVICE PORT LOG [NAME=VALUE...] -- [ARGUMENT...]
serve() {
  local device=$1 server_port=$2 log=$3
  shift 3
  local -a assignments=()
  while [[ $1 != -- ]]; do assignments+=("$1"); shift; done
  shift
  # shellcheck disable=SC2086
  env CUDA_VISIBLE_DEVICES="$device" HF_HUB_OFFLINE=1 HF_HOME="$script_dir/ext/hf" PYTHONHASHSEED=0 \
    TRITON_CUDART_PATH="$cuda_include" TRITON_CUDACRT_PATH="$cuda_include" "${assignments[@]}" \
    "$venv/bin/python" -m stator_vllm.serve "$model_link" --port "$server_port" \
    --served-model-name stator-share \
    --max-model-len "$context" --max-num-seqs "$agents" \
    --kv-cache-memory-bytes "$(( kv_cache_gib * 1073741824 ))" $extra_args "$@" >"$log" 2>&1 &
  pids+=($!)
  local started
  started=$(date +%s%N)
  # Ready, dead, or out of time.
  while kill -0 "${pids[-1]}" 2>/dev/null && (( ($(date +%s%N) - started) / 1000000000 < start_limit_s )); do
    if python3 "$client" wait "$server_port" 1; then return 0; fi
  done
  return 1
}

# step NAME COMMAND...: the pages that the kernel faults in for the GPU while
# the command runs; what the command prints goes to the log before the count.
step() {
  local name=$1 pages started code=0
  shift
  pages=$(fault_pages)
  started=$(date +%s%N)
  "$@" >>"$raw_log" || code=$?
  printf 'STEP mode=%s name=%s pages=%s wall_ms=%s exit=%s\n' "$mode" "$name" \
    "$(( $(fault_pages) - pages ))" "$(( ($(date +%s%N) - started) / 1000000 ))" "$code" >>"$raw_log"
  return "$code"
}
agent() {  # PORT TASK
  python3 "$client" agents stator-share "$prefix_file" "$tasks_file" "$n_gen" 1 "$1" "$2"
}

read -r -a mode_list <<<"$modes"
for run in $(seq 1 "$repetitions"); do
  printf 'BEGIN_VFAULT run=%s\n' "$run" >>"$raw_log"
  for position in $(seq 0 $(( ${#mode_list[@]} - 1 ))); do
    mode=${mode_list[$(( (position + run - 1) % ${#mode_list[@]} ))]}
    env1=() env2=() arguments=()
    case $mode in
      vllm) ;;
      stator)
        env1=(STATOR_WEIGHTS_PUBLISH="$mount_dir/weights" STATOR_KV_PUBLISH="$mount_dir/kv")
        env2=(STATOR_WEIGHTS_ATTACH="$mount_dir/weights" STATOR_KV_ATTACH="$mount_dir/kv")
        arguments=(--kv-transfer-config "$connector") ;;
      *) echo "unknown mode $mode" >&2; exit 2 ;;
    esac
    ports_free || { echo "the ports of the servers are in use" >&2; exit 1; }
    failed=0
    printf 'CASE mode=%s kv_cache_gib=%s\n' "$mode" "$kv_cache_gib" >>"$raw_log"
    if step start1 serve "$first_mig" "$port" "$result_dir/server1.$mode.$run.log" "${env1[@]}" -- "${arguments[@]}" &&
       step prefix python3 "$client" prefix stator-share "$prefix_file" "$port"; then
      # The publish runs when the request of the prefix has finished.
      sleep 2
      step publisher_first agent "$port" 30 || failed=1
      step publisher_second agent "$port" 31 || failed=1
      if step start2 serve "$second_mig" "$(( port + 1 ))" "$result_dir/server2.$mode.$run.log" "${env2[@]}" -- "${arguments[@]}"; then
        step attacher_first agent "$(( port + 1 ))" 32 || failed=1
        step attacher_second agent "$(( port + 1 ))" 33 || failed=1
      else
        failed=1
        tail -n 20 "$result_dir/server2.$mode.$run.log" | sed 's/^/SERVER_LOG slot=2 /' >>"$raw_log"
      fi
    else
      failed=1
      tail -n 20 "$result_dir/server1.$mode.$run.log" | sed 's/^/SERVER_LOG slot=1 /' >>"$raw_log"
    fi
    for slot in 1 2; do
      log="$result_dir/server$slot.$mode.$run.log"
      [[ -f $log ]] || continue
      grep -o 'stator: .*' "$log" | sed "s/^stator: /PLUGIN slot=$slot /" >>"$raw_log" || true
    done
    stop_servers
    rm -f "$mount_dir"/weights "$mount_dir"/weights.json "$mount_dir"/kv "$mount_dir"/kv.json 2>/dev/null || true
    printf 'END_CASE failed=%s\n' "$failed" >>"$raw_log"
    sleep 3
  done
done

# vLLM writes the network address of the host into its log.
sed -i -E 's#tcp://[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:#tcp://HOST-ADDRESS:#g' "$result_dir"/server*.log
awk -f "$script_dir/summarize_vllm_faultcount.awk" "$raw_log" >"$result_dir/vllm_faultcount_summary.csv"
(
  cd "$script_dir"
  sha256sum openai_client.py vllm_share_client.py summarize_vllm_faultcount.awk run_vllm_faultcount.sh \
    vllm_stator/pyproject.toml vllm_stator/stator_vllm/__init__.py vllm_stator/stator_vllm/connector.py \
    vllm_stator/stator_vllm/serve.py
) >"$result_dir/source_hashes.txt"
printf 'vllm_faultcount_result_dir=%s\n' "$result_dir"
