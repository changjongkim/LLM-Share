#!/usr/bin/env bash
# Extents under the stock server, published at a token boundary. This is
# run_engine_kvserver.sh with one change in the client: the publisher
# evaluates and saves the prefix together with the two line breaks that
# begin every task, so that every published token is a token of the prompt
# of every agent. In the first campaign the last published token was not
# (the tokenizer joins the end of the prefix with the line breaks), the
# server evaluated it again, and extents placed it in another cell than the
# copy does; 36 of 72 responses then differed from the copy after at least
# 22 tokens. Expectation, fixed before this campaign: with every published
# token kept, gate V2 holds for every response.
#
# A publisher is one llama-server process
# that evaluates the prefix of the agent workload in its slot and saves the
# slot through the HTTP interface of the server. N agent servers, every
# second one in the other MIG instance, restore the slot through the same
# interface and each answer one request, prefix + task, with the prompt
# cache on. The server binary and its requests are the same in both modes;
# only the environment of the engine differs:
#
#   copy    the slot file carries the rows; every agent server copies them
#           into its own device memory
#   extent  the cache of the publisher is a file on a tmpfs with 2 MiB
#           pages and the slot file only names its rows; every agent server
#           maps the rows read-only and keeps its own rows in private memory
#           that follows use
#
# The servers run with one slot, without warm-up, without the prompt cache
# in host memory and without context checkpoints, since those features save
# the state of a slot on their own. The weights are read in place.
#
# Kill gates, fixed before the campaign:
#   V1  every request of every server succeeds;
#   V2  every agent server on extents returns the text of the agent server
#       that restored a copy in the same repetition;
#   V3  an agent server evaluates at most 256 tokens of its prompt in either
#       mode (the restored prefix is reused, not computed again);
#   V4  the agent servers on extents need less memory than those on copies;
#   V5  restoring the slot is not slower with extents than with copies, and
#       saving it is not slower either.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
agent_counts=${AGENT_COUNTS:-"4 8"}
prefix_file=${PREFIX_FILE:-"$script_dir/workloads/agent_prefix.txt"}
tasks_file=${TASKS_FILE:-"$script_dir/workloads/agent_tasks.txt"}
context=${CONTEXT:-32768}
n_gen=${N_GEN:-64}
grow=${GROW_ROWS:-256}
base_port=${BASE_PORT:-18080}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
engine=${CHAIN_ENGINE:-"$script_dir/llama.cpp-chain"}
server="$engine/build/bin/llama-server"
client="$script_dir/kv_server_client2.py"
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-kvserver2"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
for file in "$server" "$model" "$prefix_file" "$tasks_file" "$client"; do
  [[ -r "$file" ]] || { echo "missing $file" >&2; exit 2; }
done
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
live_pids=()
stop_servers() {
  if (( ${#live_pids[@]} )); then
    kill "${live_pids[@]}" 2>/dev/null || true
    local pid
    for pid in "${live_pids[@]}"; do
      for _ in $(seq 1 100); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.05
      done
      kill -9 "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    done
  fi
  live_pids=()
}
cleanup() {
  stop_servers
  if (( mounted_huge )); then sudo -n umount "$huge_dir" || true; fi
  rmdir "$huge_dir" "$mount_root" 2>/dev/null || true
  if [[ -n "$work" ]]; then rm -rf "$work"; fi
}
trap cleanup EXIT
mkdir -p "$huge_dir" "$result_dir"
sudo -n mount -t tmpfs -o "huge=always,size=8192m,uid=$(id -u),gid=$(id -g)" \
  tmpfs "$huge_dir"
mounted_huge=1
work=$(mktemp -d "$result_dir/work.XXXXXX")
mkdir "$work/slots"

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\nagent_counts=%s\n' \
    "$mig_a" "$mig_b" "$repetitions" "$agent_counts"
  printf 'context=%s\nn_gen=%s\ngrow_rows=%s\n' "$context" "$n_gen" "$grow"
  printf 'prefix_sha256=%s\ntasks_sha256=%s\n' \
    "$(sha256sum "$prefix_file" | cut -d' ' -f1)" \
    "$(sha256sum "$tasks_file" | cut -d' ' -f1)"
  printf 'model=%s\n' "$(basename "$model")"
  printf 'engine_commit=%s\n' "$(git -C "$engine" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$server" "$engine/build/bin/libllama.so" \
    "$engine/build/bin/libggml-cuda.so"
} >"$result_dir/metadata.txt"

start_server() {  # mig port log env...
  local mig=$1 port=$2 log=$3
  shift 3
  env CUDA_VISIBLE_DEVICES="$mig" GGML_CUDA_HOST_PTR=1 "$@" "$server" \
    -m "$model" -c "$context" -ngl 99 -np 1 --host 127.0.0.1 --port "$port" \
    --slot-save-path "$work/slots" --cache-ram 0 --ctx-checkpoints 0 \
    --no-webui --no-warmup >"$log" 2>&1 &
  live_pids+=($!)
}

run_case() {  # mode agents
  local mode=$1 agents=$2 index mig failed=0
  local publisher_env=() agent_env=()
  rm -f "$work"/slots/* "$huge_dir"/* "$work"/server.*
  if [[ "$mode" == extent ]]; then
    publisher_env=(LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_GROW="$grow")
  fi
  printf 'CASE mode=%s agents=%s\n' "$mode" "$agents" >>"$raw_log"
  start_server "$mig_a" "$base_port" "$work/server.publisher" "${publisher_env[@]}"
  local published=""
  if python3 "$client" wait "$base_port" 300; then
    published=$(python3 "$client" publish "$base_port" "$prefix_file" prefix.bin \
      2>"$work/publish.err" || true)
  fi
  if [[ -z "$published" ]]; then
    {
      tail -n 5 "$work/server.publisher" | sed 's/^/STDERR /'
      printf 'END_CASE failed=%s\n' "$((agents + 1))"
    } >>"$raw_log"
    stop_servers
    return 0
  fi
  printf '%s\n' "$published" >>"$raw_log"
  local tokens
  tokens=$(sed -n 's/^PUBLISHED prefix_tokens=\([0-9]*\).*/\1/p' <<<"$published")
  if [[ "$mode" == extent ]]; then
    agent_env=(LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_PREFIX="$tokens"
      LLAMA_KV_GROW="$grow")
  fi

  local before lowest sample ports=()
  before=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
  lowest=$before
  for index in $(seq 0 $((agents - 1))); do
    if (( index % 2 == 0 )); then mig=$mig_a; else mig=$mig_b; fi
    ports+=("$((base_port + 1 + index))")
    start_server "$mig" "$((base_port + 1 + index))" "$work/server.$index" \
      "${agent_env[@]}"
  done
  for index in "${!ports[@]}"; do
    python3 "$client" wait "${ports[$index]}" 300 || failed=$((failed + 1))
  done
  python3 "$client" agents "$prefix_file" "$tasks_file" "$n_gen" prefix.bin \
    "${ports[@]}" >"$work/agents.out" 2>"$work/agents.err" &
  local client_pid=$!
  while kill -0 "$client_pid" 2>/dev/null; do
    sample=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
    (( sample < lowest )) && lowest=$sample
    sleep 0.1
  done
  wait "$client_pid" || true
  {
    cat "$work/agents.out"
    failed=$(( failed + $(grep -c ' exit=1' "$work/agents.out" || true) ))
    if (( failed > 0 )); then
      for index in "${!ports[@]}"; do
        grep -i -E 'error|cannot|failed' "$work/server.$index" | tail -n 2 |
          sed "s/^/STDERR server=$index /" || true
      done
    fi
    printf 'MEMORY mem_available_drop_mib=%s file_mib=%s\n' \
      "$(( (before - lowest) / 1024 ))" \
      "$(( $(du -sk "$huge_dir" | cut -f1) / 1024 ))"
    printf 'END_CASE failed=%s\n' "$failed"
  } >>"$raw_log"
  stop_servers
}

for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then modes=(copy extent); else modes=(extent copy); fi
  printf 'BEGIN_KVSERVER run=%s context=%s prefix_bytes=%s\n' "$run" "$context" \
    "$(stat -c %s "$prefix_file")" >>"$raw_log"
  for agents in $agent_counts; do
    for mode in "${modes[@]}"; do
      run_case "$mode" "$agents"
    done
  done
  printf 'END_KVSERVER\n' >>"$raw_log"
done

awk -f "$script_dir/summarize_engine_kvserver.awk" "$raw_log" \
  >"$result_dir/engine_kvserver_summary.csv"
(
  cd "$script_dir"
  sha256sum kv_chain.patch kv_server_client2.py summarize_engine_kvserver.awk \
    run_engine_kvserver2.sh
) >"$result_dir/source_hashes.txt"
printf 'engine_kvserver_result_dir=%s\n' "$result_dir"
