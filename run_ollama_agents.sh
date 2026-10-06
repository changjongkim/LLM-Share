#!/usr/bin/env bash
# A deployed serving system as a baseline: Ollama, which serves the requests
# of all agents in one process per model. Eight agents ask for the
# continuation of the same prefix, each with its own task, in two rounds: the
# first with an empty server (cold), the second with the same prompts again,
# when every slot of the server holds the prefix (warm). Two configurations:
#
#   one  one server in the 12-SM MIG instance, eight parallel requests
#   two  one server in each MIG instance, four parallel requests each
#
# The servers are instances of their own (ports and model directory of this
# campaign); the Ollama service of the machine and its models are not used.
# The model is the GGUF file of the other campaigns, imported unchanged.
# Memory is the drop of MemAvailable from before the servers start, so that
# it contains what a server copies of the weights.
#
# Validity criteria, fixed before the campaign:
#   O1  every request completes and produces at least one token;
#   O2  every isolated server reports a non-zero GPU allocation through its
#       own /api/ps endpoint;
#   O3  the machine's Ollama service stays alive and is never addressed by
#       this runner; its PID is recorded before and after the campaign;
#   O4  cold and warm prompt work, memory, first token and throughput are
#       reported. Whether Ollama reuses the common prefix is a result, not a
#       gate: prompt tokens evaluated in the warm round decide it.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
agents=${AGENTS:-8}
paragraphs=${PARAGRAPHS:-320}
context=${CONTEXT:-32768}
n_gen=${N_GEN:-64}
base_port=${BASE_PORT:-11500}
ollama=${OLLAMA:-/usr/local/bin/ollama}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
store=${OLLAMA_STORE:-"$script_dir/ollama_models"}
ollama_library=${OLLAMA_LLM_LIBRARY:-cuda_v13}
name=stator-baseline
client="$script_dir/ollama_client.py"
visible_shim="$script_dir/ollama_mig_visible.so"
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-ollama-agents"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
for file in "$ollama" "$model" "$client" "$visible_shim"; do
  [[ -r "$file" ]] || { echo "missing $file" >&2; exit 2; }
done
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi
system_ollama_pid_before=$(systemctl show ollama -p MainPID --value 2>/dev/null || true)
for port in "$base_port" "$((base_port + 1))"; do
  if python3 -c 'import socket,sys; s=socket.socket(); sys.exit(s.connect_ex(("127.0.0.1", int(sys.argv[1]))))' "$port"; then
    echo "port is already in use: $port" >&2
    exit 2
  fi
done

work=
live_pids=()
stop_servers() {
  if (( ${#live_pids[@]} )); then
    local pid
    for pid in "${live_pids[@]}"; do
      kill -- "-$pid" 2>/dev/null || true
      for _ in $(seq 1 200); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.05
      done
      kill -9 -- "-$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    done
  fi
  live_pids=()
}
cleanup() {
  stop_servers
  if [[ -n "$work" ]]; then rm -rf "$work"; fi
}
trap cleanup EXIT
mkdir -p "$result_dir" "$store"
work=$(mktemp -d "$result_dir/work.XXXXXX")

paragraph="The scheduler keeps a run queue for every processor and moves a task between queues when the load is uneven. A page fault enters the kernel, which finds the mapping, allocates a frame, fills it, and updates the page table before the task resumes. "
prefix="$work/prefix"
: >"$prefix"
for _ in $(seq 1 "$paragraphs"); do printf '%s' "$paragraph" >>"$prefix"; done

start_server() {  # mig port parallel
  setsid env CUDA_VISIBLE_DEVICES="$1" OLLAMA_MIG_VISIBLE_DEVICE="$1" \
    LD_PRELOAD="$visible_shim${LD_PRELOAD:+:$LD_PRELOAD}" \
    OLLAMA_HOST="127.0.0.1:$2" OLLAMA_MODELS="$store" \
    OLLAMA_LLM_LIBRARY="$ollama_library" OLLAMA_CONTEXT_LENGTH="$context" \
    OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=f16 \
    OLLAMA_NUM_PARALLEL="$3" OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=30m \
    "$ollama" serve >"$work/server.$2" 2>&1 &
  live_pids+=($!)
  python3 "$client" wait "$2" 60
}

# The model is imported once, through a server of this campaign.
start_server "$mig_a" "$base_port" 1
if ! env OLLAMA_HOST="127.0.0.1:$base_port" "$ollama" list 2>/dev/null | grep -q "^$name"; then
  printf 'FROM %s\n' "$model" >"$work/Modelfile"
  env OLLAMA_HOST="127.0.0.1:$base_port" "$ollama" create "$name" -f "$work/Modelfile" \
    >"$work/create.log" 2>&1
fi
stop_servers

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\nagents=%s\n' "$mig_a" "$mig_b" \
    "$repetitions" "$agents"
  printf 'paragraphs=%s\ncontext=%s\nn_gen=%s\n' "$paragraphs" "$context" "$n_gen"
  printf 'model=%s\nollama=%s\nollama_library=%s\n' "$(basename "$model")" \
    "$("$ollama" --version 2>&1 | tail -n 1)" "$ollama_library"
  printf 'system_ollama_pid_before=%s\n' "$system_ollama_pid_before"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$ollama" "$visible_shim"
} >"$result_dir/metadata.txt"

run_case() {  # config
  local config=$1 ports=() unique_ports=() index before lowest sample round
  before=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
  lowest=$before
  if [[ "$config" == one ]]; then
    start_server "$mig_a" "$base_port" "$agents"
    for index in $(seq 1 "$agents"); do ports+=("$base_port"); done
  else
    start_server "$mig_a" "$base_port" "$(( agents / 2 ))"
    start_server "$mig_b" "$((base_port + 1))" "$(( agents / 2 ))"
    for index in $(seq 1 "$agents"); do ports+=("$(( base_port + index % 2 ))"); done
  fi
  mapfile -t unique_ports < <(printf '%s\n' "${ports[@]}" | sort -n -u)
  printf 'CASE config=%s agents=%s\n' "$config" "$agents" >>"$raw_log"
  for round in cold warm; do
    python3 "$client" agents "$name" "$prefix" "$n_gen" "$context" "${ports[@]}" \
      >"$work/agents.out" 2>"$work/agents.err" &
    local client_pid=$!
    while kill -0 "$client_pid" 2>/dev/null; do
      sample=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
      (( sample < lowest )) && lowest=$sample
      sleep 0.1
    done
    wait "$client_pid" || true
    : >"$work/status.out"
    for index in "${unique_ports[@]}"; do
      python3 "$client" status "$index" >>"$work/status.out" 2>>"$work/agents.err" || true
    done
    {
      sed "s/^AGENT /AGENT round=$round /" "$work/agents.out"
      cat "$work/status.out"
      printf 'ROUND round=%s failed=%s mem_available_drop_mib=%s servers=%s gpu_servers=%s vram_mib=%s\n' \
        "$round" "$(grep -c ' exit=1' "$work/agents.out" || true)" \
        "$(( (before - lowest) / 1024 ))" \
        "${#unique_ports[@]}" "$(awk '{ for (i=1;i<=NF;i++) if ($i=="gpu=1") n++ } END { print n+0 }' "$work/status.out")" \
        "$(awk '{ for (i=1;i<=NF;i++) if ($i ~ /^vram_bytes=/) { split($i,a,"="); n+=a[2] } } END { printf "%.0f", n/1048576 }' "$work/status.out")"
    } >>"$raw_log"
  done
  printf 'END_CASE\n' >>"$raw_log"
  stop_servers
  sleep 2
}

for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then configs=(one two); else configs=(two one); fi
  printf 'BEGIN_OLLAMA run=%s\n' "$run" >>"$raw_log"
  for config in "${configs[@]}"; do run_case "$config"; done
  printf 'END_OLLAMA\n' >>"$raw_log"
done

printf 'SYSTEM_OLLAMA pid_after=%s alive=%s\n' \
  "$(systemctl show ollama -p MainPID --value 2>/dev/null || true)" \
  "$(systemctl is-active --quiet ollama 2>/dev/null && echo 1 || echo 0)" >>"$raw_log"

awk -f "$script_dir/summarize_ollama_agents.awk" "$raw_log" \
  >"$result_dir/ollama_agents_summary.csv"
(
  cd "$script_dir"
  sha256sum ollama_client.py ollama_mig_visible.c summarize_ollama_agents.awk \
    run_ollama_agents.sh
) >"$result_dir/source_hashes.txt"
printf 'ollama_agents_result_dir=%s\n' "$result_dir"
