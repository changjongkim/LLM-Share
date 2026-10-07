#!/usr/bin/env bash
# A pipeline of agents on the tools of a public function-calling benchmark.
# A planner computes the prompt prefix (a system prompt and the schemas of
# the tools, workloads/bfcl_prefix.txt) and publishes it. G leaders attach to
# it, read the brief of their group, write a plan of N_GEN tokens and publish
# the prefix with their brief and plan. L workers per leader attach to what
# their leader published and work through TURNS turns: a request of the
# benchmark (workloads/bfcl_tasks.txt), then tool results, each followed by
# N_GEN generated tokens. The planner and the leaders generate while the
# workers run. Three stacks:
#
#   stock  the engine at the upstream commit without any patch of this
#          repository: every process copies the weights to device memory and
#          every hand-over is the engine's state file with the rows
#   copy   the weights are read in place; every hand-over is the state file
#          with the rows, and every process holds its copy in device memory
#   chain  the weights are read in place; the planner and the leaders keep
#          the rows they add in files and publish them, and a process maps
#          the segments before its own read-only (extent chains)
#
# The planner and the even leaders run in the 12-SM MIG instance, the odd
# leaders in the 6-SM instance, and the workers of a leader alternate between
# the two. While a case runs, the power rails of the board are sampled with
# tegrastats every POWER_MS; the energy of a phase is the sum of the samples
# of the input rail (VIN) and of the GPU rail (VDD_GPU) between its bounds:
#
#   prefix   from the start until the planner has published
#   leaders  until every leader has published
#   workers  until every process has left
#
# Kill gates, fixed before the campaign:
#   P1  no process fails in any stack;
#   P2  every worker of copy and of chain writes the text of the worker of
#       stock in the same repetition, over all turns;
#   P3  the pipeline holds less memory on chain than on copy, and less on
#       copy than on stock;
#   P4  the pipeline completes on chain within 1.10 of the time on stock;
#   P5  the pipeline takes on chain no more than 1.10 of the energy of the
#       input rail on stock.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
groups=${GROUPS_COUNT:-2}
workers=${WORKERS:-4}
turns=${TURNS:-3}
prefix_file=${PREFIX_FILE:-"$script_dir/workloads/bfcl_prefix.txt"}
tasks_file=${TASKS_FILE:-"$script_dir/workloads/bfcl_tasks.txt"}
context=${CONTEXT:-16384}
n_gen=${N_GEN:-64}
grow=${GROW_ROWS:-256}
limit_s=${CASE_LIMIT_S:-1200}
power_ms=${POWER_MS:-200}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
engine=${CHAIN_ENGINE:-"$script_dir/llama.cpp-chain"}
stock_engine=${STOCK_ENGINE:-"$script_dir/llama.cpp-stock"}
driver=${DRIVER:-"$script_dir/kv_tree"}
stock_driver=${STOCK_DRIVER:-"$script_dir/kv_tree_stock"}
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-kvpipe"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
if [[ ! -x "$driver" || ! -x "$stock_driver" || ! -r "$model" ]]; then
  echo "missing driver or model; run make kv_tree kv_tree_stock" >&2
  exit 2
fi
for file in "$prefix_file" "$tasks_file"; do
  if [[ ! -r "$file" ]]; then echo "cannot read $file" >&2; exit 2; fi
done
if [[ -n "$(git -C "$stock_engine" status --porcelain --untracked-files=no)" ]]; then
  echo "the engine of the stock stack is not unmodified" >&2
  exit 2
fi
if ! command -v tegrastats >/dev/null; then
  echo "tegrastats is required for the power samples" >&2
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

huge_dir="$mount_root/kv_huge"
mounted_huge=0
work=
live_pids=()
power_pid=
cleanup() {
  if (( ${#live_pids[@]} )); then kill -9 "${live_pids[@]}" 2>/dev/null || true; fi
  if [[ -n "$power_pid" ]]; then pkill -P "$power_pid" 2>/dev/null || true; kill "$power_pid" 2>/dev/null || true; fi
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
task_lines=$(wc -l <"$tasks_file")

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\ngroups=%s\nworkers=%s\nturns=%s\n' \
    "$mig_a" "$mig_b" "$repetitions" "$groups" "$workers" "$turns"
  printf 'context=%s\nn_gen=%s\ngrow_rows=%s\npower_ms=%s\n' "$context" "$n_gen" \
    "$grow" "$power_ms"
  printf 'prefix_file=%s\nprefix_sha256=%s\ntasks_file=%s\ntasks_sha256=%s\n' \
    "$(basename "$prefix_file")" "$(sha256sum "$prefix_file" | cut -d' ' -f1)" \
    "$(basename "$tasks_file")" "$(sha256sum "$tasks_file" | cut -d' ' -f1)"
  printf 'model=%s\n' "$(basename "$model")"
  printf 'engine_commit=%s\nstock_engine_commit=%s\n' \
    "$(git -C "$engine" rev-parse HEAD)" "$(git -C "$stock_engine" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$driver" "$engine/build/bin/libllama.so" \
    "$engine/build/bin/libggml-cuda.so" "$stock_driver" \
    "$stock_engine/build/bin/libllama.so" "$stock_engine/build/bin/libggml-cuda.so"
} >"$result_dir/metadata.txt"

planner_task=$'\n\nRequest to the planner: name the three tools above that a travel assistant needs most.\n'
group_brief() {  # group
  printf '\n\nBrief for the leader of group %s: your workers answer user requests with the tools above. Write the rules they follow when they choose a tool and fill its parameters.\n' "$1"
}
leader_task() {  # group
  printf '\n\nRequest to the leader of group %s: state in one sentence what your workers must check before a call.\n' "$1"
}
worker_turn() {  # worker turn
  if (( $2 == 0 )); then
    printf '\n\nUser request for worker %s: %s\nCall:\n' "$1" \
      "$(sed -n "$(( $1 % task_lines + 1 ))p" "$tasks_file")"
  else
    printf '\n\nTool result %s for worker %s: {"status": "ok", "items": %s}\nNext call or final answer:\n' \
      "$2" "$1" "$(( $1 + $2 ))"
  fi
}
leader_mig() { if (( $1 % 2 == 0 )); then echo "$mig_a"; else echo "$mig_b"; fi; }
worker_mig() {  # group worker
  if (( ($1 + $2) % 2 == 0 )); then echo "$mig_a"; else echo "$mig_b"; fi
}
text_of() { { grep '^TEXT ' "$1" || true; } | sha256sum | cut -c1-16; }
field_of() {  # name line-prefix file
  { grep "^$2 " "$3" || true; } | head -n 1 |
    sed -n "s/^.* $1=\\([^ ]*\\).*\$/\\1/p"
}
wait_line() {  # line-prefix file pid: until the file has the line or the process left
  until grep -q "^$1 " "$2" 2>/dev/null; do
    kill -0 "$3" 2>/dev/null || return 0
    sleep 0.02
  done
}
# The energy of the samples of a rail between two times, in joules: every
# sample stands for the time until the next one.
energy_between() {  # trace rail from to
  awk -v rail="$2" -v from="$3" -v to="$4" '
    {
      for (i = 2; i < NF; ++i) if ($i == rail) { split($(i + 1), part, "mW"); watts = part[1] / 1000 }
      if (seen && last >= from && last < to) joules += last_watts * ($1 - last)
      seen = 1; last = $1; last_watts = watts
    }
    END { printf "%.1f", joules }' "$1"
}

run_case() {  # mode
  local mode=$1 group worker number mig turn program=$driver
  local base_env=(GGML_CUDA_HOST_PTR=1) planner_env=()
  case $mode in
    stock) program=$stock_driver base_env=() ;;
    copy) ;;
    chain) planner_env=(LLAMA_KV_HOST="$huge_dir/r" LLAMA_KV_GROW="$grow") ;;
    *) echo "unknown mode $mode" >&2; exit 2 ;;
  esac
  rm -f "$work"/state.* "$work/go" "$work"/out.* "$work/power" "$huge_dir"/*
  ( tegrastats --interval "$power_ms" | while IFS= read -r line; do
      printf '%s %s\n' "$EPOCHREALTIME" "$line"
    done >"$work/power" ) &
  power_pid=$!
  disown "$power_pid"
  sleep 1
  local before lowest sample started t_start t_prefix t_leaders t_end
  before=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
  lowest=$before
  started=$(date +%s%N)
  t_start=$EPOCHREALTIME

  env CUDA_VISIBLE_DEVICES="$mig_a" "${base_env[@]}" "${planner_env[@]}" \
    "$program" "$model" "$context" "$n_gen" "text:$prefix_file" \
    "save:$work/state.root" "wait:$work/go" "say:$planner_task" gen \
    >"$work/out.root" 2>"$work/out.root.err" &
  local root_pid=$!
  live_pids=("$root_pid")
  wait_line PUBLISHED "$work/out.root" "$root_pid"
  t_prefix=$EPOCHREALTIME
  local root_tokens
  root_tokens=$(field_of tokens PUBLISHED "$work/out.root")

  local leader_pids=() leader_tokens=()
  for group in $(seq 0 $((groups - 1))); do
    local leader_env=()
    if [[ "$mode" == chain ]]; then
      leader_env=(LLAMA_KV_CHAIN="$huge_dir/r:${root_tokens:-0}"
        LLAMA_KV_HOST="$huge_dir/m$group" LLAMA_KV_GROW="$grow")
    fi
    env CUDA_VISIBLE_DEVICES="$(leader_mig "$group")" "${base_env[@]}" \
      "${leader_env[@]}" "$program" "$model" "$context" "$n_gen" \
      "load:$work/state.root" "say:$(group_brief "$group")" gen \
      "save:$work/state.leader.$group" "wait:$work/go" \
      "say:$(leader_task "$group")" gen \
      >"$work/out.leader.$group" 2>"$work/out.leader.$group.err" &
    leader_pids+=($!)
  done
  live_pids+=("${leader_pids[@]}")
  for group in $(seq 0 $((groups - 1))); do
    wait_line PUBLISHED "$work/out.leader.$group" "${leader_pids[$group]}"
    leader_tokens+=("$(field_of tokens PUBLISHED "$work/out.leader.$group")")
  done
  t_leaders=$EPOCHREALTIME

  local worker_pids=() worker_groups=() worker_migs=() worker_files=()
  number=0
  for group in $(seq 0 $((groups - 1))); do
    for worker in $(seq 0 $((workers - 1))); do
      mig=$(worker_mig "$group" "$worker")
      local worker_env=() steps=("load:$work/state.leader.$group")
      if [[ "$mode" == chain ]]; then
        worker_env=(LLAMA_KV_CHAIN="$huge_dir/r:${root_tokens:-0},$huge_dir/m$group:${leader_tokens[$group]:-0}"
          LLAMA_KV_HOST=anon LLAMA_KV_GROW="$grow")
      fi
      for turn in $(seq 0 $((turns - 1))); do
        steps+=("say:$(worker_turn "$number" "$turn")" gen)
      done
      env CUDA_VISIBLE_DEVICES="$mig" "${base_env[@]}" "${worker_env[@]}" \
        "$program" "$model" "$context" "$n_gen" "${steps[@]}" \
        >"$work/out.worker.$number" 2>"$work/out.worker.$number.err" &
      worker_pids+=($!)
      worker_groups+=("$group")
      worker_migs+=("$mig")
      worker_files+=("$work/out.worker.$number")
      number=$((number + 1))
    done
  done
  live_pids+=("${worker_pids[@]}")
  touch "$work/go"

  local alive=1 timed_out=0 pid
  while (( alive > 0 )); do
    alive=0
    for pid in "${live_pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then alive=$((alive + 1)); fi
    done
    sample=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
    (( sample < lowest )) && lowest=$sample
    if (( ($(date +%s%N) - started) / 1000000000 > limit_s )); then
      timed_out=1
      kill -9 "${live_pids[@]}" 2>/dev/null || true
    fi
    sleep 0.1
  done
  t_end=$EPOCHREALTIME
  local wall_ms=$(( ($(date +%s%N) - started) / 1000000 ))
  sleep 0.5
  pkill -P "$power_pid" 2>/dev/null || true
  kill "$power_pid" 2>/dev/null || true
  power_pid=
  {
    printf 'CASE mode=%s groups=%s workers=%s turns=%s\n' "$mode" "$groups" \
      "$workers" "$turns"
    local code=0 failed=0
    wait "$root_pid" || code=$?
    if (( code != 0 )) || ! grep -q '^RESULT ' "$work/out.root"; then
      failed=$((failed + 1))
      sed 's/^/STDERR /' "$work/out.root.err" | tail -n 5
    fi
    printf 'PLANNER exit=%s text=%s tokens=%s publish_ms=%s state_bytes=%s generation_tps=%s\n' \
      "$code" "$(text_of "$work/out.root")" "${root_tokens:-0}" \
      "$(field_of publish_ms PUBLISHED "$work/out.root")" \
      "$(field_of state_bytes PUBLISHED "$work/out.root")" \
      "$(field_of generation_tps RESULT "$work/out.root")"
    for group in $(seq 0 $((groups - 1))); do
      code=0
      wait "${leader_pids[$group]}" || code=$?
      if (( code != 0 )) || ! grep -q '^RESULT ' "$work/out.leader.$group"; then
        failed=$((failed + 1))
        sed 's/^/STDERR /' "$work/out.leader.$group.err" | tail -n 5
      fi
      printf 'LEADER group=%s mig=%s exit=%s text=%s tokens=%s context_ms=%s load_ms=%s publish_ms=%s state_bytes=%s generation_tps=%s\n' \
        "$group" "$(leader_mig "$group" | cut -c5-12)" "$code" \
        "$(text_of "$work/out.leader.$group")" "${leader_tokens[$group]:-0}" \
        "$(field_of context_ms ATTACHED "$work/out.leader.$group")" \
        "$(field_of load_ms ATTACHED "$work/out.leader.$group")" \
        "$(field_of publish_ms PUBLISHED "$work/out.leader.$group")" \
        "$(field_of state_bytes PUBLISHED "$work/out.leader.$group")" \
        "$(field_of generation_tps RESULT "$work/out.leader.$group")"
    done
    for number in "${!worker_pids[@]}"; do
      code=0
      wait "${worker_pids[$number]}" || code=$?
      if (( code != 0 )) || ! grep -q '^RESULT ' "${worker_files[$number]}"; then
        failed=$((failed + 1))
        sed 's/^/STDERR /' "${worker_files[$number]}.err" | tail -n 5
      fi
      printf 'WORKER index=%s group=%s mig=%s exit=%s text=%s tokens=%s context_ms=%s load_ms=%s decode_ms=%s first_token_ms=%s generated=%s generation_tps=%s total_ms=%s\n' \
        "$number" "${worker_groups[$number]}" "${worker_migs[$number]:4:8}" "$code" \
        "$(text_of "${worker_files[$number]}")" \
        "$(field_of tokens RESULT "${worker_files[$number]}")" \
        "$(field_of context_ms ATTACHED "${worker_files[$number]}")" \
        "$(field_of load_ms ATTACHED "${worker_files[$number]}")" \
        "$(field_of decode_ms RESULT "${worker_files[$number]}")" \
        "$(field_of first_token_ms RESULT "${worker_files[$number]}")" \
        "$(field_of generated RESULT "${worker_files[$number]}")" \
        "$(field_of generation_tps RESULT "${worker_files[$number]}")" \
        "$(field_of total_ms RESULT "${worker_files[$number]}")"
    done
    printf 'MEMORY pipeline_drop_mib=%s files_mib=%s\n' \
      "$(( (before - lowest) / 1024 ))" \
      "$(df -k --output=used "$huge_dir" | awk 'NR > 1 { printf "%d", $1 / 1024 }')"
    local phase from to
    for phase in prefix leaders workers; do
      case $phase in
        prefix) from=$t_start to=$t_prefix ;;
        leaders) from=$t_prefix to=$t_leaders ;;
        workers) from=$t_leaders to=$t_end ;;
      esac
      printf 'ENERGY phase=%s seconds=%s vin_joules=%s gpu_joules=%s\n' "$phase" \
        "$(awk -v a="$from" -v b="$to" 'BEGIN { printf "%.2f", b - a }')" \
        "$(energy_between "$work/power" VIN "$from" "$to")" \
        "$(energy_between "$work/power" VDD_GPU "$from" "$to")"
    done
    printf 'END_CASE failed=%s timed_out=%s wall_ms=%s power_samples=%s\n' "$failed" \
      "$timed_out" "$wall_ms" "$(wc -l <"$work/power")"
  } >>"$raw_log"
  live_pids=()
  rm -f "$work"/state.* "$work/go" "$huge_dir"/*
}

read -r -a forward <<<"${MODES:-stock copy chain}"
backward=()
for mode in "${forward[@]}"; do backward=("$mode" "${backward[@]}"); done
for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then modes=("${forward[@]}"); else modes=("${backward[@]}"); fi
  printf 'BEGIN_KVPIPE run=%s context=%s\n' "$run" "$context" >>"$raw_log"
  for mode in "${modes[@]}"; do
    run_case "$mode"
  done
  printf 'END_KVPIPE\n' >>"$raw_log"
done

awk -f "$script_dir/summarize_engine_kvpipe.awk" "$raw_log" \
  >"$result_dir/engine_kvpipe_summary.csv"
(
  cd "$script_dir"
  sha256sum kv_chain.patch kv_tree.cpp summarize_engine_kvpipe.awk \
    run_engine_kvpipe.sh "$(realpath --relative-to=. "$prefix_file")" \
    "$(realpath --relative-to=. "$tasks_file")"
) >"$result_dir/source_hashes.txt"
printf 'engine_kvpipe_result_dir=%s\n' "$result_dir"
