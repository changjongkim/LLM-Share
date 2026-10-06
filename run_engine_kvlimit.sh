#!/usr/bin/env bash
# Memory limits per agent. Four agents attach to one published prefix, every
# second one in the other MIG instance, each in a control group of its own.
# Two ways to obtain the prefix:
#
#   restore      the agent copies it into device memory (the unmodified
#                engine); the cache of its whole context is device memory
#   extent_lazy  the agent maps the rows read-only; its own rows are private
#                host memory that the CPU populates ahead of the GPU
#
# and two phases per repetition:
#
#   account  no limit. Records what the control group of each agent was
#            charged at its peak (memory.peak) next to what the agents
#            took from the device (drop of MemAvailable).
#   limit    every agent gets memory.max = the mean peak of the extent
#            agents in the account phase + HEADROOM_MIB. Agent 0 then
#            receives a task with RUNAWAY_PARAGRAPHS of extra text, which
#            needs more memory for its rows than the headroom; the other
#            agents receive their normal task.
#
# Kill gates, fixed before the campaign:
#   L1  limit: the charge of agent 0 on extents reaches its limit (peak
#       within 16 MiB of memory.max), and the charge of agent 0 that
#       restores stays below it although its cache holds the same rows,
#       because device memory is not charged. The charged share of the
#       memory that the agents take (account phase) is reported, not gated:
#       the compute buffers and the CUDA context of a process are device
#       memory in both modes;
#   L2  limit, extent_lazy: agent 0 is ended by its own control group (one
#       oom_kill event, no event in any other group), the other three agents
#       complete and write the text of the account phase;
#   L3  limit, restore: agent 0 completes although it holds more memory than
#       its limit, because the limit does not see device memory;
#   L4  no process outside the control group of agent 0 fails in any case.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
agents=${AGENTS:-4}
paragraphs=${PARAGRAPHS:-320}
runaway_paragraphs=${RUNAWAY_PARAGRAPHS:-80}
headroom_mib=${HEADROOM_MIB:-100}
n_gen=${N_GEN:-64}
grow=${GROW_ROWS:-256}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
engine=${KV_ENGINE:-"$script_dir/llama.cpp-kv"}
driver=${DRIVER:-"$script_dir/kv_fork"}
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-kvlimit"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"
group_root="/sys/fs/cgroup/llmshare_kvlimit_$$"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
if [[ ! -x "$driver" || ! -r "$model" ]]; then
  echo "missing driver or model; run make and see README.md" >&2
  exit 2
fi
if ! sudo -n true 2>/dev/null; then
  echo "passwordless sudo is required for the tmpfs mount and the control groups" >&2
  exit 2
fi
if ! grep -qw memory /sys/fs/cgroup/cgroup.subtree_control; then
  echo "the memory controller is not delegated below the root control group" >&2
  exit 2
fi
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi

huge_dir="$mount_root/kv_huge"
mounted_huge=0
made_groups=0
work=
live_pids=()
cleanup() {
  if (( ${#live_pids[@]} )); then kill -9 "${live_pids[@]}" 2>/dev/null || true; fi
  sleep 0.3
  if (( made_groups )); then
    sudo -n rmdir "$group_root"/a* 2>/dev/null || true
    sudo -n rmdir "$group_root" 2>/dev/null || true
  fi
  if (( mounted_huge )); then sudo -n umount "$huge_dir" || true; fi
  rmdir "$huge_dir" "$mount_root" 2>/dev/null || true
  if [[ -n "$work" ]]; then rm -rf "$work"; fi
}
trap cleanup EXIT
mkdir -p "$huge_dir" "$result_dir"
sudo -n mount -t tmpfs -o "huge=always,size=4096m,uid=$(id -u),gid=$(id -g)" \
  tmpfs "$huge_dir"
mounted_huge=1
work=$(mktemp -d "$result_dir/work.XXXXXX")
sudo -n mkdir "$group_root"
made_groups=1
echo +memory | sudo -n tee "$group_root/cgroup.subtree_control" >/dev/null
for index in $(seq 0 $((agents - 1))); do sudo -n mkdir "$group_root/a$index"; done

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\nagents=%s\n' "$mig_a" "$mig_b" \
    "$repetitions" "$agents"
  printf 'paragraphs=%s\nrunaway_paragraphs=%s\nheadroom_mib=%s\nn_gen=%s\ngrow_rows=%s\n' \
    "$paragraphs" "$runaway_paragraphs" "$headroom_mib" "$n_gen" "$grow"
  printf 'model=%s\n' "$(basename "$model")"
  printf 'engine_commit=%s\n' "$(git -C "$engine" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  uname -r
  nvidia-smi -L
  sha256sum "$driver" "$engine/build/bin/libllama.so" \
    "$engine/build/bin/libggml-cuda.so"
} >"$result_dir/metadata.txt"

paragraph="The scheduler keeps a run queue for every processor and moves a task between queues when the load is uneven. A page fault enters the kernel, which finds the mapping, allocates a frame, fills it, and updates the page table before the task resumes. "
context=4096
while (( context < (paragraphs + runaway_paragraphs) * 52 + 1024 )); do
  context=$(( context * 2 ))
done
prefix="$work/prefix"
: >"$prefix"
for _ in $(seq 1 "$paragraphs"); do printf '%s' "$paragraph" >>"$prefix"; done
runaway=""
for _ in $(seq 1 "$runaway_paragraphs"); do runaway+=$paragraph; done
suffix_of() {  # agent index, phase
  if [[ "$2" == limit && "$1" == 0 ]]; then printf '\n\n%s' "$runaway"; fi
  printf '\n\nTask for agent %s: summarize the text above in %s sentences.\n' \
    "$1" "$(( $1 + 2 ))"
}
field_of() {  # name file
  sed -n "s/^.* $1=\\([^ ]*\\).*\$/\\1/p" "$2" | head -n 1
}

publish() {  # label mig state-file env...
  local label=$1 mig=$2 state=$3
  shift 3
  rm -f "$state"
  env CUDA_VISIBLE_DEVICES="$mig" GGML_CUDA_HOST_PTR=1 "$@" \
    "$driver" parent "$model" "$context" "$prefix" "$state" "" 0 \
    >"$work/publish.out" 2>"$work/publish.err" || true
  printf 'PUBLISH store=%s %s\n' "$label" \
    "$({ grep '^PUBLISHED ' "$work/publish.out" || true; } | sed 's/^PUBLISHED //')"
}

# Leaves the mean peak of the agents of the case in mean_peak, in MiB.
run_case() {  # phase mode prefix-tokens limit-mib
  local phase=$1 mode=$2 tokens=$3 limit=$4 index mig state envs=()
  if [[ "$mode" == restore ]]; then
    state="$work/state.device"
  else
    state="$work/state.huge"
    envs=(LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_PREFIX="$tokens" LLAMA_KV_GROW="$grow")
  fi
  # A control group keeps its peak and its events; new ones start from zero.
  for index in $(seq 0 $((agents - 1))); do
    sudo -n rmdir "$group_root/a$index"
    sudo -n mkdir "$group_root/a$index"
    if [[ "$limit" == max ]]; then
      echo max | sudo -n tee "$group_root/a$index/memory.max" >/dev/null
    else
      echo "$(( limit * 1048576 ))" | sudo -n tee "$group_root/a$index/memory.max" >/dev/null
    fi
  done
  local before lowest sample pids=() migs=()
  before=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
  lowest=$before
  for index in $(seq 0 $((agents - 1))); do
    if (( index % 2 == 0 )); then mig=$mig_a; else mig=$mig_b; fi
    migs+=("$mig")
    # The shell joins the control group of the agent and becomes the agent.
    # shellcheck disable=SC2016
    bash -c 'echo $$ | sudo -n tee "$0/cgroup.procs" >/dev/null && exec "$@"' \
      "$group_root/a$index" \
      env CUDA_VISIBLE_DEVICES="$mig" GGML_CUDA_HOST_PTR=1 "${envs[@]}" \
      "$driver" child "$model" "$context" "$state" "$(suffix_of "$index" "$phase")" \
      "$n_gen" >"$work/agent.$index" 2>"$work/agent.$index.err" &
    pids+=($!)
  done
  live_pids=("${pids[@]}")
  while kill -0 "${pids[@]}" 2>/dev/null; do
    sample=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
    (( sample < lowest )) && lowest=$sample
    sleep 0.05
  done
  local alive=1
  while (( alive > 0 )); do
    alive=0
    for index in "${pids[@]}"; do
      if kill -0 "$index" 2>/dev/null; then alive=$((alive + 1)); fi
    done
    sample=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
    (( sample < lowest )) && lowest=$sample
    sleep 0.05
  done
  live_pids=()
  local peaks=0 failed=0
  {
    printf 'CASE phase=%s mode=%s limit_mib=%s\n' "$phase" "$mode" "$limit"
    for index in "${!pids[@]}"; do
      local code=0 text result peak kills
      wait "${pids[$index]}" || code=$?
      result=$(grep '^RESULT ' "$work/agent.$index" || true)
      text=$({ grep '^TEXT ' "$work/agent.$index" || true; } | sha256sum | cut -c1-16)
      if (( code != 0 )) || [[ -z "$result" ]]; then
        failed=$((failed + 1))
        text=none
      fi
      peak=$(( $(cat "$group_root/a$index/memory.peak") / 1048576 ))
      kills=$(awk '$1 == "oom_kill" { print $2 }' "$group_root/a$index/memory.events")
      peaks=$((peaks + peak))
      printf 'AGENT index=%s mig=%s exit=%s text=%s peak_mib=%s oom_kill=%s %s\n' \
        "$index" "${migs[$index]:4:8}" "$code" "$text" "$peak" "${kills:-0}" \
        "${result#RESULT }"
    done
    printf 'MEMORY mem_available_drop_mib=%s charged_peak_sum_mib=%s\n' \
      "$(( (before - lowest) / 1024 ))" "$peaks"
    printf 'END_CASE failed=%s\n' "$failed"
  } >>"$raw_log"
  mean_peak=$(( peaks / agents ))
}

mean_peak=0
for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then modes=(restore extent_lazy); else modes=(extent_lazy restore); fi
  rm -f "$huge_dir/kv.0"
  {
    printf 'BEGIN_KVLIMIT run=%s paragraphs=%s context=%s\n' "$run" "$paragraphs" "$context"
    publish device "$mig_a" "$work/state.device"
    publish huge "$mig_a" "$work/state.huge" LLAMA_KV_HOST="$huge_dir/kv" \
      LLAMA_KV_GROW="$grow"
  } >>"$raw_log"
  chmod 0400 "$huge_dir/kv.0"
  tokens=$(field_of prefix_tokens "$work/publish.out")
  extent_peak=0
  for mode in "${modes[@]}"; do
    run_case account "$mode" "$tokens" max
    if [[ "$mode" == extent_lazy ]]; then extent_peak=$mean_peak; fi
  done
  for mode in "${modes[@]}"; do
    run_case limit "$mode" "$tokens" "$(( extent_peak + headroom_mib ))"
  done
  printf 'END_KVLIMIT\n' >>"$raw_log"
  rm -f "$huge_dir/kv.0" "$work"/state.*
done

awk -f "$script_dir/summarize_engine_kvlimit.awk" "$raw_log" \
  >"$result_dir/engine_kvlimit_summary.csv"
(
  cd "$script_dir"
  sha256sum kv_extents.patch kv_fork.cpp summarize_engine_kvlimit.awk \
    run_engine_kvlimit.sh
) >"$result_dir/source_hashes.txt"
printf 'engine_kvlimit_result_dir=%s\n' "$result_dir"
