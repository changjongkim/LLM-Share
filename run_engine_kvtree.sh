#!/usr/bin/env bash
# A tree of agents on one prompt prefix. A root computes the prefix and
# publishes it. G group leaders attach to it, append the context of their
# group and publish again. L leaves per leader attach to what their leader
# published, append their task and generate. The root and the leaders keep
# generating while the leaves run. Compares four ways to hand state down:
#
#   copy         every hand-over is the engine's state file with the rows;
#                every process holds its own copy in device memory
#   flat         one level of extents: every process maps the rows of the
#                root read-only; a leader cannot publish, and each leaf
#                computes the context of its group again
#   chain        extent chains: a leader keeps the rows it appends in a file
#                of its own and publishes them; a leaf maps the segment of
#                the root and the segment of its leader read-only
#   chain_small  the same with the files of the leaders on 4 KiB pages
#
# The file of the root is on a tmpfs with 2 MiB pages. The root and the even
# leaders run in the 12-SM MIG instance, the odd leaders in the 6-SM
# instance, and the leaves of a leader alternate between the two. In the
# extent modes the files are removed from their directory as soon as every
# process has attached.
#
# Kill gates, fixed before the campaign:
#   T1  no process fails in any mode;
#   T2  every leaf of flat, chain and chain_small writes the text of the
#       leaf that received copies in the same repetition;
#   T3  the tree on chain needs less memory than on copy and than on flat;
#   T4  a leader pauses no longer to publish with chain than with copy, and
#       a leaf attaches no slower with chain than with copy;
#   T5  the first token of a leaf is not later with chain than with flat;
#   T6  after the files are removed every process still completes, and the
#       tmpfs holds no page once the last process has left.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
groups=${GROUPS_COUNT:-2}
leaves=${LEAVES:-4}
paragraphs=${PARAGRAPHS:-320}
group_paragraphs=${GROUP_PARAGRAPHS:-20}
n_gen=${N_GEN:-64}
grow=${GROW_ROWS:-256}
limit_s=${CASE_LIMIT_S:-900}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
engine=${CHAIN_ENGINE:-"$script_dir/llama.cpp-chain"}
driver=${DRIVER:-"$script_dir/kv_tree"}
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-kvtree"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
if [[ ! -x "$driver" || ! -r "$model" ]]; then
  echo "missing driver or model; run make kv_tree and see README.md" >&2
  exit 2
fi
if ! sudo -n true 2>/dev/null; then
  echo "passwordless sudo is required for the tmpfs mounts" >&2
  exit 2
fi
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi

huge_dir="$mount_root/kv_huge"
small_dir="$mount_root/kv_small"
mounted_huge=0
mounted_small=0
work=
live_pids=()
cleanup() {
  if (( ${#live_pids[@]} )); then kill -9 "${live_pids[@]}" 2>/dev/null || true; fi
  if (( mounted_huge )); then sudo -n umount "$huge_dir" || true; fi
  if (( mounted_small )); then sudo -n umount "$small_dir" || true; fi
  rmdir "$huge_dir" "$small_dir" "$mount_root" 2>/dev/null || true
  if [[ -n "$work" ]]; then rm -rf "$work"; fi
}
trap cleanup EXIT
mkdir -p "$huge_dir" "$small_dir" "$result_dir"
sudo -n mount -t tmpfs -o "huge=always,size=8192m,uid=$(id -u),gid=$(id -g)" \
  tmpfs "$huge_dir"
mounted_huge=1
sudo -n mount -t tmpfs -o "huge=never,size=4096m,uid=$(id -u),gid=$(id -g)" \
  tmpfs "$small_dir"
mounted_small=1
work=$(mktemp -d "$result_dir/work.XXXXXX")

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\ngroups=%s\nleaves=%s\n' \
    "$mig_a" "$mig_b" "$repetitions" "$groups" "$leaves"
  printf 'paragraphs=%s\ngroup_paragraphs=%s\nn_gen=%s\ngrow_rows=%s\n' \
    "$paragraphs" "$group_paragraphs" "$n_gen" "$grow"
  printf 'model=%s\n' "$(basename "$model")"
  printf 'engine_commit=%s\n' "$(git -C "$engine" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$driver" "$engine/build/bin/libllama.so" \
    "$engine/build/bin/libggml-cuda.so"
} >"$result_dir/metadata.txt"

paragraph="The scheduler keeps a run queue for every processor and moves a task between queues when the load is uneven. A page fault enters the kernel, which finds the mapping, allocates a frame, fills it, and updates the page table before the task resumes. "
context=4096
while (( context < (paragraphs + group_paragraphs) * 52 + 512 )); do
  context=$(( context * 2 ))
done
prefix="$work/prefix"
: >"$prefix"
for _ in $(seq 1 "$paragraphs"); do printf '%s' "$paragraph" >>"$prefix"; done
root_task=$'\n\nTask for the root: list the kernel mechanisms named above.'
group_context() {  # group
  local text="" count
  for count in $(seq 1 "$group_paragraphs"); do
    text+="Group $1 note $count: the group reviews how the kernel handles memory pressure, which pages it reclaims first, and how a task that waits for a page is resumed. "
  done
  printf '\n\n%s' "$text"
}
leader_task() {  # group
  printf '\n\nTask for the leader of group %s: state the goal of the group in one sentence.' "$1"
}
leaf_task() {  # leaf number in the tree
  printf '\n\nTask for agent %s: summarize the text above in %s sentences.' \
    "$1" "$(( $1 + 2 ))"
}
leader_mig() { if (( $1 % 2 == 0 )); then echo "$mig_a"; else echo "$mig_b"; fi; }
leaf_mig() {  # group leaf
  if (( ($1 + $2) % 2 == 0 )); then echo "$mig_a"; else echo "$mig_b"; fi
}
text_of() { { grep '^TEXT ' "$1" || true; } | sha256sum | cut -c1-16; }
field_of() {  # name line-prefix file
  { grep "^$2 " "$3" || true; } | head -n 1 |
    sed -n "s/^.* $1=\\([^ ]*\\).*\$/\\1/p"
}
used_mib() {  # the pages that the two tmpfs hold
  df -k --output=used "$huge_dir" "$small_dir" | awk 'NR > 1 { sum += $1 } END { printf "%d", sum / 1024 }'
}
wait_line() {  # line-prefix file pid: until the file has the line or the process left
  until grep -q "^$1 " "$2" 2>/dev/null; do
    kill -0 "$3" 2>/dev/null || return 0
    sleep 0.02
  done
}

run_case() {  # mode
  local mode=$1 group leaf number mig
  local root_env=() leader_file_dir="$huge_dir"
  case $mode in
    flat | chain | chain_small) root_env=(LLAMA_KV_HOST="$huge_dir/r" LLAMA_KV_GROW="$grow") ;;
  esac
  [[ "$mode" == chain_small ]] && leader_file_dir="$small_dir"
  rm -f "$work"/state.* "$work/go" "$work"/out.* "$huge_dir"/* "$small_dir"/*
  local before lowest sample started
  before=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
  lowest=$before
  started=$(date +%s%N)

  env CUDA_VISIBLE_DEVICES="$mig_a" GGML_CUDA_HOST_PTR=1 "${root_env[@]}" \
    "$driver" "$model" "$context" "$n_gen" "text:$prefix" \
    "save:$work/state.root" "wait:$work/go" "say:$root_task" gen \
    >"$work/out.root" 2>"$work/out.root.err" &
  local root_pid=$!
  live_pids=("$root_pid")
  wait_line PUBLISHED "$work/out.root" "$root_pid"
  local root_tokens
  root_tokens=$(field_of tokens PUBLISHED "$work/out.root")

  local leader_pids=() leader_tokens=()
  for group in $(seq 0 $((groups - 1))); do
    local leader_env=() steps=("load:$work/state.root" "say:$(group_context "$group")")
    case $mode in
      flat)
        leader_env=(LLAMA_KV_HOST="$huge_dir/r" LLAMA_KV_PREFIX="${root_tokens:-0}"
          LLAMA_KV_GROW="$grow") ;;
      chain | chain_small)
        leader_env=(LLAMA_KV_CHAIN="$huge_dir/r:${root_tokens:-0}"
          LLAMA_KV_HOST="$leader_file_dir/m$group" LLAMA_KV_GROW="$grow") ;;
    esac
    if [[ "$mode" != flat ]]; then steps+=("save:$work/state.leader.$group"); fi
    steps+=("wait:$work/go" "say:$(leader_task "$group")" gen)
    env CUDA_VISIBLE_DEVICES="$(leader_mig "$group")" GGML_CUDA_HOST_PTR=1 \
      "${leader_env[@]}" "$driver" "$model" "$context" "$n_gen" "${steps[@]}" \
      >"$work/out.leader.$group" 2>"$work/out.leader.$group.err" &
    leader_pids+=($!)
  done
  live_pids+=("${leader_pids[@]}")
  for group in $(seq 0 $((groups - 1))); do
    if [[ "$mode" == flat ]]; then
      wait_line ATTACHED "$work/out.leader.$group" "${leader_pids[$group]}"
      leader_tokens+=("${root_tokens:-0}")
    else
      wait_line PUBLISHED "$work/out.leader.$group" "${leader_pids[$group]}"
      leader_tokens+=("$(field_of tokens PUBLISHED "$work/out.leader.$group")")
    fi
  done

  local leaf_pids=() leaf_groups=() leaf_migs=() leaf_files=()
  number=0
  for group in $(seq 0 $((groups - 1))); do
    for leaf in $(seq 0 $((leaves - 1))); do
      mig=$(leaf_mig "$group" "$leaf")
      local leaf_env=() steps=()
      case $mode in
        copy) steps=("load:$work/state.leader.$group") ;;
        flat)
          leaf_env=(LLAMA_KV_HOST="$huge_dir/r" LLAMA_KV_PREFIX="${root_tokens:-0}"
            LLAMA_KV_GROW="$grow")
          steps=("load:$work/state.root" "say:$(group_context "$group")") ;;
        chain | chain_small)
          leaf_env=(LLAMA_KV_CHAIN="$huge_dir/r:${root_tokens:-0},$leader_file_dir/m$group:${leader_tokens[$group]:-0}"
            LLAMA_KV_HOST=anon LLAMA_KV_GROW="$grow")
          steps=("load:$work/state.leader.$group") ;;
      esac
      steps+=("say:$(leaf_task "$number")" gen)
      env CUDA_VISIBLE_DEVICES="$mig" GGML_CUDA_HOST_PTR=1 "${leaf_env[@]}" \
        "$driver" "$model" "$context" "$n_gen" "${steps[@]}" \
        >"$work/out.leaf.$number" 2>"$work/out.leaf.$number.err" &
      leaf_pids+=($!)
      leaf_groups+=("$group")
      leaf_migs+=("$mig")
      leaf_files+=("$work/out.leaf.$number")
      number=$((number + 1))
    done
  done
  live_pids+=("${leaf_pids[@]}")
  touch "$work/go"
  for number in "${!leaf_pids[@]}"; do
    wait_line ATTACHED "${leaf_files[$number]}" "${leaf_pids[$number]}"
  done
  # Every process that shares a file has it mapped; the names can go.
  local files_attached files_removed=0
  files_attached=$(used_mib)
  if [[ "$mode" != copy ]]; then
    rm -f "$huge_dir"/* "$small_dir"/*
    files_removed=1
  fi

  local alive=1 timed_out=0 number_alive
  while (( alive > 0 )); do
    alive=0
    for number_alive in "${live_pids[@]}"; do
      if kill -0 "$number_alive" 2>/dev/null; then alive=$((alive + 1)); fi
    done
    sample=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
    (( sample < lowest )) && lowest=$sample
    if (( ($(date +%s%N) - started) / 1000000000 > limit_s )); then
      timed_out=1
      kill -9 "${live_pids[@]}" 2>/dev/null || true
    fi
    sleep 0.1
  done
  local files_end
  files_end=$(used_mib)
  {
    printf 'CASE mode=%s groups=%s leaves=%s\n' "$mode" "$groups" "$leaves"
    local code=0 failed=0
    wait "$root_pid" || code=$?
    if (( code != 0 )) || ! grep -q '^RESULT ' "$work/out.root"; then
      failed=$((failed + 1))
      sed 's/^/STDERR /' "$work/out.root.err" | tail -n 5
    fi
    printf 'ROOT exit=%s text=%s tokens=%s publish_ms=%s state_bytes=%s generation_tps=%s\n' \
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
    for number in "${!leaf_pids[@]}"; do
      code=0
      wait "${leaf_pids[$number]}" || code=$?
      if (( code != 0 )) || ! grep -q '^RESULT ' "${leaf_files[$number]}"; then
        failed=$((failed + 1))
        sed 's/^/STDERR /' "${leaf_files[$number]}.err" | tail -n 5
      fi
      printf 'LEAF index=%s group=%s mig=%s exit=%s text=%s context_ms=%s load_ms=%s decode_ms=%s first_token_ms=%s generation_tps=%s\n' \
        "$number" "${leaf_groups[$number]}" "${leaf_migs[$number]:4:8}" "$code" \
        "$(text_of "${leaf_files[$number]}")" \
        "$(field_of context_ms ATTACHED "${leaf_files[$number]}")" \
        "$(field_of load_ms ATTACHED "${leaf_files[$number]}")" \
        "$(field_of decode_ms RESULT "${leaf_files[$number]}")" \
        "$(field_of first_token_ms RESULT "${leaf_files[$number]}")" \
        "$(field_of generation_tps RESULT "${leaf_files[$number]}")"
    done
    printf 'MEMORY tree_drop_mib=%s files_attached_mib=%s files_removed=%s files_end_mib=%s\n' \
      "$(( (before - lowest) / 1024 ))" "$files_attached" "$files_removed" \
      "$files_end"
    printf 'END_CASE failed=%s timed_out=%s wall_ms=%s\n' "$failed" "$timed_out" \
      "$(( ($(date +%s%N) - started) / 1000000 ))"
  } >>"$raw_log"
  live_pids=()
  rm -f "$work"/state.* "$work/go" "$huge_dir"/* "$small_dir"/*
}

forward=(copy flat chain chain_small)
backward=(chain_small chain flat copy)
for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then modes=("${forward[@]}"); else modes=("${backward[@]}"); fi
  {
    printf 'BEGIN_KVTREE run=%s paragraphs=%s group_paragraphs=%s context=%s\n' \
      "$run" "$paragraphs" "$group_paragraphs" "$context"
    # What a leaf must write: the same tokens computed alone in its instance.
    number=0
    for group in $(seq 0 $((groups - 1))); do
      for leaf in $(seq 0 $((leaves - 1))); do
        mig=$(leaf_mig "$group" "$leaf")
        env CUDA_VISIBLE_DEVICES="$mig" GGML_CUDA_HOST_PTR=1 "$driver" "$model" \
          "$context" "$n_gen" "text:$prefix" "say:$(group_context "$group")" \
          "say:$(leaf_task "$number")" gen >"$work/alone" 2>/dev/null || true
        # A leaf whose leader runs in the same instance as the root and as
        # the leaf itself reads rows that only its own instance computed.
        own=0
        if [[ "$mig" == "$mig_a" && "$(leader_mig "$group")" == "$mig_a" ]]; then own=1; fi
        printf 'REFERENCE index=%s own_instance_path=%s text=%s\n' "$number" \
          "$own" "$(text_of "$work/alone")"
        number=$((number + 1))
      done
    done
  } >>"$raw_log"
  for mode in "${modes[@]}"; do
    run_case "$mode"
  done
  printf 'END_KVTREE\n' >>"$raw_log"
done

awk -f "$script_dir/summarize_engine_kvtree.awk" "$raw_log" \
  >"$result_dir/engine_kvtree_summary.csv"
(
  cd "$script_dir"
  sha256sum kv_chain.patch kv_tree.cpp summarize_engine_kvtree.awk \
    run_engine_kvtree.sh
) >"$result_dir/source_hashes.txt"
printf 'engine_kvtree_result_dir=%s\n' "$result_dir"
