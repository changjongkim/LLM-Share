#!/usr/bin/env bash
# A serving process that forks its state while it keeps running. The parent
# computes a prefix, hands its key-value cache to N children, half of them in
# the other MIG instance, and then generates its own continuation while the
# children generate theirs. Compares two ways to hand the state over:
#
#   copy    the parent saves the engine's state file (the rows of the cache),
#           each child copies it into its own device memory
#   extent  the cache of the parent is a file on a tmpfs with 2 MiB pages;
#           the parent freezes the rows of the prefix and saves a state file
#           that only names them, each child maps the rows read-only and
#           keeps the rows it writes in private memory that follows use
#
# Kill gates, fixed before the campaign: the parent and every child must
# produce the text of a process that computes the same tokens alone; handing
# the state over (the pause of the parent) and attaching must not take longer
# with extents than with copies; the children must hold one copy of the
# prefix between them.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
children=${CHILDREN:-4}
paragraphs=${PARAGRAPHS:-80}
n_gen=${N_GEN:-64}
grow=${GROW_ROWS:-256}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
engine=${KV_ENGINE:-"$script_dir/llama.cpp-kv"}
driver=${DRIVER:-"$script_dir/kv_fork"}
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-kvfork"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
if [[ ! -x "$driver" || ! -r "$model" ]]; then
  echo "missing driver or model; run make and see README.md" >&2
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
parent_pid=
cleanup() {
  if [[ -n "$parent_pid" ]]; then kill "$parent_pid" 2>/dev/null || true; fi
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

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\nchildren=%s\n' \
    "$mig_a" "$mig_b" "$repetitions" "$children"
  printf 'paragraphs=%s\nn_gen=%s\ngrow_rows=%s\n' "$paragraphs" "$n_gen" "$grow"
  printf 'model=%s\n' "$(basename "$model")"
  printf 'engine_commit=%s\n' "$(git -C "$engine" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$driver" "$engine/build/bin/libllama.so" \
    "$engine/build/bin/libggml-cuda.so"
} >"$result_dir/metadata.txt"

paragraph="The scheduler keeps a run queue for every processor and moves a task between queues when the load is uneven. A page fault enters the kernel, which finds the mapping, allocates a frame, fills it, and updates the page table before the task resumes. "
parent_suffix=$'\n\nTask for the parent: list the kernel mechanisms named above.'
suffix_of() {  # child index
  printf '\n\nTask for agent %s: summarize the text above in %s sentences.\n' \
    "$1" "$(( $1 + 2 ))"
}
context=4096
while (( context < paragraphs * 52 + 512 )); do context=$(( context * 2 )); done
prefix="$work/prefix"
: >"$prefix"
for _ in $(seq 1 "$paragraphs"); do printf '%s' "$paragraph" >>"$prefix"; done
text_of() {  # output file
  { grep '^TEXT ' "$1" || true; } | sha256sum | cut -c1-16
}

run_case() {  # mode first-mig other-mig
  local mode=$1 first=$2 other=$3 index mig
  local parent_env=() child_env=() state="$work/state.$mode"
  if [[ "$mode" == extent ]]; then
    parent_env=(LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_GROW="$grow")
  fi
  rm -f "$state" "$work/go" "$huge_dir/kv.0"
  env CUDA_VISIBLE_DEVICES="$first" GGML_CUDA_HOST_PTR=1 "${parent_env[@]}" \
    "$driver" parent "$model" "$context" "$prefix" "$state" "$parent_suffix" \
    "$n_gen" "$work/go" >"$work/parent" 2>"$work/parent.err" &
  parent_pid=$!
  until grep -q '^PUBLISHED ' "$work/parent" 2>/dev/null; do
    kill -0 "$parent_pid" 2>/dev/null || break
    sleep 0.02
  done
  local published tokens
  published=$(grep '^PUBLISHED ' "$work/parent" || true)
  tokens=$(sed -n 's/^PUBLISHED prefix_tokens=\([0-9]*\).*/\1/p' "$work/parent")
  if [[ "$mode" == extent ]]; then
    child_env=(LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_PREFIX="${tokens:-0}"
      LLAMA_KV_GROW="$grow")
  fi
  local before lowest sample pss=0 rss=0 pids=() migs=() failed=0
  before=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
  lowest=$before
  for index in $(seq 0 $((children - 1))); do
    if (( index % 2 == 0 )); then mig=$first; else mig=$other; fi
    migs+=("$mig")
    env CUDA_VISIBLE_DEVICES="$mig" GGML_CUDA_HOST_PTR=1 "${child_env[@]}" \
      "$driver" child "$model" "$context" "$state" "$(suffix_of "$index")" \
      "$n_gen" >"$work/child.$index" 2>"$work/child.$index.err" &
    pids+=($!)
  done
  touch "$work/go"
  local smaps=() tick=0
  for index in "${pids[@]}"; do smaps+=("/proc/$index/smaps"); done
  while kill -0 "${pids[@]}" "$parent_pid" 2>/dev/null; do
    sample=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
    (( sample < lowest )) && lowest=$sample
    if [[ "$mode" == extent ]] && (( tick % 2 == 1 )); then
      # A child may leave between the liveness test and the read.
      # shellcheck disable=SC2002
      sample=$(cat "${smaps[@]}" 2>/dev/null |
        awk -v file="$huge_dir/kv.0" '
          /^[0-9a-f]+-[0-9a-f]+ / { inside = ($6 == file) }
          inside && /^Pss:/ { pss += $2 }
          inside && /^Rss:/ { rss += $2 }
          END { printf "%d %d", pss / 1024, rss / 1024 }' || true)
      if [[ -n "$sample" ]]; then
        (( ${sample% *} > pss )) && pss=${sample% *}
        (( ${sample#* } > rss )) && rss=${sample#* }
      fi
    fi
    tick=$((tick + 1))
    sleep 0.1
  done
  {
    printf 'CASE mode=%s children=%s\n' "$mode" "$children"
    local code=0 result
    wait "$parent_pid" || code=$?
    parent_pid=
    result=$(grep '^RESULT ' "$work/parent" || true)
    if (( code != 0 )) || [[ -z "$result" ]]; then
      failed=$((failed + 1))
      sed 's/^/STDERR /' "$work/parent.err" | tail -n 5
    fi
    printf 'PARENT mig=%s exit=%s text=%s %s %s\n' "${first:4:8}" "$code" \
      "$(text_of "$work/parent")" "${published#PUBLISHED }" \
      "$(sed 's/^RESULT role=parent prefix_tokens=[0-9]* //;s/ state_ms=[0-9.]*//;s/ prefix_ms=[0-9.]*//' <<<"$result")"
    for index in "${!pids[@]}"; do
      code=0
      wait "${pids[$index]}" || code=$?
      result=$(grep '^RESULT ' "$work/child.$index" || true)
      if (( code != 0 )) || [[ -z "$result" ]]; then
        failed=$((failed + 1))
        sed 's/^/STDERR /' "$work/child.$index.err" | tail -n 5
      fi
      printf 'CHILD index=%s mig=%s exit=%s text=%s %s\n' "$index" \
        "${migs[$index]:4:8}" "$code" "$(text_of "$work/child.$index")" \
        "${result#RESULT }"
    done
    printf 'MEMORY mem_available_drop_mib=%s prefix_pss_mib=%s prefix_rss_mib=%s\n' \
      "$(( (before - lowest) / 1024 ))" "$pss" "$rss"
    printf 'END_CASE failed=%s\n' "$failed"
  } >>"$raw_log"
  rm -f "$state" "$work/go" "$huge_dir/kv.0"
}

for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then
    first=$mig_a other=$mig_b modes=(copy extent)
  else
    first=$mig_b other=$mig_a modes=(extent copy)
  fi
  {
    printf 'BEGIN_KVFORK run=%s paragraphs=%s context=%s parent_mig=%s\n' \
      "$run" "$paragraphs" "$context" "${first:4:8}"
    # What each process must write: the same tokens computed alone.
    env CUDA_VISIBLE_DEVICES="$first" GGML_CUDA_HOST_PTR=1 "$driver" alone \
      "$model" "$context" "$prefix" "$parent_suffix" "$n_gen" \
      >"$work/alone" 2>/dev/null || true
    printf 'REFERENCE who=parent text=%s\n' "$(text_of "$work/alone")"
    for index in $(seq 0 $((children - 1))); do
      if (( index % 2 == 0 )); then mig=$first; else mig=$other; fi
      env CUDA_VISIBLE_DEVICES="$mig" GGML_CUDA_HOST_PTR=1 "$driver" alone \
        "$model" "$context" "$prefix" "$(suffix_of "$index")" "$n_gen" \
        >"$work/alone" 2>/dev/null || true
      printf 'REFERENCE who=%s text=%s\n' "$index" "$(text_of "$work/alone")"
    done
  } >>"$raw_log"
  for mode in "${modes[@]}"; do
    run_case "$mode" "$first" "$other"
  done
  printf 'END_KVFORK\n' >>"$raw_log"
done

awk -f "$script_dir/summarize_engine_kvfork.awk" "$raw_log" \
  >"$result_dir/engine_kvfork_summary.csv"
(
  cd "$script_dir"
  sha256sum kv_extents.patch kv_fork.cpp summarize_engine_kvfork.awk \
    run_engine_kvfork.sh
) >"$result_dir/source_hashes.txt"
printf 'engine_kvfork_result_dir=%s\n' "$result_dir"
