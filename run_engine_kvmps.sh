#!/usr/bin/env bash
# Extents under every way the GPU can be shared. A parent computes a prefix,
# hands its key-value cache to N children and keeps generating while they
# generate. The processes share the GPU in one of four configurations:
#
#   timeslice  every process in the 12-SM MIG instance, time-sliced
#   mps        every process a client of one MPS server in that instance
#   mig        the parent and every second child in the 12-SM instance, the
#              other children in the 6-SM instance
#   mig_mps    the same placement, every process a client of the MPS server
#              of its instance
#
# and the state is handed over in one of two ways:
#
#   copy    the parent saves the engine's state file, each child copies the
#           rows into its own device memory
#   extent  the parent freezes the rows in its cache file (tmpfs, 2 MiB
#           pages), each child maps them read-only and keeps the rows it
#           writes in private memory that follows use
#
# Kill gates, fixed before the campaign:
#   G1  no process fails in any cell;
#   G2  every child on extents writes the text of the child that received a
#       copy in the same configuration and repetition;
#   G3  the children on extents hold one copy of the prefix between them
#       (proportional share of the cache file at most 1.1 times its size);
#   G4  the children on extents need less memory than the children on copies
#       in every cell;
#   G5  the pause of the parent and the attach of a child are not longer
#       with extents than with copies in any cell;
#   G6  the summed generation speed of the children on extents is at least
#       0.97 of that on copies in timeslice and mps, and at least 0.92 in
#       mig and mig_mps, which include the 6-SM instance where a cache in
#       host memory is known to cost up to 5.8%.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
child_counts=${CHILD_COUNTS:-"4 8"}
paragraphs=${PARAGRAPHS:-"80 320"}
n_gen=${N_GEN:-64}
grow=${GROW_ROWS:-256}
limit_s=${CASE_LIMIT_S:-900}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
engine=${KV_ENGINE:-"$script_dir/llama.cpp-kv"}
driver=${DRIVER:-"$script_dir/kv_fork"}
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-kvmps"}
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
if pgrep -f '(^|/)nvidia-cuda-mps-control( |$)' >/dev/null ||
   pgrep -f '(^|/)nvidia-cuda-mps-server( |$)' >/dev/null; then
  echo "an MPS daemon is already running; refusing to reuse it" >&2
  exit 2
fi

# The MPS control socket needs a short path; MPS_ROOT names the directory.
if [[ -n "${MPS_ROOT:-}" ]]; then
  mps_root=$MPS_ROOT
  mkdir -p "$mps_root"
  [[ -z "$(ls -A "$mps_root")" ]] || { echo "MPS_ROOT is not empty" >&2; exit 2; }
else
  mps_root=$(mktemp -d "${TMPDIR:-/tmp}/km.XXXXXX")
fi
(( ${#mps_root} <= 84 )) || { echo "MPS directory path too long" >&2; exit 2; }
pipe_a="$mps_root/a"
pipe_b="$mps_root/b"
mkdir -p "$pipe_a" "$pipe_b" "$mps_root/la" "$mps_root/lb"
running_a=0
running_b=0
start_mps() {
  if [[ "$1" == a && "$running_a" -eq 0 ]]; then
    env CUDA_VISIBLE_DEVICES="$mig_a" CUDA_MPS_PIPE_DIRECTORY="$pipe_a" \
      CUDA_MPS_LOG_DIRECTORY="$mps_root/la" nvidia-cuda-mps-control -d
    running_a=1
  elif [[ "$1" == b && "$running_b" -eq 0 ]]; then
    env CUDA_VISIBLE_DEVICES="$mig_b" CUDA_MPS_PIPE_DIRECTORY="$pipe_b" \
      CUDA_MPS_LOG_DIRECTORY="$mps_root/lb" nvidia-cuda-mps-control -d
    running_b=1
  fi
}
stop_mps() {
  if [[ "$1" == a && "$running_a" -eq 1 ]]; then
    env CUDA_MPS_PIPE_DIRECTORY="$pipe_a" CUDA_MPS_LOG_DIRECTORY="$mps_root/la" \
      bash -c 'echo quit | nvidia-cuda-mps-control' >/dev/null 2>&1 || true
    running_a=0
  elif [[ "$1" == b && "$running_b" -eq 1 ]]; then
    env CUDA_MPS_PIPE_DIRECTORY="$pipe_b" CUDA_MPS_LOG_DIRECTORY="$mps_root/lb" \
      bash -c 'echo quit | nvidia-cuda-mps-control' >/dev/null 2>&1 || true
    running_b=0
  fi
}
prepare() {  # configuration
  case $1 in
    mig | timeslice) stop_mps a; stop_mps b ;;
    mps) start_mps a; stop_mps b ;;
    mig_mps) start_mps a; start_mps b ;;
  esac
}
# Prints "MIG PIPE_DIR" for a process of a configuration; index -1 is the
# parent. PIPE_DIR is "-" for a process that is not an MPS client.
placement() {  # configuration index
  local config=$1 index=$2
  case $config in
    timeslice) echo "$mig_a -" ;;
    mps) echo "$mig_a $pipe_a" ;;
    mig) if (( index % 2 != 1 )); then echo "$mig_a -"; else echo "$mig_b -"; fi ;;
    mig_mps)
      if (( index % 2 != 1 )); then echo "$mig_a $pipe_a"; else echo "$mig_b $pipe_b"; fi ;;
  esac
}

huge_dir="$mount_root/kv_huge"
mounted_huge=0
work=
live_pids=()
cleanup() {
  if (( ${#live_pids[@]} )); then kill -9 "${live_pids[@]}" 2>/dev/null || true; fi
  stop_mps a
  stop_mps b
  rm -rf "$mps_root"
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
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\nchild_counts=%s\n' \
    "$mig_a" "$mig_b" "$repetitions" "$child_counts"
  printf 'paragraphs=%s\nn_gen=%s\ngrow_rows=%s\ncase_limit_s=%s\n' \
    "$paragraphs" "$n_gen" "$grow" "$limit_s"
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
context_of() {  # paragraphs
  local need=$(( $1 * 52 + 512 )) size=4096
  while (( size < need )); do size=$(( size * 2 )); done
  printf '%s' "$size"
}
text_of() {  # output file
  { grep '^TEXT ' "$1" || true; } | sha256sum | cut -c1-16
}
# Runs the driver as a process of a configuration, in the background.
launch() {  # configuration index output env-and-arguments...
  local mig pipe output=$3
  read -r mig pipe <<<"$(placement "$1" "$2")"
  shift 3
  local mps_env=()
  if [[ "$pipe" != - ]]; then mps_env=(CUDA_MPS_PIPE_DIRECTORY="$pipe"); fi
  env -u CUDA_MPS_PIPE_DIRECTORY CUDA_VISIBLE_DEVICES="$mig" \
    GGML_CUDA_HOST_PTR=1 "${mps_env[@]}" "$@" >"$output" 2>"$output.err" &
}

run_case() {  # configuration mode children context prefix-file
  local config=$1 mode=$2 children=$3 context=$4 prefix=$5 index mig pipe
  local parent_env=() child_env=() state="$work/state.$mode"
  if [[ "$mode" == extent ]]; then
    parent_env=(LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_GROW="$grow")
  fi
  rm -f "$state" "$work/go" "$huge_dir/kv.0"
  launch "$config" -1 "$work/parent" "${parent_env[@]}" \
    "$driver" parent "$model" "$context" "$prefix" "$state" "$parent_suffix" \
    "$n_gen" "$work/go"
  local parent_pid=$!
  live_pids=("$parent_pid")
  until grep -q '^PUBLISHED ' "$work/parent" 2>/dev/null; do
    kill -0 "$parent_pid" 2>/dev/null || break
    sleep 0.02
  done
  local published tokens file_mib=0
  published=$(grep '^PUBLISHED ' "$work/parent" || true)
  tokens=$(sed -n 's/^PUBLISHED prefix_tokens=\([0-9]*\).*/\1/p' "$work/parent")
  if [[ "$mode" == extent ]]; then
    child_env=(LLAMA_KV_HOST="$huge_dir/kv" LLAMA_KV_PREFIX="${tokens:-0}"
      LLAMA_KV_GROW="$grow")
    file_mib=$(( $(du -k "$huge_dir/kv.0" 2>/dev/null | cut -f1) / 1024 ))
  fi
  local before lowest sample pss=0 rss=0 pids=() migs=() failed=0
  before=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
  lowest=$before
  local started
  started=$(date +%s%N)
  for index in $(seq 0 $((children - 1))); do
    read -r mig pipe <<<"$(placement "$config" "$index")"
    migs+=("$mig")
    launch "$config" "$index" "$work/child.$index" "${child_env[@]}" \
      "$driver" child "$model" "$context" "$state" "$(suffix_of "$index")" \
      "$n_gen"
    pids+=($!)
  done
  live_pids=("$parent_pid" "${pids[@]}")
  touch "$work/go"
  # Lowest available memory, and the pages of the parent's cache file that
  # the children map, until every process has left. A cell that exceeds the
  # limit is ended and counted as failed.
  local smaps=() tick=0 alive=1 timed_out=0 finished=0
  for index in "${pids[@]}"; do smaps+=("/proc/$index/smaps"); done
  while (( alive > 0 )); do
    alive=0
    for index in "${live_pids[@]}"; do
      if kill -0 "$index" 2>/dev/null; then alive=$((alive + 1)); fi
    done
    if (( finished == 0 )); then
      local left=0
      for index in "${pids[@]}"; do
        if kill -0 "$index" 2>/dev/null; then left=$((left + 1)); fi
      done
      if (( left == 0 )); then finished=$(date +%s%N); fi
    fi
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
    if (( ($(date +%s%N) - started) / 1000000000 > limit_s )); then
      timed_out=1
      kill -9 "${live_pids[@]}" 2>/dev/null || true
    fi
    tick=$((tick + 1))
    sleep 0.1
  done
  (( finished > 0 )) || finished=$(date +%s%N)
  {
    printf 'CASE config=%s mode=%s children=%s\n' "$config" "$mode" "$children"
    local code=0 result
    wait "$parent_pid" || code=$?
    result=$(grep '^RESULT ' "$work/parent" || true)
    if (( code != 0 )) || [[ -z "$result" ]]; then
      failed=$((failed + 1))
      sed 's/^/STDERR /' "$work/parent.err" | tail -n 5
    fi
    printf 'PARENT mig=%s exit=%s text=%s %s %s\n' "${mig_a:4:8}" "$code" \
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
    printf 'MEMORY mem_available_drop_mib=%s prefix_pss_mib=%s prefix_rss_mib=%s file_mib=%s\n' \
      "$(( (before - lowest) / 1024 ))" "$pss" "$rss" "$file_mib"
    printf 'END_CASE failed=%s timed_out=%s children_wall_ms=%s\n' "$failed" \
      "$timed_out" "$(( (finished - started) / 1000000 ))"
  } >>"$raw_log"
  live_pids=()
  rm -f "$state" "$work/go" "$huge_dir/kv.0"
}

for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then
    configs=(timeslice mps mig mig_mps) modes=(copy extent)
  else
    configs=(mig_mps mig mps timeslice) modes=(extent copy)
  fi
  for count in $paragraphs; do
    context=$(context_of "$count")
    prefix="$work/prefix.$count"
    : >"$prefix"
    for _ in $(seq 1 "$count"); do printf '%s' "$paragraph" >>"$prefix"; done
    {
      printf 'BEGIN_KVMPS run=%s paragraphs=%s context=%s\n' "$run" "$count" \
        "$context"
      # What a child must write: the same tokens computed alone in its MIG
      # instance, without MPS. Odd children also run in the 6-SM instance.
      stop_mps a
      stop_mps b
      highest=0
      for children in $child_counts; do
        (( children > highest )) && highest=$children
      done
      for index in $(seq 0 $((highest - 1))); do
        for mig in "$mig_a" "$mig_b"; do
          if [[ "$mig" == "$mig_b" ]] && (( index % 2 == 0 )); then continue; fi
          env CUDA_VISIBLE_DEVICES="$mig" GGML_CUDA_HOST_PTR=1 "$driver" alone \
            "$model" "$context" "$prefix" "$(suffix_of "$index")" "$n_gen" \
            >"$work/alone" 2>/dev/null || true
          printf 'REFERENCE who=%s mig=%s text=%s\n' "$index" "${mig:4:8}" \
            "$(text_of "$work/alone")"
        done
      done
    } >>"$raw_log"
    for config in "${configs[@]}"; do
      prepare "$config"
      for children in $child_counts; do
        for mode in "${modes[@]}"; do
          run_case "$config" "$mode" "$children" "$context" "$prefix"
        done
      done
    done
    printf 'END_KVMPS\n' >>"$raw_log"
  done
done
stop_mps a
stop_mps b

awk -f "$script_dir/summarize_engine_kvmps.awk" "$raw_log" \
  >"$result_dir/engine_kvmps_summary.csv"
(
  cd "$script_dir"
  sha256sum kv_extents.patch kv_fork.cpp summarize_engine_kvmps.awk \
    run_engine_kvmps.sh
) >"$result_dir/source_hashes.txt"
printf 'engine_kvmps_result_dir=%s\n' "$result_dir"
