#!/usr/bin/env bash
# Why STATOR generates at 0.86 of Copy with a 64K-token prefix (098, ctx64k,
# four agents), when the ratio is 0.97 at 32K and 0.99 at 16K. One agent at a
# time, so that no other process shares the GPU, on the publishers and the
# grouped decode attention of 098 (kv_fork3, llama.cpp-kv3,
# GGML_CUDA_FATTN_HOST_GQA=2):
#
#   restore      the prefix copied from the state file into device memory
#   extent_lazy  the prefix mapped read-only from the publisher's file on a
#                tmpfs with 2 MiB pages, the agent's own rows private
#
# Measured per agent: the RESULT line (model, context, state, suffix and
# generation time), the page faults of the driver's bottom-half threads while
# the agent runs (an anchor process keeps the threads alive), and for
# extent_lazy the largest Rss, ShmemPmdMapped and FilePmdMapped of the
# mappings of the prefix file in /proc/PID/smaps (whether the GPU reads the
# prefix through 2 MiB or 4 KiB entries). Then one agent of each mode under
# nsys gives the GPU time per kernel. Questions:
#   C1  does one agent alone reproduce the gap (a read-path cost), or only
#       four agents (contention)?
#   C2  does the GPU fault while it decodes on extents?
#   C3  is the prefix mapping PMD-mapped?
#   C4  which kernels take the extra time?
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
gpu=${PRODUCER_MIG:?}
repetitions=${REPETITIONS:-3}
paragraphs=${PARAGRAPHS:-1280}
context=${CONTEXT:-131072}
n_gen=${N_GEN:-64}
grow=${GROW_ROWS:-256}
model=${MODEL:-"$script_dir/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"}
driver=${DRIVER:-"$script_dir/kv_fork3"}
nsys_bin=${NSYS:-/usr/local/cuda-13.0/bin/nsys}
result_dir="$script_dir/results/${RESULT_TAG:-$(date +%Y%m%d)-engine-kvctx-cause-v1}"
raw="$result_dir/raw.log"
[[ -e "$result_dir" ]] && { echo "exists: $result_dir" >&2; exit 2; }
mkdir -p "$result_dir"
huge="$script_dir/mounts/kv_cause"
mkdir -p "$huge"
sudo -n mount -t tmpfs -o "huge=always,size=12288m,uid=$(id -u),gid=$(id -g)" tmpfs "$huge"
work=$(mktemp -d "$result_dir/work.XXXXXX")
anchor=
cleanup() {
  [[ -n "$anchor" ]] && kill "$anchor" 2>/dev/null || true
  sudo -n umount "$huge" 2>/dev/null || true
  rmdir "$huge" 2>/dev/null || true
  rm -f "$work"/state.* 2>/dev/null || true
}
trap cleanup EXIT
export GGML_CUDA_FATTN_HOST_GQA=2

paragraph="The scheduler keeps a run queue for every processor and moves a task between queues when the load is uneven. A page fault enters the kernel, which finds the mapping, allocates a frame, fills it, and updates the page table before the task resumes. "
prefix="$work/prefix"
: >"$prefix"
for _ in $(seq 1 "$paragraphs"); do printf '%s' "$paragraph" >>"$prefix"; done
suffix=$'\n\nTask for agent 0: summarize the text above in 2 sentences.\n'

fault_pages() {
  local total=0 line rest
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
smaps_of() {  # pid file: Rss, ShmemPmdMapped, FilePmdMapped (KiB) over the mappings of file
  awk -v file="$2" '
    /^[0-9a-f]+-[0-9a-f]+ / { inside = ($6 == file) }
    inside && /^Rss:/ { rss += $2 }
    inside && /^ShmemPmdMapped:/ { shm += $2 }
    inside && /^FilePmdMapped:/ { fpm += $2 }
    END { printf "%d %d %d", rss, shm, fpm }' "/proc/$1/smaps" 2>/dev/null || printf '0 0 0'
}

{
  printf 'timestamp=%s\nparagraphs=%s\ncontext=%s\nn_gen=%s\nrepetitions=%s\n' "$(date --iso-8601=seconds)" \
    "$paragraphs" "$context" "$n_gen" "$repetitions"
  printf 'model=%s\nfattn_host_gqa=%s\n' "$(basename "$model")" "$GGML_CUDA_FATTN_HOST_GQA"
  sha256sum "$driver" "$script_dir/llama.cpp-kv3/build/bin/libllama.so" "$script_dir/llama.cpp-kv3/build/bin/libggml-cuda.so" "$0"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
} >"$result_dir/metadata.txt"

CUDA_VISIBLE_DEVICES="$gpu" "$script_dir/linux_share_probe" hold 1000000 >/dev/null 2>&1 &
anchor=$!
for _ in $(seq 1 100); do grep -qs 'UVM GPU' /proc/[0-9]*/comm && break; sleep 0.1; done

sync
sudo -n sh -c 'echo 1 >/proc/sys/vm/drop_caches'
cat "$model" >/dev/null
publish() {  # label state env...
  local label=$1 state=$2
  shift 2
  env CUDA_VISIBLE_DEVICES="$gpu" GGML_CUDA_HOST_PTR=1 "$@" \
    "$driver" parent "$model" "$context" "$prefix" "$state" "" 0 >"$work/publish.$label" 2>"$work/publish.$label.err"
  printf 'PUBLISH store=%s %s\n' "$label" "$(grep '^PUBLISHED ' "$work/publish.$label" | sed 's/^PUBLISHED //')" >>"$raw"
}
publish device "$work/state.device"
publish huge "$work/state.huge" LLAMA_KV_HOST="$huge/kv" LLAMA_KV_GROW="$grow"
chmod 0400 "$huge/kv.0"
tokens=$(sed -n 's/^PUBLISHED prefix_tokens=\([0-9]*\).*/\1/p' "$work/publish.huge")
printf 'FILE kv0_kib=%s prefix_tokens=%s\n' "$(du -k "$huge/kv.0" | cut -f1)" "$tokens" >>"$raw"

agent() {  # mode run [nsys-output]
  local mode=$1 run=$2 report=${3:-} envs=() state="$work/state.device" pid f0 f1 best="0 0 0" s
  if [[ "$mode" == extent_lazy ]]; then
    envs=(LLAMA_KV_HOST="$huge/kv" LLAMA_KV_PREFIX="$tokens" LLAMA_KV_GROW="$grow")
    state="$work/state.huge"
  fi
  local cmd=("$driver" child "$model" "$context" "$state" "$suffix" "$n_gen")
  [[ -n "$report" ]] && cmd=("$nsys_bin" profile --trace=cuda --sample=none --cpuctxsw=none \
    --force-overwrite=true -o "$report" "${cmd[@]}")
  f0=$(fault_pages)
  env CUDA_VISIBLE_DEVICES="$gpu" GGML_CUDA_HOST_PTR=1 "${envs[@]}" "${cmd[@]}" \
    >"$work/agent.out" 2>"$work/agent.err" &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    if [[ "$mode" == extent_lazy ]]; then
      # the agent itself (under nsys: the child that maps the file)
      for p in $pid $(pgrep -P "$pid" 2>/dev/null) ; do
        s=$(smaps_of "$p" "$huge/kv.0")
        (( ${s%% *} > ${best%% *} )) && best=$s
      done
    fi
    sleep 0.25
  done
  local code=0
  wait "$pid" || code=$?
  f1=$(fault_pages)
  printf 'AGENT mode=%s run=%s nsys=%s exit=%s gpu_fault_pages=%s kv_rss_kib=%s shmem_pmd_kib=%s file_pmd_kib=%s %s\n' \
    "$mode" "$run" "$([[ -n "$report" ]] && echo 1 || echo 0)" "$code" "$(( f1 - f0 ))" ${best} \
    "$(grep '^RESULT ' "$work/agent.out" | sed 's/^RESULT //')" >>"$raw"
  if (( code != 0 )); then sed 's/^/STDERR /' "$work/agent.err" | tail -n 5 >>"$raw"; fi
}

for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then modes=(restore extent_lazy); else modes=(extent_lazy restore); fi
  for mode in "${modes[@]}"; do agent "$mode" "$run"; done
done
for mode in restore extent_lazy; do
  agent "$mode" nsys "$result_dir/nsys.$mode"
  "$nsys_bin" stats --report cuda_gpu_kern_sum --format csv --force-export=true \
    -o "$result_dir/kern.$mode" "$result_dir/nsys.$mode.nsys-rep" >/dev/null 2>&1 || true
done
printf 'END\n' >>"$raw"
echo "result_dir=$result_dir"
