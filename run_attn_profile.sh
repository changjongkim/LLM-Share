#!/usr/bin/env bash
# D2 of PLAN-20261015: where a KV cache in host memory costs time, by CUDA
# kernel. One process per case (kv_fork alone: compute a 16,321-token prefix,
# then generate N_GEN tokens) runs under Nsight Systems with the KV cache in
# device memory (`device`) or in private host memory with 2 MiB pages (`host`,
# LLAMA_KV_HOST=anon, the form host_copy_huge of the locality set). The weights
# are read in place in both. The kernel summary of each case is kept; prefill
# and decode use different kernels, so their times separate by name.
#
# Kill gates, fixed before the campaign:
#   A1  every case completes and writes its kernel summary.
# Reported, not gated: the time of every kernel by model and placement, and the
# ratio host/device.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig=${PRODUCER_MIG:-GPU-a7c66ad2-6dbb-0ab8-c1a2-37ba6dba3600}
repetitions=${REPETITIONS:-2}
n_gen=${N_GEN:-256}
driver=${DRIVER:-"$script_dir/kv_fork_tuned2"}
nsys=${NSYS:-$(command -v nsys || echo /usr/local/cuda-13.0/bin/nsys)}
read -r -a models <<<"${MODELS:-Qwen2.5-7B-Instruct-Q4_K_M.gguf Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf}"
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-attn-profile"}
result_dir="$result_root/$result_tag"
[[ -x "$driver" && -x "$nsys" ]] || { echo "missing driver or nsys" >&2; exit 2; }
[[ -e "$result_dir" ]] && { echo "result directory already exists: $result_dir" >&2; exit 2; }
mkdir -p "$result_dir"
work=$(mktemp -d)
trap 'rm -rf "${work:?}"' EXIT
paragraph="The scheduler keeps a run queue for every processor and moves a task between queues when the load is uneven. A page fault enters the kernel, which finds the mapping, allocates a frame, fills it, and updates the page table before the task resumes. "
for _ in $(seq 1 320); do printf '%s' "$paragraph"; done >"$work/prefix"
{
  printf 'timestamp=%s\nmig=%s\nrepetitions=%s\nn_gen=%s\nmodels=%s\n' "$(date --iso-8601=seconds)" "$mig" \
    "$repetitions" "$n_gen" "${models[*]}"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  "$nsys" --version
  sha256sum "$driver"
} >"$result_dir/metadata.txt"
export CUDA_VISIBLE_DEVICES="$mig"
for run in $(seq 1 "$repetitions"); do
  for model in "${models[@]}"; do
    for kv in device host; do
      envs=(GGML_CUDA_HOST_PTR=1)
      [[ "$kv" == host ]] && envs+=(LLAMA_KV_HOST=anon)
      name="${model%%-Instruct*}.$kv.$run"
      env "${envs[@]}" "$nsys" profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true \
        -o "$work/$name" "$driver" alone "$script_dir/models/$model" 32768 "$work/prefix" \
        "Task for agent 0: summarize the text above in 2 sentences." "$n_gen" >"$result_dir/$name.out" 2>&1 ||
        echo "FAILED $name" >>"$result_dir/failed.txt"
      "$nsys" stats --report cuda_gpu_kern_sum --format csv --output "$result_dir/$name" "$work/$name.nsys-rep" \
        >/dev/null 2>&1 || echo "NOSTATS $name" >>"$result_dir/failed.txt"
    done
  done
done
python3 "$script_dir/summarize_attn_profile.py" "$result_dir" >"$result_dir/attn_profile_summary.csv"
( cd "$script_dir" && sha256sum run_attn_profile.sh summarize_attn_profile.py kv_fork.cpp kv_tuned2.patch ) >"$result_dir/source_hashes.txt"
printf 'attn_profile_result_dir=%s\n' "$result_dir"
