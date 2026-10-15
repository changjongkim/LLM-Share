#!/usr/bin/env bash
# The read pattern of attention over a KV cache in device memory and in host
# memory of several page sizes (gpu_attn_probe.cu), with the row of Qwen2.5-7B
# (4 KV heads, 1 KiB) and of Llama-3.1-8B (8 KV heads, 2 KiB). Every kind runs
# in REPETITIONS processes of ROUNDS rounds each, in the order of KINDS.
#
# hugetlb pages are reserved for the campaign (2 MiB pages for SIZE_MIB plus
# 64 MiB, two 1 GiB pages) and the previous reservation is restored at exit.
#
# Kill gates, fixed before the campaign:
#   H1  every kind that maps its buffer reports every round; a kind whose
#       buffer cannot be mapped (no hugetlb page) is recorded as unavailable.
# Reported, not gated: the rate of each kind and pattern, and its ratio to
# device memory.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig=${PRODUCER_MIG:-GPU-a7c66ad2-6dbb-0ab8-c1a2-37ba6dba3600}
repetitions=${REPETITIONS:-3}
rounds=${ROUNDS:-6}
size_mib=${SIZE_MIB:-1024}
read -r -a kinds <<<"${KINDS:-device host_4k host_thp tlb_2m tlb_1g file_ro}"
read -r -a row_set <<<"${ROWS:-1024 2048}"
probe=${PROBE:-"$script_dir/gpu_attn_probe"}
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-gpu-attn-probe"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"
[[ -x "$probe" ]] || { echo "missing probe; run make gpu_attn_probe" >&2; exit 2; }
sudo -n true 2>/dev/null || { echo "passwordless sudo is required" >&2; exit 2; }
[[ -e "$result_dir" ]] && { echo "result directory already exists: $result_dir" >&2; exit 2; }

h2=/sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
h1g=/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
old2=$(cat "$h2"); old1g=$(cat "$h1g")
memfs_dir="$mount_root/attn_memfs"
mounted=0
cleanup() {
  pkill -9 -P $$ 2>/dev/null || true
  echo "$old2" | sudo -n tee "$h2" >/dev/null || true
  echo "$old1g" | sudo -n tee "$h1g" >/dev/null || true
  if (( mounted )); then rm -f "$memfs_dir/kv"; sudo -n umount "$memfs_dir" || true; fi
  rmdir "$memfs_dir" "$mount_root" 2>/dev/null || true
}
trap cleanup EXIT
mkdir -p "$memfs_dir" "$result_dir"
sudo -n mount -t tmpfs -o "huge=always,size=$(( size_mib + 512 ))m,uid=$(id -u),gid=$(id -g)" tmpfs "$memfs_dir"
mounted=1
echo $(( size_mib / 2 + 32 )) | sudo -n tee "$h2" >/dev/null
echo $(( size_mib / 1024 + 1 )) | sudo -n tee "$h1g" >/dev/null
{
  printf 'timestamp=%s\nmig=%s\nrepetitions=%s\nrounds=%s\nsize_mib=%s\nkinds=%s\nrows=%s\n' \
    "$(date --iso-8601=seconds)" "$mig" "$repetitions" "$rounds" "$size_mib" "${kinds[*]}" "${row_set[*]}"
  printf 'hugepages_2m=%s hugepages_1g=%s (reserved for the campaign)\n' "$(cat "$h2")" "$(cat "$h1g")"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$probe"
} >"$result_dir/metadata.txt"

export CUDA_VISIBLE_DEVICES="$mig"
for run in $(seq 1 "$repetitions"); do
  for row in "${row_set[@]}"; do
    for kind in "${kinds[@]}"; do
      printf 'CASE run=%s kind=%s row_bytes=%s\n' "$run" "$kind" "$row" >>"$raw_log"
      timeout -s KILL 600 "$probe" "$kind" "$size_mib" "$row" "$rounds" "$memfs_dir/kv" >>"$raw_log" 2>&1 ||
        printf 'FAILED kind=%s row_bytes=%s\n' "$kind" "$row" >>"$raw_log"
      rm -f "$memfs_dir/kv"
    done
  done
done
awk -f "$script_dir/summarize_gpu_attn_probe.awk" "$raw_log" >"$result_dir/gpu_attn_probe_summary.csv"
(
  cd "$script_dir"
  sha256sum gpu_attn_probe.cu summarize_gpu_attn_probe.awk run_gpu_attn_probe.sh
) >"$result_dir/source_hashes.txt"
printf 'gpu_attn_probe_result_dir=%s\n' "$result_dir"
