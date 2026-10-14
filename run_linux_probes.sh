#!/usr/bin/env bash
# The kernel's mechanisms for sharing memory between processes, on memory that
# the GPU reads in place (linux_share_probe.cu), in the 12-SM instance:
#
#   fork gpu / cpu    a zygote fills SIZE_MIB on the CPU and forks before CUDA;
#                     the child reads it on the GPU directly, or after a CPU
#                     read of every page (private memory of the child before
#                     and after)
#   fork_after_init   a child forked after its parent initialized CUDA tries
#                     to use CUDA
#   register          a process maps the model file shared and read-only and
#                     registers the mapping with CUDA, once with the file in
#                     the page cache of ext4 and once on a tmpfs with 2 MiB
#                     pages; REGISTER_PROCS processes do it at the same time
#
# Kill gates, fixed before the campaign:
#   P1  every fork and register case exits and reports its line;
#   P2  in fork gpu, the child's private memory grows by at least 90% of
#       SIZE_MIB, and in fork cpu by less than 5%;
#   P3  fork_after_init never lets the child allocate device memory.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
repetitions=${REPETITIONS:-6}
size_mib=${SIZE_MIB:-1024}
register_procs=${REGISTER_PROCS:-4}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
probe=${PROBE:-"$script_dir/linux_share_probe"}
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-linux-probes"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
if [[ ! -x "$probe" || ! -r "$model" ]]; then
  echo "missing probe or model; run make linux_share_probe" >&2
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
memfs_dir="$mount_root/probe_memfs"
mounted=0
cleanup() {
  pkill -9 -P $$ 2>/dev/null || true
  if (( mounted )); then sudo -n umount "$memfs_dir" || true; fi
  rmdir "$memfs_dir" "$mount_root" 2>/dev/null || true
}
trap cleanup EXIT
mkdir -p "$memfs_dir" "$result_dir"
sudo -n mount -t tmpfs -o "huge=always,size=8192m,uid=$(id -u),gid=$(id -g)" tmpfs "$memfs_dir"
mounted=1
dd if="$model" of="$memfs_dir/model.gguf" bs=64M iflag=direct status=none

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig=%s\nrepetitions=%s\nsize_mib=%s\nregister_procs=%s\n' "$mig" "$repetitions" \
    "$size_mib" "$register_procs"
  printf 'model=%s\nmodel_fs=%s\n' "$(basename "$model")" "$(stat -f -c %T "$model")"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$probe"
} >"$result_dir/metadata.txt"

export CUDA_VISIBLE_DEVICES="$mig"
for run in $(seq 1 "$repetitions"); do
  printf 'BEGIN_PROBES run=%s\n' "$run" >>"$raw_log"
  order=(gpu cpu)
  (( run % 2 == 0 )) && order=(cpu gpu)
  for kind in "${order[@]}"; do
    timeout -s KILL 300 "$probe" fork "$size_mib" "$kind" >>"$raw_log" 2>&1 ||
      printf 'FORK mode=%s failed=1\n' "$kind" >>"$raw_log"
  done
  timeout -s KILL 120 "$probe" fork_after_init 64 >>"$raw_log" 2>&1 ||
    printf 'FORK_AFTER_INIT failed=1\n' >>"$raw_log"
  for where in pagecache memfs; do
    file=$model
    if [[ "$where" == pagecache ]]; then
      python3 -c 'import os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)' "$model"
    else
      file="$memfs_dir/model.gguf"
    fi
    pids=()
    for index in $(seq 0 $((register_procs - 1))); do
      timeout -s KILL 600 "$probe" register "$file" >"$result_dir/register.$index" 2>&1 &
      pids+=($!)
    done
    for index in "${!pids[@]}"; do
      wait "${pids[$index]}" || true
      sed "s/^REGISTER /REGISTER where=$where procs=$register_procs index=$index /" \
        "$result_dir/register.$index" | grep '^REGISTER ' >>"$raw_log" ||
        printf 'REGISTER where=%s index=%s failed=1\n' "$where" "$index" >>"$raw_log"
    done
    rm -f "$result_dir"/register.*
  done
  printf 'END_PROBES\n' >>"$raw_log"
done
awk -f "$script_dir/summarize_linux_probes.awk" "$raw_log" >"$result_dir/linux_probes_summary.csv"
(
  cd "$script_dir"
  sha256sum linux_share_probe.cu summarize_linux_probes.awk run_linux_probes.sh
) >"$result_dir/source_hashes.txt"
printf 'linux_probes_result_dir=%s\n' "$result_dir"
