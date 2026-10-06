#!/usr/bin/env bash
# Can a sharer change shared state through the GPU? Two routes:
#
#   host  a file on a tmpfs with 2 MiB pages holds a pattern. A process maps
#         it read-only and private, as an agent maps a published prefix,
#         reads it on the GPU and then writes its first word on the GPU. A
#         second process in the other MIG instance then does the same.
#   vmm   a process shares device memory through the CUDA virtual memory
#         management interface with its own mapping read-only. The importer
#         maps it read-only, raises its own mapping to read-write and writes
#         the first word.
#
# Expectations, fixed before the campaign: on the host route the GPU read
# returns every word, the GPU write fails in the writing process, the file
# keeps the pattern, and the second process still reads every word (P1). On
# the vmm route the importer chooses the access of its mapping: if the raise
# and the write succeed, the exporter reads the changed word (P2). Either
# outcome of P2 is reported; it decides what the record may say about the
# integrity of device memory that is shared across processes.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
size=${SIZE_MIB:-64}
probe="$script_dir/cuda_protect_probe"
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-protect"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

[[ -x "$probe" ]] || { echo "missing $probe; run make cuda_protect_probe" >&2; exit 2; }
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
cleanup() {
  if (( mounted_huge )); then sudo -n umount "$huge_dir" || true; fi
  rmdir "$huge_dir" "$mount_root" 2>/dev/null || true
}
trap cleanup EXIT
mkdir -p "$huge_dir" "$result_dir"
sudo -n mount -t tmpfs -o "huge=always,size=1024m,uid=$(id -u),gid=$(id -g)" \
  tmpfs "$huge_dir"
mounted_huge=1
{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\nsize_mib=%s\n' "$mig_a" "$mig_b" \
    "$repetitions" "$size"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$probe"
} >"$result_dir/metadata.txt"

file="$huge_dir/pattern"
python3 -c "
import sys
word = (0x5eedf00d).to_bytes(4, 'little')
with open(sys.argv[1], 'wb') as handle:
    for _ in range(int(sys.argv[2])):
        handle.write(word * (1 << 18))
" "$file" "$size"
chmod 0400 "$file"

for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then first=$mig_a second=$mig_b; else first=$mig_b second=$mig_a; fi
  {
    for mig in "$first" "$second"; do
      line=$(CUDA_VISIBLE_DEVICES="$mig" "$probe" host "$file" "$size" 2>/dev/null || true)
      printf 'PROBE run=%s mig=%s %s\n' "$run" "${mig:4:8}" "${line#RESULT }"
    done
    line=$(CUDA_VISIBLE_DEVICES="$first" "$probe" vmm "$size" 2>/dev/null || true)
    printf 'PROBE run=%s mig=%s %s\n' "$run" "${first:4:8}" "${line#RESULT }"
  } >>"$raw_log"
done

awk -f "$script_dir/summarize_protect.awk" "$raw_log" \
  >"$result_dir/protect_summary.csv"
(
  cd "$script_dir"
  sha256sum cuda_protect_probe.cu summarize_protect.awk run_protect.sh
) >"$result_dir/source_hashes.txt"
printf 'protect_result_dir=%s\n' "$result_dir"
