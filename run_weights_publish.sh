#!/usr/bin/env bash
# Putting a model file on 2 MiB pages: how long it takes and what it leaves
# in memory. The weights are read in place fastest from a file whose pages
# are 2 MiB (a tmpfs mounted with huge=always). Three ways to put a model
# file from storage there:
#
#   cp       a buffered copy, which fills the page cache with the file as well
#   dd       a copy with direct reads of the source (dd iflag=direct), one
#            reader, through a buffer of the program
#   publish  weights_publish: direct reads of several readers into the shared
#            mapping of the target, without a buffer in between
#
# Before every case the page cache is dropped, so that the source is read
# from the device, and the target is a tmpfs of its own. Measured: the time,
# the page cache that the load leaves behind (file pages that are not shared
# memory), the largest drop of MemFree while loading, and whether the target
# equals the source.
#
# FILES lists the sources. SYNTHETIC_COPIES > 0 adds a file that is that many
# copies of the last source, to show a size that no model of this repository
# has; it is created in SYNTHETIC_DIR and removed at the end.
#
# Gates, fixed before the campaign:
#   L1  every target equals its source;
#   L2  publish and dd leave less than 5% of the size of the file in the page
#       cache, and cp leaves more than 90% of it;
#   L3  publish takes no longer than cp for every file.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repetitions=${REPETITIONS:-6}
files=${FILES:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf $script_dir/models/Qwen2.5-14B-Instruct-Q4_K_M.gguf"}
synthetic_copies=${SYNTHETIC_COPIES:-4}
synthetic_dir=${SYNTHETIC_DIR:-"$script_dir/models"}
threads=${THREADS:-8}
chunk_mib=${CHUNK_MIB:-32}
program=${PROGRAM:-"$script_dir/weights_publish"}
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-weights-publish"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
[[ -x "$program" ]] || { echo "missing $program; run make weights_publish" >&2; exit 2; }
for file in $files; do
  [[ -r "$file" ]] || { echo "cannot read $file" >&2; exit 2; }
done
if ! sudo -n true 2>/dev/null; then
  echo "passwordless sudo is required for the tmpfs mount and the cache drop" >&2
  exit 2
fi
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi

target_dir="$mount_root/weights"
mounted=0
synthetic=
cleanup() {
  if (( mounted )); then sudo -n umount "$target_dir" || true; fi
  rmdir "$target_dir" "$mount_root" 2>/dev/null || true
  if [[ -n "$synthetic" ]]; then rm -f "$synthetic"; fi
}
trap cleanup EXIT
mkdir -p "$target_dir" "$result_dir"

if (( synthetic_copies > 0 )); then
  last=${files##* }
  synthetic="$synthetic_dir/synthetic-${synthetic_copies}x-$(basename "$last")"
  : >"$synthetic"
  for _ in $(seq 1 "$synthetic_copies"); do cat "$last" >>"$synthetic"; done
  files="$files $synthetic"
fi

meminfo_kib() { awk -v name="$1:" '$1 == name { print $2 }' /proc/meminfo; }
# File pages that are not shared memory: the page cache proper.
page_cache_kib() { echo $(( $(meminfo_kib Cached) - $(meminfo_kib Shmem) )); }

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'repetitions=%s\nthreads=%s\nchunk_mib=%s\nsynthetic_copies=%s\n' \
    "$repetitions" "$threads" "$chunk_mib" "$synthetic_copies"
  for file in $files; do
    printf 'file=%s bytes=%s\n' "$(basename "$file")" "$(stat -c %s "$file")"
  done
  printf 'source_filesystem=%s\n' "$(df --output=fstype,source "$script_dir/models" | tail -n 1 | xargs)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  sha256sum "$program"
} >"$result_dir/metadata.txt"

run_case() {  # run file method
  local run=$1 file=$2 method=$3 bytes target
  bytes=$(stat -c %s "$file")
  target="$target_dir/model"
  sudo -n mount -t tmpfs -o "huge=always,size=$(( bytes / 1048576 + 64 ))m,uid=$(id -u),gid=$(id -g)" \
    tmpfs "$target_dir"
  mounted=1
  sync
  sudo -n sh -c 'echo 1 >/proc/sys/vm/drop_caches'
  local cache_before free_before lowest sample started seconds
  cache_before=$(page_cache_kib)
  free_before=$(meminfo_kib MemFree)
  lowest=$free_before
  started=$(date +%s%N)
  case $method in
    cp) cp "$file" "$target" & ;;
    dd) dd if="$file" of="$target" iflag=direct bs="${chunk_mib}M" status=none & ;;
    publish) "$program" "$file" "$target" "$threads" "$chunk_mib" >/dev/null & ;;
    *) echo "unknown method $method" >&2; exit 2 ;;
  esac
  local pid=$! code=0
  while kill -0 "$pid" 2>/dev/null; do
    sample=$(meminfo_kib MemFree)
    (( sample < lowest )) && lowest=$sample
    sleep 0.02
  done
  wait "$pid" || code=$?
  seconds=$(awk -v ns="$(( $(date +%s%N) - started ))" 'BEGIN { printf "%.3f", ns / 1e9 }')
  local cache_after identical=0
  cache_after=$(page_cache_kib)
  # The comparison reads the source through the page cache; it comes last.
  if (( code == 0 )) && cmp -s "$file" "$target"; then identical=1; fi
  printf 'PUBLISH run=%s file=%s bytes=%s method=%s exit=%s seconds=%s gib_per_s=%s page_cache_left_mib=%s free_drop_mib=%s identical=%s\n' \
    "$run" "$(basename "$file")" "$bytes" "$method" "$code" "$seconds" \
    "$(awk -v b="$bytes" -v s="$seconds" 'BEGIN { printf "%.2f", (s > 0 ? b / 1073741824 / s : 0) }')" \
    "$(( (cache_after - cache_before) / 1024 ))" "$(( (free_before - lowest) / 1024 ))" \
    "$identical" >>"$raw_log"
  sudo -n umount "$target_dir"
  mounted=0
}

for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then methods=(cp dd publish); else methods=(publish dd cp); fi
  for file in $files; do
    for method in "${methods[@]}"; do
      run_case "$run" "$file" "$method"
    done
  done
done

awk -f "$script_dir/summarize_weights_publish.awk" "$raw_log" \
  >"$result_dir/weights_publish_summary.csv"
(
  cd "$script_dir"
  sha256sum weights_publish.c summarize_weights_publish.awk run_weights_publish.sh
) >"$result_dir/source_hashes.txt"
printf 'weights_publish_result_dir=%s\n' "$result_dir"
