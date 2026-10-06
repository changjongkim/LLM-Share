#!/usr/bin/env bash
# Does the page size of the mapped model file explain the cost of reading
# weights in place? The same model is served from the ext4 page cache
# (4 KiB pages), from a tmpfs mounted with huge=always (transparent 2 MiB
# pages), and from hugetlbfs (2 MiB pages from a reserved pool), against the
# upstream device copy. The mounts and the pool are removed on exit.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig_a=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
mig_b=${CONSUMER_MIG:-MIG-31ffbfe4-9a85-5f4e-b406-90ffbfa82ca3}
repetitions=${REPETITIONS:-6}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
bin_dir=${BIN_DIR:-"$script_dir/llama.cpp/build/bin"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-pages"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi
sudo -n true 2>/dev/null || {
  echo "passwordless sudo is required for the mounts and the hugetlb pool" >&2
  exit 2
}
model_bytes=$(stat -c %s "$model")
huge_pages=$(( (model_bytes + 2097151) / 2097152 ))
# The model is held three times: page cache, tmpfs, and hugetlbfs.
needed_kb=$(( 3 * model_bytes / 1024 + 24 * 1024 * 1024 ))
available_kb=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
(( available_kb > needed_kb )) || { echo "not enough available memory" >&2; exit 2; }

thp_dir="$mount_root/thp"
htlb_dir="$mount_root/hugetlb"
original_pages=$(cat /proc/sys/vm/nr_hugepages)
mounted_thp=0
mounted_htlb=0
cleanup() {
  if (( mounted_thp )); then sudo -n umount "$thp_dir" || true; fi
  if (( mounted_htlb )); then sudo -n umount "$htlb_dir" || true; fi
  sudo -n sh -c "echo $original_pages > /proc/sys/vm/nr_hugepages" || true
  rmdir "$thp_dir" "$htlb_dir" "$mount_root" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

mkdir -p "$result_dir" "$thp_dir" "$htlb_dir"
sudo -n mount -t tmpfs -o "huge=always,size=$(( model_bytes / 1048576 + 64 ))m,uid=$(id -u),gid=$(id -g)" \
  tmpfs "$thp_dir"
mounted_thp=1
cp "$model" "$thp_dir/model.gguf"
sudo -n sh -c "echo $(( original_pages + huge_pages + 8 )) > /proc/sys/vm/nr_hugepages"
(( $(cat /proc/sys/vm/nr_hugepages) >= original_pages + huge_pages )) || {
  echo "could not reserve $huge_pages huge pages" >&2
  exit 2
}
sudo -n mount -t hugetlbfs -o "pagesize=2M,uid=$(id -u),gid=$(id -g)" none "$htlb_dir"
mounted_htlb=1
# hugetlbfs has no write(); the file is filled through a mapping and is
# padded with zeros to a multiple of 2 MiB.
python3 - "$model" "$htlb_dir/model.gguf" "$huge_pages" <<'PYTHON'
import mmap, os, sys
source, target, pages = sys.argv[1], sys.argv[2], int(sys.argv[3])
size = pages * 2097152
fd = os.open(target, os.O_CREAT | os.O_RDWR, 0o644)
os.ftruncate(fd, size)
view = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
with open(source, "rb") as handle:
    offset = 0
    while True:
        chunk = handle.read(64 << 20)
        if not chunk:
            break
        view[offset:offset + len(chunk)] = chunk
        offset += len(chunk)
view.close()
os.close(fd)
PYTHON

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig_a=%s\nmig_b=%s\nrepetitions=%s\n' "$mig_a" "$mig_b" "$repetitions"
  printf 'model=%s\nmodel_bytes=%s\nhuge_pages=%s\n' "$(basename "$model")" \
    "$model_bytes" "$huge_pages"
  printf 'engine_commit=%s\n' "$(git -C "$script_dir/llama.cpp" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$bin_dir/llama-bench" "$bin_dir/llama-completion"
} >"$result_dir/metadata.txt"

# mode -> model path and environment
cases=(copy inplace_4k inplace_thp inplace_hugetlb)
path_of() {
  case $1 in
    copy | inplace_4k) printf '%s' "$model" ;;
    inplace_thp) printf '%s' "$thp_dir/model.gguf" ;;
    inplace_hugetlb) printf '%s' "$htlb_dir/model.gguf" ;;
  esac
}
migs=("$mig_a" "$mig_b")
prompt="Write a detailed technical essay about how operating systems manage memory."
for run in $(seq 1 "$repetitions"); do
  mig=${migs[$((run % 2))]}
  for offset in 0 1 2 3; do
    mode=${cases[$(((run + offset) % 4))]}
    path=$(path_of "$mode")
    extra=()
    if [[ "$mode" != copy ]]; then extra=(GGML_CUDA_HOST_PTR=1); fi
    {
      printf 'BEGIN_ENGINE_PAGES run=%s mig=%s mode=%s\n' "$run" "${mig:4:8}" "$mode"
      env CUDA_VISIBLE_DEVICES="$mig" "${extra[@]}" "$bin_dir/llama-bench" \
        -m "$path" -ngl 99 -p 512 -n 128 -r 3 -o jsonl 2>/dev/null |
        sed 's/^/BENCH /'
      status=0
      env CUDA_VISIBLE_DEVICES="$mig" "${extra[@]}" "$bin_dir/llama-completion" \
        -m "$path" -ngl 99 -n 256 -c 4096 --temp 0 --seed 1 -no-cnv --ignore-eos \
        -p "$prompt" >"$result_dir/text.tmp" 2>"$result_dir/perf.tmp" || status=$?
      grep -E 'load time|  eval time' "$result_dir/perf.tmp" |
        sed 's/^.*common_perf_print: */PERF /'
      printf 'OUTPUT bytes=%s sha256=%s\n' "$(stat -c %s "$result_dir/text.tmp")" \
        "$(sha256sum "$result_dir/text.tmp" | cut -d' ' -f1)"
      printf 'END_ENGINE_PAGES status=%s\n' "$status"
    } >>"$raw_log"
  done
done
rm -f "$result_dir/text.tmp" "$result_dir/perf.tmp"

awk -f "$script_dir/summarize_engine_pages.awk" "$raw_log" \
  >"$result_dir/engine_pages_summary.csv"
(
  cd "$script_dir"
  sha256sum inplace_weights.patch summarize_engine_pages.awk run_engine_pages.sh
) >"$result_dir/source_hashes.txt"
printf 'engine_pages_result_dir=%s\n' "$result_dir"
