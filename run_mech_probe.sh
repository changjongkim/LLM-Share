#!/usr/bin/env bash
# The sharing mechanisms of the kernel when the CPU or the GPU accesses the
# shared pages (mech_probe.cu), on the whole GPU. Every case runs one process
# that initializes CUDA; only the accessor differs (cpu or gpu):
#
#   private   a private writable file mapping: read every word, then write one
#             word per page (pages copied, time, faults)
#   shared    a shared writable file mapping: read every word
#   fork      memory inherited from a zygote: read every word, then write one
#             word per page in the child
#   ksm       SIZE_MIB in COPIES buffers with the same content, merged by ksmd:
#             read every word, then write one word per page (pages that remain
#             merged)
#   reclaim   a hot file (SIZE_MIB, read repeatedly for RECLAIM_S seconds)
#             and a cold file (SIZE_MIB, read once from the CPU and never again)
#             mapped in one process, without reclaim (none) and while the
#             runner asks the kernel to reclaim RECLAIM_STEP_MIB from the
#             control group of the process every second until it has asked for
#             RECLAIM_TOTAL_MIB (pressure), more than the cold file holds. The
#             hot file is read by the CPU or the GPU in the page cache (none),
#             by the GPU in the page cache locked with mlock (lock), or by the
#             GPU from a memory file system with 2 MiB pages (memfs, the
#             placement of STATOR, which swap alone could reclaim and the
#             device has no swap). Reports the share of each file in memory at
#             the end, which shows what reclaim chose, the memory that the
#             requests freed, and the time of each request.
#
# The file is SIZE_MIB on ext4. It is in the page cache before the private and
# shared cases, and dropped from it before a reclaim case, so that the process
# brings it into the page cache of its own control group.
#
# After REPETITIONS timed repetitions, one traced repetition counts kernel
# events with bpftrace (notifier invalidations, SMMU invalidations,
# write-protect faults, unmaps, accessed-flag scans, fork page-table copies,
# KSM merges, and the faults of the process and of the threads of the driver);
# private and fork run there once with the read only and once with both steps.
#
# Gates, fixed before the campaign:
#   M1  every case exits 0 and every read returns the right sum;
#   M2  a CPU read of the private mapping and of the inherited memory adds less
#       than 1% of SIZE_MIB of private memory;
#   M3  without reclaim, the hot and the cold file remain at least 99% in memory.
# Reported, not gated: everything else.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
gpu=${PRODUCER_MIG:-0}
repetitions=${REPETITIONS:-6}
size_mib=${SIZE_MIB:-1024}
copies=${COPIES:-4}
reclaim_s=${RECLAIM_S:-20}
reclaim_step_mib=${RECLAIM_STEP_MIB:-256}
reclaim_total_mib=${RECLAIM_TOTAL_MIB:-1536}
ksm_pages=${KSM_PAGES:-20000}
ksm_sleep_ms=${KSM_SLEEP_MS:-1}
probe=${PROBE:-"$script_dir/mech_probe"}
anchor_probe=${ANCHOR_PROBE:-"$script_dir/linux_share_probe"}
work_dir=${WORK_DIR:-"$script_dir/mech_work"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-mech-probe"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"
ksm=/sys/kernel/mm/ksm

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
for binary in "$probe" "$anchor_probe"; do
  [[ -x "$binary" ]] || { echo "missing $binary" >&2; exit 2; }
done
if ! sudo -n true 2>/dev/null || [[ ! -e "$ksm/run" ]] || ! command -v bpftrace >/dev/null; then
  echo "passwordless sudo, KSM and bpftrace are required" >&2
  exit 2
fi
if [[ "$(cat "$ksm/pages_sharing")" != 0 || "$(cat "$ksm/run")" != 0 ]]; then
  echo "KSM is in use; it must be stopped with no merged pages" >&2
  exit 2
fi
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi

file="$work_dir/mech_buffer.bin"
cold="$work_dir/mech_cold.bin"
memfs="$work_dir/memfs"
mounted=0
pages_before=$(cat "$ksm/pages_to_scan")
sleep_before=$(cat "$ksm/sleep_millisecs")
ksm_set() { echo "$2" | sudo -n tee "$ksm/$1" >/dev/null; }
drop_cache() {
  python3 -c 'import os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)' "$1"
}
anchor_pid=
probe_pid=
scope_pid=
trace_pid=
fifo=
cleanup() {
  if [[ -n "$trace_pid" ]]; then sudo -n pkill -INT -x bpftrace || true; fi
  if [[ -n "$scope_pid" ]]; then kill -9 "$scope_pid" 2>/dev/null || true; fi
  if [[ -n "$probe_pid" ]] && kill -0 "$probe_pid" 2>/dev/null; then
    ksm_set run 2 || true
    for _ in $(seq 1 600); do [[ "$(cat "$ksm/pages_sharing")" == 0 ]] && break; sleep 0.2; done
    kill "$probe_pid" 2>/dev/null || true
  fi
  ksm_set run 0 || true
  ksm_set pages_to_scan "$pages_before" || true
  ksm_set sleep_millisecs "$sleep_before" || true
  if [[ -n "$anchor_pid" ]]; then kill "$anchor_pid" 2>/dev/null || true; fi
  [[ -z "$fifo" ]] || rm -f "$fifo"
  if (( mounted )); then sudo -n umount "$memfs" || true; fi
  rmdir "$memfs" 2>/dev/null || true
  rm -f "$file" "$cold"
  rmdir "$work_dir" 2>/dev/null || true
}
trap cleanup EXIT
mkdir -p "$result_dir" "$work_dir"
head -c "$((size_mib << 20))" /dev/urandom >"$file"
head -c "$((size_mib << 20))" /dev/urandom >"$cold"
mkdir -p "$memfs"
sudo -n mount -t tmpfs -o "huge=always,size=$((size_mib + 512))m,uid=$(id -u),gid=$(id -g)" tmpfs "$memfs"
mounted=1
cp "$file" "$memfs/hot.bin"

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'gpu=%s\nrepetitions=%s\nsize_mib=%s\ncopies=%s\nreclaim_s=%s\nreclaim_step_mib=%s\nreclaim_total_mib=%s\n' \
    "$gpu" "$repetitions" "$size_mib" "$copies" "$reclaim_s" "$reclaim_step_mib" "$reclaim_total_mib"
  printf 'ksm_pages_to_scan=%s\nksm_sleep_millisecs=%s\n' "$ksm_pages" "$ksm_sleep_ms"
  printf 'file=%s\nfile_fs=%s\n' "$file" "$(stat -f -c %T "$work_dir")"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  uname -r
  nvidia-smi -L
  sha256sum "$probe"
} >"$result_dir/metadata.txt"

# The threads of the driver that serve GPU faults exist while a context does.
CUDA_VISIBLE_DEVICES="$gpu" "$anchor_probe" hold 1000000 >/dev/null 2>&1 &
anchor_pid=$!
for _ in $(seq 1 100); do grep -qs 'UVM GPU' /proc/[0-9]*/comm && break; sleep 0.1; done
grep -qs 'UVM GPU' /proc/[0-9]*/comm || { echo "no fault thread of the driver" >&2; exit 2; }

cat >"$result_dir/events.bt" <<'BT'
BEGIN { printf("ATTACHED\n"); }
kprobe:__mmu_notifier_invalidate_range_start { @n["notifier_invalidate"] = count(); }
kprobe:__mmu_notifier_clear_flush_young, kprobe:__mmu_notifier_clear_young { @n["notifier_young"] = count(); }
kprobe:arm_smmu_mm_arch_invalidate_secondary_tlbs { @n["smmu_invalidate"] = count(); }
kprobe:do_wp_page /comm == "mech_probe" || strncmp(comm, "UVM GPU", 7) == 0/ { @n["wp_fault"] = count(); }
kprobe:try_to_unmap_one { @n["unmap"] = count(); }
kprobe:folio_referenced_one { @n["referenced_scan"] = count(); }
kprobe:copy_page_range /comm == "mech_probe"/ { @n["fork_copy"] = count(); }
kprobe:try_to_merge_one_page { @n["ksm_merge"] = count(); }
kprobe:handle_mm_fault /comm == "mech_probe"/ { @n["cpu_fault"] = count(); }
kprobe:handle_mm_fault /strncmp(comm, "UVM GPU", 7) == 0/ { @n["gpu_fault"] = count(); }
END { print(@n); clear(@n); }
BT

wait_for() {  # word file pid seconds
  local waited=0
  until grep -q "^$1" "$2" 2>/dev/null; do
    kill -0 "$3" 2>/dev/null || return 1
    (( waited >= $4 * 10 )) && return 1
    sleep 0.1
    waited=$((waited + 1))
  done
}

trace_start() {
  : >"$result_dir/events.out"
  sudo -n bpftrace -q "$result_dir/events.bt" >"$result_dir/events.out" 2>&1 &
  trace_pid=$!
  wait_for ATTACHED "$result_dir/events.out" "$trace_pid" 60
}
trace_stop() {  # prints EVENTS k=v ...
  sudo -n pkill -INT -x bpftrace || true
  wait "$trace_pid" || true
  trace_pid=
  printf 'EVENTS'
  sed -nE 's/^@n\[([a-z_]+)\]: ([0-9]+)$/ \1=\2/p' "$result_dir/events.out" | tr -d '\n'
  printf '\n'
}

run_plain() {  # mech access [read]
  CUDA_VISIBLE_DEVICES="$gpu" "$probe" "$1" "$file" "$2" ${3:+"$3"}
}
run_fork() {  # access [read]
  CUDA_VISIBLE_DEVICES="$gpu" "$probe" fork "$size_mib" "$1" ${2:+"$2"}
}
run_ksm() {  # access
  local out status=0
  out=$(mktemp "$result_dir/out.XXXXXX")
  fifo=$(mktemp -u "$result_dir/fifo.XXXXXX")
  mkfifo "$fifo"
  ksm_set run 0
  ksm_set pages_to_scan "$ksm_pages"
  ksm_set sleep_millisecs "$ksm_sleep_ms"
  CUDA_VISIBLE_DEVICES="$gpu" "$probe" ksm "$size_mib" "$copies" "$1" <"$fifo" >"$out" 2>&1 &
  probe_pid=$!
  exec 8>"$fifo"
  wait_for READY "$out" "$probe_pid" 120 || status=1
  ksm_set run 1
  wait_for MERGED "$out" "$probe_pid" 960 || status=1
  ksm_set run 0
  echo >&8
  wait_for DONE "$out" "$probe_pid" 300 || status=1
  ksm_set run 2
  for _ in $(seq 1 600); do [[ "$(cat "$ksm/pages_sharing")" == 0 ]] && break; sleep 0.2; done
  ksm_set run 0
  echo >&8
  exec 8>&-
  wait "$probe_pid" || status=1
  probe_pid=
  rm -f "$fifo"
  fifo=
  grep -E '^(MERGE|KSMSTEP) ' "$out" || true
  rm -f "$out"
  return "$status"
}
memcg_mib() { awk '{ printf "%d", $1 / 1048576 }' "$1/memory.current"; }
run_reclaim() {  # access pin pressure
  local out unit cg status=0 hot=$file flag=none
  out=$(mktemp "$result_dir/out.XXXXXX")
  unit="stator-mech-$$-$RANDOM"
  [[ "$2" == memfs ]] && hot="$memfs/hot.bin"
  [[ "$2" == lock ]] && flag=lock
  drop_cache "$file"
  drop_cache "$cold"
  systemd-run --user --scope --quiet --unit="$unit" env CUDA_VISIBLE_DEVICES="$gpu" \
    "$probe" reclaim "$hot" "$cold" "$1" "$reclaim_s" "$flag" >"$out" 2>&1 &
  scope_pid=$!
  cg=
  for _ in $(seq 1 100); do
    cg=$(systemctl --user show -p ControlGroup --value "$unit.scope" 2>/dev/null || true)
    [[ -n "$cg" && -e "/sys/fs/cgroup$cg/memory.reclaim" ]] && break
    sleep 0.05
  done
  cg="/sys/fs/cgroup$cg"
  wait_for READY "$out" "$scope_pid" 120 || status=1
  local attempts=0 succeeded=0 asked=0 write_ms=0 write_max=0 t0 took before after=
  before=$(memcg_mib "$cg")
  while kill -0 "$scope_pid" 2>/dev/null; do
    if [[ "$3" == pressure ]] && (( asked < reclaim_total_mib )); then
      attempts=$((attempts + 1))
      t0=$(date +%s%N)
      if echo "${reclaim_step_mib}M" >"$cg/memory.reclaim" 2>/dev/null; then succeeded=$((succeeded + 1)); fi
      took=$(( ($(date +%s%N) - t0) / 1000000 ))
      write_ms=$((write_ms + took)); (( took > write_max )) && write_max=$took
      asked=$((asked + reclaim_step_mib))
      (( asked >= reclaim_total_mib )) && after=$(memcg_mib "$cg" 2>/dev/null || echo 0)
    fi
    sleep 1
  done
  [[ -n "$after" ]] || after=$before
  wait "$scope_pid" || status=1
  scope_pid=
  grep -E '^RECLAIM' "$out" | sed "s/\$/ pressure=$3 reclaim_attempts=$attempts reclaim_succeeded=$succeeded reclaim_write_ms=$write_ms reclaim_write_max_ms=$write_max memcg_before_mib=$before memcg_after_mib=$after/" || true
  rm -f "$out"
  return "$status"
}

run_case() {  # traced mech access [extra] [pressure]
  local traced=$1 mech=$2 access=$3 extra=${4:-} pressure=${5:-} status=0 body
  case "$mech" in
    private|shared) cat "$file" >/dev/null ;;
  esac
  if (( traced )); then trace_start || status=1; fi
  case "$mech" in
    private|shared) body=$(run_plain "$mech" "$access" "$extra" 2>&1) || status=1 ;;
    fork) body=$(run_fork "$access" "$extra" 2>&1) || status=1 ;;
    ksm) body=$(run_ksm "$access" 2>&1) || status=1 ;;
    reclaim) body=$(run_reclaim "$access" "$extra" "$pressure" 2>&1) || status=1 ;;
  esac
  {
    printf 'CASE mech=%s access=%s variant=%s traced=%s\n' "$mech" "$access" "${extra:-all}${pressure:+-$pressure}" "$traced"
    printf '%s\n' "$body"
    if (( traced )); then trace_stop; fi
    printf 'END_CASE status=%s\n' "$status"
  } >>"$raw_log"
}

cases=(
  "private cpu" "private gpu" "shared cpu" "shared gpu" "fork cpu" "fork gpu"
  "ksm cpu" "ksm gpu"
  "reclaim cpu none none" "reclaim gpu none none" "reclaim gpu lock none" "reclaim gpu memfs none"
  "reclaim cpu none pressure" "reclaim gpu none pressure" "reclaim gpu lock pressure"
  "reclaim gpu memfs pressure"
)
for run in $(seq 1 "$repetitions"); do
  printf 'BEGIN_RUN run=%s traced=0\n' "$run" >>"$raw_log"
  if (( run % 2 == 1 )); then order=("${cases[@]}"); else
    order=(); for c in "${cases[@]}"; do order=("$c" "${order[@]}"); done
  fi
  for c in "${order[@]}"; do
    # shellcheck disable=SC2086
    run_case 0 $c
  done
  printf 'END_RUN\n' >>"$raw_log"
done

printf 'BEGIN_RUN run=trace traced=1\n' >>"$raw_log"
for c in "private cpu read" "private gpu read" "fork cpu read" "fork gpu read" "${cases[@]}"; do
  # shellcheck disable=SC2086
  run_case 1 $c
done
printf 'END_RUN\n' >>"$raw_log"

python3 "$script_dir/summarize_mech_probe.py" "$raw_log" "$size_mib" >"$result_dir/mech_probe_summary.csv"
(
  cd "$script_dir"
  sha256sum mech_probe.cu run_mech_probe.sh summarize_mech_probe.py
) >"$result_dir/source_hashes.txt"
printf 'mech_probe_result_dir=%s\n' "$result_dir"
