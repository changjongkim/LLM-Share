#!/usr/bin/env bash
# K3 of PLAN-20261015: the page cache of the model file locked with mlock, under
# the reclaim of the kernel, next to the page cache without the lock and to the
# tmpfs with 2 MiB pages of STATOR. This is run_engine_reclaim.sh with one more
# mode:
#
#   mlock      the weights are read in place from the page cache of the model
#              file on ext4, which every agent locks (llama.cpp --load-mode mmap+mlock), as a
#              system that shares weights through the page cache and keeps them
#              resident would
#
# The other modes and the measurements are those of run_engine_reclaim.sh.
#
# Kill gates, fixed before the campaign:
#   R1  no process fails, and every agent writes the text of the same agent of
#       the same mode without pressure in the same repetition;
#   M1  with pressure, the GPU takes fewer page faults with mlock than with
#       pagecache.
# Reported, not gated: the speed under pressure, the reclaim requests, the
# memory that the lock charges.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mig=${PRODUCER_MIG:-MIG-fafc828a-2a0d-5988-a009-030a36930090}
repetitions=${REPETITIONS:-6}
read -r -a modes_forward <<<"${MODES:-pagecache mlock memfs}"
read -r -a pressures <<<"${PRESSURES:-none reclaim}"
agents=${AGENTS:-4}
n_gen=${N_GEN:-512}
reclaim=${RECLAIM:-512M}
reclaim_every_s=${RECLAIM_EVERY_S:-1}
start_s=${START_S:-10}
limit_s=${CASE_LIMIT_S:-1200}
model=${MODEL:-"$script_dir/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
bin_dir=${BIN_DIR:-"$script_dir/llama.cpp/build/bin"}
probe=${PROBE:-"$script_dir/linux_share_probe"}
mount_root=${MOUNT_ROOT:-"$script_dir/mounts"}
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-reclaim-mlock"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
if [[ ! -x "$bin_dir/llama-completion" || ! -x "$probe" || ! -r "$model" ]]; then
  echo "missing llama-completion, probe or model" >&2
  exit 2
fi
if ! sudo -n true 2>/dev/null || ! systemd-run --user --scope --quiet true; then
  echo "passwordless sudo and a systemd user manager are required" >&2
  exit 2
fi
if [[ "$(stat -f -c %T "$model")" == tmpfs ]]; then
  echo "the model must be on a disk file system for the pagecache mode" >&2
  exit 2
fi
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi

fault_counters() {
  local pages=0 ticks=0 stat line rest
  for stat in /proc/[0-9]*/stat; do
    { read -r line <"$stat"; } 2>/dev/null || continue
    [[ "$line" == *"(UVM GPU"*" BH)"* ]] || continue
    rest=${line##*) }
    # shellcheck disable=SC2086
    set -- $rest
    pages=$(( pages + $8 + ${10} ))
    ticks=$(( ticks + ${12} + ${13} ))
  done
  printf '%s %s' "$pages" "$ticks"
}
vmstat_of() { awk -v k="$1" '$1 == k { print $2 }' /proc/vmstat; }
stat_of() { awk -v k="$2" '$1 == k { print $2 }' "$1/memory.stat" 2>/dev/null || printf '0'; }
drop_cache() {  # file: drops its clean, unmapped pages from the page cache
  python3 -c 'import os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)' "$1"
}
tick_ms=$(( 1000 / $(getconf CLK_TCK) ))

memfs_dir="$mount_root/reclaim_memfs"
mounted=0
work=
anchor_pid=
scope_pid=
cleanup() {
  if [[ -n "$scope_pid" ]]; then kill -9 "$scope_pid" 2>/dev/null || true; fi
  pkill -9 -f "^$bin_dir/llama-completion" 2>/dev/null || true
  if [[ -n "$anchor_pid" ]]; then kill "$anchor_pid" 2>/dev/null || true; fi
  if (( mounted )); then sudo -n umount "$memfs_dir" || true; fi
  rmdir "$memfs_dir" "$mount_root" 2>/dev/null || true
  if [[ -n "$work" ]]; then rm -rf "$work"; fi
}
trap cleanup EXIT
mkdir -p "$memfs_dir" "$result_dir"
sudo -n mount -t tmpfs -o "huge=always,size=8192m,uid=$(id -u),gid=$(id -g)" tmpfs "$memfs_dir"
mounted=1
work=$(mktemp -d "$result_dir/work.XXXXXX")

CUDA_VISIBLE_DEVICES="$mig" "$probe" hold 1000000 >"$work/anchor" 2>&1 &
anchor_pid=$!
for _ in $(seq 1 100); do grep -qs 'UVM GPU' /proc/[0-9]*/comm && break; sleep 0.1; done
if ! grep -qs 'UVM GPU' /proc/[0-9]*/comm; then
  echo "no bottom-half thread of the driver found" >&2
  exit 2
fi

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'mig=%s\nrepetitions=%s\nmodes=%s\npressures=%s\n' "$mig" "$repetitions" \
    "${modes_forward[*]}" "${pressures[*]}"
  printf 'agents=%s\nn_gen=%s\nreclaim=%s\nreclaim_every_s=%s\nstart_s=%s\n' \
    "$agents" "$n_gen" "$reclaim" "$reclaim_every_s" "$start_s"
  printf 'model=%s\nmodel_fs=%s\n' "$(basename "$model")" "$(stat -f -c %T "$model")"
  printf 'engine_commit=%s\n' "$(git -C "$script_dir/llama.cpp" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$bin_dir/llama-completion" "$probe"
} >"$result_dir/metadata.txt"

prompt="Write a detailed technical essay about how operating systems manage memory."

run_case() {  # mode pressure
  local mode=$1 pressure=$2 unit path extra=() cg
  unit="stator-reclaim-$$-$RANDOM"
  path=$model
  [[ "$mode" == memfs ]] && path="$memfs_dir/model.gguf"
  [[ "$mode" == device ]] || extra=(GGML_CUDA_HOST_PTR=1)
  local lock_flag=
  [[ "$mode" == mlock ]] && lock_flag="--load-mode mmap+mlock"
  rm -f "$memfs_dir/model.gguf"
  drop_cache "$model"
  rm -rf "$work/out"; mkdir -p "$work/out"
  # The agents and, for memfs, the copy of the model run in one scope.
  cat >"$work/scope.sh" <<SCOPE
#!/usr/bin/env bash
if [[ "$mode" == memfs ]]; then
  dd if="$model" of="$path" bs=64M iflag=direct status=none || exit 9
fi
for i in \$(seq 0 $((agents - 1))); do
  env CUDA_VISIBLE_DEVICES="$mig" ${extra[*]} "$bin_dir/llama-completion" -m "$path" -ngl 99 \
    -n "$n_gen" -c 4096 --temp 0 --seed 1 -no-cnv --ignore-eos ${lock_flag} -p "$prompt" \
    >"$work/out/agent.\$i" 2>"$work/out/agent.\$i.err" &
done
fail=0
for job in \$(jobs -p); do wait "\$job" || fail=\$((fail + 1)); done
exit "\$fail"
SCOPE
  chmod +x "$work/scope.sh"
  local faults_before faults_after major_before major_after started now attempts=0 succeeded=0
  local code=0 timed_out=0 refault_before refault_after file_before file_after
  major_before=$(vmstat_of pgmajfault)
  systemd-run --user --scope --quiet --unit="$unit" "$work/scope.sh" &
  scope_pid=$!
  cg=
  for _ in $(seq 1 100); do
    cg=$(systemctl --user show -p ControlGroup --value "$unit.scope" 2>/dev/null || true)
    [[ -n "$cg" && -e "/sys/fs/cgroup$cg/memory.reclaim" ]] && break
    sleep 0.05
  done
  cg="/sys/fs/cgroup$cg"
  started=$(date +%s)
  sleep "$start_s"
  faults_before=$(fault_counters)
  refault_before=$(stat_of "$cg" workingset_refault_file)
  file_before=$(stat_of "$cg" file)
  while kill -0 "$scope_pid" 2>/dev/null; do
    now=$(date +%s)
    if (( now - started > limit_s )); then
      timed_out=1
      pkill -9 -f "^$bin_dir/llama-completion" 2>/dev/null || true
      break
    fi
    if [[ "$pressure" == reclaim ]]; then
      attempts=$((attempts + 1))
      if echo "$reclaim" >"$cg/memory.reclaim" 2>/dev/null; then succeeded=$((succeeded + 1)); fi
      refault_after=$(stat_of "$cg" workingset_refault_file)
      file_after=$(stat_of "$cg" file)
    fi
    sleep "$reclaim_every_s"
  done
  refault_after=${refault_after:-$(stat_of "$cg" workingset_refault_file)}
  file_after=${file_after:-$(stat_of "$cg" file)}
  wait "$scope_pid" || code=$?
  scope_pid=
  faults_after=$(fault_counters)
  major_after=$(vmstat_of pgmajfault)
  {
    printf 'CASE mode=%s pressure=%s agents=%s\n' "$mode" "$pressure" "$agents"
    local i tps text
    for i in $(seq 0 $((agents - 1))); do
      tps=$({ grep ' eval time' "$work/out/agent.$i.err" || true; } | grep -v 'prompt eval' |
        sed -E 's/.*, *([0-9.]+) tokens per second.*/\1/' | tail -n 1)
      text=$(sha256sum "$work/out/agent.$i" | cut -c1-16)
      printf 'AGENT index=%s tps=%s text=%s bytes=%s\n' "$i" "${tps:-0}" "$text" \
        "$(stat -c %s "$work/out/agent.$i")"
    done
    read -r pages_before ticks_before <<<"$faults_before"
    read -r pages_after ticks_after <<<"$faults_after"
    printf 'KERNEL gpu_fault_pages=%s fault_thread_ms=%s major_faults=%s refault_file=%s file_before_mib=%s file_last_mib=%s\n' \
      "$(( pages_after - pages_before ))" "$(( (ticks_after - ticks_before) * tick_ms ))" \
      "$(( major_after - major_before ))" "$(( ${refault_after:-0} - ${refault_before:-0} ))" \
      "$(( ${file_before:-0} / 1048576 ))" "$(( ${file_after:-0} / 1048576 ))"
    printf 'RECLAIM attempts=%s succeeded=%s\n' "$attempts" "$succeeded"
    printf 'END_CASE exit=%s timed_out=%s\n' "$code" "$timed_out"
  } >>"$raw_log"
  rm -f "$memfs_dir/model.gguf"
}

modes_backward=()
for mode in "${modes_forward[@]}"; do modes_backward=("$mode" "${modes_backward[@]}"); done
for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then modes=("${modes_forward[@]}"); else modes=("${modes_backward[@]}"); fi
  printf 'BEGIN_RECLAIM run=%s\n' "$run" >>"$raw_log"
  for mode in "${modes[@]}"; do
    for pressure in "${pressures[@]}"; do run_case "$mode" "$pressure"; done
  done
  printf 'END_RECLAIM\n' >>"$raw_log"
done

awk -v order="${modes_forward[*]}" -f "$script_dir/summarize_engine_reclaim.awk" "$raw_log" \
  >"$result_dir/engine_reclaim_summary.csv"
(
  cd "$script_dir"
  sha256sum linux_share_probe.cu summarize_engine_reclaim.awk run_engine_reclaim_mlock.sh
) >"$result_dir/source_hashes.txt"
printf 'engine_reclaim_result_dir=%s\n' "$result_dir"
