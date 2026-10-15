#!/usr/bin/env bash
# K9 of PLAN-20261015: the kernel paths of each sharing mechanism, counted with
# tracepoints (tp_count.c, counting events of perf_event_open over the whole
# system) around one run of an existing runner per scenario, together with the
# page faults that the driver thread takes on behalf of the GPU. The kernel of
# this device has no kprobes and no function tracer, so the invalidations of the
# SMMU are not counted here (their cost per entry comes from the hostmm campaign).
#
# Scenarios (SCENARIOS):
#   kv:MODE        run_engine_kvlinux.sh, 8 children on the 16,321-token prefix,
#                  2 repetitions, one mode of copy shared_rw cow_plain ro_plain extent
#   reclaim:MODE   run_engine_reclaim.sh under reclaim, 2 repetitions, pagecache or memfs
#   fork:KIND      linux_share_probe fork 1024 KIND (gpu or cpu), 2 times
#   idle           nothing for 60 s (the background of the counters)
#
# Kill gates, fixed before the campaign:
#   C1  every scenario completes and the script prints its counters;
#   C2  in kv:extent the driver thread takes fewer GPU faults than in kv:ro_plain
#       and kv:cow_plain, and kv:cow_plain allocates more pages than kv:extent.
# Reported, not gated: every counter per scenario.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
read -r -a scenarios <<<"${SCENARIOS:-idle kv:copy kv:shared_rw kv:cow_plain kv:ro_plain kv:extent reclaim:pagecache reclaim:memfs fork:gpu fork:cpu}"
result_root=${RESULT_ROOT:-"$script_dir/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-kernel-costs"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"
sudo -n true 2>/dev/null || { echo "passwordless sudo is required" >&2; exit 2; }
counter=${COUNTER:-"$script_dir/tp_count"}
[[ -x "$counter" ]] || { echo "missing tp_count; run make tp_count" >&2; exit 2; }
events=(kmem:mm_page_alloc kmem:mm_page_free filemap:mm_filemap_add_to_page_cache
        filemap:mm_filemap_delete_from_page_cache vmscan:mm_vmscan_lru_shrink_inactive
        vmscan:mm_vmscan_memcg_reclaim_begin tlb:tlb_flush iommu:io_page_fault
        mmap_lock:mmap_lock_start_locking migrate:mm_migrate_pages ksm:ksm_merge_one_page
        pagemap:mm_lru_insertion faults-minor faults-major)
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
tick_ms=$(( 1000 / $(getconf CLK_TCK) ))
[[ -e "$result_dir" ]] && { echo "result directory already exists: $result_dir" >&2; exit 2; }
mkdir -p "$result_dir"
tracer=
anchor_pid=
cleanup() {
  if [[ -n "$tracer" ]]; then sudo -n pkill -INT -x tp_count 2>/dev/null || true; fi
  if [[ -n "$anchor_pid" ]]; then kill "$anchor_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
{
  printf 'timestamp=%s\nscenarios=%s\n' "$(date --iso-8601=seconds)" "${scenarios[*]}"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  uname -r
  printf 'events=%s\n' "${events[*]}"
  sha256sum "$counter"
} >"$result_dir/metadata.txt"

# The driver keeps the threads that serve GPU faults, and their counters, while
# a process holds a context; this one holds it for the whole campaign.
CUDA_VISIBLE_DEVICES=${PRODUCER_MIG:-GPU-a7c66ad2-6dbb-0ab8-c1a2-37ba6dba3600} \
  "$script_dir/linux_share_probe" hold 1000000 >/dev/null 2>&1 &
anchor_pid=$!
sleep 3
for scenario in "${scenarios[@]}"; do
  kind=${scenario%%:*}
  arg=${scenario#*:}
  out="$result_dir/trace.${scenario//:/_}"
  sudo -n "$counter" "${events[@]}" >"$out" 2>&1 &
  tracer=$!
  until grep -q TP_READY "$out" 2>/dev/null; do sleep 0.2; done
  before=$(fault_counters)
  status=0
  case "$kind" in
    idle) sleep 60 ;;
    kv) env MODES="$arg" CELLS="same:8:320" REPETITIONS=2 RESULT_TAG="$result_tag-sub-kv-$arg" \
          "$script_dir/run_engine_kvlinux.sh" >"$result_dir/sub.${scenario//:/_}.log" 2>&1 || status=$? ;;
    reclaim) env MODES="$arg" PRESSURES=reclaim REPETITIONS=2 RESULT_TAG="$result_tag-sub-reclaim-$arg" \
          "$script_dir/run_engine_reclaim.sh" >"$result_dir/sub.${scenario//:/_}.log" 2>&1 || status=$? ;;
    fork) for _ in 1 2; do "$script_dir/linux_share_probe" fork 1024 "$arg" >>"$result_dir/sub.${scenario//:/_}.log" 2>&1 || status=$?; done ;;
    *) echo "unknown scenario $scenario" >&2; exit 2 ;;
  esac
  after=$(fault_counters)
  sudo -n pkill -INT -x tp_count 2>/dev/null || true
  wait "$tracer" 2>/dev/null || true
  tracer=
  read -r -a b <<<"$before"
  read -r -a a <<<"$after"
  printf 'SCENARIO name=%s status=%s\n' "$scenario" "$status" >>"$raw_log"
  printf 'COUNT name=gpu_fault_pages value=%s\nCOUNT name=fault_thread_ms value=%s\n' \
    "$(( a[0] - b[0] ))" "$(( (a[1] - b[1]) * tick_ms ))" >>"$raw_log"
  awk '/^TP / { split($2, n, "="); split($3, v, "="); printf "COUNT name=%s value=%s\n", n[2], v[2] }' "$out" >>"$raw_log"
  printf 'END_SCENARIO\n' >>"$raw_log"
done
awk -f "$script_dir/summarize_kernel_costs.awk" "$raw_log" >"$result_dir/kernel_costs_summary.csv"
(
  cd "$script_dir"
  sha256sum tp_count.c summarize_kernel_costs.awk run_kernel_costs.sh run_engine_kvlinux.sh run_engine_reclaim.sh linux_share_probe.cu
) >"$result_dir/source_hashes.txt"
printf 'kernel_costs_result_dir=%s\n' "$result_dir"
