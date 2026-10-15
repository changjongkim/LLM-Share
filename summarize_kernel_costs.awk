# Summarizes run_kernel_costs.sh: one row per scenario, one column per counter
# (the GPU faults of the driver thread and the tracepoints of tp_count.c).
function field(name,    i, pair) {
  for (i = 1; i <= NF; ++i) { split($i, pair, "="); if (pair[1] == name) return pair[2] }
  return ""
}
BEGIN { m = split("gpu_fault_pages fault_thread_ms faults-minor faults-major kmem:mm_page_alloc kmem:mm_page_free filemap:mm_filemap_add_to_page_cache filemap:mm_filemap_delete_from_page_cache vmscan:mm_vmscan_lru_shrink_inactive vmscan:mm_vmscan_memcg_reclaim_begin tlb:tlb_flush iommu:io_page_fault mmap_lock:mmap_lock_start_locking migrate:mm_migrate_pages ksm:ksm_merge_one_page pagemap:mm_lru_insertion", names, " ") }
/^SCENARIO / { s = field("name"); order[++n] = s; status[s] = field("status"); next }
/^COUNT / { value[s, field("name")] = field("value"); next }
END {
  printf "scenario,status"
  for (j = 1; j <= m; ++j) printf ",%s", names[j]
  printf "\n"
  for (i = 1; i <= n; ++i) {
    s = order[i]; printf "%s,%s", s, status[s]
    for (j = 1; j <= m; ++j) printf ",%s", ((s, names[j]) in value) ? value[s, names[j]] : 0
    printf "\n"
  }
}
