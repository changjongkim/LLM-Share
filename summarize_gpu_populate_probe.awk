# Summarizes run_gpu_populate_probe.sh: one row per mode, file system and step.
function field(name,    i, pair) {
  for (i = 1; i <= NF; ++i) { split($i, pair, "="); if (pair[1] == name) return pair[2] }
  return ""
}
/^CASE / { key = field("mode") "," field("fs") "," field("step"); if (!(key in runs)) order[++n] = key; runs[key]++; next }
/^PREPARE / {
  if (field("failed") != "") { failed[key]++; next }
  prep[key] += field("prepare_ms"); gpu[key] += field("gpu_ms"); good[key] += field("sum_ok"); done[key]++; next
}
/^ERROR / { failed[key]++; next }
/^KERNEL / { faults[key] += field("gpu_fault_pages"); fms[key] += field("fault_thread_ms"); next }
END {
  print "mode,fs,step,runs,failed,checked,prepare_ms,gpu_ms,gpu_fault_pages,fault_thread_ms"
  for (i = 1; i <= n; ++i) {
    k = order[i]
    printf "%s,%d,%d,%d,%.2f,%.2f,%.0f,%.0f\n", k, runs[k], failed[k] + 0, good[k] + 0,
      done[k] ? prep[k] / done[k] : 0, done[k] ? gpu[k] / done[k] : 0, faults[k] / runs[k], fms[k] / runs[k]
  }
}
