# KSM on memory that the GPU reads and writes (run_ksm_probe.sh). One row per
# kind of process (nocuda, cuda): the cases, the merge rate and its time, the
# CPU time of ksmd, and for cuda the share of merged pages that a GPU read and
# a GPU write leave merged, with the time of the write per written page.
# Means over the cases with the half-width of the 95% confidence interval.
function field(name, i, pair) {
  for (i = 1; i <= NF; ++i) { split($i, pair, "="); if (pair[1] == name) return pair[2] }
  return ""
}
function t95(samples, df) {
  df = samples - 1
  if (df <= 1) return 12.706
  if (df == 2) return 4.303
  if (df == 3) return 3.182
  if (df == 4) return 2.776
  if (df == 5) return 2.571
  if (df == 6) return 2.447
  return 2.365
}
function add(key, metric, value) {
  n[key, metric]++; sum[key, metric] += value; sq[key, metric] += value * value
}
function mean(key, metric) { return n[key, metric] ? sum[key, metric] / n[key, metric] : 0 }
function ci(key, metric, m, v) {
  if (n[key, metric] < 2) return 0
  m = mean(key, metric)
  v = (sq[key, metric] - n[key, metric] * m * m) / (n[key, metric] - 1)
  return v > 0 ? t95(n[key, metric]) * sqrt(v / n[key, metric]) : 0
}

/^CASE / {
  key = field("kind"); cases[key]++
  if (field("status") != 0) failed[key]++
  add(key, "ksmd_ms", field("ksmd_ms"))
  next
}
/^MERGE / {
  if (field("merged_pages") >= 0.99 * field("of")) merged_all[key]++
  add(key, "merge_s", field("merge_s")); add(key, "rate", field("pages_per_s"))
  next
}
/^GPU / {
  before = field("merged_before")
  if (before > 0) {
    add(key, "kept_read", field("merged_after_gpu_read") / before)
    add(key, "kept_cpu_read", field("merged_after_cpu_gpu_read") / before)
    # The share of the written pages that are still merged after the write.
    left = field("merged_after_gpu_write") - (field("merged_after_cpu_gpu_read") - field("written_pages"))
    add(key, "kept_write", left / field("written_pages"))
  }
  add(key, "write_us_per_page", field("gpu_write_s") * 1e6 / field("written_pages"))
  if (field("right") == 1 && field("right2") == 1) right[key]++
  next
}

END {
  print "kind,cases,failed,merged_all,right,merge_s,merge_s_ci95,pages_per_s,pages_per_s_ci95," \
        "ksmd_ms,ksmd_ms_ci95,kept_after_gpu_read,kept_after_cpu_gpu_read," \
        "written_kept_after_gpu_write,gpu_write_us_per_page,gpu_write_us_ci95"
  split("nocuda cuda", kinds, " ")
  for (k = 1; k <= 2; ++k) {
    key = kinds[k]
    if (!(key in cases)) continue
    printf "%s,%d,%d,%d,%d,%.2f,%.2f,%.0f,%.0f,%.0f,%.0f,%.4f,%.4f,%.4f,%.1f,%.1f\n", key, \
           cases[key], failed[key] + 0, merged_all[key] + 0, right[key] + 0, \
           mean(key, "merge_s"), ci(key, "merge_s"), mean(key, "rate"), ci(key, "rate"), \
           mean(key, "ksmd_ms"), ci(key, "ksmd_ms"), mean(key, "kept_read"), \
           mean(key, "kept_cpu_read"), mean(key, "kept_write"), mean(key, "write_us_per_page"), \
           ci(key, "write_us_per_page")
  }
}
