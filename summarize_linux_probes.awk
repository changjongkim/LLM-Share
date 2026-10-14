# The kernel's mechanisms for sharing memory, on memory that the GPU reads
# (run_linux_probes.sh). One row per probe: fork with a GPU read or a CPU read
# first, fork after CUDA initialization, and registration of a shared
# read-only mapping in the page cache or on a memory file system. Means over
# the cases with the half-width of the 95% confidence interval.
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
  if (df <= 10) return 2.228
  if (df <= 20) return 2.086
  return 2.042
}
function add(key, metric, value) {
  n[key, metric]++; sum[key, metric] += value; sq[key, metric] += value * value
  if (!(key in seen)) { seen[key] = 1; order[++keys] = key }
}
function mean(key, metric) { return n[key, metric] ? sum[key, metric] / n[key, metric] : 0 }
function ci(key, metric, m, v) {
  if (n[key, metric] < 2) return 0
  m = mean(key, metric)
  v = (sq[key, metric] - n[key, metric] * m * m) / (n[key, metric] - 1)
  return v > 0 ? t95(n[key, metric]) * sqrt(v / n[key, metric]) : 0
}

/^FORK mode=/ {
  key = "fork_" field("mode")
  cases[key]++
  if (field("failed") == 1) { failed[key]++; next }
  if (field("right") == 1) right[key]++
  add(key, "size_mib", field("size_mib"))
  add(key, "growth_mib", field("child_private_after_mib") - field("child_private_before_mib"))
  add(key, "gpu_read_s", field("gpu_read_s"))
  add(key, "cpu_read_s", field("cpu_read_s"))
  next
}
/^FORK_AFTER_INIT / {
  key = "fork_after_init"
  if (field("failed") == 1) { cases[key]++; failed[key]++; next }
  # The child's line; its error string starts the value of child_malloc.
  if (field("child_malloc") != "") {
    cases[key]++
    if (field("child_malloc") != "no") refused[key]++
    if (!(key in seen)) { seen[key] = 1; order[++keys] = key }
  }
  next
}
/^REGISTER / {
  key = "register_" field("where")
  cases[key]++
  if (field("failed") == 1) { failed[key]++; next }
  if (field("register") == "cudaSuccess") registered[key]++
  if (field("right") == 1) right[key]++
  add(key, "size_mib", field("size_mib"))
  add(key, "register_s", field("register_s"))
  add(key, "gpu_read_s", field("gpu_read_s"))
  add(key, "unregister_s", field("unregister_s"))
  add(key, "locked_mib", field("locked_mib"))
  add(key, "pinned_mib", field("pinned_mib"))
  next
}

END {
  print "probe,cases,failed,right,registered,child_cuda_refused,size_mib,child_growth_mib," \
        "child_growth_ci95,gpu_read_s,gpu_read_ci95,cpu_read_s,register_s,register_ci95," \
        "unregister_s,locked_mib,pinned_mib"
  for (i = 1; i <= keys; ++i) {
    key = order[i]
    printf "%s,%d,%d,%d,%d,%d,%.0f,%.1f,%.1f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.1f,%.1f\n", key, \
           cases[key], failed[key] + 0, right[key] + 0, registered[key] + 0, refused[key] + 0, \
           mean(key, "size_mib"), mean(key, "growth_mib"), ci(key, "growth_mib"), \
           mean(key, "gpu_read_s"), ci(key, "gpu_read_s"), mean(key, "cpu_read_s"), \
           mean(key, "register_s"), ci(key, "register_s"), mean(key, "unregister_s"), \
           mean(key, "locked_mib"), mean(key, "pinned_mib")
  }
}
