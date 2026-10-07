# Putting a model file on 2 MiB pages. One row per file and method: means
# over the repetitions, the half-width of the 95% confidence interval of the
# time, and the paired ratio of the time of cp to that of the method.
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

/^PUBLISH / {
  run = field("run") + 0; if (run > runs) runs = run
  file = field("file"); method = field("method"); key = file SUBSEP method
  if (!(file in seen)) { seen[file] = 1; order[++files] = file; bytes[file] = field("bytes") }
  cases[key]++
  if (field("exit") != 0) { failed[key]++; next }
  seconds[key, run] = field("seconds")
  total[key] += field("seconds"); square[key] += field("seconds") * field("seconds")
  rate[key] += field("gib_per_s")
  cache[key] += field("page_cache_left_mib")
  drop[key] += field("free_drop_mib")
  same[key] += field("identical")
  next
}

END {
  print "file,gib,method,runs,failed,seconds,seconds_ci95,gib_per_s,speedup_vs_cp," \
        "page_cache_left_mib,page_cache_left_share,free_drop_mib,identical"
  split("cp dd publish", methods, " ")
  for (f = 1; f <= files; ++f) for (m = 1; m <= 3; ++m) {
    key = order[f] SUBSEP methods[m]
    if (!(key in cases)) continue
    n = cases[key] - failed[key]
    if (n < 1) n = 1
    variance = n > 1 ? (square[key] - total[key] * total[key] / n) / (n - 1) : 0
    if (variance < 0) variance = 0
    ratio = 0; pairs = 0
    for (r = 1; r <= runs; ++r) {
      base = order[f] SUBSEP "cp" SUBSEP r
      if ((base in seconds) && ((key, r) in seconds) && seconds[key, r] > 0) {
        ratio += log(seconds[base] / seconds[key, r]); pairs++
      }
    }
    mib = bytes[order[f]] / 1048576
    printf "%s,%.1f,%s,%d,%d,%.3f,%.3f,%.2f,%.3f,%.0f,%.3f,%.0f,%d\n", order[f], mib / 1024, \
           methods[m], cases[key], failed[key] + 0, total[key] / n, \
           (n > 1 ? t95(n) * sqrt(variance / n) : 0), rate[key] / n, \
           (pairs ? exp(ratio / pairs) : 0), cache[key] / n, cache[key] / n / mib, \
           drop[key] / n, same[key] + 0
  }
}
