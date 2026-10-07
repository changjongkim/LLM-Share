# Scattered reads and writes by the kind of memory. One row per number of
# SMs, buffer size, stride and kind: the mean rate of each kernel and its
# paired ratio to device memory in the same repetition (geometric mean and
# 95% confidence interval of the log-ratio).
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
function ratio(series, key, base, r, n, total, sq, value, average, variance, margin) {
  n = 0; total = 0; sq = 0
  for (r = 1; r <= runs; ++r) {
    if (!(series[key, r] > 0 && series[base, r] > 0)) continue
    value = log(series[key, r] / series[base, r])
    n++; total += value; sq += value * value
  }
  if (n == 0) { ratio_mean = ratio_low = ratio_high = 0; return }
  average = total / n
  variance = n > 1 ? (sq - total * total / n) / (n - 1) : 0
  if (variance < 0) variance = 0
  margin = n > 1 ? t95(n) * sqrt(variance / n) : 0
  ratio_mean = exp(average); ratio_low = exp(average - margin); ratio_high = exp(average + margin)
}

/^PROBE / {
  run = field("run") + 0; if (run > runs) runs = run
  if (field("kind") == "") { failed++; next }
  cell = field("sms") SUBSEP field("size_mib") SUBSEP field("stride_kib")
  if (!(cell in seen)) { seen[cell] = 1; order[++cells] = cell }
  key = cell SUBSEP field("kind")
  count[key]++
  scan[key, run] = field("scan_mwords_s"); scan_sum[key] += field("scan_mwords_s")
  gather[key, run] = field("gather_mwords_s"); gather_sum[key] += field("gather_mwords_s")
  scatter[key, run] = field("scatter_mwords_s"); scatter_sum[key] += field("scatter_mwords_s")
  next
}

END {
  print "sms,size_mib,stride_kib,kind,runs,scan_mwords_s,gather_mwords_s," \
        "scatter_mwords_s,scan_vs_device,gather_vs_device,gather_low,gather_high," \
        "scatter_vs_device,scatter_low,scatter_high,failed_probes"
  split("device host_huge host_small", kinds, " ")
  for (c = 1; c <= cells; ++c) for (k = 1; k <= 3; ++k) {
    key = order[c] SUBSEP kinds[k]
    if (!(key in count)) continue
    base = order[c] SUBSEP "device"
    split(order[c], name, SUBSEP)
    n = count[key]
    printf "%s,%s,%s,%s,%d,%.1f,%.1f,%.1f,", name[1], name[2], name[3], kinds[k], n, \
           scan_sum[key] / n, gather_sum[key] / n, scatter_sum[key] / n
    ratio(scan, key, base); printf "%.4f,", ratio_mean
    ratio(gather, key, base); printf "%.4f,%.4f,%.4f,", ratio_mean, ratio_low, ratio_high
    ratio(scatter, key, base); printf "%.4f,%.4f,%.4f,%d\n", ratio_mean, ratio_low, ratio_high, failed + 0
  }
}
