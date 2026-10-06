# The read path by the kind of memory. One row per MIG instance, buffer size
# and kind of memory: means over the repetitions, and the bandwidth of the
# full pass relative to device memory of the same repetition with a 95%
# confidence interval (Student t).
function field(name, i, pair) {
  for (i = 1; i <= NF; ++i) { split($i, pair, "="); if (pair[1] == name) return pair[2] }
  return ""
}
function t95(n) {
  split("12.706 4.303 3.182 2.776 2.571 2.447 2.365 2.306 2.262", table, " ")
  return n < 2 ? 0 : (n - 1 <= 9 ? table[n - 1] : 2.0)
}

/^FAILED / { failed[field("instance"), field("size_mib"), field("kind")]++; next }
/^PROBE / {
  cell = field("instance") SUBSEP field("size_mib")
  key = cell SUBSEP field("kind")
  cells[cell] = 1
  runs[key]++
  full[key] += field("full_gib_per_s")
  full_ms[key] += field("full_ms")
  page_ms[key] += field("page_ms")
  per_page[key] += field("ns_per_page")
  block_ms[key] += field("block_ms")
  value[key, field("run")] = field("full_gib_per_s")
  if (field("run") + 0 > last) last = field("run") + 0
  next
}

END {
  print "instance,size_mib,kind,runs,failed,full_gib_per_s,full_ms,page_ms," \
        "ns_per_page,block_ms,full_vs_device,lower_ci95,upper_ci95"
  split("12sm 6sm", instances, " ")
  split("device host_huge host_small", kinds, " ")
  for (i = 1; i <= 2; ++i) for (size = 2; size <= 65536; size += 2) {
    cell = instances[i] SUBSEP size
    if (!(cell in cells)) continue
    for (k = 1; k <= 3; ++k) {
      key = cell SUBSEP kinds[k]
      if (!(key in runs)) continue
      n = runs[key]
      pairs = 0; sum = 0; sumsq = 0
      for (r = 1; r <= last; ++r) {
        if (!((key, r) in value) || !((cell, "device", r) in value)) continue
        ratio = value[key, r] / value[cell, "device", r]
        pairs++; sum += ratio; sumsq += ratio * ratio
      }
      mean = pairs > 0 ? sum / pairs : 0
      sd = pairs > 1 ? sqrt((sumsq - pairs * mean * mean) / (pairs - 1)) : 0
      if (sd != sd) sd = 0
      half = pairs > 1 ? t95(pairs) * sd / sqrt(pairs) : 0
      printf "%s,%d,%s,%d,%d,%.2f,%.3f,%.3f,%.2f,%.3f,%.4f,%.4f,%.4f\n", \
             instances[i], size, kinds[k], n, failed[key] + 0, full[key] / n, \
             full_ms[key] / n, page_ms[key] / n, per_page[key] / n, \
             block_ms[key] / n, mean, mean - half, mean + half
    }
  }
}
