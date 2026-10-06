# CUDA VMM attach cost by allocation granule. One row per granule: means over
# successful repetitions, with all failures and incorrect words counted.
function field(name, i, pair) {
  for (i = 1; i <= NF; ++i) {
    split($i, pair, "=")
    if (pair[1] == name) return pair[2]
  }
  return ""
}

/^FAILED / {
  failed[field("granule_mib") + 0]++
  seen[field("granule_mib") + 0] = 1
  next
}
/^PROBE / {
  g = field("granule_mib") + 0
  seen[g] = 1
  runs[g]++
  handles[g] = field("handles")
  export_ms[g] += field("export_ms")
  attach[g] += field("attach_ms")
  per[g] += field("attach_us_per_handle")
  wrong[g] += field("wrong")
  next
}

END {
  print "granule_mib,handles,runs,failed,export_ms,attach_ms,attach_us_per_handle,wrong_words"
  for (g = 2; g <= 65536; g += 2) {
    if (!(g in seen)) continue
    n = runs[g] > 0 ? runs[g] : 1
    printf "%d,%d,%d,%d,%.3f,%.3f,%.1f,%d\n", g, handles[g], runs[g] + 0, \
           failed[g] + 0, export_ms[g] / n, attach[g] / n, per[g] / n, wrong[g] + 0
  }
}
