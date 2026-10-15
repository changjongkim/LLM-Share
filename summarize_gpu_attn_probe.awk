# Summarizes run_gpu_attn_probe.sh: one row per kind and row size, with the
# mean rate of each pattern over all rounds and its ratio to device memory.
function field(name,    i, pair) {
  for (i = 1; i <= NF; ++i) { split($i, pair, "="); if (pair[1] == name) return pair[2] }
  return ""
}
/^CASE / { key = field("kind") "," field("row_bytes"); if (!(key in cases)) order[++n] = key; cases[key]++; next }
/^ATTN / { rounds[key]++; h[key] += field("heads_gbs"); r[key] += field("rows_gbs"); x[key] += field("random_gbs"); next }
/^(ERROR|FAILED) / { failed[key]++; next }
END {
  print "kind,row_bytes,processes,rounds,failed,heads_gbs,rows_gbs,random_gbs,heads_vs_device,rows_vs_device,random_vs_device"
  for (i = 1; i <= n; ++i) {
    k = order[i]; split(k, part, ",")
    if (rounds[k]) { mh[k] = h[k] / rounds[k]; mr[k] = r[k] / rounds[k]; mx[k] = x[k] / rounds[k] }
  }
  for (i = 1; i <= n; ++i) {
    k = order[i]; split(k, part, ","); d = "device," part[2]
    printf "%s,%d,%d,%d,%.2f,%.2f,%.2f,%.4f,%.4f,%.4f\n", k, cases[k], rounds[k] + 0, failed[k] + 0,
      mh[k], mr[k], mx[k], (mh[d] ? mh[k] / mh[d] : 0), (mr[d] ? mr[k] / mr[d] : 0), (mx[d] ? mx[k] / mx[d] : 0)
  }
}
