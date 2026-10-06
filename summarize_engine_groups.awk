# One row per weights mode and sequences per server: the generation rate of
# the two servers together, and the memory they hold.
# A ROW line is "ROW instance=x | PP | TG | B | N_KV | T_PP | S_PP | T_TG | S_TG | T | S |".
function field(name, i, pair) {
  for (i = 1; i <= NF; ++i) { split($i, pair, "="); if (pair[1] == name) return pair[2] }
  return ""
}
/^BEGIN_ENGINE_GROUPS / { key = field("mode") "," field("sequences"); total = 0; rows = 0; next }
/^ROW / {
  split($0, cell, "|")
  total += cell[9]
  rows++
  if ($2 == "instance=a") first[key] += cell[9]; else second[key] += cell[9]
  next
}
/^MEMORY / {
  drop[key] += field("mem_available_drop_mib")
  mapped[key] += field("model_mapped_mib")
  next
}
/^END_ENGINE_GROUPS / {
  count[key]++
  if (field("failed_servers") != 0 || rows != 2) bad[key]++
  sum[key] += total
  sumsq[key] += total * total
  next
}
END {
  print "mode,sequences_per_server,runs,failed_runs,generation_tps_total," \
        "generation_tps_sd,instance_a_tps,instance_b_tps,device_memory_mib," \
        "model_mapped_mib,memory_total_mib"
  split("4 8 16 32", sizes, " ")
  split("copy inplace", modes, " ")
  for (s = 1; s <= 4; ++s) {
    for (m = 1; m <= 2; ++m) {
      key = modes[m] "," sizes[s]
      if (!(key in count)) continue
      n = count[key]
      mean = sum[key] / n
      variance = 0
      if (n > 1) variance = (sumsq[key] - sum[key] * sum[key] / n) / (n - 1)
      if (variance < 0) variance = 0
      printf "%s,%d,%d,%.2f,%.2f,%.2f,%.2f,%.0f,%.0f,%.0f\n", key, n, \
             bad[key] + 0, mean, sqrt(variance), first[key] / n, \
             second[key] / n, drop[key] / n, mapped[key] / n, \
             (drop[key] + mapped[key]) / n
    }
  }
}
