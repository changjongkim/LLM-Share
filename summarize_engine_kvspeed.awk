# One row per number of processes and mode: the generation speed summed over
# the processes, and its paired ratio to `device` of the same repetition
# (geometric mean and 95% confidence interval of the log-ratio).
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
/^BEGIN_KVSPEED / { run = field("run") + 0; if (run > runs) runs = run; tokens = field("prefix_tokens"); next }
/^CASE / { mode = field("mode"); processes = field("processes") + 0; counts[processes] = 1; next }
/^AGENT / {
  key = processes SUBSEP mode
  if (field("exit") != 0 || field("generation_tps") == "") next
  speed[key, run] += field("generation_tps")
  suffix[key] += field("suffix_ms")
  done[key]++
  next
}
/^END_CASE / { key = processes SUBSEP mode; cases[key]++; failed[key] += field("failed_agents"); next }
END {
  print "processes,mode,runs,failed_agents,prefix_tokens,generation_tps_total," \
        "generation_vs_device,lower_ci95,upper_ci95,suffix_ms"
  split("device anon anon_lazy restore extent extent_lazy", order, " ")
  for (processes = 1; processes <= 64; ++processes) {
    if (!(processes in counts)) continue
    baseline = processes SUBSEP "device"
    for (m = 1; m <= 6; ++m) {
      key = processes SUBSEP order[m]
      if (!(key in cases)) continue
      n = cases[key]
      samples = 0; total = 0; totalsq = 0; sum = 0
      for (r = 1; r <= runs; ++r) {
        sum += speed[key, r]
        if (speed[key, r] > 0 && speed[baseline, r] > 0) {
          ratio = log(speed[key, r] / speed[baseline, r])
          samples++; total += ratio; totalsq += ratio * ratio
        }
      }
      average = samples ? total / samples : 0
      variance = 0
      if (samples > 1) variance = (totalsq - total * total / samples) / (samples - 1)
      if (variance < 0) variance = 0
      margin = 0
      if (samples > 1) margin = t95(samples) * sqrt(variance / samples)
      finished = done[key] > 0 ? done[key] : 1
      printf "%d,%s,%d,%d,%d,%.2f,%.4f,%.4f,%.4f,%.1f\n", processes, order[m], n, \
             failed[key] + 0, tokens, sum / n, exp(average), exp(average - margin), \
             exp(average + margin), suffix[key] / finished
    }
  }
}
