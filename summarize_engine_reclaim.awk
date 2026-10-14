# Weights read in place under the reclaim of the kernel (run_engine_reclaim.sh).
# One row per mode, in the order of -v order="MODE MODE ...", and pressure:
# means over the repetitions with the half-width of the 95% confidence
# interval, the summed generation speed of the agents relative to the same
# mode without pressure in the same repetition (geometric mean and its 95%
# confidence interval), and the texts compared with the same agent without
# pressure.
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
function spread(series, key, r, n, sum, sq, variance) {
  n = 0; sum = 0; sq = 0
  for (r = 1; r <= runs; ++r) {
    if (!((key, r) in series)) continue
    n++; sum += series[key, r]; sq += series[key, r] * series[key, r]
  }
  spread_mean = n ? sum / n : 0
  variance = n > 1 ? (sq - sum * sum / n) / (n - 1) : 0
  if (variance < 0) variance = 0
  spread_ci = n > 1 ? t95(n) * sqrt(variance / n) : 0
}
function ratios(series, key, base, r, samples, total, sq, value, variance, margin) {
  samples = 0; total = 0; sq = 0
  for (r = 1; r <= runs; ++r) {
    if (!(series[key, r] > 0 && series[base, r] > 0)) continue
    value = log(series[key, r] / series[base, r])
    samples++; total += value; sq += value * value
  }
  if (!samples) { ratio_mean = ratio_low = ratio_high = 0; return }
  variance = samples > 1 ? (sq - total * total / samples) / (samples - 1) : 0
  if (variance < 0) variance = 0
  margin = samples > 1 ? t95(samples) * sqrt(variance / samples) : 0
  ratio_mean = exp(total / samples)
  ratio_low = exp(total / samples - margin); ratio_high = exp(total / samples + margin)
}

/^BEGIN_RECLAIM / { run = field("run") + 0; if (run > runs) runs = run; next }
/^CASE / {
  mode = field("mode"); pressure = field("pressure"); key = mode SUBSEP pressure
  seen[key] = 1; case_tps = 0
  next
}
/^AGENT / {
  case_tps += field("tps")
  text[key, run, field("index")] = field("text")
  agents_n[key]++
  if (field("tps") + 0 > 0) agents_ok[key]++
  next
}
/^KERNEL / {
  faults[key, run] = field("gpu_fault_pages"); fault_ms[key, run] = field("fault_thread_ms")
  major[key, run] = field("major_faults"); refault[key, run] = field("refault_file")
  next
}
/^RECLAIM / { attempts[key] += field("attempts"); succeeded[key] += field("succeeded"); next }
/^END_CASE / {
  cases[key]++
  if (field("exit") != 0) failed[key]++
  timed[key] += field("timed_out")
  tps[key, run] = case_tps
  next
}

END {
  print "mode,pressure,runs,failed,timed_out,agents,agents_with_speed,texts_equal_to_none," \
        "texts_compared,tps_sum,tps_ci95,tps_vs_none,vs_none_low,vs_none_high," \
        "gpu_fault_pages,gpu_fault_pages_ci95,fault_thread_ms,fault_thread_ci95," \
        "major_faults,major_faults_ci95,refault_file_pages,refault_file_ci95," \
        "reclaim_attempts,reclaim_succeeded"
  modes_n = split(order, modes, " ")
  split("none reclaim", kinds, " ")
  for (m = 1; m <= modes_n; ++m) {
    for (k = 1; k <= 2; ++k) {
      key = modes[m] SUBSEP kinds[k]
      if (!(key in seen)) continue
      base = modes[m] SUBSEP "none"
      equal = 0; compared = 0
      for (item in text) {
        split(item, part, SUBSEP)
        if (part[1] SUBSEP part[2] != key) continue
        other = base SUBSEP part[3] SUBSEP part[4]
        if (!(other in text)) continue
        compared++
        if (text[item] == text[other]) equal++
      }
      printf "%s,%s,%d,%d,%d,%d,%d,%d,%d,", modes[m], kinds[k], cases[key], failed[key] + 0, \
             timed[key] + 0, agents_n[key] + 0, agents_ok[key] + 0, equal, compared
      spread(tps, key); printf "%.2f,%.2f,", spread_mean, spread_ci
      ratios(tps, key, base); printf "%.4f,%.4f,%.4f,", ratio_mean, ratio_low, ratio_high
      spread(faults, key); printf "%.0f,%.0f,", spread_mean, spread_ci
      spread(fault_ms, key); printf "%.0f,%.0f,", spread_mean, spread_ci
      spread(major, key); printf "%.0f,%.0f,", spread_mean, spread_ci
      spread(refault, key); printf "%.0f,%.0f,", spread_mean, spread_ci
      printf "%d,%d\n", attempts[key] + 0, succeeded[key] + 0
    }
  }
}
