# A latency-critical agent of the unmodified engine and background agents by
# placement, without a fault and with one. One row per instance of the
# critical agent, placement and fault: how often the critical agent
# completes, how many of the background agents that the fault did not name
# still generate when it has finished or left, and, over the runs in which it
# completes, its generation time and rate with the half-width of the 95%
# confidence interval and the ratio to the critical agent alone in the same
# instance, paired within the repetition (geometric mean, interval on the
# logarithm, and median). `big` names the MIG instance with 12 SMs.
function field(name, i, pair) {
  for (i = 1; i <= NF; ++i) { split($i, pair, "="); if (pair[1] == name) return pair[2] }
  return ""
}
function part_of(list, name, n, items, i, pair) {
  n = split(list, items, ",")
  for (i = 1; i <= n; ++i) { split(items[i], pair, "="); if (pair[1] == name) return pair[2] }
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
function median(values, n, i, j, swap) {
  if (n < 1) return 0
  for (i = 2; i <= n; ++i)
    for (j = i; j > 1 && values[j - 1] > values[j]; --j) {
      swap = values[j]; values[j] = values[j - 1]; values[j - 1] = swap
    }
  return n % 2 ? values[(n + 1) / 2] : (values[n / 2] + values[n / 2 + 1]) / 2
}
function spread(series, key, r, n, sum, sq, variance) {
  n = 0; sum = 0; sq = 0
  for (r = 1; r <= runs; ++r) {
    if (!((key, r) in series)) continue
    n++; sum += series[key, r]; sq += series[key, r] * series[key, r]
  }
  spread_n = n
  spread_mean = n ? sum / n : 0
  variance = n > 1 ? (sq - sum * sum / n) / (n - 1) : 0
  if (variance < 0) variance = 0
  spread_ci = n > 1 ? t95(n) * sqrt(variance / n) : 0
}
function ratios(series, key, base, r, samples, total, sq, value, list, variance, half) {
  samples = 0; total = 0; sq = 0
  for (r = 1; r <= runs; ++r) {
    if (!(series[key, r] > 0 && series[base, r] > 0)) continue
    value = series[key, r] / series[base, r]
    list[++samples] = value
    total += log(value); sq += log(value) * log(value)
  }
  ratio_mean = samples ? exp(total / samples) : 0
  variance = samples > 1 ? (sq - total * total / samples) / (samples - 1) : 0
  if (variance < 0) variance = 0
  half = samples > 1 ? t95(samples) * sqrt(variance / samples) : 0
  ratio_low = samples ? exp(total / samples - half) : 0
  ratio_high = samples ? exp(total / samples + half) : 0
  ratio_median = median(list, samples)
}

BEGIN { if (big == "") big = "fafc828a" }
/^BEGIN_KVPLACE / { run = field("run") + 0; if (run > runs) runs = run; next }
/^CASE / {
  instance = field("critical_mig") == big ? "12sm" : "8sm"
  key = instance SUBSEP field("placement") SUBSEP field("fault")
  if (!(key in seen)) { seen[key] = 1; order[++rows] = key }
  count[key] = field("background") + 0
  ok = 0
  next
}
/^FAULT / {
  faults[key]++
  alive_critical[key] += field("critical_alive")
  alive_background[key] += field("background_alive")
  if (field("victim") == "probe") {
    probe = $0
    sub(/^.* probe=/, "", probe)
    error = part_of(probe, "gpu_write")
    if (error != "" && error != "cudaSuccess") refused[key]++
    if (error == "") error = probe
    if (index("," errors[key] ",", "," error ",") == 0)
      errors[key] = errors[key] == "" ? error : errors[key] "," error
  }
  next
}
/^CRITICAL / {
  if (field("completed") == 1 && field("generation_ms") > 0) {
    ok = 1
    completed[key]++
    wait_now = field("decoded_ms") - field("released_ms")
    time_now = field("generation_ms") + 0
    rate_now = field("generated") * 1000 / field("generation_ms")
    text_now = field("text")
  }
  next
}
/^BACKGROUND / {
  others[key] += field("others")
  generating[key] += field("generating_at_end")
  next
}
/^MEMORY / { memory_now = field("resident_drop_mib") + 0; next }
/^END_CASE / {
  cases[key]++
  guard[key] += field("guard"); timed_out[key] += field("timed_out"); died[key] += field("died_early")
  if (ok) {
    wait_ms[key, run] = wait_now; time_ms[key, run] = time_now; rate[key, run] = rate_now
    text[key, run] = text_now
  }
  memory[key, run] = memory_now
  next
}

END {
  print "critical_instance,placement,fault,background,runs,critical_completed," \
        "others_started,others_generating_at_end,guard_runs,timed_out_runs," \
        "died_early_runs,faults,critical_alive_after_fault,background_alive_after_fault," \
        "writes_refused,write_error,release_to_decoded_ms,release_to_decoded_ci95," \
        "generation_ms,generation_ci95,generation_tps,generation_tps_ci95," \
        "time_vs_alone,time_vs_alone_low,time_vs_alone_high,time_vs_alone_median," \
        "rate_vs_alone,rate_vs_alone_low,rate_vs_alone_high,memory_mib," \
        "texts_equal_to_alone,texts_compared"
  for (i = 1; i <= rows; ++i) {
    key = order[i]
    split(key, name, SUBSEP)
    base = name[1] SUBSEP "alone" SUBSEP "none"
    equal = 0; compared = 0
    for (r = 1; r <= runs; ++r) {
      if (!((key, r) in text) || !((base, r) in text)) continue
      compared++
      if (text[key, r] == text[base, r]) equal++
    }
    printf "%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,", name[1], name[2], name[3], count[key], \
           cases[key] + 0, completed[key] + 0, others[key] + 0, generating[key] + 0, \
           guard[key] + 0, timed_out[key] + 0, died[key] + 0
    printf "%d,%d,%d,%d,%s,", faults[key] + 0, alive_critical[key] + 0, \
           alive_background[key] + 0, refused[key] + 0, errors[key] == "" ? "none" : errors[key]
    spread(wait_ms, key); printf "%.1f,%.1f,", spread_mean, spread_ci
    spread(time_ms, key); printf "%.0f,%.0f,", spread_mean, spread_ci
    spread(rate, key); printf "%.2f,%.2f,", spread_mean, spread_ci
    ratios(time_ms, key, base)
    printf "%.3f,%.3f,%.3f,%.3f,", ratio_mean, ratio_low, ratio_high, ratio_median
    ratios(rate, key, base); printf "%.4f,%.4f,%.4f,", ratio_mean, ratio_low, ratio_high
    spread(memory, key); printf "%.0f,", spread_mean
    printf "%d,%d\n", equal, compared
  }
}
