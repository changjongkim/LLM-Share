# KSM against copies and extents (run_engine_kvksm.sh). One row per mode, in
# the order of -v order="MODE MODE ...": means over the repetitions with the
# half-width of the 95% confidence interval. The highest memory is the largest
# drop of available memory; the memory that KSM merges is the largest number
# of pages that share a merged page, and the memory at that sample is the drop
# when KSM shares the most. The times are from the start of the children to
# the first sample at which KSM shares half and 90% of its largest count. The
# generation speed is paired with host_copy in the same repetition (geometric
# mean of the ratio), and texts are compared with the child of host_copy.
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
function ratio(series, key, base, r, samples, total) {
  samples = 0; total = 0
  for (r = 1; r <= runs; ++r) {
    if (!(series[key, r] > 0 && series[base, r] > 0)) continue
    samples++; total += log(series[key, r] / series[base, r])
  }
  return samples ? exp(total / samples) : 0
}

/^BEGIN_KVKSM / { run = field("run") + 0; if (run > runs) runs = run; next }
/^CASE / {
  key = field("mode")
  case_done = 0; case_tps = 0; case_first = 0; n_samples = 0
  next
}
/^PARENT / { if (field("exit") == 0 && field("generation_tps") != "") parents[key]++; next }
/^CHILD / {
  started[key]++
  if (field("exit") != 0 || field("generation_tps") == "") next
  finished[key]++
  case_tps += field("generation_tps"); case_first += field("first_token_ms"); case_done++
  text[key, run, field("index")] = field("text")
  next
}
/^SAMPLE / {
  ++n_samples
  sample_t[n_samples] = field("t") + 0
  sample_drop[n_samples] = field("drop_mib") + 0
  sample_sharing[n_samples] = field("pages_sharing") + 0
  next
}
/^KERNEL / {
  faults[key, run] = field("gpu_fault_pages"); ksmd[key, run] = field("ksmd_ms")
  next
}
/^MEMORY / { peak[key, run] = field("mem_available_drop_mib"); next }
/^END_CASE / {
  cases[key]++
  failed[key] += field("failed"); guard[key] += field("guard"); timed[key] += field("timed_out")
  if (case_done > 0) { tps[key, run] = case_tps; first[key, run] = case_first / case_done }
  most = 0; at = 0
  for (i = 1; i <= n_samples; ++i) if (sample_sharing[i] > most) { most = sample_sharing[i]; at = i }
  merged[key, run] = most * 4 / 1024
  if (at > 0) {
    at_merge[key, run] = sample_drop[at]
    half = 0; most90 = 0
    for (i = 1; i <= n_samples; ++i) {
      if (!half && sample_sharing[i] >= most * 0.5) half = sample_t[i]
      if (!most90 && sample_sharing[i] >= most * 0.9) most90 = sample_t[i]
    }
    t50[key, run] = half; t90[key, run] = most90
  }
  next
}

END {
  print "mode,runs,failed,guard_cases,timed_out,children_started,children_finished," \
        "texts_equal_to_host_copy,texts_compared,peak_mib,peak_ci95,merged_mib,merged_ci95," \
        "memory_at_most_merged_mib,memory_at_most_merged_ci95,t50_s,t50_ci95,t90_s,t90_ci95," \
        "ksmd_ms,ksmd_ci95,gpu_fault_pages,gpu_fault_pages_ci95,children_tps_sum," \
        "tps_vs_host_copy,child_first_token_ms,child_first_token_ci95"
  modes_n = split(order, modes, " ")
  for (m = 1; m <= modes_n; ++m) {
    key = modes[m]
    if (!(key in cases)) continue
    equal = 0; compared = 0
    for (pair in text) {
      split(pair, part, SUBSEP)
      if (part[1] != key || !(("host_copy", part[2], part[3]) in text)) continue
      compared++
      if (text[pair] == text["host_copy", part[2], part[3]]) equal++
    }
    printf "%s,%d,%d,%d,%d,%d,%d,%d,%d,", key, cases[key], failed[key] + 0, guard[key] + 0, \
           timed[key] + 0, started[key] + 0, finished[key] + 0, equal, compared
    spread(peak, key); printf "%.0f,%.0f,", spread_mean, spread_ci
    spread(merged, key); printf "%.0f,%.0f,", spread_mean, spread_ci
    spread(at_merge, key); printf "%.0f,%.0f,", spread_mean, spread_ci
    spread(t50, key); printf "%.1f,%.1f,", spread_mean, spread_ci
    spread(t90, key); printf "%.1f,%.1f,", spread_mean, spread_ci
    spread(ksmd, key); printf "%.0f,%.0f,", spread_mean, spread_ci
    spread(faults, key); printf "%.0f,%.0f,", spread_mean, spread_ci
    spread(tps, key); printf "%.2f,", spread_mean
    printf "%.4f,", ratio(tps, key, "host_copy")
    spread(first, key); printf "%.0f,%.0f\n", spread_mean, spread_ci
  }
}
