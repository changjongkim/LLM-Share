# Resident agents and released agents. One row per count of resident agents:
# means over the repetitions with the half-width of the 95% confidence
# interval. The generation rate of the released agents is summed within a
# repetition, and its ratio to the smallest count is paired within the
# repetition (geometric mean, interval on the logarithm, and median). The
# text of a released agent is compared with its text at the smallest count in
# the same repetition. `big` names the MIG instance with 12 SMs.
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
/^BEGIN_KVRESIDENT / {
  run = field("run") + 0; if (run > runs) runs = run
  prefix_tokens = field("prefix_tokens"); file_mib += field("file_used_mib"); begun++
  next
}
/^SKIPPED / { skipped[field("residents") + 0]++; counts[field("residents") + 0] = 1; next }
/^CASE / {
  n = field("residents") + 0; counts[n] = 1; active = field("active") + 0
  rate = 0; rate_big = 0; rate_small = 0; response = 0; input = 0; done = 0
  next
}
/^AGENT / {
  if (field("role") != "active") next
  if (field("exit") != 0 || !(field("generation_ms") > 0)) next
  value = field("generated") * 1000 / field("generation_ms")
  rate += value
  if (field("mig") == big) rate_big += value; else rate_small += value
  response += field("decoded_ms") - field("released_ms")
  input += field("input_ms")
  done++
  text[n, run, field("index")] = field("text")
  next
}
/^MEMORY / {
  memory[n, run] = field("resident_drop_mib")
  per[n, run] = field("resident_drop_mib") / n
  remaining = field("remaining_without_model_mib") + 0
  if (!(n in least) || remaining < least[n]) least[n] = remaining
  anon[n] += field("pss_anon_mib"); file[n] += field("pss_file_mib")
  shmem[n] += field("pss_shmem_mib"); device[n] += field("device_mib")
  next
}
/^END_CASE / {
  cases[n]++
  failed[n] += field("failed"); guard[n] += field("guard"); timed_out[n] += field("timed_out")
  ready[n, run] = field("ready_s"); active_s[n, run] = field("active_s")
  if (done == active) {
    tps[n, run] = rate; tps_big[n, run] = rate_big; tps_small[n, run] = rate_small
    wait_ms[n, run] = response / done; input_ms[n, run] = input / done
  }
  next
}

END {
  print "residents,active,runs,skipped_runs,failed_processes,guard_runs,timed_out_runs," \
        "prefix_tokens,cache_file_mib,ready_s,memory_mib,memory_ci95,memory_per_resident_mib," \
        "remaining_without_model_min_mib,pss_anon_mib,pss_file_mib,pss_shmem_mib,device_mib," \
        "release_to_decoded_ms,release_to_decoded_ci95,input_decode_ms,active_s," \
        "active_tps_sum,active_tps_ci95,vs_smallest,vs_smallest_low,vs_smallest_high," \
        "vs_smallest_median,tps_12sm,tps_8sm,vs_smallest_12sm,vs_smallest_8sm," \
        "texts_equal_to_smallest,texts_compared"
  total = 0
  for (n in counts) order[++total] = n + 0
  for (i = 2; i <= total; ++i)
    for (j = i; j > 1 && order[j - 1] > order[j]; --j) {
      swap = order[j]; order[j] = order[j - 1]; order[j - 1] = swap
    }
  base = order[1]
  for (i = 1; i <= total; ++i) {
    n = order[i]
    c = cases[n] > 0 ? cases[n] : 1
    equal = 0; compared = 0
    for (key in text) {
      split(key, part, SUBSEP)
      if (part[1] + 0 != n) continue
      if (!((base SUBSEP part[2] SUBSEP part[3]) in text)) continue
      compared++
      if (text[key] == text[base SUBSEP part[2] SUBSEP part[3]]) equal++
    }
    printf "%d,%d,%d,%d,%d,%d,%d,%s,%.0f,", n, active, cases[n] + 0, skipped[n] + 0, \
           failed[n] + 0, guard[n] + 0, timed_out[n] + 0, prefix_tokens, \
           begun ? file_mib / begun : 0
    spread(ready, n); printf "%.1f,", spread_mean
    spread(memory, n); printf "%.0f,%.0f,", spread_mean, spread_ci
    spread(per, n); printf "%.1f,", spread_mean
    printf "%d,%.0f,%.0f,%.0f,%.0f,", least[n] + 0, anon[n] / c, file[n] / c, \
           shmem[n] / c, device[n] / c
    spread(wait_ms, n); printf "%.1f,%.1f,", spread_mean, spread_ci
    spread(input_ms, n); printf "%.1f,", spread_mean
    spread(active_s, n); printf "%.2f,", spread_mean
    spread(tps, n); printf "%.2f,%.2f,", spread_mean, spread_ci
    ratios(tps, n, base); printf "%.4f,%.4f,%.4f,%.4f,", ratio_mean, ratio_low, ratio_high, ratio_median
    spread(tps_big, n); printf "%.2f,", spread_mean
    spread(tps_small, n); printf "%.2f,", spread_mean
    ratios(tps_big, n, base); printf "%.4f,", ratio_mean
    ratios(tps_small, n, base); printf "%.4f,", ratio_mean
    printf "%d,%d\n", equal, compared
  }
}
