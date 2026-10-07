# Mechanisms in one engine. One row per placement, number of children,
# length of the prefix and mode, the cells in the order of the log and the
# modes in the order of -v order="MODE MODE ...": means over the repetitions
# with the half-width of the 95% confidence interval, and the paired ratio of
# the generation speed to that of `copy` in the same repetition (geometric
# mean, 95% confidence interval of the log-ratio, and the median of the
# repetitions). The text of a child is compared with the child of the same
# placement and repetition that received a copy.
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
# Mean and half-width of series[key, 1..runs] over the runs that have a value.
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
function ratios(series, key, base, r, samples, total, totalsq, value, average, variance, margin, list) {
  samples = 0; total = 0; totalsq = 0
  for (r = 1; r <= runs; ++r) {
    if (!(series[key, r] > 0 && series[base, r] > 0)) continue
    value = series[key, r] / series[base, r]
    list[++samples] = value
    total += log(value); totalsq += log(value) * log(value)
  }
  if (samples == 0) { ratio_mean = ratio_low = ratio_high = ratio_median = 0; return }
  average = total / samples
  variance = samples > 1 ? (totalsq - total * total / samples) / (samples - 1) : 0
  if (variance < 0) variance = 0
  margin = samples > 1 ? t95(samples) * sqrt(variance / samples) : 0
  ratio_mean = exp(average); ratio_low = exp(average - margin)
  ratio_high = exp(average + margin); ratio_median = median(list, samples)
}

/^BEGIN_KVMECH / { run = field("run") + 0; if (run > runs) runs = run; next }
/^CASE / {
  cell = field("placement") SUBSEP field("children") SUBSEP field("paragraphs")
  mode = field("mode")
  key = cell SUBSEP mode
  if (!(cell in seen)) { seen[cell] = 1; cell_order[++cells_n] = cell }
  case_attach = 0; case_first = 0; case_done = 0; case_tps = 0
  next
}
/^PARENT / {
  if (field("exit") != 0 || field("generation_tps") == "") next
  parents[key]++
  tokens[key] += field("prefix_tokens")
  publish[key, run] = field("publish_ms")
  state[key] += field("state_bytes")
  parent_tps[key, run] = field("generation_tps")
  next
}
/^CHILD / {
  started[key]++
  if (field("own_instance") == 1) own_started[key]++; else other_started[key]++
  if (field("exit") != 0 || field("generation_tps") == "") next
  finished[key]++
  if (field("own_instance") == 1) own_finished[key]++; else other_finished[key]++
  case_attach += field("context_ms") + field("state_ms")
  case_first += field("first_token_ms")
  case_tps += field("generation_tps")
  case_done++
  text[key, run, field("index")] = field("text")
  pairs[cell, run, field("index")] = 1
  next
}
/^MEMORY / { memory[key, run] = field("mem_available_drop_mib"); next }
/^END_CASE / {
  cases[key]++
  timed_out[key] += field("timed_out")
  if (case_done > 0) {
    attach[key, run] = case_attach / case_done
    first[key, run] = case_first / case_done
    tps[key, run] = case_tps
  }
  next
}

END {
  print "placement,children,paragraphs,mode,runs,parents_completed,timed_out,prefix_tokens," \
        "publish_ms,publish_ci95,state_mib,children_started,children_finished," \
        "own_instance_finished,own_instance_started,other_instance_finished," \
        "other_instance_started,child_attach_ms,child_attach_ci95," \
        "child_first_token_ms,child_first_token_ci95,children_tps_sum," \
        "children_tps_vs_copy,vs_copy_low,vs_copy_high,vs_copy_median," \
        "parent_tps,parent_tps_vs_copy,parent_vs_copy_median," \
        "children_memory_mib,children_memory_ci95,texts_equal_to_copy," \
        "texts_compared"
  modes_n = split(order, modes, " ")
  for (c = 1; c <= cells_n; ++c) {
    cell = cell_order[c]
    split(cell, name, SUBSEP)
    for (m = 1; m <= modes_n; ++m) {
      key = cell SUBSEP modes[m]
      if (!(key in cases)) continue
      base = cell SUBSEP "copy"
      n = cases[key]
      done = parents[key] > 0 ? parents[key] : 1
      equal = 0; compared = 0
      for (pair in pairs) {
        split(pair, part, SUBSEP)
        if (part[1] SUBSEP part[2] SUBSEP part[3] != cell) continue
        index_key = part[4] SUBSEP part[5]
        if (!((key SUBSEP index_key) in text)) continue
        if (!((base SUBSEP index_key) in text)) continue
        compared++
        if (text[key SUBSEP index_key] == text[base SUBSEP index_key]) equal++
      }
      printf "%s,%d,%d,%s,%d,%d,%d,%.0f,", name[1], name[2], name[3], modes[m], n, \
             parents[key] + 0, timed_out[key] + 0, tokens[key] / done
      spread(publish, key)
      printf "%.2f,%.2f,%.2f,%d,%d,%d,%d,%d,%d,", spread_mean, spread_ci, \
             state[key] / done / 1048576, started[key] + 0, finished[key] + 0, \
             own_finished[key] + 0, own_started[key] + 0, other_finished[key] + 0, \
             other_started[key] + 0
      spread(attach, key); printf "%.1f,%.1f,", spread_mean, spread_ci
      spread(first, key); printf "%.0f,%.0f,", spread_mean, spread_ci
      spread(tps, key); printf "%.2f,", spread_mean
      ratios(tps, key, base)
      printf "%.4f,%.4f,%.4f,%.4f,", ratio_mean, ratio_low, ratio_high, ratio_median
      spread(parent_tps, key); printf "%.3f,", spread_mean
      ratios(parent_tps, key, base)
      printf "%.4f,%.4f,", ratio_mean, ratio_median
      spread(memory, key)
      printf "%.0f,%.0f,%d,%d\n", spread_mean, spread_ci, equal, compared
    }
  }
}
