# The memory of the whole serving stack. Three tables.
#
# Default: one row per stack and number of agents, with means over the
# repetitions, the half-width of the 95% confidence interval of the memory,
# and the paired ratio of the summed generation speed to that of `none` and
# to that of `weights` in the same repetition (geometric mean, 95% confidence
# interval of the log-ratio, and the median of the repetitions).
#
# -v table=fit: one row per stack. The memory of a case is fitted by a line
# over the number of agents in every repetition; the row has the mean slope
# (memory per added agent) and intercept, the ratio of the slope to that of
# `none` in the same repetition (mean and largest), the files the agents
# share, and the number of agents that fit by the line: the largest N with
# intercept + N * slope within the memory that was available before the
# cases, less the model file when the weights are read in place (its pages
# count as available although they are in use). -v model_bytes=N gives the
# size of the model file.
#
# -v table=texts: one row per repetition, with the number of agent indices
# that occur in more than one case and the number of those for which every
# case wrote one text.
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
# The paired ratio of the speed of `key` to that of `base`: sets ratio_mean,
# ratio_low, ratio_high and ratio_median.
function ratios(key, base, r, samples, total, totalsq, value, average, variance, margin, list) {
  samples = 0; total = 0; totalsq = 0
  for (r = 1; r <= runs; ++r) {
    if (!(speed[key, r] > 0 && speed[base, r] > 0)) continue
    value = speed[key, r] / speed[base, r]
    list[++samples] = value
    total += log(value); totalsq += log(value) * log(value)
  }
  if (samples == 0) { ratio_mean = ratio_low = ratio_high = ratio_median = 0; return 0 }
  average = total / samples
  variance = samples > 1 ? (totalsq - total * total / samples) / (samples - 1) : 0
  if (variance < 0) variance = 0
  margin = samples > 1 ? t95(samples) * sqrt(variance / samples) : 0
  ratio_mean = exp(average); ratio_low = exp(average - margin)
  ratio_high = exp(average + margin); ratio_median = median(list, samples)
  return samples
}

/^BEGIN_KVSTACK / { run = field("run") + 0; if (run > runs) runs = run; next }
/^PUBLISH / {
  if (field("store") == "huge") { kv_file_kib += field("file_used_kib"); kv_files++ }
  if (field("store") == "stock") { state_bytes += field("state_bytes"); states++ }
  next
}
/^SKIPPED / { skipped[field("mode") SUBSEP (field("agents") + 0)]++; next }
/^CASE / { mode = field("mode"); agents = field("agents") + 0; counts[agents] = 1; next }
/^AGENT / {
  key = mode SUBSEP agents
  if (field("exit") != 0 || field("generation_tps") == "") next
  done[key]++
  index_of_agent = field("index") + 0
  if (!((run, index_of_agent) in first_text)) {
    first_text[run, index_of_agent] = field("text")
    distinct[run, index_of_agent] = 1
  } else if (field("text") != first_text[run, index_of_agent]) {
    distinct[run, index_of_agent]++
  }
  occurrences[run, index_of_agent]++
  if (index_of_agent > last_index) last_index = index_of_agent
  load[key] += field("model_ms")
  attach[key] += field("context_ms") + field("state_ms")
  first[key] += field("first_token_ms")
  speed[key, run] += field("generation_tps")
  next
}
/^MEMORY / {
  key = mode SUBSEP agents
  memory[key, run] = field("mem_available_drop_mib")
  available[mode] += field("mem_available_before_mib"); available_n[mode]++
  anon[key] += field("pss_anon_mib")
  file[key] += field("pss_file_mib")
  shmem[key] += field("pss_shmem_mib")
  device[key] += field("device_mib")
  sampled[key] += (field("samples") > 0)
  next
}
/^END_CASE / {
  key = mode SUBSEP agents
  cases[key]++
  failed[key] += field("failed_agents")
  wall[key] += field("wall_ms")
  next
}

END {
  split("none kv weights both", order, " ")
  if (table == "texts") {
    print "run,indices_in_more_than_one_case,indices_with_one_text,texts"
    for (r = 1; r <= runs; ++r) {
      shared = 0; same = 0; texts = 0
      for (i = 0; i <= last_index; ++i) {
        if (!((r, i) in occurrences)) continue
        texts += occurrences[r, i]
        if (occurrences[r, i] < 2) continue
        shared++
        if (distinct[r, i] == 1) same++
      }
      printf "%d,%d,%d,%d\n", r, shared, same, texts
    }
    exit
  }
  if (table == "fit") {
    print "mode,runs,counts,largest_count,slope_mib_per_agent,slope_ci95," \
          "intercept_mib,slope_vs_none,slope_vs_none_max,model_file_shared_mib," \
          "cache_file_shared_mib,memory_available_mib,agents_that_fit_by_line"
    for (m = 1; m <= 4; ++m) {
      mode = order[m]
      slope_sum = 0; slope_sq = 0; intercept_sum = 0; fitted = 0
      versus_sum = 0; versus_max = 0; versus_n = 0; points = 0; largest = 0
      for (r = 1; r <= runs; ++r) {
        n = 0; sx = 0; sy = 0; sxx = 0; sxy = 0
        for (agents = 1; agents <= 4096; ++agents) {
          if (!((mode, agents, r) in memory)) continue
          n++; sx += agents; sy += memory[mode, agents, r]
          sxx += agents * agents; sxy += agents * memory[mode, agents, r]
          if (agents > largest) largest = agents
        }
        if (n < 2) continue
        slope[mode, r] = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        intercept = (sy - slope[mode, r] * sx) / n
        fitted++; points = n
        slope_sum += slope[mode, r]; slope_sq += slope[mode, r] * slope[mode, r]
        intercept_sum += intercept
        if (("none", r) in slope && slope["none", r] > 0) {
          versus = slope[mode, r] / slope["none", r]
          versus_sum += versus; versus_n++
          if (versus > versus_max) versus_max = versus
        }
      }
      if (fitted == 0) continue
      mean_slope = slope_sum / fitted
      variance = fitted > 1 ? (slope_sq - slope_sum * slope_sum / fitted) / (fitted - 1) : 0
      if (variance < 0) variance = 0
      margin = fitted > 1 ? t95(fitted) * sqrt(variance / fitted) : 0
      in_place = (mode == "weights" || mode == "both")
      mapped = (mode == "kv" || mode == "both")
      model_mib = in_place ? model_bytes / 1048576 : 0
      cache_mib = mapped && kv_files ? kv_file_kib / kv_files / 1024 : 0
      room = available_n[mode] ? available[mode] / available_n[mode] : 0
      fit = int((room - model_mib - intercept_sum / fitted) / mean_slope)
      printf "%s,%d,%d,%d,%.1f,%.1f,%.1f,%.4f,%.4f,%.0f,%.0f,%.0f,%d\n", mode, \
             fitted, points, largest, mean_slope, margin, intercept_sum / fitted, \
             versus_n ? versus_sum / versus_n : 0, versus_max, model_mib, cache_mib, \
             room, fit
    }
    exit
  }
  print "mode,agents,runs,skipped_runs,failed_agents,weights_load_ms,attach_ms," \
        "first_token_ms,generation_tps_total,vs_none,vs_none_low,vs_none_high," \
        "vs_none_median,vs_weights,vs_weights_low,vs_weights_high," \
        "vs_weights_median,memory_mib,memory_ci95,memory_per_agent_mib," \
        "pss_anon_mib,pss_file_mib,pss_shmem_mib,device_mib,sampled_runs,wall_s"
  for (agents = 1; agents <= 4096; ++agents) {
    if (!(agents in counts)) continue
    for (m = 1; m <= 4; ++m) {
      key = order[m] SUBSEP agents
      if (!(key in cases)) {
        if (key in skipped)
          printf "%s,%d,0,%d,,,,,,,,,,,,,,,,,,,,,,\n", order[m], agents, skipped[key]
        continue
      }
      n = cases[key]
      finished = done[key] > 0 ? done[key] : 1
      speed_sum = 0; memory_sum = 0; memory_sq = 0
      for (r = 1; r <= runs; ++r) {
        speed_sum += speed[key, r]
        memory_sum += memory[key, r]; memory_sq += memory[key, r] * memory[key, r]
      }
      variance = n > 1 ? (memory_sq - memory_sum * memory_sum / n) / (n - 1) : 0
      if (variance < 0) variance = 0
      margin = n > 1 ? t95(n) * sqrt(variance / n) : 0
      printf "%s,%d,%d,%d,%d,%.0f,%.1f,%.0f,%.2f,", order[m], agents, n, \
             skipped[key] + 0, failed[key] + 0, load[key] / finished, \
             attach[key] / finished, first[key] / finished, speed_sum / n
      ratios(key, "none" SUBSEP agents)
      printf "%.4f,%.4f,%.4f,%.4f,", ratio_mean, ratio_low, ratio_high, ratio_median
      ratios(key, "weights" SUBSEP agents)
      printf "%.4f,%.4f,%.4f,%.4f,", ratio_mean, ratio_low, ratio_high, ratio_median
      s = sampled[key] > 0 ? sampled[key] : 1
      printf "%.0f,%.0f,%.1f,%.0f,%.0f,%.0f,%.0f,%d,%.1f\n", memory_sum / n, margin, \
             memory_sum / n / agents, anon[key] / s, file[key] / s, shmem[key] / s, \
             device[key] / s, sampled[key] + 0, wall[key] / n / 1000
    }
  }
}
