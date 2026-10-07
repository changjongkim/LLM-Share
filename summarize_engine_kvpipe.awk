# A pipeline of agents. One row per stack: means over the repetitions with
# the half-width of the 95% confidence interval, and the paired ratio to
# `stock` in the same repetition (geometric mean and median) of the time to
# complete the pipeline and of its energy on the input rail. The text of a
# worker or of a leader is compared with that of `stock` in the same
# repetition.
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
function ratios(series, key, base, r, samples, total, value, list) {
  samples = 0; total = 0
  for (r = 1; r <= runs; ++r) {
    if (!(series[key, r] > 0 && series[base, r] > 0)) continue
    value = series[key, r] / series[base, r]
    list[++samples] = value
    total += log(value)
  }
  ratio_mean = samples ? exp(total / samples) : 0
  ratio_median = median(list, samples)
}

/^BEGIN_KVPIPE / { run = field("run") + 0; if (run > runs) runs = run; next }
/^CASE / {
  mode = field("mode"); seen[mode] = 1
  shape = field("groups") "x" field("workers") "x" field("turns")
  attach_sum = 0; first_sum = 0; tps_sum = 0; done = 0
  next
}
/^PLANNER / {
  planner_tokens[mode] = field("tokens")
  planner_publish[mode, run] = field("publish_ms")
  next
}
/^LEADER / {
  leaders[mode]++
  leader_tokens[mode] += field("tokens")
  leader_publish[mode] += field("publish_ms")
  leader_attach[mode] += field("context_ms") + field("load_ms")
  leader_text[mode, run, field("group")] = field("text")
  next
}
/^WORKER / {
  started[mode]++
  if (field("exit") != 0 || field("generation_tps") == "") next
  finished[mode]++
  attach_sum += field("context_ms") + field("load_ms")
  first_sum += field("first_token_ms")
  tps_sum += field("generation_tps")
  done++
  worker_tokens[mode] += field("tokens")
  generated[mode] += field("generated")
  worker_text[mode, run, field("index")] = field("text")
  next
}
/^MEMORY / { memory[mode, run] = field("pipeline_drop_mib"); files[mode] += field("files_mib"); next }
/^ENERGY / {
  phase = field("phase")
  vin[mode SUBSEP phase, run] = field("vin_joules")
  gpu[mode SUBSEP phase, run] = field("gpu_joules")
  seconds[mode SUBSEP phase, run] = field("seconds")
  vin_total[mode, run] += field("vin_joules")
  gpu_total[mode, run] += field("gpu_joules")
  next
}
/^END_CASE / {
  cases[mode]++
  failed[mode] += field("failed")
  timed_out[mode] += field("timed_out")
  wall[mode, run] = field("wall_ms") / 1000
  if (done > 0) {
    attach[mode, run] = attach_sum / done
    first[mode, run] = first_sum / done
    tps[mode, run] = tps_sum
  }
  next
}

END {
  print "mode,shape,runs,failed_processes,timed_out,planner_tokens,leader_tokens," \
        "worker_tokens,workers_finished,workers_started,wall_s,wall_ci95," \
        "wall_vs_stock,wall_vs_stock_median,memory_mib,memory_ci95,files_mib," \
        "planner_publish_ms,leader_attach_ms,leader_publish_ms,worker_attach_ms," \
        "worker_first_token_ms,workers_tps_sum,tps_vs_stock,tps_vs_stock_median," \
        "prefix_s,leaders_s,workers_s,vin_prefix_j,vin_leaders_j,vin_workers_j," \
        "vin_total_j,vin_total_ci95,vin_vs_stock,vin_vs_stock_median,gpu_total_j," \
        "generated_tokens_per_vin_kj,worker_texts_equal_to_stock,worker_texts_compared," \
        "leader_texts_equal_to_stock,leader_texts_compared"
  split("stock copy chain", order, " ")
  for (m = 1; m <= 3; ++m) {
    mode = order[m]
    if (!(mode in seen)) continue
    n = cases[mode]
    l = leaders[mode] > 0 ? leaders[mode] : 1
    f = finished[mode] > 0 ? finished[mode] : 1
    equal = 0; compared = 0; leader_equal = 0; leader_compared = 0
    for (key in worker_text) {
      split(key, part, SUBSEP)
      if (part[1] != mode) continue
      if (!(("stock" SUBSEP part[2] SUBSEP part[3]) in worker_text)) continue
      compared++
      if (worker_text[key] == worker_text["stock" SUBSEP part[2] SUBSEP part[3]]) equal++
    }
    for (key in leader_text) {
      split(key, part, SUBSEP)
      if (part[1] != mode) continue
      if (!(("stock" SUBSEP part[2] SUBSEP part[3]) in leader_text)) continue
      leader_compared++
      if (leader_text[key] == leader_text["stock" SUBSEP part[2] SUBSEP part[3]]) leader_equal++
    }
    printf "%s,%s,%d,%d,%d,%d,%.0f,%.0f,%d,%d,", mode, shape, n, failed[mode] + 0, \
           timed_out[mode] + 0, planner_tokens[mode], leader_tokens[mode] / l, \
           worker_tokens[mode] / f, finished[mode] + 0, started[mode] + 0
    spread(wall, mode); printf "%.1f,%.1f,", spread_mean, spread_ci
    ratios(wall, mode, "stock"); printf "%.4f,%.4f,", ratio_mean, ratio_median
    spread(memory, mode); printf "%.0f,%.0f,%.0f,", spread_mean, spread_ci, files[mode] / n
    spread(planner_publish, mode)
    printf "%.2f,%.1f,%.2f,", spread_mean, leader_attach[mode] / l, leader_publish[mode] / l
    spread(attach, mode); printf "%.1f,", spread_mean
    spread(first, mode); printf "%.0f,", spread_mean
    spread(tps, mode); printf "%.2f,", spread_mean
    ratios(tps, mode, "stock"); printf "%.4f,%.4f,", ratio_mean, ratio_median
    spread(seconds, mode SUBSEP "prefix"); printf "%.1f,", spread_mean
    spread(seconds, mode SUBSEP "leaders"); printf "%.1f,", spread_mean
    spread(seconds, mode SUBSEP "workers"); printf "%.1f,", spread_mean
    spread(vin, mode SUBSEP "prefix"); printf "%.0f,", spread_mean
    spread(vin, mode SUBSEP "leaders"); printf "%.0f,", spread_mean
    spread(vin, mode SUBSEP "workers"); printf "%.0f,", spread_mean
    spread(vin_total, mode); printf "%.0f,%.0f,", spread_mean, spread_ci
    total_vin = spread_mean
    ratios(vin_total, mode, "stock"); printf "%.4f,%.4f,", ratio_mean, ratio_median
    spread(gpu_total, mode); printf "%.0f,", spread_mean
    printf "%.1f,%d,%d,%d,%d\n", (total_vin > 0 ? generated[mode] / n / (total_vin / 1000) : 0), \
           equal, compared, leader_equal, leader_compared
  }
}
