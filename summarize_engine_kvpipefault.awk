# A pipeline of agents in which one process fails. One row per configuration
# and fault: counts over the repetitions of the processes that complete (the
# killed process excepted), of the workers whose text equals that of the case
# without a fault in the same repetition and configuration, and of the GPU
# writes that are refused; the time to complete the pipeline with the
# half-width of its 95% confidence interval, and its ratio to the case
# without a fault, paired within the repetition (geometric mean, interval on
# the logarithm, and median).
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

/^BEGIN_KVPIPEFAULT / { run = field("run") + 0; if (run > runs) runs = run; next }
/^CASE / {
  config = field("config"); fault = field("fault"); key = config SUBSEP fault
  if (!(key in seen)) { seen[key] = 1; order[++rows] = key }
  shape = field("groups") "x" field("workers") "x" field("turns")
  next
}
/^PLANNER / {
  started[key]++
  if (field("completed") == 1) { completed[key]++; planner_done[key]++ }
  next
}
/^LEADER / {
  if (fault == "kill_leader" && field("group") == 0) next
  started[key]++; leaders[key]++
  if (field("completed") == 1) { completed[key]++; leaders_done[key]++ }
  next
}
/^WORKER / {
  if (fault == "kill_worker" && field("index") == 0) next
  started[key]++; workers[key]++
  if (field("completed") == 1) {
    completed[key]++; workers_done[key]++
    text[key, run, field("index")] = field("text")
  }
  next
}
/^FAULT / {
  if (field("victim_alive") == 1) victim_alive[key]++
  if (fault == "write") {
    injected[key]++
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
/^END_CASE / {
  cases[key]++
  timed_out[key] += field("timed_out")
  servers[key] += field("mps_servers")
  wall[key, run] = field("wall_ms") / 1000
  phase[key, run] = field("workers_ms") / 1000
  next
}

END {
  print "config,fault,shape,runs,victim_alive_runs,others_started,others_completed," \
        "planner_completed,leaders_started,leaders_completed,workers_started," \
        "workers_completed,workers_with_reference_text,workers_compared," \
        "writes_injected,writes_refused,write_error,timed_out,mps_servers_after," \
        "wall_s,wall_ci95,wall_vs_none,wall_vs_none_low,wall_vs_none_high," \
        "wall_vs_none_median,workers_phase_s"
  for (i = 1; i <= rows; ++i) {
    key = order[i]
    split(key, name, SUBSEP)
    base = name[1] SUBSEP "none"
    n = cases[key] > 0 ? cases[key] : 1
    equal = 0; compared = 0
    for (item in text) {
      split(item, part, SUBSEP)
      if (part[1] != name[1] || part[2] != name[2]) continue
      compared++
      reference = base SUBSEP part[3] SUBSEP part[4]
      if ((reference in text) && text[item] == text[reference]) equal++
    }
    printf "%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,", name[1], name[2], shape, \
           cases[key] + 0, victim_alive[key] + 0, started[key] + 0, completed[key] + 0, \
           planner_done[key] + 0, leaders[key] + 0, leaders_done[key] + 0, \
           workers[key] + 0, workers_done[key] + 0, equal, compared
    printf "%d,%d,%s,%d,%.1f,", injected[key] + 0, refused[key] + 0, \
           errors[key] == "" ? "none" : errors[key], timed_out[key] + 0, servers[key] / n
    spread(wall, key); printf "%.1f,%.1f,", spread_mean, spread_ci
    ratios(wall, key, base)
    printf "%.4f,%.4f,%.4f,%.4f,", ratio_mean, ratio_low, ratio_high, ratio_median
    spread(phase, key); printf "%.1f\n", spread_mean
  }
}
