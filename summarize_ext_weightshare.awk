# Weight sharing through CUDA IPC as a baseline. One row per placement and
# stack: means over the repetitions, the half-width of the 95% confidence
# interval of the memory, the roles that the library reported, and the texts
# that equal those of `stock` in the same placement and repetition.
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

/^BEGIN_WEIGHTSHARE / { run = field("run") + 0; if (run > runs) runs = run; next }
/^CASE / {
  placement = field("placement"); stack = field("stack"); key = placement SUBSEP stack
  if (!(placement in seen)) { seen[placement] = 1; order[++placements] = placement }
  case_tps = 0
  next
}
/^AGENT / {
  started[key]++
  if (field("own_instance") == 1) own_started[key]++; else other_started[key]++
  if (field("exit") != 0 || field("generation_tps") == "") next
  finished[key]++
  if (field("own_instance") == 1) own_finished[key]++; else other_finished[key]++
  load[key] += field("model_ms")
  first[key] += field("first_token_ms")
  case_tps += field("generation_tps")
  text[key, run, field("index")] = field("text")
  next
}
/^MEMORY / {
  value = field("mem_available_drop_mib")
  memory[key] += value; memory_sq[key] += value * value
  next
}
/^END_CASE / {
  cases[key]++
  timed_out[key] += field("timed_out")
  workers[key] += field("workers"); masters[key] += field("masters")
  fallbacks[key] += field("fallbacks")
  tps[key] += case_tps
  next
}

END {
  print "placement,stack,runs,agents_started,agents_finished,own_instance_finished," \
        "own_instance_started,other_instance_finished,other_instance_started," \
        "library_workers,library_masters,library_fallbacks,weights_load_ms," \
        "first_token_ms,agents_tps_sum,memory_mib,memory_ci95,timed_out," \
        "texts_equal_to_stock,texts_compared"
  split("stock ipc inplace", stacks, " ")
  for (p = 1; p <= placements; ++p) for (s = 1; s <= 3; ++s) {
    key = order[p] SUBSEP stacks[s]
    if (!(key in cases)) continue
    base = order[p] SUBSEP "stock"
    n = cases[key]
    f = finished[key] > 0 ? finished[key] : 1
    equal = 0; compared = 0
    for (item in text) {
      split(item, part, SUBSEP)
      if (part[1] SUBSEP part[2] != key) continue
      if (!((base SUBSEP part[3] SUBSEP part[4]) in text)) continue
      compared++
      if (text[item] == text[base SUBSEP part[3] SUBSEP part[4]]) equal++
    }
    variance = n > 1 ? (memory_sq[key] - memory[key] * memory[key] / n) / (n - 1) : 0
    if (variance < 0) variance = 0
    printf "%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.0f,%.0f,%.2f,%.0f,%.0f,%d,%d,%d\n", \
           order[p], stacks[s], n, started[key] + 0, finished[key] + 0, \
           own_finished[key] + 0, own_started[key] + 0, other_finished[key] + 0, \
           other_started[key] + 0, workers[key] + 0, masters[key] + 0, fallbacks[key] + 0, \
           load[key] / f, first[key] / f, tps[key] / n, memory[key] / n, \
           (n > 1 ? t95(n) * sqrt(variance / n) : 0), timed_out[key] + 0, equal, compared
  }
}
