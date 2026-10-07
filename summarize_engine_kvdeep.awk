# Deeper and wider trees of agents. One row per way of handing state down:
# means over the repetitions, with the half-width of the 95% confidence
# interval for the memory of the tree and the first token of a leaf. The
# text of a leaf is compared with the leaf of the same repetition that
# received copies. The pages of the tmpfs are given while every process
# runs, after the first subtree has left, and after the last process.
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

/^BEGIN_KVDEEP / { run = field("run") + 0; if (run > runs) runs = run; next }
/^CASE / {
  mode = field("mode"); seen[mode] = 1
  shape[mode] = field("fanouts"); processes[mode] = field("processes"); leaves[mode] = field("leaves")
  first_sum = 0; attach_sum = 0; tps_sum = 0; done = 0
  next
}
/^ROOT / { root_publish[mode] += field("publish_ms"); root_state[mode] += field("state_bytes"); roots[mode]++; next }
/^INNER / {
  if (field("exit") != 0) next
  inner[mode]++
  inner_attach[mode] += field("context_ms") + field("load_ms")
  inner_publish[mode] += field("publish_ms")
  inner_state[mode] += field("state_bytes")
  next
}
/^LEAF / {
  started[mode]++
  if (field("exit") != 0 || field("generation_tps") == "") next
  finished[mode]++
  attach_sum += field("context_ms") + field("load_ms")
  first_sum += field("first_token_ms")
  tps_sum += field("generation_tps")
  done++
  text[mode, run, field("node")] = field("text")
  next
}
/^MEMORY / {
  memory[mode, run] = field("tree_drop_mib")
  files[mode] += field("files_attached_mib")
  files_all[mode] += field("files_all_alive_mib")
  if (field("files_first_subtree_left_mib") >= 0) {
    files_left[mode] += field("files_first_subtree_left_mib"); left_runs[mode]++
  }
  files_end[mode] += field("files_end_mib")
  next
}
/^END_CASE / {
  cases[mode]++
  failed[mode] += field("failed")
  timed_out[mode] += field("timed_out")
  wall[mode] += field("wall_ms")
  if (done > 0) {
    attach[mode, run] = attach_sum / done
    first[mode, run] = first_sum / done
    tps[mode, run] = tps_sum
  }
  next
}

END {
  print "mode,fanouts,processes,leaves,runs,failed_processes,timed_out," \
        "root_publish_ms,root_state_mib,inner_attach_ms,inner_publish_ms," \
        "inner_state_mib,leaf_attach_ms,leaf_first_token_ms,leaf_first_token_ci95," \
        "leaf_tps_sum,tree_memory_mib,tree_memory_ci95,files_attached_mib," \
        "files_all_alive_mib,files_first_subtree_left_mib,files_end_mib," \
        "leaves_finished,leaf_texts_equal_to_copy,leaf_texts_compared,wall_s"
  split("copy chain", order, " ")
  for (m = 1; m <= 2; ++m) {
    mode = order[m]
    if (!(mode in seen)) continue
    n = cases[mode]
    r = roots[mode] > 0 ? roots[mode] : 1
    i = inner[mode] > 0 ? inner[mode] : 1
    equal = 0; compared = 0
    for (key in text) {
      split(key, part, SUBSEP)
      if (part[1] != mode) continue
      if (!(("copy" SUBSEP part[2] SUBSEP part[3]) in text)) continue
      compared++
      if (text[key] == text["copy" SUBSEP part[2] SUBSEP part[3]]) equal++
    }
    printf "%s,%s,%d,%d,%d,%d,%d,%.2f,%.2f,%.1f,%.2f,%.2f,", mode, shape[mode], \
           processes[mode], leaves[mode], n, failed[mode] + 0, timed_out[mode] + 0, \
           root_publish[mode] / r, root_state[mode] / r / 1048576, \
           inner_attach[mode] / i, inner_publish[mode] / i, inner_state[mode] / i / 1048576
    spread(attach, mode); printf "%.1f,", spread_mean
    spread(first, mode); printf "%.0f,%.0f,", spread_mean, spread_ci
    spread(tps, mode); printf "%.2f,", spread_mean
    spread(memory, mode); printf "%.0f,%.0f,", spread_mean, spread_ci
    printf "%.0f,%.0f,%.0f,%.0f,%d,%d,%d,%.1f\n", files[mode] / n, files_all[mode] / n, \
           (left_runs[mode] > 0 ? files_left[mode] / left_runs[mode] : -1), \
           files_end[mode] / n, finished[mode] + 0, equal, compared, wall[mode] / n / 1000
  }
}
