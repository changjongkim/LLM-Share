# Copy-on-write as the way to obtain a prefix. One row per prefix length,
# mode and number of agents: means over the repetitions, the texts compared
# agent by agent with the agent that restores the same state by copy, and the
# paired ratio of the generation speed to `restore` of the same repetition
# (geometric mean and 95% confidence interval of the log-ratio).
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
/^BEGIN_KVCOW / {
  run = field("run") + 0; size = field("paragraphs") + 0
  if (run > runs) runs = run
  sizes[size] = 1
  next
}
/^PUBLISH / {
  if (field("store") == "device") state_mib[size, run] = field("state_bytes") / 1048576
  next
}
/^CASE / { mode = field("mode"); agents = field("agents") + 0; counts[agents] = 1; next }
/^AGENT / {
  key = size SUBSEP mode SUBSEP agents
  if (field("exit") != 0 || field("generation_tps") == "") next
  done[key]++
  text[key, run, field("index")] = field("text")
  tokens[key] += field("prefix_tokens")
  attach[key] += field("context_ms") + field("state_ms")
  suffix[key] += field("suffix_ms")
  first[key] += field("first_token_ms")
  speed[key, run] += field("generation_tps")
  next
}
/^MEMORY / {
  key = size SUBSEP mode SUBSEP agents
  memory[key, run] = field("mem_available_drop_mib")
  pss[key] += field("prefix_pss_mib")
  rss[key] += field("prefix_rss_mib")
  next
}
/^END_CASE / {
  key = size SUBSEP mode SUBSEP agents
  cases[key]++
  failed[key] += field("failed_agents")
  next
}
END {
  print "paragraphs,mode,agents,runs,failed_agents,prefix_tokens," \
        "texts_equal_to_restore,texts_compared,attach_ms,suffix_ms,first_token_ms," \
        "generation_tps_total,generation_vs_restore,lower_ci95,upper_ci95," \
        "memory_mib,memory_per_agent_above_extent_mib,prefix_state_mib," \
        "prefix_pss_mib,prefix_rss_mib"
  split("restore cow cow_noread cow_small cow_small_noread extent_lazy", order, " ")
  for (size = 1; size <= 100000; ++size) {
    if (!(size in sizes)) continue
    for (agents = 1; agents <= 64; ++agents) {
      if (!(agents in counts)) continue
      baseline = size SUBSEP "restore" SUBSEP agents
      extent = size SUBSEP "extent_lazy" SUBSEP agents
      for (m = 1; m <= 6; ++m) {
        key = size SUBSEP order[m] SUBSEP agents
        if (!(key in cases)) continue
        n = cases[key]
        finished = done[key] > 0 ? done[key] : 1
        equal = 0; compared = 0
        samples = 0; total = 0; totalsq = 0
        speed_sum = 0; memory_sum = 0; above = 0; state_sum = 0
        for (r = 1; r <= runs; ++r) {
          for (agent = 0; agent < agents; ++agent) {
            if ((baseline, r, agent) in text) {
              compared++
              if (text[key, r, agent] == text[baseline, r, agent]) equal++
            }
          }
          speed_sum += speed[key, r]
          memory_sum += memory[key, r]
          above += (memory[key, r] - memory[extent, r]) / agents
          state_sum += state_mib[size, r]
          if (speed[key, r] > 0 && speed[baseline, r] > 0) {
            ratio = log(speed[key, r] / speed[baseline, r])
            samples++; total += ratio; totalsq += ratio * ratio
          }
        }
        average = samples ? total / samples : 0
        variance = 0
        if (samples > 1) variance = (totalsq - total * total / samples) / (samples - 1)
        if (variance < 0) variance = 0
        margin = 0
        if (samples > 1) margin = t95(samples) * sqrt(variance / samples)
        printf "%d,%s,%d,%d,%d,%.0f,%d,%d,%.1f,%.0f,%.0f,%.2f,%.4f,%.4f,%.4f,%.0f,%.0f,%.0f,%.0f,%.0f\n", \
               size, order[m], agents, n, failed[key] + 0, tokens[key] / finished, \
               equal, compared, attach[key] / finished, suffix[key] / finished, \
               first[key] / finished, speed_sum / n, exp(average), \
               exp(average - margin), exp(average + margin), memory_sum / n, \
               above / n, state_sum / n, pss[key] / n, rss[key] / n
      }
    }
  }
}
