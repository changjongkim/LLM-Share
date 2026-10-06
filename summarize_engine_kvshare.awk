# Agents that start from one computed prefix. With -v table=publish: one row
# per prefix length and store of the publisher. Otherwise: one row per prefix
# length, mode and number of agents, with means over the repetitions and the
# paired ratio of the generation speed of each mode to `restore` of the same
# repetition (geometric mean and 95% confidence interval of the log-ratio).
# Texts are compared agent by agent within a repetition: with the agent that
# restores the same state by copy, and with the agent that recomputes the
# prefix, separately for agents in the publisher's MIG instance and in the
# other one (the agent that recomputes does so in its own instance).
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

/^BEGIN_KVSHARE / {
  run = field("run") + 0; size = field("paragraphs") + 0
  publisher = field("publisher_mig")
  if (run > runs) runs = run
  sizes[size] = 1
  next
}
/^PUBLISH / {
  store = field("store")
  key = size SUBSEP store
  published[key]++
  if (field("prefix_tokens") == "") { publish_failed[key]++; next }
  publish_tokens[key] += field("prefix_tokens")
  publish_prefix[key] += field("prefix_ms")
  publish_ms[key] += field("publish_ms")
  publish_bytes[key] += field("state_bytes")
  publish_file[key] += field("file_used_kib")
  if (store == "device") state_mib[size, run] = field("state_bytes") / 1048576
  next
}
/^CASE / { mode = field("mode"); agents = field("agents") + 0; counts[agents] = 1; next }
/^AGENT / {
  key = size SUBSEP mode SUBSEP agents
  seen[key]++
  if (field("exit") != 0 || field("generation_tps") == "") next
  done[key]++
  text[key, run, field("index")] = field("text")
  own[size, run, agents, field("index")] = (field("mig") == publisher)
  tokens[key] += field("prefix_tokens")
  attach[key] += field("context_ms") + field("state_ms")
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
  if (table == "publish") {
    print "paragraphs,store,runs,failed_runs,prefix_tokens,prefix_ms,publish_ms," \
          "state_mib,file_used_mib"
    split("device huge small", stores, " ")
    for (size = 1; size <= 100000; ++size) {
      if (!(size in sizes)) continue
      for (s = 1; s <= 3; ++s) {
        key = size SUBSEP stores[s]
        if (!(key in published)) continue
        n = published[key] - publish_failed[key]
        if (n < 1) n = 1
        printf "%d,%s,%d,%d,%.0f,%.0f,%.2f,%.2f,%.0f\n", size, stores[s], \
               published[key], publish_failed[key] + 0, publish_tokens[key] / n, \
               publish_prefix[key] / n, publish_ms[key] / n, \
               publish_bytes[key] / n / 1048576, publish_file[key] / n / 1024
      }
    }
    exit
  }
  print "paragraphs,mode,agents,runs,failed_agents,prefix_tokens," \
        "texts_equal_to_restore,texts_compared," \
        "own_instance_equal_to_recompute,own_instance_compared," \
        "other_instance_equal_to_recompute,other_instance_compared," \
        "attach_ms,first_token_ms," \
        "generation_tps_total,generation_vs_restore,lower_ci95,upper_ci95," \
        "memory_mib,memory_saved_vs_restore_mib,prefix_state_mib," \
        "prefix_pss_mib,prefix_rss_mib"
  split("recompute restore cow cow_small extent extent_lazy extent_lazy_small", order, " ")
  for (size = 1; size <= 100000; ++size) {
    if (!(size in sizes)) continue
    for (agents = 1; agents <= 64; ++agents) {
      if (!(agents in counts)) continue
      reference = size SUBSEP "recompute" SUBSEP agents
      baseline = size SUBSEP "restore" SUBSEP agents
      for (m = 1; m <= 7; ++m) {
        key = size SUBSEP order[m] SUBSEP agents
        if (!(key in cases)) continue
        n = cases[key]
        finished = done[key] > 0 ? done[key] : 1
        equal = 0; compared = 0
        own_equal = 0; own_compared = 0; other_equal = 0; other_compared = 0
        samples = 0; total = 0; totalsq = 0
        speed_sum = 0; memory_sum = 0; saved = 0; state_sum = 0
        for (r = 1; r <= runs; ++r) {
          for (agent = 0; agent < agents; ++agent) {
            if ((baseline, r, agent) in text) {
              compared++
              if (text[key, r, agent] == text[baseline, r, agent]) equal++
            }
            if (!((reference, r, agent) in text)) continue
            match_reference = (text[key, r, agent] == text[reference, r, agent])
            if (own[size, r, agents, agent]) {
              own_compared++; own_equal += match_reference
            } else {
              other_compared++; other_equal += match_reference
            }
          }
          speed_sum += speed[key, r]
          memory_sum += memory[key, r]
          saved += memory[baseline, r] - memory[key, r]
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
        printf "%d,%s,%d,%d,%d,%.0f,%d,%d,%d,%d,%d,%d,%.1f,%.0f,%.2f,%.4f,%.4f,%.4f,%.0f,%.0f,%.0f,%.0f,%.0f\n", \
               size, order[m], agents, n, failed[key] + 0, tokens[key] / finished, \
               equal, compared, own_equal, own_compared, other_equal, \
               other_compared, attach[key] / finished, first[key] / finished, \
               speed_sum / n, exp(average), exp(average - margin), \
               exp(average + margin), memory_sum / n, saved / n, state_sum / n, \
               pss[key] / n, rss[key] / n
      }
    }
  }
}
