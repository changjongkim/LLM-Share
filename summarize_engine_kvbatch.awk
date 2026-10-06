# Eight agents on one prefix when agents may share a process. One row per
# configuration: means over the repetitions. "Ready" is the time from the
# start of a case until a server has the first token of all its sequences;
# for a server that starts after the first has published, the time it
# started is included. Texts are compared agent by agent with the
# configuration in which the second server copies the state.
function field(name, i, pair) {
  for (i = 1; i <= NF; ++i) { split($i, pair, "="); if (pair[1] == name) return pair[2] }
  return ""
}
/^BEGIN_KVBATCH / { run = field("run") + 0; if (run > runs) runs = run; next }
/^CASE / { config = field("config"); next }
/^SERVER / {
  servers[config]++
  if (field("exit") != 0 || field("generation_tps") == "") next
  name = field("name")
  ready = field("started_ms") + field("first_token_ms")
  ready_sum[config, name] += ready
  if (ready > last[config, run]) last[config, run] = ready
  tps[config, name] += field("generation_tps")
  total[config] += field("generation_tps")
  tokens[config] = field("prefix_tokens")
  if (name == "b") attach[config] += field("context_ms") + field("state_ms")
  next
}
/^PUBLISH / { publish[config] += field("publish_ms"); state[config] += field("state_bytes"); next }
/^AGENT / { text[config, run, field("index")] = field("text"); agents_seen[field("index")] = 1; next }
/^MEMORY / { memory[config] += field("mem_available_drop_mib"); next }
/^END_CASE / { cases[config]++; failed[config] += field("failed"); next }
END {
  print "config,runs,failed_servers,prefix_tokens,ready_first_server_ms," \
        "ready_all_servers_ms,generation_tps_total,generation_tps_12sm," \
        "generation_tps_6sm,second_server_attach_ms,publish_ms,state_mib," \
        "memory_mib,texts_equal_to_copy,texts_compared"
  split("one_server two_servers_compute two_servers_copy two_servers_extent", order, " ")
  for (m = 1; m <= 4; ++m) {
    config = order[m]
    if (!(config in cases)) continue
    n = cases[config]
    all = 0; equal = 0; compared = 0
    for (r = 1; r <= runs; ++r) {
      all += last[config, r]
      for (agent in agents_seen) {
        if (!((config, r, agent) in text) || !(("two_servers_copy", r, agent) in text)) continue
        compared++
        if (text[config, r, agent] == text["two_servers_copy", r, agent]) equal++
      }
    }
    printf "%s,%d,%d,%d,%.0f,%.0f,%.2f,%.2f,%.2f,%.1f,%.2f,%.2f,%.0f,%d,%d\n", config, n, \
           failed[config] + 0, tokens[config], ready_sum[config, "a"] / n, all / n, \
           total[config] / n, tps[config, "a"] / n, tps[config, "b"] / n, \
           attach[config] / n, publish[config] / n, state[config] / n / 1048576, \
           memory[config] / n, equal, compared
  }
}
