# Eight agents on one prefix split over N batching servers. One row per
# configuration (N:mode): means over the repetitions. "Ready" is the time from
# the start of a case until the last server has the first token of all its
# sequences. Texts are compared agent by agent with N:copy of the same N, and
# for one server with 1:none.
function field(name, i, pair) {
  for (i = 1; i <= NF; ++i) { split($i, pair, "="); if (pair[1] == name) return pair[2] }
  return ""
}
/^BEGIN_KVBATCH / { run = field("run") + 0; if (run > runs) runs = run; next }
/^CASE / { config = field("config"); nserv[config] = field("servers"); mode[config] = field("mode"); next }
/^SERVER / {
  if (field("exit") != 0 || field("generation_tps") == "") next
  ready = field("started_ms") + field("first_token_ms")
  if (ready > last[config, run]) last[config, run] = ready
  total[config] += field("generation_tps")
  tokens[config] = field("prefix_tokens")
  if (field("name") != "s0") { attach[config] += field("context_ms") + field("state_ms"); children[config]++ }
  next
}
/^PUBLISH / { publish[config] += field("publish_ms"); state[config] += field("state_bytes"); next }
/^AGENT / { text[config, run, field("index")] = field("text"); agents_seen[field("index")] = 1; next }
/^MEMORY / { memory[config] += field("mem_available_drop_mib"); next }
/^END_CASE / { cases[config]++; failed[config] += field("failed"); order[++norder] = config; next }
END {
  print "config,servers,mode,runs,failed_servers,prefix_tokens,ready_all_servers_ms," \
        "generation_tps_total,child_attach_ms,publish_ms,state_mib,memory_mib," \
        "texts_equal_to_copy,texts_compared"
  for (k = 1; k <= norder; ++k) {
    config = order[k]
    if (done[config]++) continue
    n = cases[config]
    all = 0; equal = 0; compared = 0
    ref = nserv[config] == 1 && mode[config] != "none" ? "1:none" : nserv[config] ":copy"
    for (r = 1; r <= runs; ++r) {
      all += last[config, r]
      for (agent in agents_seen) {
        if (!((config, r, agent) in text) || !((ref, r, agent) in text)) continue
        compared++
        if (text[config, r, agent] == text[ref, r, agent]) equal++
      }
    }
    printf "%s,%d,%s,%d,%d,%d,%.0f,%.2f,%.1f,%.2f,%.2f,%.0f,%d,%d\n", config, nserv[config], mode[config], \
           n, failed[config] + 0, tokens[config], all / n, total[config] / n, \
           children[config] ? attach[config] / children[config] : 0, publish[config] / n, \
           state[config] / n / 1048576, memory[config] / n, equal, compared
  }
}
