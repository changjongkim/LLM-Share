# Extents under the stock server. One row per mode and number of agent
# servers: means over the repetitions. The text of an agent server is
# compared with the agent server of the same repetition that restored a copy.
function field(name, i, pair) {
  for (i = 1; i <= NF; ++i) { split($i, pair, "="); if (pair[1] == name) return pair[2] }
  return ""
}

/^BEGIN_KVSERVER / { run = field("run"); next }
/^CASE / { mode = field("mode"); agents = field("agents"); key = mode SUBSEP agents; seen[agents] = 1; next }
/^PUBLISHED / {
  published[key]++
  tokens[key] += field("prefix_tokens")
  prefix_ms[key] += field("prefix_ms")
  publish_ms[key] += field("publish_ms")
  save_ms[key] += field("server_save_ms")
  state[key] += field("state_bytes")
  next
}
/^AGENT / {
  started[key]++
  if (field("exit") != 0) next
  finished[key]++
  restore[key] += field("restore_ms")
  evaluated[key] += field("prompt_evaluated")
  if (field("prompt_evaluated") + 0 > most[key] + 0) most[key] = field("prompt_evaluated")
  first[key] += field("first_token_ms")
  case_tps += field("generation_tps")
  text[key, run, field("index")] = field("text")
  token_ids[key, run, field("index")] = field("token_ids")
  pairs[agents, run, field("index")] = 1
  next
}
/^MEMORY / { memory[key] += field("mem_available_drop_mib"); file[key] += field("file_mib"); next }
/^END_CASE / { cases[key]++; failed[key] += field("failed"); tps[key] += case_tps; case_tps = 0; next }

END {
  print "mode,agents,runs,failed,prefix_tokens,prefix_ms,save_request_ms," \
        "server_save_ms,slot_file_mib,restore_request_ms," \
        "prompt_tokens_evaluated,most_prompt_tokens_evaluated," \
        "first_token_ms,generation_tps_sum,agents_memory_mib,cache_file_mib," \
        "agents_finished,texts_equal_to_copy,min_common_tokens_to_copy," \
        "token_sequences_compared"
  split("copy extent", order, " ")
  for (agents = 1; agents <= 64; ++agents) {
    if (!(agents in seen)) continue
    for (m = 1; m <= 2; ++m) {
      key = order[m] SUBSEP agents
      if (!(key in cases)) continue
      n = cases[key]
      p = published[key] > 0 ? published[key] : 1
      f = finished[key] > 0 ? finished[key] : 1
      equal = 0
      compared = 0
      min_common = -1
      for (pair in pairs) {
        if (index(pair, agents SUBSEP) != 1) continue
        rest = substr(pair, length(agents SUBSEP) + 1)
        if ((key SUBSEP rest) in text && \
            text[key SUBSEP rest] == text["copy" SUBSEP agents SUBSEP rest]) equal++
        current = token_ids[key SUBSEP rest]
        reference = token_ids["copy" SUBSEP agents SUBSEP rest]
        if (current != "" && reference != "") {
          delete current_tokens
          delete reference_tokens
          current_n = split(current, current_tokens, ":")
          reference_n = split(reference, reference_tokens, ":")
          common = 0
          while (common < current_n && common < reference_n && \
                 current_tokens[common + 1] == reference_tokens[common + 1]) common++
          if (min_common < 0 || common < min_common) min_common = common
          compared++
        }
      }
      printf "%s,%d,%d,%d,%.0f,%.0f,%.2f,%.2f,%.2f,%.1f,%.1f,%d,%.0f,%.2f,%.0f,%.0f,%d,%d,%d,%d\n", \
             order[m], agents, n, failed[key] + 0, tokens[key] / p, \
             prefix_ms[key] / p, publish_ms[key] / p, save_ms[key] / p, \
             state[key] / p / 1048576, restore[key] / f, evaluated[key] / f, \
             most[key] + 0, first[key] / f, tps[key] / n, memory[key] / n, \
             file[key] / n, finished[key] + 0, equal, min_common, compared
    }
  }
}
