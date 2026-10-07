# vLLM as an in-process baseline. One row per round: means over the
# repetitions of the agents that completed.
function field(name, i, pair) {
  for (i = 1; i <= NF; ++i) { split($i, pair, "="); if (pair[1] == name) return pair[2] }
  return ""
}

/^BEGIN_VLLM / { runs++; next }
/^SERVER / {
  ready += field("ready"); startup += field("startup_ms"); ready_drop += field("ready_drop_mib")
  next
}
/^ROUND / { round = field("name"); rounds[round]++; case_tps = 0; next }
/^AGENT / {
  started[round]++
  if (field("exit") != 0) next
  finished[round]++
  first[round] += field("first_token_ms")
  prompt[round] += field("prompt_tokens")
  cached[round] += field("prompt_cached")
  generated[round] += field("generated")
  tps[round] += field("generation_tps")
  request[round] += field("request_ms")
  next
}
/^MEMORY / { memory[field("round")] += field("peak_drop_mib"); next }

END {
  print "round,runs,servers_ready,agents_started,agents_finished,startup_ms," \
        "ready_memory_mib,first_token_ms,request_ms,prompt_tokens,prompt_cached," \
        "generated,agents_tps_sum,peak_memory_mib"
  split("cold warm", order, " ")
  for (r = 1; r <= 2; ++r) {
    name = order[r]
    if (!(name in rounds)) continue
    n = rounds[name]
    f = finished[name] > 0 ? finished[name] : 1
    printf "%s,%d,%d,%d,%d,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.1f,%.2f,%.0f\n", name, runs, ready, \
           started[name] + 0, finished[name] + 0, startup / runs, ready_drop / runs, \
           first[name] / f, request[name] / f, prompt[name] / f, cached[name] / f, \
           generated[name] / f, tps[name] / n, memory[name] / n
  }
}
