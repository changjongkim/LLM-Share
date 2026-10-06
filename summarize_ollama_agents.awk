# Ollama baseline. One row per server placement and cold or warm round: means
# over repetitions. Text is compared across repetitions and, for a warm
# request, with the same request in the cold round.
function field(name, i, pair) {
  for (i = 1; i <= NF; ++i) {
    split($i, pair, "=")
    if (pair[1] == name) return pair[2]
  }
  return ""
}

/^BEGIN_OLLAMA / { run = field("run"); next }
/^CASE / { config = field("config"); next }
/^AGENT / {
  key = config SUBSEP field("round")
  started[key]++
  if (field("exit") != 0) next
  done[key]++
  evaluated[key] += field("prompt_evaluated")
  first[key] += field("first_token_ms")
  request[key] += field("request_ms")
  tps[key] += field("generation_tps")
  generated[key] += field("generated")
  text[key, run, field("index")] = field("text")
  pair[key, run, field("index")] = 1
  next
}
/^ROUND / {
  key = config SUBSEP field("round")
  rounds[key]++
  failed[key] += field("failed")
  memory[key] += field("mem_available_drop_mib")
  servers[key] += field("servers")
  gpu[key] += field("gpu_servers")
  vram[key] += field("vram_mib")
  next
}

END {
  print "config,round,runs,agents_started,agents_finished,prompt_tokens_evaluated," \
        "first_token_ms,request_ms,generation_tps_sum,memory_mib,servers,gpu_servers," \
        "vram_mib,generated_per_agent,texts_equal_to_first,texts_equal_to_cold"
  split("one two", configs, " ")
  split("cold warm", names, " ")
  for (c = 1; c <= 2; ++c) for (r = 1; r <= 2; ++r) {
    key = configs[c] SUBSEP names[r]
    if (!(key in rounds)) continue
    n = rounds[key]
    d = done[key] > 0 ? done[key] : 1
    equal_first = equal_cold = compared = 0
    for (p in pair) {
      if (index(p, key SUBSEP) != 1) continue
      rest = substr(p, length(key SUBSEP) + 1)
      split(rest, parts, SUBSEP)
      if ((key SUBSEP 1 SUBSEP parts[2]) in text) {
        compared++
        if (text[key SUBSEP rest] == text[key SUBSEP 1 SUBSEP parts[2]]) equal_first++
      }
      if (names[r] == "warm" && \
          ((configs[c] SUBSEP "cold" SUBSEP rest) in text) && \
          text[key SUBSEP rest] == text[configs[c] SUBSEP "cold" SUBSEP rest]) equal_cold++
    }
    printf "%s,%s,%d,%d,%d,%.0f,%.0f,%.0f,%.2f,%.0f,%.1f,%.1f,%.0f,%.1f,%d/%d,%s\n", \
           configs[c], names[r], n, started[key] + 0, done[key] + 0, \
           evaluated[key] / d, first[key] / d, request[key] / d, tps[key] / n, \
           memory[key] / n, servers[key] / n, gpu[key] / n, vram[key] / n, \
           generated[key] / d, equal_first, compared, \
           names[r] == "warm" ? equal_cold "/" done[key] : "-"
  }
}
