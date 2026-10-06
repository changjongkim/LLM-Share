# Baselines in one engine. One row per placement, number of children and way
# of handing the cache over: means over the repetitions. The text of a child
# is compared with the child of the same placement and repetition that
# received a copy.
function field(name, i, pair) {
  for (i = 1; i <= NF; ++i) { split($i, pair, "="); if (pair[1] == name) return pair[2] }
  return ""
}

/^BEGIN_KVSOTA / { run = field("run"); next }
/^CASE / {
  cell = field("placement") SUBSEP field("children")
  mode = field("mode")
  key = cell SUBSEP mode
  seen[cell] = 1
  next
}
/^PARENT / {
  if (field("exit") != 0 || field("generation_tps") == "") next
  parents[key]++
  tokens[key] += field("prefix_tokens")
  publish[key] += field("publish_ms")
  state[key] += field("state_bytes")
  parent_tps[key] += field("generation_tps")
  next
}
/^CHILD / {
  started[key]++
  if (field("own_instance") == 1) own_started[key]++; else other_started[key]++
  if (field("exit") != 0 || field("generation_tps") == "") next
  finished[key]++
  if (field("own_instance") == 1) own_finished[key]++; else other_finished[key]++
  attach[key] += field("context_ms") + field("state_ms")
  first[key] += field("first_token_ms")
  case_tps += field("generation_tps")
  text[key, run, field("index")] = field("text")
  pairs[cell, run, field("index")] = 1
  next
}
/^MEMORY / { memory[key] += field("mem_available_drop_mib"); next }
/^END_CASE / {
  cases[key]++
  timed_out[key] += field("timed_out")
  tps[key] += case_tps
  case_tps = 0
  next
}

END {
  print "placement,children,mode,runs,parents_completed,timed_out,prefix_tokens," \
        "publish_ms,state_mib,children_started,children_finished," \
        "own_instance_finished,own_instance_started,other_instance_finished," \
        "other_instance_started,child_attach_ms,child_first_token_ms," \
        "children_tps_sum,parent_tps,children_memory_mib,texts_equal_to_copy"
  split("same cross", placements, " ")
  split("copy demand device cow extent", modes, " ")
  for (p = 1; p <= 2; ++p) for (count = 1; count <= 64; ++count) {
    cell = placements[p] SUBSEP count
    if (!(cell in seen)) continue
    for (m = 1; m <= 5; ++m) {
      key = cell SUBSEP modes[m]
      if (!(key in cases)) continue
      n = cases[key]
      pd = parents[key] > 0 ? parents[key] : 1
      f = finished[key] > 0 ? finished[key] : 1
      equal = 0
      for (pair in pairs) {
        if (index(pair, cell SUBSEP) != 1) continue
        rest = substr(pair, length(cell SUBSEP) + 1)
        if ((key SUBSEP rest) in text && \
            text[key SUBSEP rest] == text[cell SUBSEP "copy" SUBSEP rest]) equal++
      }
      printf "%s,%d,%s,%d,%d,%d,%.0f,%.2f,%.2f,%d,%d,%d,%d,%d,%d,%.1f,%.0f,%.2f,%.2f,%.0f,%d\n", \
             placements[p], count, modes[m], n, parents[key] + 0, \
             timed_out[key] + 0, tokens[key] / pd, publish[key] / pd, \
             state[key] / pd / 1048576, started[key] + 0, finished[key] + 0, \
             own_finished[key] + 0, own_started[key] + 0, \
             other_finished[key] + 0, other_started[key] + 0, attach[key] / f, \
             first[key] / f, tps[key] / n, parent_tps[key] / pd, memory[key] / n, equal
    }
  }
}
