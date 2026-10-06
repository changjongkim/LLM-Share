# Extents under every way of sharing the GPU. One row per configuration,
# number of children, prefix and way of handing the state over: means over
# the repetitions. The text of a child is compared with the child of the same
# cell and repetition that received a copy, and with a process that computes
# the same tokens alone in the child's own MIG instance without MPS.
function field(name, i, pair) {
  for (i = 1; i <= NF; ++i) { split($i, pair, "="); if (pair[1] == name) return pair[2] }
  return ""
}

/^BEGIN_KVMPS / { delete reference; run = field("run"); para = field("paragraphs"); next }
/^REFERENCE / { reference[field("who"), field("mig")] = field("text"); next }
/^CASE / {
  cell = field("config") SUBSEP field("children") SUBSEP para
  mode = field("mode")
  key = cell SUBSEP mode
  seen[cell] = 1
  next
}
/^PARENT / {
  if (field("exit") != 0 || field("generation_tps") == "") next
  tokens[key] += field("prefix_tokens")
  publish[key] += field("publish_ms")
  state[key] += field("state_bytes")
  parent_tps[key] += field("generation_tps")
  parent_done[key]++
  next
}
/^CHILD / {
  started[key]++
  if (field("exit") != 0 || field("generation_tps") == "") next
  finished[key]++
  attach[key] += field("context_ms") + field("state_ms")
  first[key] += field("first_token_ms")
  case_tps += field("generation_tps")
  child_text[key, run, field("index")] = field("text")
  pairs[cell, run, field("index")] = 1
  alone_compared[key]++
  if (field("text") == reference[field("index"), field("mig")]) alone_equal[key]++
  next
}
/^MEMORY / {
  memory[key] += field("mem_available_drop_mib")
  pss[key] += field("prefix_pss_mib")
  file[key] += field("file_mib")
  next
}
/^END_CASE / {
  cases[key]++
  failed[key] += field("failed")
  timed_out[key] += field("timed_out")
  wall[key] += field("children_wall_ms")
  sum_tps[key] += case_tps
  case_tps = 0
  next
}

END {
  print "config,children,prefix_tokens,mode,runs,failed_processes,timed_out," \
        "publish_ms,state_mib,child_attach_ms,child_first_token_ms," \
        "children_tps_sum,parent_tps,children_wall_s,children_memory_mib," \
        "prefix_pss_mib,cache_file_mib,children_finished," \
        "texts_equal_to_copy,texts_equal_to_alone,texts_compared_to_alone"
  split("timeslice mps mig mig_mps", configs, " ")
  split("4 8 16", counts, " ")
  split("80 320 640", paras, " ")
  split("copy extent", modes, " ")
  for (c = 1; c <= 4; ++c) for (p = 1; p <= 3; ++p) for (k = 1; k <= 3; ++k) {
    cell = configs[c] SUBSEP counts[k] SUBSEP paras[p]
    if (!(cell in seen)) continue
    for (m = 1; m <= 2; ++m) {
      key = cell SUBSEP modes[m]
      if (!(key in cases)) continue
      n = cases[key]
      pd = parent_done[key] > 0 ? parent_done[key] : 1
      cd = finished[key] > 0 ? finished[key] : 1
      equal = 0
      for (pair in pairs) {
        if (index(pair, cell SUBSEP) != 1) continue
        rest = substr(pair, length(cell SUBSEP) + 1)
        if ((key SUBSEP rest) in child_text && \
            child_text[key SUBSEP rest] == child_text[cell SUBSEP "copy" SUBSEP rest]) equal++
      }
      printf "%s,%s,%.0f,%s,%d,%d,%d,%.2f,%.2f,%.1f,%.0f,%.2f,%.2f,%.2f,%.0f,%.0f,%.0f,%d,%d,%d,%d\n", \
             configs[c], counts[k], tokens[key] / pd, modes[m], n, \
             failed[key] + 0, timed_out[key] + 0, publish[key] / pd, \
             state[key] / pd / 1048576, attach[key] / cd, first[key] / cd, \
             sum_tps[key] / n, parent_tps[key] / pd, wall[key] / n / 1000, \
             memory[key] / n, pss[key] / n, file[key] / n, finished[key] + 0, \
             equal, alone_equal[key] + 0, alone_compared[key] + 0
    }
  }
}
