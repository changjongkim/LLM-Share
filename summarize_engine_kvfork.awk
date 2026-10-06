# A parent that forks its state while it keeps running. One row per way of
# handing the state over: means over the repetitions. The text of a child is
# compared with the child of the same repetition that received the state by
# copy, and with a process that computes the same tokens alone in the child's
# own MIG instance, separately for children in the parent's instance and in
# the other one.
function field(name, i, pair) {
  for (i = 1; i <= NF; ++i) { split($i, pair, "="); if (pair[1] == name) return pair[2] }
  return ""
}

/^BEGIN_KVFORK / { delete reference; run = field("run"); parent_mig = field("parent_mig"); next }
/^REFERENCE / { reference[field("who")] = field("text"); next }
/^CASE / { mode = field("mode"); children[mode] = field("children"); next }
/^PARENT / {
  parents[mode]++
  if (field("exit") != 0 || field("generation_tps") == "") next
  tokens[mode] += field("prefix_tokens")
  publish[mode] += field("publish_ms")
  state[mode] += field("state_bytes")
  parent_tps[mode] += field("generation_tps")
  parent_done[mode]++
  if (field("text") == reference["parent"]) parent_equal[mode]++
  next
}
/^CHILD / {
  started[mode]++
  if (field("exit") != 0 || field("generation_tps") == "") next
  finished[mode]++
  attach[mode] += field("context_ms") + field("state_ms")
  first[mode] += field("first_token_ms")
  child_tps[mode] += field("generation_tps")
  child_text[mode, run, field("index")] = field("text")
  pairs[run, field("index")] = 1
  if (field("mig") == parent_mig) {
    own_compared[mode]++
    if (field("text") == reference[field("index")]) own_equal[mode]++
  } else {
    other_compared[mode]++
    if (field("text") == reference[field("index")]) other_equal[mode]++
  }
  next
}
/^MEMORY / {
  memory[mode] += field("mem_available_drop_mib")
  pss[mode] += field("prefix_pss_mib")
  rss[mode] += field("prefix_rss_mib")
  next
}
/^END_CASE / { cases[mode]++; failed[mode] += field("failed"); next }

END {
  print "mode,runs,children,failed_processes,prefix_tokens,publish_ms," \
        "state_mib,parent_texts_equal,parent_generation_tps,children_started," \
        "child_texts_equal_to_copy,own_instance_equal_to_alone," \
        "own_instance_compared,other_instance_equal_to_alone," \
        "other_instance_compared,child_attach_ms,child_first_token_ms," \
        "child_generation_tps,children_memory_mib,prefix_pss_mib,prefix_rss_mib"
  split("copy extent", order, " ")
  for (m = 1; m <= 2; ++m) {
    mode = order[m]
    if (!(mode in cases)) continue
    n = cases[mode]
    p = parent_done[mode] > 0 ? parent_done[mode] : 1
    c = finished[mode] > 0 ? finished[mode] : 1
    equal = 0
    for (pair in pairs) {
      if ((mode SUBSEP pair) in child_text && \
          child_text[mode SUBSEP pair] == child_text["copy" SUBSEP pair]) equal++
    }
    printf "%s,%d,%d,%d,%.0f,%.2f,%.2f,%d,%.2f,%d,%d,%d,%d,%d,%d,%.1f,%.0f,%.2f,%.0f,%.0f,%.0f\n", \
           mode, n, children[mode], failed[mode] + 0, tokens[mode] / p, \
           publish[mode] / p, state[mode] / p / 1048576, parent_equal[mode] + 0, \
           parent_tps[mode] / p, started[mode] + 0, equal, own_equal[mode] + 0, \
           own_compared[mode] + 0, other_equal[mode] + 0, \
           other_compared[mode] + 0, attach[mode] / c, first[mode] / c, \
           child_tps[mode] / c, memory[mode] / n, pss[mode] / n, rss[mode] / n
  }
}
