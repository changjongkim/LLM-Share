# A tree of agents. One row per way of handing state down: means over the
# repetitions. The text of a leaf is compared with the leaf of the same
# repetition that received copies, and with a process that computes the same
# tokens alone in the leaf's own MIG instance, separately for leaves whose
# rows were all computed in their own instance and for the others.
function field(name, i, pair) {
  for (i = 1; i <= NF; ++i) { split($i, pair, "="); if (pair[1] == name) return pair[2] }
  return ""
}

/^BEGIN_KVTREE / { delete reference; delete own; run = field("run"); next }
/^REFERENCE / {
  reference[field("index")] = field("text")
  own[field("index")] = field("own_instance_path")
  next
}
/^CASE / { mode = field("mode"); groups[mode] = field("groups"); leaves[mode] = field("leaves"); next }
/^ROOT / {
  if (field("exit") != 0 || field("generation_tps") == "") next
  roots[mode]++
  root_tokens[mode] += field("tokens")
  root_publish[mode] += field("publish_ms")
  root_state[mode] += field("state_bytes")
  next
}
/^LEADER / {
  if (field("exit") != 0 || field("generation_tps") == "") next
  leaders[mode]++
  leader_tokens[mode] += field("tokens")
  leader_attach[mode] += field("context_ms") + field("load_ms")
  if (field("publish_ms") != "") {
    leader_publish[mode] += field("publish_ms")
    leader_state[mode] += field("state_bytes")
    published[mode]++
  }
  next
}
/^LEAF / {
  started[mode]++
  if (field("exit") != 0 || field("generation_tps") == "") next
  finished[mode]++
  attach[mode] += field("context_ms") + field("load_ms")
  decode[mode] += field("decode_ms")
  first[mode] += field("first_token_ms")
  tps[mode] += field("generation_tps")
  leaf_text[mode, run, field("index")] = field("text")
  pairs[run, field("index")] = 1
  if (own[field("index")] == 1) {
    own_compared[mode]++
    if (field("text") == reference[field("index")]) own_equal[mode]++
  } else {
    other_compared[mode]++
    if (field("text") == reference[field("index")]) other_equal[mode]++
  }
  next
}
/^MEMORY / {
  memory[mode] += field("tree_drop_mib")
  files[mode] += field("files_attached_mib")
  files_end[mode] += field("files_end_mib")
  removed[mode] += field("files_removed")
  next
}
/^END_CASE / {
  cases[mode]++
  failed[mode] += field("failed")
  timed_out[mode] += field("timed_out")
  wall[mode] += field("wall_ms")
  next
}

END {
  print "mode,runs,groups,leaves_per_group,failed_processes,timed_out," \
        "root_tokens,leader_tokens,root_publish_ms,root_state_mib," \
        "leader_attach_ms,leader_publish_ms,leader_state_mib," \
        "leaf_attach_ms,leaf_decode_ms,leaf_first_token_ms,leaf_tps_sum," \
        "tree_memory_mib,segment_files_mib,files_removed_runs," \
        "files_left_at_end_mib,leaves_finished,leaf_texts_equal_to_copy," \
        "own_path_equal_to_alone,own_path_compared," \
        "other_path_equal_to_alone,other_path_compared,wall_s"
  split("copy flat chain chain_small", order, " ")
  for (m = 1; m <= 4; ++m) {
    mode = order[m]
    if (!(mode in cases)) continue
    n = cases[mode]
    r = roots[mode] > 0 ? roots[mode] : 1
    l = leaders[mode] > 0 ? leaders[mode] : 1
    p = published[mode] > 0 ? published[mode] : 1
    f = finished[mode] > 0 ? finished[mode] : 1
    equal = 0
    for (pair in pairs) {
      if ((mode SUBSEP pair) in leaf_text && \
          leaf_text[mode SUBSEP pair] == leaf_text["copy" SUBSEP pair]) equal++
    }
    printf "%s,%d,%d,%d,%d,%d,%.0f,%.0f,%.2f,%.2f,%.1f,%.2f,%.2f,%.1f,%.0f,%.0f,%.2f,%.0f,%.0f,%d,%.0f,%d,%d,%d,%d,%d,%d,%.1f\n", \
           mode, n, groups[mode], leaves[mode], failed[mode] + 0, \
           timed_out[mode] + 0, root_tokens[mode] / r, leader_tokens[mode] / l, \
           root_publish[mode] / r, root_state[mode] / r / 1048576, \
           leader_attach[mode] / l, leader_publish[mode] / p, \
           leader_state[mode] / p / 1048576, attach[mode] / f, decode[mode] / f, \
           first[mode] / f, tps[mode] / n, memory[mode] / n, files[mode] / n, \
           removed[mode] + 0, files_end[mode] / n, finished[mode] + 0, equal, \
           own_equal[mode] + 0, own_compared[mode] + 0, other_equal[mode] + 0, \
           other_compared[mode] + 0, wall[mode] / n / 1000
  }
}
