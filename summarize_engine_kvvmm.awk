# The device-memory counterpart of extents. One row per prefix length, MIG
# instance of the parent, and way of handing the cache over: means over the
# repetitions. The text of a child is compared with the child of the same
# repetition that received the state by copy. For the two cases with a child
# in the other MIG instance the row reports how many children ran. Instances
# appear in the order in which they first hold the parent.
function field(name, i, pair) {
  for (i = 1; i <= NF; ++i) { split($i, pair, "="); if (pair[1] == name) return pair[2] }
  return ""
}
/^BEGIN_KVVMM / {
  run = field("run"); size = field("paragraphs") + 0; instance = field("parent_mig")
  sizes[size] = 1
  if (!(instance in instances)) { instances[instance] = 1; names[++n_names] = instance }
  next
}
/^CASE / { mode = field("mode"); key = size SUBSEP instance SUBSEP mode; next }
/^PARENT / {
  if (field("exit") != 0 || field("generation_tps") == "") next
  parents[key]++
  tokens[key] = field("prefix_tokens")
  publish[key] += field("publish_ms")
  state[key] += field("state_bytes")
  parent_tps[key] += field("generation_tps")
  parent_text[key, run] = field("text")
  next
}
/^CHILD / {
  started[key]++
  if (field("exit") != 0 || field("generation_tps") == "") next
  finished[key]++
  attach[key] += field("context_ms") + field("state_ms")
  first[key] += field("first_token_ms")
  child_tps[key] += field("generation_tps")
  child_text[key, run, field("index")] = field("text")
  next
}
/^MEMORY / { memory[key] += field("mem_available_drop_mib"); next }
/^END_CASE / { cases[key]++; next }
END {
  print "paragraphs,parent_mig,mode,runs,prefix_tokens,publish_ms,state_mib," \
        "parent_generation_tps,parent_texts_equal_to_copy,children_started," \
        "children_finished,child_texts_equal_to_copy,child_attach_ms," \
        "child_first_token_ms,child_generation_tps,children_memory_mib"
  split("copy extent vmm extent_cross vmm_cross", order, " ")
  for (size = 1; size <= 100000; ++size) {
    if (!(size in sizes)) continue
    for (k = 1; k <= n_names; ++k) {
      instance = names[k]
      for (m = 1; m <= 5; ++m) {
        key = size SUBSEP instance SUBSEP order[m]
        if (!(key in cases)) continue
        copy = size SUBSEP instance SUBSEP "copy"
        n = cases[key]
        p = parents[key] > 0 ? parents[key] : 1
        c = finished[key] > 0 ? finished[key] : 1
        parent_equal = 0; child_equal = 0
        for (r = 1; r <= 64; ++r) {
          if ((key, r) in parent_text && parent_text[key, r] == parent_text[copy, r]) parent_equal++
          for (i = 0; i < 64; ++i) {
            if ((key, r, i) in child_text && child_text[key, r, i] == child_text[copy, r, i]) child_equal++
          }
        }
        printf "%d,%s,%s,%d,%d,%.2f,%.2f,%.2f,%d,%d,%d,%d,%.1f,%.0f,%.2f,%.0f\n", size, \
               instance, order[m], n, tokens[key], publish[key] / p, \
               state[key] / p / 1048576, parent_tps[key] / p, parent_equal, \
               started[key] + 0, finished[key] + 0, child_equal, attach[key] / c, \
               first[key] / c, child_tps[key] / c, memory[key] / n
      }
    }
  }
}
