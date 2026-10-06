# One row per weights mode: how many runs gave five distinct texts (base and
# four adapters), how many texts equal the device-copy text of the same run
# and adapter, and the memory the five processes held.
function field(name, i, pair) {
  for (i = 1; i <= NF; ++i) { split($i, pair, "="); if (pair[1] == name) return pair[2] }
  return ""
}
/^BEGIN_ENGINE_ADAPTERS / { run = field("run"); mode = field("mode"); delete seen; distinct = 0; next }
/^TEXT / {
  sha = field("sha256")
  text[mode, run, field("adapter")] = sha
  if (!(sha in seen)) { seen[sha] = 1; distinct++ }
  next
}
/^MEMORY / {
  drop[mode] += field("mem_available_drop_mib")
  mapped[mode] += field("model_mapped_mib")
  next
}
/^END_ENGINE_ADAPTERS / {
  count[mode]++
  if (field("failed_processes") != 0) failed[mode]++
  if (distinct == 5) all_distinct[mode]++
  if (run > runs) runs = run
  next
}
END {
  print "mode,runs,failed_runs,runs_with_five_distinct_texts,texts_equal_to_copy," \
        "texts_compared,device_memory_mib,model_mapped_mib,memory_total_mib"
  split("copy inplace", order, " ")
  for (m = 1; m <= 2; ++m) {
    mode = order[m]
    if (!(mode in count)) continue
    equal = 0; compared = 0
    for (r = 1; r <= runs; ++r) {
      for (a = 0; a <= 4; ++a) {
        if (text[mode, r, a] == "" || text["copy", r, a] == "") continue
        compared++
        if (text[mode, r, a] == text["copy", r, a]) equal++
      }
    }
    printf "%s,%d,%d,%d,%d,%d,%.0f,%.0f,%.0f\n", mode, count[mode], \
           failed[mode] + 0, all_distinct[mode] + 0, equal, compared, \
           drop[mode] / count[mode], mapped[mode] / count[mode], \
           (drop[mode] + mapped[mode]) / count[mode]
  }
}
