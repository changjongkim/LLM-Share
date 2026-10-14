# Summarizes run_gpu_read_trace.sh: one row per case (mapping, read, procs)
# over the repetitions.
#   reader_rounds            CHECK lines (expected: runs x procs x rounds)
#   differ_rounds            reader rounds with a differing word
#   differ_words             words that differ, summed
#   cpu_wrong_rounds         differing rounds in which the CPU also reads a differing word
#   again_wrong_rounds       differing rounds in which a second GPU pass reads one again
#   examples                 examples (up to four per differing round)
#   examples_from_first_page examples whose value is the word at the same offset of the
#                            first 4 KiB page of the same 2 MiB page
#   examples_cpu_ok          examples that the CPU reads right
#   pmd_kib                  mean FilePmdMapped of a reader after its GPU pass
function field(name,    i, pair) {
  for (i = 1; i <= NF; ++i) {
    split($i, pair, "=")
    if (pair[1] == name) return pair[2]
  }
  return ""
}
/^CASE / {
  key = field("mapping") "," field("read") "," field("procs")
  if (!(key in runs)) order[++n] = key
  runs[key]++
  next
}
/^CHECK / {
  rounds_seen[key]++
  words[key] += field("words")
  if (field("differ") + 0 > 0) {
    differ_rounds[key]++
    differ[key] += field("differ")
    if (field("cpu_differ") + 0 > 0) cpu_wrong[key]++
    if (field("again_differ") + 0 > 0) again_wrong[key]++
  }
  pmd[key] += field("pmd_kib")
  ms[key] += field("gpu_ms")
  next
}
/^EXAMPLE / {
  examples[key]++
  index_word = field("index") + 0
  source = field("source") + 0
  if (field("source_page_in_block") == "0" && source >= 0 &&
      int(source / 262144) == int(index_word / 262144) && source % 512 == index_word % 512) first_page[key]++
  if (field("cpu_value_ok") == "1") cpu_ok[key]++
  next
}
/^ERROR / { errors[key]++; next }
/^KERNEL / {
  faults[key] += field("gpu_fault_pages")
  fault_ms[key] += field("fault_thread_ms")
  next
}
/^END_CASE / { failed[key] += field("failed"); next }
END {
  print "mapping,read,procs,runs,failed,errors,reader_rounds,words_checked,differ_rounds,differ_words," \
        "cpu_wrong_rounds,again_wrong_rounds,examples,examples_from_first_page,examples_cpu_ok," \
        "pmd_kib,gpu_ms,gpu_fault_pages,fault_thread_ms"
  for (i = 1; i <= n; ++i) {
    k = order[i]
    printf "%s,%d,%d,%d,%d,%.0f,%d,%.0f,%d,%d,%d,%d,%d,%.0f,%.1f,%.0f,%.0f\n", k, runs[k], failed[k] + 0,
      errors[k] + 0, rounds_seen[k] + 0, words[k] + 0, differ_rounds[k] + 0, differ[k] + 0,
      cpu_wrong[k] + 0, again_wrong[k] + 0, examples[k] + 0, first_page[k] + 0, cpu_ok[k] + 0,
      rounds_seen[k] ? pmd[k] / rounds_seen[k] : 0, rounds_seen[k] ? ms[k] / rounds_seen[k] : 0,
      faults[k] / runs[k], fault_ms[k] / runs[k]
  }
}
