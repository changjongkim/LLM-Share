# Summarizes run_gpu_read_check.sh: one row per case (fs, writer, read, procs)
# over the repetitions.
#   reader_rounds     CHECK lines (expected: runs x procs x rounds)
#   differ_words      words that differ from their expected value, summed
#   zero_words        of those, words that read as zero
#   differ_pages      4 KiB pages with a differing word, summed over rounds
#   differ_rounds     reader rounds with a differing word
#   examples_shifted  examples whose value belongs to another word of the file
#   gpu_ms            mean time of a reader round on the GPU
#   gpu_fault_pages   mean per case, from the driver threads
function field(name,    i, pair) {
  for (i = 1; i <= NF; ++i) {
    split($i, pair, "=")
    if (pair[1] == name) return pair[2]
  }
  return ""
}
/^CASE / {
  key = field("fs") "," field("writer") "," field("read") "," field("procs")
  if (!(key in runs)) order[++n] = key
  runs[key]++
  next
}
/^CHECK / {
  rounds_seen[key]++
  words[key] += field("words")
  differ[key] += field("differ")
  zero[key] += field("zero")
  pages[key] += field("pages")
  if (field("differ") + 0 > 0) differ_rounds[key]++
  ms[key] += field("gpu_ms")
  next
}
/^EXAMPLE / {
  examples[key]++
  if (field("source") + 0 >= 0 && field("source") != field("index")) shifted[key]++
  next
}
/^ERROR / { errors[key]++; next }
/^PUBLISH / {
  if (field("failed") != "" || field("head_differ_cpu") + 0 != 0) publish_bad[key]++
  next
}
/^KERNEL / {
  faults[key] += field("gpu_fault_pages")
  fault_ms[key] += field("fault_thread_ms")
  next
}
/^END_CASE / { failed[key] += field("failed"); next }
END {
  print "fs,writer,read,procs,runs,failed,errors,publish_bad,reader_rounds,words_checked," \
        "differ_words,zero_words,differ_pages,differ_rounds,examples,examples_shifted," \
        "gpu_ms,gpu_fault_pages,fault_thread_ms"
  for (i = 1; i <= n; ++i) {
    k = order[i]
    printf "%s,%d,%d,%d,%d,%d,%.0f,%.0f,%.0f,%.0f,%d,%d,%d,%.1f,%.0f,%.0f\n", k, runs[k],
      failed[k] + 0, errors[k] + 0, publish_bad[k] + 0, rounds_seen[k] + 0, words[k] + 0,
      differ[k] + 0, zero[k] + 0, pages[k] + 0, differ_rounds[k] + 0, examples[k] + 0,
      shifted[k] + 0, rounds_seen[k] ? ms[k] / rounds_seen[k] : 0, faults[k] / runs[k],
      fault_ms[k] / runs[k]
  }
}
