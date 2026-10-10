# One row per mode from the log of run_engine_kvfaultcount.sh: the pages that
# the kernel faulted in on behalf of the GPU (mean, smallest and largest over
# the repetitions), the first token and the summed generation rate of the
# agents, and how many agents wrote the text of the agent of copy with the
# same index in the same repetition. For a publisher that keeps generating,
# the pages before and after it has published, the time of the publish and the
# time of the task that it decodes next.
function field(name,    i, kv) {
  for (i = 1; i <= NF; ++i) { split($i, kv, "="); if (kv[1] == name) return kv[2] }
  return ""
}
/^BEGIN_KVFAULTCOUNT / { run = field("run") }
/^CASE / { mode = field("mode"); if (!(mode in seen)) { seen[mode] = 1; order[++n] = mode } }
/^AGENT / {
  text[run, mode, field("index")] = field("text"); indices[field("index")] = 1
  first[mode] += field("first_token_ms"); count[mode]++; tps[mode] += field("generation_tps")
}
/^END_CASE / {
  f = field("fault_pages"); runs[mode]++; failed[mode] += field("failed"); sum[mode] += f
  if (!(mode in low) || f < low[mode]) low[mode] = f
  if (f > high[mode]) high[mode] = f
}
/^FORK / {
  m = field("mode"); if (!(m in fseen)) { fseen[m] = 1; forder[++fn] = m }
  fruns[m]++; if (field("exit") != 0) ffailed[m]++
  before[m] += field("fault_pages_before_publish"); after[m] += field("fault_pages_after_publish")
  a = field("fault_pages_after_publish"); if (!(m in alow) || a < alow[m]) alow[m] = a; if (a > ahigh[m]) ahigh[m] = a
  pub[m] += field("publish_ms"); suf[m] += field("suffix_ms")
}
END {
  print "kind,mode,runs,failed,fault_pages,fault_pages_min,fault_pages_max,first_token_ms,generation_tps_total,texts_equal_to_copy,texts_compared,fault_pages_before_publish,publish_ms,next_task_ms"
  for (i = 1; i <= n; ++i) {
    m = order[i]; equal = 0; compared = 0
    for (key in text) {
      split(key, parts, SUBSEP)
      if (parts[2] != m) continue
      compared++
      if (text[key] != "none" && text[key] == text[parts[1], "copy", parts[3]]) equal++
    }
    printf "agents,%s,%d,%d,%.0f,%d,%d,%.0f,%.2f,%d,%d,,,\n", m, runs[m], failed[m], sum[m] / runs[m], low[m], high[m],
           first[m] / count[m], tps[m] / runs[m], equal, compared
  }
  for (i = 1; i <= fn; ++i) {
    m = forder[i]
    printf "publisher,%s,%d,%d,%.0f,%d,%d,,,,,%.0f,%.2f,%.1f\n", m, fruns[m], ffailed[m] + 0, after[m] / fruns[m], alow[m], ahigh[m],
           before[m] / fruns[m], pub[m] / fruns[m], suf[m] / fruns[m]
  }
}
