# One row per mode and step from the log of run_vllm_faultcount.sh: the pages
# that the kernel faulted in on behalf of the GPU during the step (mean,
# smallest and largest over the repetitions), and for a step that is a
# request, its first token, its time, the prompt tokens that the server
# served from its cache, and whether the agent wrote the text of the agent
# of `vllm` in the same step of the same repetition.
function field(name,    i, kv) {
  for (i = 1; i <= NF; ++i) { split($i, kv, "="); if (kv[1] == name) return kv[2] }
  return ""
}
/^BEGIN_VFAULT / { run = field("run") }
/^CASE / { mode = field("mode"); if (!(mode in seen)) { seen[mode] = 1; order[++n] = mode }; request = 0 }
/^(AGENT|PREFIX) / {
  request = 1; request_exit = field("exit"); request_first = field("first_token_ms")
  request_ms = field("request_ms"); request_cached = field("prompt_cached"); request_text = field("text")
}
/^STEP / {
  name = field("name"); key = mode SUBSEP name
  if (!(name in step_seen)) { step_seen[name] = 1; steps[++s] = name }
  pages = field("pages"); runs[key]++; sum[key] += pages
  if (!(key in low) || pages < low[key]) low[key] = pages
  if (pages > high[key]) high[key] = pages
  if (field("exit") != 0 || (request && request_exit != 0)) failed[key]++
  if (request) {
    requests[key]++; first[key] += request_first; time[key] += request_ms; cached[key] += request_cached
    text[run, mode, name] = request_text
  }
  request = 0
}
END {
  print "mode,step,runs,failed,fault_pages,fault_pages_min,fault_pages_max,first_token_ms,request_ms,prompt_cached,texts_equal_to_vllm,texts_compared"
  for (i = 1; i <= n; ++i) {
    for (j = 1; j <= s; ++j) {
      m = order[i]; key = m SUBSEP steps[j]
      if (!(key in runs)) continue
      equal = 0; compared = 0
      for (id in text) {
        split(id, parts, SUBSEP)
        if (parts[2] != m || parts[3] != steps[j] || !((parts[1], "vllm", parts[3]) in text)) continue
        compared++
        if (text[id] == text[parts[1], "vllm", parts[3]]) equal++
      }
      printf "%s,%s,%d,%d,%.0f,%d,%d", m, steps[j], runs[key], failed[key] + 0, sum[key] / runs[key], low[key], high[key]
      if (requests[key] > 0)
        printf ",%.1f,%.1f,%.0f,%d,%d\n", first[key] / requests[key], time[key] / requests[key],
               cached[key] / requests[key], equal, compared
      else
        printf ",,,,,\n"
    }
  }
}
