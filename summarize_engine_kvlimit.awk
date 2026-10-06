# Memory limits per agent. One row per phase and mode: means over the
# repetitions. Agent 0 is the agent that receives the long task in the limit
# phase; the texts of the other agents are compared with the account phase
# of the same repetition and mode.
function field(name, i, pair) {
  for (i = 1; i <= NF; ++i) { split($i, pair, "="); if (pair[1] == name) return pair[2] }
  return ""
}

/^BEGIN_KVLIMIT / { run = field("run"); next }
/^CASE / { phase = field("phase"); mode = field("mode"); key = phase SUBSEP mode; limit[key] += field("limit_mib"); next }
/^AGENT / {
  agents[key]++
  if (field("index") == 0) {
    first_peak[key] += field("peak_mib")
    first_kills[key] += field("oom_kill")
    if (field("exit") == 0 && field("generation_tps") != "") first_done[key]++
    first_tokens[key] += field("suffix_tokens")
    next
  }
  others[key]++
  other_peak[key] += field("peak_mib")
  other_kills[key] += field("oom_kill")
  if (field("exit") == 0 && field("generation_tps") != "") other_done[key]++
  text[key, run, field("index")] = field("text")
  pairs[mode, run, field("index")] = 1
  next
}
/^MEMORY / { drop[key] += field("mem_available_drop_mib"); charged[key] += field("charged_peak_sum_mib"); next }
/^END_CASE / { cases[key]++; next }

END {
  print "phase,mode,runs,limit_mib,agent0_completed,agent0_oom_kills," \
        "agent0_peak_mib,agent0_task_tokens,others,others_completed," \
        "others_oom_kills,others_peak_mib,others_texts_equal_to_account," \
        "mem_available_drop_mib,charged_peak_sum_mib,charged_fraction"
  split("account limit", phases, " ")
  split("restore extent_lazy", modes, " ")
  for (p = 1; p <= 2; ++p) for (m = 1; m <= 2; ++m) {
    key = phases[p] SUBSEP modes[m]
    if (!(key in cases)) continue
    n = cases[key]
    o = others[key] > 0 ? others[key] : 1
    equal = 0
    for (pair in pairs) {
      if (index(pair, modes[m] SUBSEP) != 1) continue
      rest = substr(pair, length(modes[m] SUBSEP) + 1)
      if (text[key, rest] != "" && text[key SUBSEP rest] == text["account" SUBSEP modes[m] SUBSEP rest]) equal++
    }
    printf "%s,%s,%d,%s,%d,%d,%.0f,%.0f,%d,%d,%d,%.0f,%d,%.0f,%.0f,%.3f\n", \
           phases[p], modes[m], n, phases[p] == "account" ? "max" : sprintf("%.0f", limit[key] / n), \
           first_done[key] + 0, first_kills[key] + 0, first_peak[key] / n, \
           first_tokens[key] / n, others[key] + 0, other_done[key] + 0, \
           other_kills[key] + 0, other_peak[key] / o, equal, drop[key] / n, \
           charged[key] / n, drop[key] > 0 ? charged[key] / drop[key] : 0
  }
}
