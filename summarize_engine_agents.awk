# One row per compute-sharing configuration, weights mode, and agent count:
# generation and prompt tokens per second per agent and in total, and memory.
function json_number(line, key, pattern, text) {
  pattern = "\"" key "\": *-?[0-9.eE+-]+"
  if (!match(line, pattern)) return ""
  text = substr(line, RSTART, RLENGTH)
  sub(/^[^:]*: */, "", text)
  return text + 0
}
function field(name, i, pair) {
  for (i = 1; i <= NF; ++i) { split($i, pair, "="); if (pair[1] == name) return pair[2] }
  return ""
}
function t95(samples, df) {
  df = samples - 1
  if (df <= 1) return 12.706
  if (df == 2) return 4.303
  if (df == 3) return 3.182
  if (df == 4) return 2.776
  if (df == 5) return 2.571
  if (df == 6) return 2.447
  return 2.365
}

/^BEGIN_ENGINE_AGENTS / {
  group = field("config") "," field("mode") "," field("agents")
  agents = field("agents")
  tg_total = 0; pp_total = 0; reports = 0
  next
}
/^AGENT / {
  if (json_number($0, "n_gen") > 0) {
    tg_total += json_number($0, "avg_ts")
    reports++
  } else {
    pp_total += json_number($0, "avg_ts")
  }
  next
}
/^MEMORY / {
  drop[group] += field("mem_available_drop_mib")
  mapped[group] += field("model_mapped_mib")
  next
}
/^END_ENGINE_AGENTS / {
  count[group]++
  if (field("failed_agents") != 0 || reports != agents) bad[group]++
  tg_sum[group] += tg_total
  tg_sumsq[group] += tg_total * tg_total
  pp_sum[group] += pp_total
  size[group] = agents
  next
}

END {
  print "config,mode,agents,runs,failed_runs,generation_tps_per_agent," \
        "generation_tps_total,generation_total_ci95,prompt_tps_total," \
        "device_memory_mib,model_mapped_mib,memory_total_mib"
  fflush()
  sorter = "LC_ALL=C sort -t, -k1,1 -k3,3n -k2,2"
  for (group in count) {
    n = count[group]
    average = tg_sum[group] / n
    variance = 0
    if (n > 1) variance = (tg_sumsq[group] - tg_sum[group] * tg_sum[group] / n) / (n - 1)
    if (variance < 0) variance = 0
    margin = 0
    if (n > 1) margin = t95(n) * sqrt(variance / n)
    printf "%s,%d,%d,%.3f,%.3f,%.3f,%.1f,%.0f,%.0f,%.0f\n", group, n, \
           bad[group] + 0, average / size[group], average, margin, \
           pp_sum[group] / n, drop[group] / n, mapped[group] / n, \
           (drop[group] + mapped[group]) / n | sorter
  }
  close(sorter)
}
