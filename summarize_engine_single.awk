# One row per weights mode: means over the repetitions, and the paired ratio
# of each mode to the device copy of the same repetition (geometric mean and
# 95% confidence interval of the log-ratio) for generation speed.
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

/^BEGIN_ENGINE_SINGLE / { run = field("run"); mode = field("mode"); next }
/^BENCH / {
  if (json_number($0, "n_gen") > 0) {
    tg[mode, run] = json_number($0, "avg_ts")
  } else {
    pp[mode, run] = json_number($0, "avg_ts")
  }
  next
}
/^MEMORY / {
  drop[mode] += field("mem_available_drop_mib")
  mapped[mode] += field("model_pss_mib")
  next
}
/^PERF +load time/ { load[mode] += $5; next }
/^PERF +eval time/ { for (i = 1; i <= NF; ++i) if ($i == "tokens" && $(i + 1) == "per") completion_tps[mode] += $(i - 1); next }
/^OUTPUT / { text[mode, run] = field("sha256"); next }
/^END_ENGINE_SINGLE / {
  count[mode]++
  if (field("status") != 0) failed[mode]++
  if (run > runs) runs = run
  next
}

END {
  print "mode,runs,failed_runs,identical_text_runs,prompt_tps,generation_tps," \
        "generation_vs_copy,lower_ci95,upper_ci95,load_ms,device_memory_mib," \
        "model_mapped_mib"
  split("copy inplace inplace_noread", order, " ")
  for (m = 1; m <= 3; ++m) {
    mode = order[m]
    if (!(mode in count)) continue
    samples = 0; total = 0; totalsq = 0; pp_sum = 0; tg_sum = 0; same = 0
    for (r = 1; r <= runs; ++r) {
      pp_sum += pp[mode, r]
      tg_sum += tg[mode, r]
      if (text[mode, r] != "" && text[mode, r] == text["copy", r]) same++
      if (tg[mode, r] > 0 && tg["copy", r] > 0) {
        ratio = log(tg[mode, r] / tg["copy", r])
        samples++; total += ratio; totalsq += ratio * ratio
      }
    }
    average = samples ? total / samples : 0
    variance = 0
    if (samples > 1) variance = (totalsq - total * total / samples) / (samples - 1)
    if (variance < 0) variance = 0
    margin = 0
    if (samples > 1) margin = t95(samples) * sqrt(variance / samples)
    printf "%s,%d,%d,%d,%.2f,%.3f,%.4f,%.4f,%.4f,%.1f,%.0f,%.0f\n", mode, \
           count[mode], failed[mode] + 0, same, pp_sum / count[mode], \
           tg_sum / count[mode], exp(average), exp(average - margin), \
           exp(average + margin), load[mode] / count[mode], \
           drop[mode] / count[mode], mapped[mode] / count[mode]
  }
}
