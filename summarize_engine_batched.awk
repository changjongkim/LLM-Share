# One row per weights mode and number of sequences decoded in one process.
# Columns of a ROW line: PP, TG, B, N_KV, T_PP s, S_PP t/s, T_TG s, S_TG t/s,
# T s, S t/s.
BEGIN { FS = "|" }
  /^BEGIN_ENGINE_BATCHED / {
    split($0, words, " ")
    for (i in words) { split(words[i], pair, "="); if (pair[1] == "mode") mode = pair[2] }
    next
  }
  /^ROW / {
    key = mode "," ($4 + 0)
    count[key]++
    prompt[key] += $7
    generation[key] += $9
    generation_sq[key] += $9 * $9
    seen[$4 + 0] = 1
  }
  /^END_ENGINE_BATCHED / { if ($0 !~ /status=0$/) failed++ }
  END {
    print "mode,sequences,runs,prompt_tps_total,generation_tps_total,generation_tps_sd,generation_tps_per_sequence"
    split("copy inplace", modes, " ")
    for (b = 1; b <= 64; ++b) {
      if (!(b in seen)) continue
      for (m = 1; m <= 2; ++m) {
        key = modes[m] "," b
        if (!(key in count)) continue
        n = count[key]
        mean = generation[key] / n
        variance = 0
        if (n > 1) variance = (generation_sq[key] - generation[key] * generation[key] / n) / (n - 1)
        if (variance < 0) variance = 0
        printf "%s,%d,%.1f,%.3f,%.3f,%.3f\n", key, n, prompt[key] / n, mean, sqrt(variance), mean / b
      }
    }
    printf "failed_runs,%d\n", failed + 0
  }