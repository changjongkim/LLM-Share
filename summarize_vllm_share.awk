# Two vLLM servers that serve one prefix. One row per mode: means over the
# repetitions with the half-width of the 95% confidence interval, and the
# paired ratio to `vllm` in the same repetition (geometric mean and median).
# The text of an agent is compared with that of `vllm` in the same
# repetition and round. Throughput is that of round `both`, also split by
# the MIG instance of the server (the servers alternate between the two).
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
function median(values, n, i, j, swap) {
  if (n < 1) return 0
  for (i = 2; i <= n; ++i)
    for (j = i; j > 1 && values[j - 1] > values[j]; --j) {
      swap = values[j]; values[j] = values[j - 1]; values[j - 1] = swap
    }
  return n % 2 ? values[(n + 1) / 2] : (values[n / 2] + values[n / 2 + 1]) / 2
}
function spread(series, key, r, n, sum, sq, variance) {
  n = 0; sum = 0; sq = 0
  for (r = 1; r <= runs; ++r) {
    if (!((key, r) in series)) continue
    n++; sum += series[key, r]; sq += series[key, r] * series[key, r]
  }
  spread_mean = n ? sum / n : 0
  variance = n > 1 ? (sq - sum * sum / n) / (n - 1) : 0
  if (variance < 0) variance = 0
  spread_ci = n > 1 ? t95(n) * sqrt(variance / n) : 0
}
function ratios(series, key, base, r, samples, total, value, list) {
  samples = 0; total = 0
  for (r = 1; r <= runs; ++r) {
    if (!(series[key, r] > 0 && series[base, r] > 0)) continue
    value = series[key, r] / series[base, r]
    list[++samples] = value
    total += log(value)
  }
  ratio_mean = samples ? exp(total / samples) : 0
  ratio_median = median(list, samples)
}
function close_round() {
  if (name == "" || done == 0) return
  ttft[mode SUBSEP name, run] = first_sum / done
  tps[mode SUBSEP name, run] = tps_sum
  if (name == "both") { tps_one[mode, run] = tps_first; tps_two[mode, run] = tps_second }
  done = 0
}

/^BEGIN_VSHARE / { run = field("run") + 0; if (run > runs) runs = run; next }
/^CASE / {
  mode = field("mode"); if (!(mode in seen)) order[++modes] = mode
  seen[mode] = 1; name = ""; done = 0
  servers = field("servers") + 0; if (servers < 2) servers = 2
  next
}
/^SERVER / {
  slot = field("slot")
  ready[mode] += field("ready")
  startup[mode SUBSEP slot, run] = field("startup_ms")
  next
}
/^PREFIX / {
  if (field("exit") != 0) { broken[mode]++; next }
  prefix_ms[mode, run] = field("request_ms")
  prefix_tokens[mode] = field("prompt_tokens")
  next
}
/^ROUND / {
  close_round()
  name = field("name"); first_sum = 0; tps_sum = 0; tps_first = 0; tps_second = 0
  lowest_fraction[mode SUBSEP name, run] = 1
  next
}
/^AGENT / {
  started[mode]++
  if (field("exit") != 0 || field("generation_tps") == "") next
  finished[mode]++
  first_sum += field("first_token_ms")
  tps_sum += field("generation_tps")
  if (field("index") % servers % 2 == 0) tps_first += field("generation_tps")
  else tps_second += field("generation_tps")
  done++
  cached = field("prompt_cached") + 0
  if (cached < 0) cached = 0
  fraction = field("prompt_tokens") > 0 ? cached / field("prompt_tokens") : 0
  if (fraction < lowest_fraction[mode SUBSEP name, run]) lowest_fraction[mode SUBSEP name, run] = fraction
  text[mode, run, name, field("index")] = field("text")
  next
}
/^MEMORY / {
  if (field("at") == "ready") { at_ready[mode, run] = field("drop_mib"); next }
  close_round()
  memory[mode, run] = field("drop_mib")
  peak[mode, run] = field("peak_drop_mib")
  files[mode, run] = field("files_mib")
  next
}
/^PLUGIN / {
  if ($3 == "weights_attach") {
    weights_ms[mode SUBSEP field("slot"), run] = field("ms")
    if (field("slot") == 2) shared_weights[mode, run] = field("bytes") / 1048576
  }
  if ($3 == "weights_publish") publish_ms[mode, run] = field("ms")
  if ($3 == "kv_cache" && field("slot") == 2) {
    kv_ms[mode, run] = field("ms")
    shared_kv[mode, run] = field("blocks") * field("block_bytes") / 1048576
  }
  if ($3 == "kv_publish") published[mode, run] = field("blocks")
  next
}
/^LMCACHE / {
  cost = $0; sub(/^.* cost /, "", cost); sub(/ ms.*$/, "", cost)
  if ($3 == "Stored" && field("slot") == 1) store_ms[mode, run] += cost
  if ($3 == "Retrieved" && field("slot") == 2 && !((mode, run) in retrieve_ms)) retrieve_ms[mode, run] = cost
  next
}
/^END_CASE / { close_round(); cases[mode]++; failed[mode] += field("failed"); next }

END {
  print "mode,servers,runs,failed_cases,servers_ready,requests_finished,requests_started," \
        "startup1_ms,startup2_ms,prefix_ms,prefix_tokens,first_token_ms,first_token_ci95," \
        "first_token_vs_vllm,first_cached_lowest,first_cached_lowest_of_any_run,both_first_token_ms,both_tps_sum," \
        "both_tps_ci95,both_tps_vs_vllm,both_tps_vs_vllm_median,tps_first_instance,tps_second_instance," \
        "ready_memory_mib,memory_mib,memory_ci95,memory_minus_vllm_mib,peak_memory_mib," \
        "files_mib,shared_weights_mib,shared_kv_mib,published_blocks,weights_publish_ms," \
        "weights_attach_ms,kv_attach_ms,lmcache_store_ms,lmcache_retrieve_ms," \
        "first_texts_equal,first_texts_compared,both_texts_equal,both_texts_compared," \
        "alone1_texts_equal,alone1_texts_compared,alone2_texts_equal,alone2_texts_compared"
  for (m = 1; m <= modes; ++m) {
    mode = order[m]
    for (key in text) {
      split(key, part, SUBSEP)
      if (part[1] != mode) continue
      base = "vllm" SUBSEP part[2] SUBSEP part[3] SUBSEP part[4]
      if (!(base in text)) continue
      compared[mode, part[3]]++
      if (text[key] == text[base]) equal[mode, part[3]]++
    }
    # The smallest cached share of a prompt in round `first`: over all
    # repetitions, and the largest of the smallest shares of each repetition.
    lowest = 1; largest_lowest = 0
    for (r = 1; r <= runs; ++r) {
      if (!(((mode SUBSEP "first"), r) in lowest_fraction)) continue
      if (lowest_fraction[mode SUBSEP "first", r] < lowest) lowest = lowest_fraction[mode SUBSEP "first", r]
      if (lowest_fraction[mode SUBSEP "first", r] > largest_lowest) largest_lowest = lowest_fraction[mode SUBSEP "first", r]
    }
    printf "%s,%d,%d,%d,%d,%d,%d,", mode, servers, cases[mode], failed[mode] + broken[mode], ready[mode], \
           finished[mode], started[mode]
    spread(startup, mode SUBSEP 1); printf "%.0f,", spread_mean
    spread(startup, mode SUBSEP 2); printf "%.0f,", spread_mean
    spread(prefix_ms, mode); printf "%.0f,%d,", spread_mean, prefix_tokens[mode]
    spread(ttft, mode SUBSEP "first"); printf "%.0f,%.0f,", spread_mean, spread_ci
    ratios(ttft, mode SUBSEP "first", "vllm" SUBSEP "first"); printf "%.4f,%.4f,%.4f,", ratio_mean, lowest, largest_lowest
    spread(ttft, mode SUBSEP "both"); printf "%.0f,", spread_mean
    spread(tps, mode SUBSEP "both"); printf "%.2f,%.2f,", spread_mean, spread_ci
    ratios(tps, mode SUBSEP "both", "vllm" SUBSEP "both"); printf "%.4f,%.4f,", ratio_mean, ratio_median
    spread(tps_one, mode); printf "%.2f,", spread_mean
    spread(tps_two, mode); printf "%.2f,", spread_mean
    spread(at_ready, mode); printf "%.0f,", spread_mean
    spread(memory, mode); printf "%.0f,%.0f,", spread_mean, spread_ci
    own = spread_mean
    difference = 0; pairs = 0
    for (r = 1; r <= runs; ++r)
      if (((mode, r) in memory) && (("vllm", r) in memory)) { difference += memory[mode, r] - memory["vllm", r]; pairs++ }
    printf "%.0f,", (pairs ? difference / pairs : 0)
    spread(peak, mode); printf "%.0f,", spread_mean
    spread(files, mode); printf "%.0f,", spread_mean
    spread(shared_weights, mode); printf "%.0f,", spread_mean
    spread(shared_kv, mode); printf "%.0f,", spread_mean
    spread(published, mode); printf "%.0f,", spread_mean
    spread(publish_ms, mode); printf "%.0f,", spread_mean
    spread(weights_ms, mode SUBSEP 2); printf "%.0f,", spread_mean
    spread(kv_ms, mode); printf "%.0f,", spread_mean
    spread(store_ms, mode); printf "%.0f,", spread_mean
    spread(retrieve_ms, mode); printf "%.0f,", spread_mean
    printf "%d,%d,%d,%d,%d,%d,%d,%d\n", equal[mode, "first"], compared[mode, "first"], equal[mode, "both"], \
           compared[mode, "both"], equal[mode, "alone1"], compared[mode, "alone1"], equal[mode, "alone2"], \
           compared[mode, "alone2"]
  }
}
