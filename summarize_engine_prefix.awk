# One row per prefix length: tokens of the prefix, the time of a process that
# recomputes it, the time of one that restores it from the prompt-cache file,
# and the size of that file.
function field(name, i, pair) {
  for (i = 1; i <= NF; ++i) { split($i, pair, "="); if (pair[1] == name) return pair[2] }
  return ""
}
/^BEGIN_ENGINE_PREFIX / { key = field("paragraphs") + 0; count[key]++; next }
/^CASE mode=recompute / { recompute[key] += field("wall_ms"); next }
/^CASE mode=save / { cache[key] += field("cache_bytes"); next }
/^CASE mode=restore / { restore[key] += field("wall_ms"); next }
/^PERF recompute +prompt eval time/ {
  for (i = 1; i <= NF; ++i) if ($i == "tokens" && $(i - 2) == "/") tokens[key] += $(i - 1)
  prefill[key] += $7
  next
}
/^PERF restore +prompt eval time/ {
  for (i = 1; i <= NF; ++i) if ($i == "tokens" && $(i - 2) == "/") restored_eval[key] += $(i - 1)
  next
}
END {
  print "paragraphs,runs,prefix_tokens,recompute_wall_ms,prefill_ms," \
        "restore_wall_ms,tokens_evaluated_after_restore,cache_mib,cache_kib_per_token"
  for (key = 1; key <= 100000; ++key) {
    if (!(key in count)) continue
    n = count[key]
    t = tokens[key] / n
    per_token = 0
    if (t > 0) per_token = cache[key] / n / 1024 / t
    printf "%d,%d,%.0f,%.0f,%.0f,%.0f,%.0f,%.1f,%.1f\n", key, n, t, \
           recompute[key] / n, prefill[key] / n, restore[key] / n, \
           restored_eval[key] / n, cache[key] / n / 1048576, per_token
  }
}
