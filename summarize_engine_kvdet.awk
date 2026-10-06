# One row per prefix length: how many distinct caches each MIG instance
# computed over the repetitions, in how many repetitions the caches of the
# two instances were the same bits, and how far apart they were.
function field(name, i, pair) {
  for (i = 1; i <= NF; ++i) { split($i, pair, "="); if (pair[1] == name) return pair[2] }
  return ""
}
/^BEGIN_KVDET / { size = field("paragraphs") + 0; sizes[size] = 1; runs[size]++; next }
/^CACHE / {
  instance = field("instance")
  tokens[size] = field("prefix_tokens")
  if (field("prefix_tokens") == "") { failed[size]++; next }
  hash = field("sha256")
  if (!((size, instance, hash) in seen)) { seen[size, instance, hash] = 1; distinct[size, instance]++ }
  next
}
/^ACROSS / {
  if (field("bytes_differ") == "") { failed[size]++; next }
  if (field("bytes_differ") == 0) same[size]++
  values[size] += field("values_differ")
  if (field("largest_difference") + 0 > largest[size]) largest[size] = field("largest_difference") + 0
  if (field("bytes_differ") > 0) {
    if (!(size in first) || field("first_tensor") + 0 < first[size]) first[size] = field("first_tensor") + 0
  }
  next
}
END {
  print "paragraphs,prefix_tokens,runs,failed,distinct_caches_instance_a," \
        "distinct_caches_instance_b,runs_with_equal_caches_across_instances," \
        "values_that_differ_mean,largest_difference,first_tensor_that_differs"
  for (size = 1; size <= 100000; ++size) {
    if (!(size in sizes)) continue
    n = runs[size]
    printf "%d,%d,%d,%d,%d,%d,%d,%.0f,%.6g,%s\n", size, tokens[size], n, \
           failed[size] + 0, distinct[size, "a"], distinct[size, "b"], \
           same[size] + 0, values[size] / n, largest[size] + 0, \
           (size in first) ? first[size] : "-"
  }
}
