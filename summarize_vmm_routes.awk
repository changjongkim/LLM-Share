# One row per placement: in how many attempts the consumer composed a range
# out of the producer's memory and its own, where the interface refused when
# it did not, what read-only access to the shared half did to a write by the
# consumer, and in how many attempts the producer's data was unchanged
# afterwards.
/^BEGIN_VMM / { seen = 0; next }
/^probe=cuda_vmm / {
  for (i = 1; i <= NF; ++i) { split($i, pair, "="); probe[pair[1]] = pair[2] }
  seen = 1
}
/^END_VMM / {
  for (i = 1; i <= NF; ++i) { split($i, pair, "="); value[pair[1]] = pair[2] }
  key = value["placement"]
  runs[key]++
  if (!seen) { note[key] = "no_result_exit_" value["exit_status"]; next }
  granularity[key] = probe["granularity"]
  if (probe["result"] == "PASS") {
    works[key]++
    readonly[key] = probe["readonly"]
    if (probe["consumer_changed_shared"] == 0) intact[key]++
  } else {
    note[key] = probe["result"] "_at_" probe["stage"] "_" probe["error"]
  }
}
END {
  print "placement,runs,works,refusal,granularity_bytes,read_only_access,producer_intact"
  split("same_instance across_instances same_mps_server", order, " ")
  for (i = 1; i <= 3; ++i) {
    key = order[i]
    if (!(key in runs)) continue
    printf "%s,%d,%d,%s,%s,%s,%d\n", key, runs[key], works[key] + 0, \
           (key in note) ? note[key] : "-", granularity[key], \
           (key in readonly) ? readonly[key] : "-", intact[key] + 0
  }
}
