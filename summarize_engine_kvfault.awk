# Faults among agents that share a prefix. One row per configuration and
# fault: over the repetitions, the agents that were started, those that were
# alive when the fault happened, those that completed, and those whose text
# is that of the case without a fault in the same configuration and
# repetition. Agent 0 is the one that is killed; `others` leaves it out. The
# write is refused when the GPU write of the injector reports an error, and
# the file is intact when its contents after the case are those before it.
function field(name, i, pair) {
  for (i = 1; i <= NF; ++i) { split($i, pair, "="); if (pair[1] == name) return pair[2] }
  return ""
}
function part_of(list, name, n, i, items, pair) {
  n = split(list, items, ",")
  for (i = 1; i <= n; ++i) { split(items[i], pair, "="); if (pair[1] == name) return pair[2] }
  return ""
}

/^BEGIN_KVFAULT / { run = field("run") + 0; next }
/^CASE / {
  config = field("config"); fault = field("fault"); key = config SUBSEP fault
  if (!(config in seen)) { seen[config] = 1; order[++configs] = config }
  next
}
/^AGENT / {
  started[key]++
  number = field("index") + 0
  if (number != 0) others_started[key]++
  if (field("completed") != 1 || field("exit") != 0) next
  completed[key]++
  if (number != 0) others_completed[key]++
  text[key, run, number] = field("text")
  next
}
/^FAULT / {
  alive[key] += field("alive_at_fault")
  if (fault == "write") {
    injected[key]++
    probe = $0
    sub(/^.* probe=/, "", probe)
    error = part_of(probe, "gpu_write")
    if (error != "" && error != "cudaSuccess") refused[key]++
    if (error == "") error = probe
    errors[key, error] = 1
  }
  next
}
/^FILE / { if (field("before") == field("after")) intact[key]++; next }
/^END_CASE / {
  cases[key]++
  timed_out[key] += field("timed_out")
  servers[key] += field("mps_servers")
  wall[key] += field("wall_s")
  next
}

END {
  print "config,fault,runs,agents_started,alive_at_fault,agents_completed," \
        "others_started,others_completed,others_with_reference_text," \
        "writes_injected,writes_refused,write_error,file_intact_runs,timed_out," \
        "mps_servers_after,wall_s"
  split("none write kill", faults, " ")
  for (c = 1; c <= configs; ++c) for (f = 1; f <= 3; ++f) {
    key = order[c] SUBSEP faults[f]
    if (!(key in cases)) continue
    base = order[c] SUBSEP "none"
    same = 0
    for (item in text) {
      split(item, part, SUBSEP)
      if (part[1] SUBSEP part[2] != key || part[4] == 0) continue
      if (text[item] == text[base SUBSEP part[3] SUBSEP part[4]]) same++
    }
    names = ""
    for (item in errors) {
      split(item, part, SUBSEP)
      if (part[1] SUBSEP part[2] == key) names = names (names == "" ? "" : "+") part[3]
    }
    n = cases[key]
    printf "%s,%s,%d,%d,%.1f,%d,%d,%d,%d,%d,%d,%s,%d,%d,%.1f,%.1f\n", order[c], \
           faults[f], n, started[key] + 0, alive[key] / n, completed[key] + 0, \
           others_started[key] + 0, others_completed[key] + 0, same, \
           injected[key] + 0, refused[key] + 0, names == "" ? "none" : names, \
           intact[key] + 0, timed_out[key] + 0, servers[key] / n, wall[key] / n
  }
}
