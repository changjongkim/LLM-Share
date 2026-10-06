# One row per route and placement: how many attempts shared memory between
# the two processes, and how CUDA IPC refused when it did.
  /^BEGIN_ROUTE / { shared = 0; passed = 0; detail = "-"; next }
  /^probe=cuda_ipc / {
    for (i = 1; i <= NF; ++i) { split($i, pair, "="); ipc[pair[1]] = pair[2] }
    if (ipc["result"] == "PASS") passed = 1
    else detail = ipc["result"] "_at_" ipc["stage"] "_" ipc["error"]
  }
  # The host route reports the memory of both tenants; one copy of 1 GiB plus
  # process overhead is below 1.5 GiB.
  /^probe=sharing_perf / {
    for (i = 1; i <= NF; ++i) {
      split($i, pair, "=")
      if (pair[1] == "memory_mib" && pair[2] < 1536) shared = 1
      if (pair[1] == "sums_ok" && pair[2] != 1) shared = 0
    }
  }
  /^END_ROUTE / {
    for (i = 1; i <= NF; ++i) { split($i, pair, "="); value[pair[1]] = pair[2] }
    key = value["route"] "," value["placement"]
    runs[key]++
    if (value["route"] == "cuda_ipc" ? passed : (shared && value["exit_status"] == 0)) works[key]++
    if (detail != "-") note[key] = detail
  }
  END {
    print "route,placement,runs,works,refusal"
    split("cuda_ipc,same_instance cuda_ipc,across_instances cuda_ipc,same_mps_server " \
          "host_page_table,same_instance host_page_table,across_instances " \
          "host_page_table,same_mps_server", order, " ")
    for (i = 1; i <= 6; ++i) {
      key = order[i]
      printf "%s,%d,%d,%s\n", key, runs[key], works[key] + 0, (key in note) ? note[key] : "-"
    }
  }