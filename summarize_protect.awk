# Protection of shared host and device mappings. The two header-and-row
# pairs retain the different fields reported for the two routes.
function field(name, i, pair) {
  for (i = 1; i <= NF; ++i) {
    split($i, pair, "=")
    if (pair[1] == name) return pair[2]
  }
  return ""
}

/^PROBE / && field("route") == "host" {
  host++
  if (field("gpu_read") == "cudaSuccess" && field("words_equal") == field("words")) read_ok++
  if (field("gpu_write") != "cudaSuccess") refused++
  errors[field("gpu_write")] = 1
  intact += field("file_intact")
  next
}
/^PROBE / && field("route") == "vmm" {
  vmm++
  if (field("importer_raise") == "CUDA_SUCCESS") raised++
  if (field("importer_write") == "CUDA_SUCCESS") written++
  unchanged += field("exporter_intact")
  next
}

END {
  names = ""
  for (name in errors) names = names (names == "" ? "" : "+") name
  print "route,attempts,gpu_reads_complete,writes_refused,write_error,shared_state_intact"
  printf "host,%d,%d,%d,%s,%d\n", host, read_ok, refused, names, intact
  print "route,attempts,importer_raised_access,importer_writes_succeeded,exporter_state_intact"
  printf "vmm,%d,%d,%d,%d\n", vmm, raised, written, unchanged
}
