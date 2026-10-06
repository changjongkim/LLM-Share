#!/usr/bin/env bash
# Verifies the packaged engine evidence without running the GPU: every
# summary is rebuilt from its raw log and compared, the pinned sources are
# checked against the tree, and the relations the research record states are
# re-evaluated.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
single_dir=${1:-"$script_dir/results/20261005-engine-single-v1"}
agents_dir=${2:-"$script_dir/results/20261005-engine-agents-v1"}
pages_dir=${3:-"$script_dir/results/20261005-engine-pages-v1"}
small_pages_dir=${4:-"$script_dir/results/20261005-engine-pages-small-v1"}
adapters_dir=${5:-"$script_dir/results/20261005-engine-adapters-v1"}
batched_dir=${6:-"$script_dir/results/20261005-engine-batched-v1"}
routes_dir=${7:-"$script_dir/results/20261005-sharing-routes-v2"}
groups_dir=${8:-"$script_dir/results/20261005-engine-groups-v1"}
prefix_dir=${9:-"$script_dir/results/20261005-engine-prefix-v1"}
kvshare_dir=${10:-"$script_dir/results/20261006-engine-kvshare-v1"}
kvfork_dir=${11:-"$script_dir/results/20261006-engine-kvfork-v1"}
kvdet_dir=${12:-"$script_dir/results/20261006-engine-kvdet-v1"}
vmm_dir=${13:-"$script_dir/results/20261006-vmm-routes-v1"}
kvspeed_dir=${14:-"$script_dir/results/20261006-engine-kvspeed-v1"}
kvcow_dir=${15:-"$script_dir/results/20261006-engine-kvcow-v1"}
kvspeed_long_dir=${16:-"$script_dir/results/20261006-engine-kvspeed-long-v1"}
kvspeed_small_dir=${17:-"$script_dir/results/20261006-engine-kvspeed-6sm-v1"}
kvspeed_small_long_dir=${18:-"$script_dir/results/20261006-engine-kvspeed-6sm-long-v1"}
kvbatch_dir=${19:-"$script_dir/results/20261006-engine-kvbatch-v1"}
kvvmm_dir=${20:-"$script_dir/results/20261006-engine-kvvmm-v1"}

fail() {
  echo "LLM-share artifact verification failed: $*" >&2
  exit 1
}
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT

# Sources of the substrate project (../thor_hostmm) are pinned by one result.
# They are checked when that project lies next to this one, and counted as
# skipped when it does not.
: >"$scratch/outside"
check_hashes() {
  [[ -f "$1/source_hashes.txt" ]] || fail "missing hashes in $1"
  local hash path
  : >"$scratch/hashes"
  while read -r hash path; do
    if [[ "$path" == ../* && ! -e "$script_dir/$path" ]]; then
      printf '%s\n' "$path" >>"$scratch/outside"
    else
      printf '%s  %s\n' "$hash" "$path" >>"$scratch/hashes"
    fi
  done <"$1/source_hashes.txt"
  (cd "$script_dir" && sha256sum -c "$scratch/hashes" >/dev/null) ||
    fail "pinned sources of $1 do not match the tree"
}
same() {
  cmp -s "$1" "$2" || fail "$2 is not reproducible from its raw log"
}

# The patch in the tree must be the difference between the engine checkout
# and its recorded commit.
if [[ -d "$script_dir/llama.cpp/.git" ]]; then
  git -C "$script_dir/llama.cpp" diff >"$scratch/patch"
  same "$scratch/patch" "$script_dir/inplace_weights.patch"
fi
if [[ -d "$script_dir/llama.cpp-kv/.git" ]]; then
  git -C "$script_dir/llama.cpp-kv" diff >"$scratch/kv_patch"
  same "$scratch/kv_patch" "$script_dir/kv_extents.patch"
fi
if [[ -d "$script_dir/llama.cpp-vmm/.git" ]]; then
  git -C "$script_dir/llama.cpp-vmm" diff >"$scratch/vmm_patch"
  same "$scratch/vmm_patch" "$script_dir/kv_vmm.patch"
fi

# --- one process -------------------------------------------------------------
check_hashes "$single_dir"
awk -f "$script_dir/summarize_engine_single.awk" "$single_dir/raw.log" \
  >"$scratch/single.csv"
same "$scratch/single.csv" "$single_dir/engine_single_summary.csv"
model_mib=$(awk -F= '$1 == "model_bytes" { printf "%d", $2 / 1048576 }' \
  "$single_dir/metadata.txt")
awk -F, -v model="$model_mib" '
  NR == 1 { next }
  { rows++; tps[$1] = $6; lower[$1] = $8; device[$1] = $11; mapped[$1] = $12 }
  $3 != 0 || $4 != $2 { bad++ }
  END {
    if (bad || rows != 3) exit 2
    # Reading in place must keep at least 90% of the generation speed, with
    # the confidence interval, and must free at least 90% of the model size
    # in device memory while mapping the model instead.
    if (lower["inplace"] < 0.90) exit 3
    if (device["copy"] - device["inplace"] < 0.9 * model) exit 4
    if (mapped["inplace"] < 0.9 * model || mapped["copy"] > 0.1 * model) exit 5
  }
' "$single_dir/engine_single_summary.csv" ||
  fail "single-process relations do not hold or a run failed"

# --- N processes under each compute-sharing configuration --------------------
check_hashes "$agents_dir"
awk -f "$script_dir/summarize_engine_agents.awk" "$agents_dir/raw.log" \
  >"$scratch/agents.csv"
same "$scratch/agents.csv" "$agents_dir/engine_agents_summary.csv"
awk -F, -v model="$model_mib" '
  NR == 1 { next }
  { rows++; total[$1 FS $2 FS $3] = $12; tps[$1 FS $2 FS $3] = $7 }
  $5 != 0 { bad++ }
  END {
    if (bad || rows != 32) exit 2
    split("mig timeslice mps mig_mps", configs, " ")
    for (c = 1; c <= 4; ++c) {
      copy = total[configs[c] FS "copy" FS 8]
      inplace = total[configs[c] FS "inplace" FS 8]
      # Eight agents reading in place must hold at least five model sizes
      # less memory than eight device copies.
      if (copy - inplace < 5 * model) exit 3
      # and must keep at least 85% of the total generation rate.
      if (tps[configs[c] FS "inplace" FS 8] < 0.85 * tps[configs[c] FS "copy" FS 8]) exit 4
    }
  }
' "$agents_dir/engine_agents_summary.csv" ||
  fail "multi-agent relations do not hold or a run failed"

# --- page size of the mapped model ------------------------------------------
check_hashes "$pages_dir"
awk -f "$script_dir/summarize_engine_pages.awk" "$pages_dir/raw.log" \
  >"$scratch/pages.csv"
same "$scratch/pages.csv" "$pages_dir/engine_pages_summary.csv"
awk -F, '
  NR == 1 { next }
  { rows++ }
  $3 != 0 || $4 != $2 { bad++ }
  END { exit (bad || rows != 4) ? 2 : 0 }
' "$pages_dir/engine_pages_summary.csv" ||
  fail "a page-size run failed or produced different text"

# In both models the 2 MiB mappings must not be slower than the 4 KiB one,
# and hugetlbfs must stay within 5% of the device copy.
for directory in "$pages_dir" "$small_pages_dir"; do
  [[ "$directory" == "$pages_dir" ]] || {
    check_hashes "$directory"
    awk -f "$script_dir/summarize_engine_pages.awk" "$directory/raw.log" \
      >"$scratch/pages_small.csv"
    same "$scratch/pages_small.csv" "$directory/engine_pages_summary.csv"
  }
  awk -F, '
    NR == 1 { next }
    $3 != 0 || $4 != $2 { bad++ }
    { ratio[$1] = $7; rows++ }
    END {
      if (bad || rows != 4) exit 2
      if (ratio["inplace_thp"] < ratio["inplace_4k"]) exit 3
      if (ratio["inplace_hugetlb"] < ratio["inplace_4k"]) exit 3
      if (ratio["inplace_hugetlb"] < 0.95) exit 4
    }
  ' "$directory/engine_pages_summary.csv" ||
    fail "page-size relations do not hold in $directory"
done

# --- adapters over one base ---------------------------------------------------
check_hashes "$adapters_dir"
awk -f "$script_dir/summarize_engine_adapters.awk" "$adapters_dir/raw.log" \
  >"$scratch/adapters.csv"
same "$scratch/adapters.csv" "$adapters_dir/engine_adapters_summary.csv"
awk -F, -v model="$model_mib" '
  NR == 1 { next }
  $3 != 0 || $4 != $2 || $5 != $6 { bad++ }
  { total[$1] = $9; rows++ }
  END {
    if (bad || rows != 2) exit 2
    # Five processes in place must hold at least three model sizes less.
    if (total["copy"] - total["inplace"] < 3 * model) exit 3
  }
' "$adapters_dir/engine_adapters_summary.csv" ||
  fail "adapter texts differ from the device copy or the memory relation fails"

# --- one process with N sequences ----------------------------------------------
check_hashes "$batched_dir"
awk -f "$script_dir/summarize_engine_batched.awk" "$batched_dir/raw.log" \
  >"$scratch/batched.csv"
same "$scratch/batched.csv" "$batched_dir/engine_batched_summary.csv"
grep -qx 'failed_runs,0' "$batched_dir/engine_batched_summary.csv" ||
  fail "a batched run failed"

# --- which route can share ----------------------------------------------------
check_hashes "$routes_dir"
awk -f "$script_dir/summarize_sharing_routes.awk" "$routes_dir/raw.log" \
  >"$scratch/routes.csv"
same "$scratch/routes.csv" "$routes_dir/sharing_routes_summary.csv"
awk -F, '
  NR == 1 { next }
  { rows++ }
  # The host page table shares in every placement.
  $1 == "host_page_table" && $4 != $3 { bad++ }
  END { exit (bad || rows != 6) ? 2 : 0 }
' "$routes_dir/sharing_routes_summary.csv" ||
  fail "the host page table did not share in every placement"

# --- a batching server per MIG instance ----------------------------------------
check_hashes "$groups_dir"
awk -f "$script_dir/summarize_engine_groups.awk" "$groups_dir/raw.log" \
  >"$scratch/groups.csv"
same "$scratch/groups.csv" "$groups_dir/engine_groups_summary.csv"
awk -F, -v model="$model_mib" '
  NR == 1 { next }
  $4 != 0 { bad++ }
  { rate[$1 FS $2] = $5; memory[$1 FS $2] = $11; rows++ }
  END {
    if (bad || rows != 4) exit 2
    split("4 8", sizes, " ")
    for (s = 1; s <= 2; ++s) {
      # In place the two servers keep at least 95% of the rate and hold at
      # least 0.9 model sizes less.
      if (rate["inplace" FS sizes[s]] < 0.95 * rate["copy" FS sizes[s]]) exit 3
      if (memory["copy" FS sizes[s]] - memory["inplace" FS sizes[s]] < 0.9 * model) exit 4
    }
  }
' "$groups_dir/engine_groups_summary.csv" ||
  fail "the two-server relations do not hold or a run failed"

# --- a shared prompt prefix ----------------------------------------------------
check_hashes "$prefix_dir"
awk -f "$script_dir/summarize_engine_prefix.awk" "$prefix_dir/raw.log" \
  >"$scratch/prefix.csv"
same "$scratch/prefix.csv" "$prefix_dir/engine_prefix_summary.csv"
awk -F, '
  NR == 1 { next }
  { rows++ }
  # The cache file holds about 56 KiB per token, and restoring evaluates one
  # token; for the long prefix restoring is faster than recomputing.
  $9 < 50 || $9 > 62 || $7 > 2 { bad++ }
  $3 > 3000 && $6 >= $4 { bad++ }
  END { exit (bad || rows != 2) ? 2 : 0 }
' "$prefix_dir/engine_prefix_summary.csv" ||
  fail "the prefix baseline is not as recorded"

# --- agents that start from one computed prefix --------------------------------
check_hashes "$kvshare_dir"
awk -f "$script_dir/summarize_engine_kvshare.awk" "$kvshare_dir/raw.log" \
  >"$scratch/kvshare.csv"
same "$scratch/kvshare.csv" "$kvshare_dir/engine_kvshare_summary.csv"
awk -v table=publish -f "$script_dir/summarize_engine_kvshare.awk" \
  "$kvshare_dir/raw.log" >"$scratch/kvshare_publish.csv"
same "$scratch/kvshare_publish.csv" "$kvshare_dir/engine_kvshare_publish.csv"
awk -F, '
  NR == 1 { next }
  {
    rows++
    key = $1 SUBSEP $3
    attach[key, $2] = $13
    # No agent fails. Every agent writes the text of the agent that obtains
    # the same state by copy, and of the agent that recomputes the prefix
    # when both run in the MIG instance of the publisher. Agents in the other
    # instance are not required to match recomputation there: the campaign
    # stated that as a gate and it does not hold, for the copy either.
    if ($5 != 0 || $8 != $4 * $3 || $9 != $10) bad++
    if ($2 != "recompute" && $7 != $8) bad++
  }
  $2 ~ /^extent/ {
    # The agents hold one copy of the prefix between them: against agents
    # that copy, they save at least 80% of a prefix each, and the pages of
    # the cache file they map amount to one prefix however many map it.
    if ($20 < 0.8 * $3 * $21) memory_bad++
    if ($22 > 1.15 * $21 || $23 < 0.75 * $3 * $21) shared_bad++
  }
  $2 == "extent_lazy" {
    lazy++
    # Generation: the gate of the campaign was 97% of the speed of agents
    # that copy. It holds in four of the six cells; 95% holds in all.
    if ($16 < 0.95) slow++
    if ($16 < 0.97) below++
    # Attaching does not take longer than copying.
    if ($13 > attach[key, "restore"]) late++
  }
  END {
    if (bad || rows == 0 || rows % 7 != 0) exit 2
    if (memory_bad) exit 3
    if (shared_bad) exit 4
    if (slow || lazy != 6 || below != 2) exit 5
    if (late) exit 6
  }
' "$kvshare_dir/engine_kvshare_summary.csv" ||
  fail "the prefix-sharing relations do not hold or an agent failed (awk exit $?)"
awk -F, '
  NR == 1 { next }
  { rows++; publish[$1, $2] = $7; state[$1, $2] = $8; sizes[$1] = 1 }
  $4 != 0 { bad++ }
  END {
    for (size in sizes) {
      # Publishing is faster than saving the state, and its state file holds
      # no rows.
      if (publish[size, "huge"] >= publish[size, "device"]) bad++
      if (publish[size, "small"] >= publish[size, "device"]) bad++
      if (state[size, "huge"] > 1 || state[size, "device"] < 100) bad++
    }
    exit (bad || rows == 0 || rows % 3 != 0) ? 2 : 0
  }
' "$kvshare_dir/engine_kvshare_publish.csv" ||
  fail "publishing is not as recorded"

# --- a parent that forks its state while it keeps running ----------------------
check_hashes "$kvfork_dir"
awk -f "$script_dir/summarize_engine_kvfork.awk" "$kvfork_dir/raw.log" \
  >"$scratch/kvfork.csv"
same "$scratch/kvfork.csv" "$kvfork_dir/engine_kvfork_summary.csv"
awk -F, '
  NR == 1 { next }
  {
    rows++
    publish[$1] = $6; state[$1] = $7; attach[$1] = $16; memory[$1] = $19
    pss[$1] = $20; children = $3
    # The parent writes the text of a process that computes the same tokens
    # alone; every child writes the text of the child that received the state
    # by copy, and of a process alone when it runs in the parent instance.
    if ($4 != 0 || $8 != $2 || $10 != $2 * $3 || $11 != $10 || $12 != $13) bad++
  }
  END {
    if (bad || rows != 2) exit 2
    # Handing the state over and attaching are faster with extents, and the
    # children hold one copy of the prefix between them.
    if (publish["extent"] >= publish["copy"]) exit 3
    if (attach["extent"] > attach["copy"]) exit 4
    if (memory["copy"] - memory["extent"] < 0.8 * children * state["copy"]) exit 5
    if (pss["extent"] > 1.15 * state["copy"]) exit 6
  }
' "$kvfork_dir/engine_kvfork_summary.csv" ||
  fail "the fork relations do not hold or a process failed (awk exit $?)"

# --- the bits of a prefix cache, across repetitions and MIG instances -----------
check_hashes "$kvdet_dir"
awk -f "$script_dir/summarize_engine_kvdet.awk" "$kvdet_dir/raw.log" \
  >"$scratch/kvdet.csv"
same "$scratch/kvdet.csv" "$kvdet_dir/engine_kvdet_summary.csv"
awk -F, '
  NR == 1 { next }
  { rows++ }
  # Each instance computes one cache, the same in every repetition; the two
  # instances never compute the same one.
  $4 != 0 || $5 != 1 || $6 != 1 || $7 != 0 { bad++ }
  END { exit (bad || rows != 3) ? 2 : 0 }
' "$kvdet_dir/engine_kvdet_summary.csv" ||
  fail "the cache bits are not as recorded"

# --- composing shared and private device memory ---------------------------------
check_hashes "$vmm_dir"
awk -f "$script_dir/summarize_vmm_routes.awk" "$vmm_dir/raw.log" \
  >"$scratch/vmm.csv"
same "$scratch/vmm.csv" "$vmm_dir/vmm_routes_summary.csv"
awk -F, '
  NR == 1 { next }
  { rows++; runs[$1] = $2; works[$1] = $3; access[$1] = $6; intact[$1] = $7 }
  END {
    if (rows != 3) exit 2
    # Inside one MIG instance, also between clients of one MPS server, the
    # virtual memory interface composes the range and keeps the shared half
    # from the consumer; it does not cross MIG instances.
    if (works["same_instance"] != runs["same_instance"]) exit 3
    if (works["same_mps_server"] != runs["same_mps_server"]) exit 3
    if (access["same_instance"] != "enforced") exit 4
    if (intact["same_instance"] != runs["same_instance"]) exit 4
    if (works["across_instances"] != 0) exit 5
  }
' "$vmm_dir/vmm_routes_summary.csv" ||
  fail "the device-memory route is not as recorded (awk exit $?)"

# --- generation speed with the cache in device and in host memory ---------------
check_hashes "$kvspeed_dir"
awk -f "$script_dir/summarize_engine_kvspeed.awk" "$kvspeed_dir/raw.log" \
  >"$scratch/kvspeed.csv"
same "$scratch/kvspeed.csv" "$kvspeed_dir/engine_kvspeed_summary.csv"
awk -F, '
  NR == 1 { next }
  { rows++ }
  $4 != 0 { bad++ }
  # Over 256 tokens a cache in host memory on 2 MiB pages, and a cache on
  # extents, keep 99% of the generation speed of a cache in device memory,
  # with one process and with two that are time-sliced; so does the copy,
  # which is device memory.
  ($2 == "anon" || $2 == "extent" || $2 == "extent_lazy" || $2 == "restore") && $7 < 0.99 { bad++ }
  END { exit (bad || rows != 12) ? 2 : 0 }
' "$kvspeed_dir/engine_kvspeed_summary.csv" ||
  fail "the generation speed of a host-memory cache is not as recorded"

# The same with the long prefix, and in the 6-SM instance: there a cache in
# host memory is slower than one in device memory, shared or not, by about
# 2% with the short prefix and about 6% with the long one.
speed_check() {  # directory lowest highest
  check_hashes "$1"
  awk -f "$script_dir/summarize_engine_kvspeed.awk" "$1/raw.log" >"$scratch/speed.csv"
  same "$scratch/speed.csv" "$1/engine_kvspeed_summary.csv"
  awk -F, -v low="$2" -v high="$3" '
    NR == 1 { next }
    { rows++ }
    $4 != 0 { bad++ }
    $2 == "restore" && ($7 < 0.99 || $7 > 1.01) { bad++ }
    ($2 == "anon" || $2 == "extent" || $2 == "extent_lazy") && ($7 < low || $7 > high) { bad++ }
    END { exit (bad || rows != 6) ? 2 : 0 }
  ' "$1/engine_kvspeed_summary.csv" ||
    fail "the generation speed in $1 is not as recorded"
}
speed_check "$kvspeed_long_dir" 0.99 1.02
speed_check "$kvspeed_small_dir" 0.96 0.99
speed_check "$kvspeed_small_long_dir" 0.92 0.96

# --- copy-on-write as the way to obtain a prefix ---------------------------------
check_hashes "$kvcow_dir"
awk -f "$script_dir/summarize_engine_kvcow.awk" "$kvcow_dir/raw.log" \
  >"$scratch/kvcow.csv"
same "$scratch/kvcow.csv" "$kvcow_dir/engine_kvcow_summary.csv"
awk -F, '
  NR == 1 { next }
  {
    rows++
    first[$3, $2] = $11
    # No agent fails, and every agent writes the text of the agent that copies.
    if ($5 != 0 || $7 != $8 || $8 != $4 * $3) bad++
  }
  # Without the CPU read pass every agent ends with a private copy of the
  # prefix; with it the agent copies a small part.
  $2 ~ /_noread$/ && $17 < 0.9 * $18 { copies_bad++ }
  ($2 == "cow" || $2 == "cow_small") && $17 > 0.3 * $18 { copies_bad++ }
  END {
    if (bad || rows != 12) exit 2
    if (copies_bad) exit 3
    for (key in first) {
      split(key, part, SUBSEP)
      # Extents reach the first token before every copy-on-write mapping,
      # and a mapping without the read pass needs at least 1.5 times as long.
      if (part[2] ~ /^cow/ && first[key] < first[part[1], "extent_lazy"]) exit 4
      if (part[2] ~ /_noread$/ && first[key] < 1.5 * first[part[1], "extent_lazy"]) exit 5
    }
  }
' "$kvcow_dir/engine_kvcow_summary.csv" ||
  fail "the copy-on-write relations do not hold or an agent failed (awk exit $?)"

# --- agents that may share a process ---------------------------------------------
check_hashes "$kvbatch_dir"
awk -f "$script_dir/summarize_engine_kvbatch.awk" "$kvbatch_dir/raw.log" \
  >"$scratch/kvbatch.csv"
same "$scratch/kvbatch.csv" "$kvbatch_dir/engine_kvbatch_summary.csv"
awk -F, '
  NR == 1 { next }
  {
    rows++
    tps[$1] = $7; ready[$1] = $6; memory[$1] = $13
    if ($3 != 0) bad++
  }
  $1 == "two_servers_extent" && ($14 != $15 || $15 != $2 * 8) { bad++ }
  END {
    if (bad || rows != 4) exit 2
    # A server in each MIG instance generates more than one server.
    if (tps["two_servers_extent"] < 1.25 * tps["one_server"]) exit 3
    # With extents the two servers hold less than 60% of what two servers
    # that copy hold, and no more than 110% of the single server; they keep
    # 95% of the speed of the two that copy.
    if (memory["two_servers_extent"] > 0.6 * memory["two_servers_copy"]) exit 4
    if (memory["two_servers_extent"] > 1.1 * memory["one_server"]) exit 4
    if (tps["two_servers_extent"] < 0.95 * tps["two_servers_copy"]) exit 5
    # All agents are ready sooner than with a copy, and a copy is sooner than
    # a second computation.
    if (ready["two_servers_extent"] >= ready["two_servers_copy"]) exit 6
    if (ready["two_servers_copy"] >= ready["two_servers_compute"]) exit 6
  }
' "$kvbatch_dir/engine_kvbatch_summary.csv" ||
  fail "the batching-server relations do not hold or a server failed (awk exit $?)"

# --- host extents against a cache in shared device memory ------------------------
check_hashes "$kvvmm_dir"
awk -f "$script_dir/summarize_engine_kvvmm.awk" "$kvvmm_dir/raw.log" \
  >"$scratch/kvvmm.csv"
same "$scratch/kvvmm.csv" "$kvvmm_dir/engine_kvvmm_summary.csv"
awk -F, '
  NR == 1 { next }
  {
    rows++
    key = $1 SUBSEP $2
    cells[key] = 1
    publish[key, $3] = $6; attach[key, $3] = $13; first[key, $3] = $14
    speed[key, $3] = $15; memory[key, $3] = $16
  }
  $3 == "copy" || $3 == "extent" || $3 == "vmm" {
    # Parent and children run, and write the texts of the copy.
    if ($9 != $4 || $10 != $4 * 4 || $11 != $10 || $12 != $11) bad++
  }
  # A child in the other MIG instance runs on host extents and is refused
  # the device memory.
  $3 == "extent_cross" && ($10 != $4 || $11 != $10) { bad++ }
  $3 == "vmm_cross" && ($10 != $4 || $11 != 0) { bad++ }
  END {
    if (bad || rows != 20) exit 2
    for (key in cells) {
      # Host extents attach in a quarter of the time of the device-memory
      # cache or less, reach the first token sooner, and hold less; both
      # hold less than the copy and pause the parent for less.
      if (attach[key, "extent"] > 0.25 * attach[key, "vmm"]) exit 3
      if (first[key, "extent"] >= first[key, "vmm"]) exit 3
      if (memory[key, "extent"] >= memory[key, "vmm"]) exit 4
      if (memory[key, "vmm"] >= memory[key, "copy"]) exit 4
      if (publish[key, "extent"] >= publish[key, "vmm"]) exit 5
      if (publish[key, "vmm"] >= publish[key, "copy"]) exit 5
      # The device-memory cache generates at the speed of the copy; host
      # extents keep at least 90% of it.
      if (speed[key, "vmm"] < 0.98 * speed[key, "copy"]) exit 6
      if (speed[key, "extent"] < 0.90 * speed[key, "copy"]) exit 6
    }
  }
' "$kvvmm_dir/engine_kvvmm_summary.csv" ||
  fail "the device-memory comparison does not hold or a process failed (awk exit $?)"

echo "llm_share_artifact_verification=PASS"
echo "substrate_sources_skipped=$(sort -u "$scratch/outside" | wc -l)"
echo "single_runs=$(grep -c '^BEGIN_ENGINE_SINGLE' "$single_dir/raw.log")"
echo "agent_runs=$(grep -c '^BEGIN_ENGINE_AGENTS' "$agents_dir/raw.log")"
echo "page_runs=$(grep -c '^BEGIN_ENGINE_PAGES' "$pages_dir/raw.log")"
echo "adapter_runs=$(grep -c '^BEGIN_ENGINE_ADAPTERS' "$adapters_dir/raw.log")"
echo "batched_runs=$(grep -c '^BEGIN_ENGINE_BATCHED' "$batched_dir/raw.log")"
echo "group_runs=$(grep -c '^BEGIN_ENGINE_GROUPS' "$groups_dir/raw.log")"
echo "prefix_runs=$(grep -c '^BEGIN_ENGINE_PREFIX' "$prefix_dir/raw.log")"
echo "route_attempts=$(grep -c '^BEGIN_ROUTE' "$routes_dir/raw.log")"
echo "prefix_sharing_cases=$(grep -c '^CASE ' "$kvshare_dir/raw.log")"
echo "fork_cases=$(grep -c '^CASE ' "$kvfork_dir/raw.log")"
echo "cache_bit_runs=$(grep -c '^BEGIN_KVDET ' "$kvdet_dir/raw.log")"
echo "device_route_attempts=$(grep -c '^BEGIN_VMM ' "$vmm_dir/raw.log")"
echo "speed_cases=$(cat "$kvspeed_dir/raw.log" "$kvspeed_long_dir/raw.log" \
  "$kvspeed_small_dir/raw.log" "$kvspeed_small_long_dir/raw.log" | grep -c '^CASE ')"
echo "copy_on_write_cases=$(grep -c '^CASE ' "$kvcow_dir/raw.log")"
echo "batching_server_cases=$(grep -c '^CASE ' "$kvbatch_dir/raw.log")"
echo "device_memory_cases=$(grep -c '^CASE ' "$kvvmm_dir/raw.log")"
