#!/usr/bin/env bash
# Verifies the campaigns run on the whole GPU with MIG disabled (result
# directories named *-full-v1, from 2026-10-14) without running the GPU: every
# summary is rebuilt from its raw log and compared, the pinned sources are
# checked against the tree, and the gates that each runner states in its
# header are evaluated again. A campaign that has not been run is reported as
# absent and does not fail the verification; a gate that the record reports
# as not met is listed as an expected exception with its measured value.
#
#   verify_stator_full.sh [RESULTS_DIR]
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
results=${1:-"$script_dir/results"}
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
verified=()
absent=()
exceptions=()

fail() {
  echo "STATOR full-GPU verification failed: $*" >&2
  exit 1
}
check_hashes() {
  [[ -f "$1/source_hashes.txt" ]] || fail "missing hashes in $1"
  (cd "$script_dir" && sed -E 's#  /.*/(workloads/[^/]+)$#  \1#' "$1/source_hashes.txt" |
    sha256sum -c - >/dev/null) || fail "pinned sources of $1 do not match the tree"
}
same() {
  cmp -s "$1" "$2" || fail "$2 is not reproducible from its raw log"
}
# campaign NAME: sets dir to the newest result directory of a runner on the
# whole GPU, or to nothing when it is absent.
dir=
campaign() {
  local found
  found=$(find "$results" -maxdepth 1 -type d -name "*-$1-full-v1" | sort | tail -n 1)
  if [[ -n "$found" && -f "$found/source_hashes.txt" ]]; then
    verified+=("$(basename "$found")")
    dir=$found
  else
    absent+=("$1")
    dir=
  fi
}
modes_of() {  # result dir: the modes that the runner recorded
  sed -n 's/^modes=//p' "$1/metadata.txt"
}
# A line that starts with "!" fails the verification; any other line is a
# gate that the record reports as not met, listed with its measured value.
gates() {  # label file program
  local label=$1 file=$2 program=$3 line
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    if [[ "$line" == '!'* ]]; then fail "$label: ${line#!}"; fi
    exceptions+=("${label}:${line// /_}")
  done < <(awk -F, "NR == 1 { for (i = 1; i <= NF; ++i) c[\$i] = i; next } $program" "$file")
}

# The patch of the engine with the shared-mapping baseline must be the
# difference between its checkout and tuned2.
if [[ -d "$script_dir/llama.cpp-linux/.git" ]]; then
  git -C "$script_dir/llama.cpp-linux" diff >"$scratch/linux.patch"
  same "$scratch/linux.patch" "$script_dir/kv_linux.patch"
fi

# --- the mechanisms of the kernel against STATOR, eight agents ----------------
campaign engine-kvlinux-mechanisms
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -v order="$(modes_of "$dir")" -f "$script_dir/summarize_engine_kvlinux.awk" \
    "$dir/raw.log" >"$scratch/kvlinux.csv"
  same "$scratch/kvlinux.csv" "$dir/engine_kvlinux_summary.csv"
  awk -F, '
    NR == 1 { for (i = 1; i <= NF; ++i) c[$i] = i; next }
    { m = $c["mode"]; n = $c["children"]
      fin[m] = $c["children_finished"]; st[m] = $c["children_started"]
      eq[m] = $c["texts_equal_to_copy"]; cmpd[m] = $c["texts_compared"]
      to[m] = $c["timed_out"] + $c["guard_cases"]; par[m] = $c["parents_completed"]; runs[m] = $c["runs"]
      flt[m] = $c["gpu_fault_pages"]; mem[m] = $c["children_memory_mib"]
      wc[m] = $c["writer_cases"]; wok[m] = $c["writer_succeeded"]; wch[m] = $c["writer_changed_file"] }
    END {
      split("copy host_copy ro_plain shared_rw cow_plain extent", base, " ")
      for (i = 1; i <= 6; ++i) {
        m = base[i]
        if (!(m in fin)) { print "K1 missing " m; bad = 1; continue }
        if (fin[m] != st[m] || to[m] != 0 || par[m] != runs[m]) { print "K1 " m; bad = 1 }
        if (m != "copy" && eq[m] != cmpd[m]) { print "K2 " m " " eq[m] "/" cmpd[m]; bad = 1 }
      }
      if (!(flt["extent"] < flt["ro_plain"] && flt["extent"] < flt["shared_rw"] && flt["extent"] < flt["cow_plain"])) {
        print "K3"; bad = 1 }
      if (!(mem["cow_plain"] >= 1.5 * mem["extent"] && mem["ro_plain"] <= 1.1 * mem["extent"] &&
            mem["shared_rw"] <= 1.1 * mem["extent"])) { print "K4"; bad = 1 }
      if (!(wc["shared_rw_write"] > 0 && wok["shared_rw_write"] == wc["shared_rw_write"] &&
            wch["shared_rw_write"] == wc["shared_rw_write"] && wc["extent_write"] > 0 &&
            wok["extent_write"] == 0 && wch["extent_write"] == 0)) { print "K5"; bad = 1 }
      if (eq["extent_write"] != cmpd["extent_write"] || cmpd["extent_write"] == 0) { print "K6"; bad = 1 }
      exit bad
    }' "$dir/engine_kvlinux_summary.csv" || fail "a gate of kvlinux-mechanisms does not hold"
fi

# --- the same agents by their number --------------------------------------------
campaign engine-kvlinux-scaling
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -v order="$(modes_of "$dir")" -f "$script_dir/summarize_engine_kvlinux.awk" \
    "$dir/raw.log" >"$scratch/kvlinux-scaling.csv"
  same "$scratch/kvlinux-scaling.csv" "$dir/engine_kvlinux_summary.csv"
  awk -F, '
    NR == 1 { for (i = 1; i <= NF; ++i) c[$i] = i; next }
    { m = $c["mode"]; n = $c["children"]
      if ($c["children_finished"] != $c["children_started"] || $c["timed_out"] + $c["guard_cases"] != 0) {
        print "K1 " m " " n; bad = 1 }
      # K2 for STATOR (Map Only) at 16 and 32 children is a gate that the
      # record reports as not met, with its counts; for every other mode it
      # must hold.
      if (m != "copy" && $c["texts_equal_to_copy"] != $c["texts_compared"]) {
        if (m == "ro_plain") print "EXCEPTION K2 ro_plain children=" n " texts=" $c["texts_equal_to_copy"] "/" $c["texts_compared"]
        else { print "K2 " m " " n; bad = 1 } }
      if (n == 8) { flt[m] = $c["gpu_fault_pages"]; mem[m] = $c["children_memory_mib"] } }
    END {
      if (!(flt["extent"] < flt["ro_plain"] && flt["extent"] < flt["cow_plain"])) { print "K3"; bad = 1 }
      if (!(mem["cow_plain"] >= 1.5 * mem["extent"] && mem["ro_plain"] <= 1.1 * mem["extent"])) { print "K4"; bad = 1 }
      exit bad
    }' "$dir/engine_kvlinux_summary.csv" >"$scratch/scaling.out" || { cat "$scratch/scaling.out" >&2; fail "a gate of kvlinux-scaling does not hold"; }
  while IFS= read -r line; do
    [[ "$line" == EXCEPTION* ]] && exceptions+=("kvlinux-scaling:${line#EXCEPTION }")
  done <"$scratch/scaling.out"
  exceptions=("${exceptions[@]// /_}")
fi

# --- the steps of STATOR left out one at a time, with the texts kept ------------
# The runner gates nothing; every child must finish and leave its text.
campaign engine-kvdiverge
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -v order="$(modes_of "$dir")" -f "$script_dir/summarize_engine_kvlinux.awk" \
    "$dir/raw.log" >"$scratch/kvdiverge.csv"
  same "$scratch/kvdiverge.csv" "$dir/engine_kvlinux_summary.csv"
  gates kvdiverge "$dir/engine_kvlinux_summary.csv" '{
    if ($c["children_finished"] != $c["children_started"] || $c["timed_out"] + $c["guard_cases"] != 0) print "!children " $c["mode"] }'
  expected_texts=$(awk -F, 'NR == 1 { for (i = 1; i <= NF; ++i) c[$i] = i; next } { t += $c["children_started"] } END { print t }' \
    "$dir/engine_kvlinux_summary.csv")
  [[ "$(find "$dir/texts" -name '*.txt' | wc -l)" == "$expected_texts" ]] || fail "kvdiverge: a text is missing"
fi

# --- every word that the GPU reads when it faults on a read-only file mapping ----
campaign gpu-read-check
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -v rounds="$(sed -n 's/^rounds=//p' "$dir/metadata.txt")" -f "$script_dir/summarize_gpu_read_check.awk" \
    "$dir/raw.log" >"$scratch/read-check.csv"
  same "$scratch/read-check.csv" "$dir/gpu_read_check_summary.csv"
  rounds=$(sed -n 's/^rounds=//p' "$dir/metadata.txt")
  gates read-check "$dir/gpu_read_check_summary.csv" "{
    if (\$c[\"failed\"] + \$c[\"errors\"] + \$c[\"publish_bad\"] != 0 || \$c[\"reader_rounds\"] != \$c[\"runs\"] * \$c[\"procs\"] * $rounds) print \"!G1 \" \$c[\"fs\"] \" \" \$c[\"writer\"] \" \" \$c[\"read\"] \" \" \$c[\"procs\"]
    if (\$c[\"read\"] == \"cpu\" && \$c[\"differ_words\"] != 0) print \"!G2 \" \$c[\"fs\"] \" \" \$c[\"writer\"] \" \" \$c[\"procs\"] }"
fi

# --- the same check, with a CPU read and a second GPU pass after a wrong read ----
campaign gpu-read-trace
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -f "$script_dir/summarize_gpu_read_trace.awk" "$dir/raw.log" >"$scratch/read-trace.csv"
  same "$scratch/read-trace.csv" "$dir/gpu_read_trace_summary.csv"
  rounds=$(sed -n 's/^rounds=//p' "$dir/metadata.txt")
  gates read-trace "$dir/gpu_read_trace_summary.csv" "{
    if (\$c[\"failed\"] + \$c[\"errors\"] != 0 || \$c[\"reader_rounds\"] != \$c[\"runs\"] * \$c[\"procs\"] * $rounds) print \"!T1 \" \$c[\"mapping\"] \" \" \$c[\"read\"] \" \" \$c[\"procs\"]
    if (\$c[\"read\"] == \"cpu\" && \$c[\"differ_words\"] != 0) print \"!T2 \" \$c[\"mapping\"] \" \" \$c[\"procs\"] }"
fi

# --- weights read in place under the reclaim of the kernel -----------------------
campaign engine-reclaim
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -v order="$(sed -n 's/^modes=//p' "$dir/metadata.txt")" \
    -f "$script_dir/summarize_engine_reclaim.awk" "$dir/raw.log" >"$scratch/reclaim.csv"
  same "$scratch/reclaim.csv" "$dir/engine_reclaim_summary.csv"
  awk -F, '
    NR == 1 { for (i = 1; i <= NF; ++i) c[$i] = i; next }
    { k = $c["mode"] SUBSEP $c["pressure"]
      if ($c["failed"] + $c["timed_out"] != 0 || $c["agents_with_speed"] != $c["agents"]) { print "R1 " $c["mode"] " " $c["pressure"]; bad = 1 }
      if ($c["pressure"] == "reclaim" && $c["texts_equal_to_none"] != $c["texts_compared"]) { print "R1 texts " $c["mode"]; bad = 1 }
      ratio[k] = $c["tps_vs_none"]; flt[k] = $c["gpu_fault_pages"] }
    END {
      if (ratio["memfs", "reclaim"] < 0.95 || ratio["device", "reclaim"] < 0.95) { print "R2"; bad = 1 }
      if (!(flt["pagecache", "reclaim"] > flt["memfs", "reclaim"])) { print "R3"; bad = 1 }
      exit bad
    }' "$dir/engine_reclaim_summary.csv" || fail "a gate of engine-reclaim does not hold"
fi

# --- fork and registration on memory that the GPU reads ---------------------------
campaign linux-probes
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -f "$script_dir/summarize_linux_probes.awk" "$dir/raw.log" >"$scratch/probes.csv"
  same "$scratch/probes.csv" "$dir/linux_probes_summary.csv"
  size=$(sed -n 's/^size_mib=//p' "$dir/metadata.txt")
  awk -F, -v size="$size" '
    NR == 1 { for (i = 1; i <= NF; ++i) c[$i] = i; next }
    { p = $c["probe"]
      if ($c["failed"] != 0) { print "P1 " p; bad = 1 }
      if (p == "fork_gpu" && $c["child_growth_mib"] < 0.9 * size) { print "P2 gpu"; bad = 1 }
      if (p == "fork_cpu" && $c["child_growth_mib"] >= 0.05 * size) { print "P2 cpu"; bad = 1 }
      if (p == "fork_after_init" && $c["child_cuda_refused"] != $c["cases"]) { print "P3"; bad = 1 } }
    END { exit bad }' "$dir/linux_probes_summary.csv" || fail "a gate of linux-probes does not hold"
fi

# --- KSM on memory that the GPU reads and writes -----------------------------------
campaign ksm-probe
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -f "$script_dir/summarize_ksm_probe.awk" "$dir/raw.log" >"$scratch/ksm.csv"
  same "$scratch/ksm.csv" "$dir/ksm_probe_summary.csv"
  awk -F, '
    NR == 1 { for (i = 1; i <= NF; ++i) c[$i] = i; next }
    { k = $c["kind"]
      if ($c["failed"] != 0 || $c["merged_all"] != $c["cases"]) { print "Q1 " k; bad = 1 }
      if (k == "cuda" && $c["kept_after_gpu_read"] < 0.99) { print "Q2"; bad = 1 }
      if (k == "cuda" && $c["written_kept_after_gpu_write"] > 0.10) { print "Q3"; bad = 1 } }
    END { exit bad }' "$dir/ksm_probe_summary.csv" || fail "a gate of ksm-probe does not hold"
fi

# --- mechanisms in one engine: baselines, ablation, locality (gates of
# run_engine_kvmech.sh; every child runs on the one GPU, so B5 asks that the
# device modes complete with every child) -----------------------------------------
verify_mech() {  # runner-name label set
  campaign "$1"
  [[ -n "$dir" ]] || return 0
  check_hashes "$dir"
  awk -v order="$(modes_of "$dir")" -f "$script_dir/summarize_engine_kvmech.awk" \
    "$dir/raw.log" >"$scratch/$2.csv"
  same "$scratch/$2.csv" "$dir/engine_kvmech_summary.csv"
  case $3 in
    baselines)
      gates "$2" "$dir/engine_kvmech_summary.csv" '
        {
          cell = $c["placement"] ":" $c["children"] ":" $c["paragraphs"]; mode = $c["mode"]
          if ($c["parents_completed"] != $c["runs"] || $c["timed_out"] != 0) print "!B1 parent " cell " " mode
          if ($c["children_finished"] != $c["children_started"]) print "!B1 children " cell " " mode
          if ($c["texts_equal_to_copy"] != $c["texts_compared"] || $c["texts_compared"] != $c["children_finished"])
            print "!B2 " cell " " mode " " $c["texts_equal_to_copy"] "/" $c["children_finished"]
          memory[cell, mode] = $c["children_memory_mib"]; attach[cell, mode] = $c["child_attach_ms"]
          pause[cell, mode] = $c["publish_ms"]; cells[cell] = 1
        }
        END {
          split("copy demand device device_tuned device_merged cow writable_no_read", others, " ")
          for (cell in cells) {
            if (!(memory[cell, "extent"] < memory[cell, "device_merged"] && memory[cell, "extent"] < memory[cell, "device_tuned"]))
              print "B3 " cell " extent=" memory[cell, "extent"] " device_tuned=" memory[cell, "device_tuned"]
            if (memory[cell, "device_tuned"] > memory[cell, "device"])
              print "B3 " cell " device_tuned=" memory[cell, "device_tuned"] " device=" memory[cell, "device"]
            if (!(memory[cell, "device"] < memory[cell, "demand"] && memory[cell, "demand"] < memory[cell, "copy"]))
              print "B3 " cell " device=" memory[cell, "device"] " demand=" memory[cell, "demand"] " copy=" memory[cell, "copy"]
            if (memory[cell, "extent"] > memory[cell, "cow"])
              print "B3 " cell " extent=" memory[cell, "extent"] " cow=" memory[cell, "cow"]
            if (!(attach[cell, "device_merged"] < 0.5 * attach[cell, "device"]))
              print "B6 " cell " device_merged=" attach[cell, "device_merged"] " device=" attach[cell, "device"]
            for (o = 1; o <= 7; ++o) {
              if (!((cell SUBSEP others[o]) in attach)) continue
              if (!(attach[cell, "extent"] < attach[cell, others[o]]))
                print "B4 attach " cell " extent=" attach[cell, "extent"] " " others[o] "=" attach[cell, others[o]]
              if (others[o] !~ /^(cow|writable_no_read)$/ && !(pause[cell, "extent"] < pause[cell, others[o]]))
                print "B4 pause " cell " extent=" pause[cell, "extent"] " " others[o] "=" pause[cell, others[o]]
            }
          }
        }' ;;
    ablation)
      gates "$2" "$dir/engine_kvmech_summary.csv" '
        {
          mode = $c["mode"]; kept = (mode !~ /^(no_read|no_populate|writable_no_read)$/)
          complete = ($c["parents_completed"] == $c["runs"] && $c["timed_out"] == 0 && $c["children_finished"] == $c["children_started"])
          equal = ($c["texts_equal_to_copy"] == $c["texts_compared"] && $c["texts_compared"] == $c["children_finished"])
          if (kept && !complete) print "!A1 " mode
          if (kept && !equal) print "!A2 " mode " " $c["texts_equal_to_copy"] "/" $c["children_finished"]
          if (!kept && !(complete && equal))
            print "counted " mode " finished=" $c["children_finished"] "/" $c["children_started"] " equal=" $c["texts_equal_to_copy"]
          memory[mode] = $c["children_memory_mib"]; attach[mode] = $c["child_attach_ms"]
        }
        END {
          if (!(memory["extent"] <= memory["grow_1024"] && memory["grow_1024"] <= memory["grow_4096"] && memory["grow_4096"] <= memory["grow_all"]))
            print "A3 extent=" memory["extent"] " grow_1024=" memory["grow_1024"] " grow_4096=" memory["grow_4096"] " grow_all=" memory["grow_all"]
          if (!(attach["extent"] < attach["small_pages"]))
            print "A4 extent=" attach["extent"] " small_pages=" attach["small_pages"]
        }' ;;
    locality)
      gates "$2" "$dir/engine_kvmech_summary.csv" '
        {
          if ($c["parents_completed"] != $c["runs"] || $c["timed_out"] != 0 || $c["children_finished"] != $c["children_started"])
            print "!L1 " $c["mode"]
          if ($c["texts_equal_to_copy"] != $c["texts_compared"] || $c["texts_compared"] != $c["children_finished"])
            print "!L2 " $c["mode"] " " $c["texts_equal_to_copy"] "/" $c["children_finished"]
        }' ;;
  esac
}
verify_mech engine-kvmech kvmech baselines
verify_mech engine-kvablate kvablate ablation
verify_mech engine-kvlocal kvlocal locality

# --- the memory of the whole stack and how many agents fit (gates K1-K5 of
# run_engine_kvstack.sh) ---------------------------------------------------------
verify_stack() {  # runner-name label gates-on-all-stacks
  campaign "$1"
  [[ -n "$dir" ]] || return 0
  local model_bytes total
  check_hashes "$dir"
  model_bytes=$(awk -F= '$1 == "model_bytes" { print $2 }' "$dir/metadata.txt")
  total=$(awk -F= '$1 == "mem_total_kib" { print $2 }' "$dir/metadata.txt")
  awk -f "$script_dir/summarize_engine_kvstack.awk" "$dir/raw.log" >"$scratch/$2.csv"
  same "$scratch/$2.csv" "$dir/engine_kvstack_summary.csv"
  awk -v table=fit -v model_bytes="$model_bytes" -v mem_total_kib="$total" \
    -f "$script_dir/summarize_engine_kvstack.awk" "$dir/raw.log" >"$scratch/$2.fit.csv"
  same "$scratch/$2.fit.csv" "$dir/engine_kvstack_fit.csv"
  awk -v table=texts -f "$script_dir/summarize_engine_kvstack.awk" "$dir/raw.log" \
    >"$scratch/$2.texts.csv"
  same "$scratch/$2.texts.csv" "$dir/engine_kvstack_texts.csv"
  if [[ "$3" == both ]]; then
    # at the capacity of the device a failed agent is a measured outcome: the
    # record reports K1 as not met with its count
    local line
    while IFS= read -r line; do
      exceptions+=("K1 campaign=$2 $line")
    done < <(awk -F, 'NR > 1 && $3 > 0 && $5 != 0 { print "mode=" $1 " agents=" $2 " failed_agents=" $5 " runs=" $3 }' \
      "$dir/engine_kvstack_summary.csv")
  else
    awk -F, 'NR > 1 && $3 > 0 { rows++; if ($5 != 0) bad++ } END { exit (bad || !rows) ? 1 : 0 }' \
      "$dir/engine_kvstack_summary.csv" || fail "K1 does not hold in $2"
  fi
  awk -F, 'NR > 1 { rows++; if ($2 != $3) bad++ } END { exit (bad || !rows) ? 1 : 0 }' \
    "$dir/engine_kvstack_texts.csv" || fail "K2 does not hold in $2"
  [[ "$3" == all ]] || return 0
  awk -F, '
    NR > 1 && $3 > 0 { memory[$1, $2] = $18; counts[$2] = 1 }
    END {
      for (n in counts) {
        if ((("both", n) in memory) && (("weights", n) in memory) && memory["both", n] >= memory["weights", n]) bad++
        if ((("both", n) in memory) && (("kv", n) in memory) && memory["both", n] >= memory["kv", n]) bad++
        if ((("weights", n) in memory) && (("none", n) in memory) && memory["weights", n] >= memory["none", n]) bad++
        if ((("kv", n) in memory) && (("none", n) in memory) && memory["kv", n] >= memory["none", n]) bad++
      }
      exit bad ? 1 : 0
    }' "$dir/engine_kvstack_summary.csv" || fail "K3 does not hold in $2"
  awk -F, '$1 == "both" { rows++; if ($9 > 0.20 || $9 <= 0) bad++ } END { exit (bad || rows != 1) ? 1 : 0 }' \
    "$dir/engine_kvstack_fit.csv" || fail "K4 does not hold in $2"
  while IFS=, read -r mode agents runs _ _ _ _ _ _ ratio _ _ median _; do
    [[ "$mode" == both && "$runs" -gt 0 ]] || continue
    awk -v r="$ratio" 'BEGIN { exit (r > 0) ? 0 : 1 }' || continue
    if awk -v r="$ratio" 'BEGIN { exit (r < 0.92) ? 0 : 1 }'; then
      exceptions+=("K5 campaign=$2 agents=$agents speed_vs_none=$ratio median=$median")
    fi
  done < <(tail -n +2 "$dir/engine_kvstack_summary.csv")
}
verify_stack engine-kvstack kvstack all
verify_stack engine-kvstack-huge kvstack-huge all
verify_stack engine-kvstack-capacity kvstack-capacity both
verify_stack engine-kvstack-capacity-huge kvstack-capacity-huge both
verify_stack engine-kvstack-qwen14b kvstack-qwen14b both
verify_stack engine-kvstack-qwen14b-none8 kvstack-qwen14b-none8 both
verify_stack engine-kvstack-qwen14b-huge kvstack-qwen14b-huge both
verify_stack engine-kvstack-capacity-qwen14b kvstack-capacity-qwen14b both

# --- extents under time slicing and MPS (gates G1-G6 of run_engine_kvmps.sh;
# G6 is 0.97 for both configurations on the whole GPU) ---------------------------
campaign engine-kvmps
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -f "$script_dir/summarize_engine_kvmps.awk" "$dir/raw.log" >"$scratch/kvmps.csv"
  same "$scratch/kvmps.csv" "$dir/engine_kvmps_summary.csv"
  gates kvmps "$dir/engine_kvmps_summary.csv" '
    {
      rows++; cell = $1 SUBSEP $2 SUBSEP $3; cells[cell] = $1
      if ($5 != 6 || $6 != 0 || $7 != 0 || $18 != $2 * $5) print "!G1 " $1 " " $4
      publish[cell, $4] = $8; attach[cell, $4] = $10; speed[cell, $4] = $12; memory[cell, $4] = $15
      if ($4 == "extent") {
        if ($19 != $18) print "!G2 " $1
        if ($16 > 1.1 * $17) print "G3 config=" $1 " prefix_pss_mib=" $16 " cache_file_mib=" $17
      }
    }
    END {
      if (rows != 4) print "!rows " rows
      for (cell in cells) {
        if (memory[cell, "extent"] >= memory[cell, "copy"]) print "!G4 " cells[cell]
        if (publish[cell, "extent"] > publish[cell, "copy"] || attach[cell, "extent"] > attach[cell, "copy"])
          print "G5 " cells[cell] " publish=" publish[cell, "extent"] " attach=" attach[cell, "extent"]
        if (speed[cell, "extent"] < 0.97 * speed[cell, "copy"])
          print "G6 " cells[cell] " ratio=" speed[cell, "extent"] / speed[cell, "copy"]
      }
    }'
fi

# --- tree of published extents (gates T1-T6 of run_engine_kvtree.sh) -----------
campaign engine-kvtree
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -f "$script_dir/summarize_engine_kvtree.awk" "$dir/raw.log" >"$scratch/kvtree.csv"
  same "$scratch/kvtree.csv" "$dir/engine_kvtree_summary.csv"
  gates kvtree "$dir/engine_kvtree_summary.csv" '
    {
      rows++; mode = $1
      if ($2 != 6 || $5 != 0 || $6 != 0 || $22 != $2 * $3 * $4) print "!T1 " mode
      if (mode != "copy" && $23 != $22) {
        if (mode == "flat" && $29 >= 32 && $30 == $22) print "T2 flat exact=" $23 "/" $22 " min_common_tokens=" $29
        else print "!T2 " mode
      }
      if (mode != "copy" && ($20 != $2 || $21 != 0)) print "!T6 " mode
      memory[mode] = $18; publish[mode] = $12; attach[mode] = $14; first[mode] = $16
    }
    END {
      if (rows != 4) print "!rows " rows
      if (memory["chain"] >= memory["copy"] || memory["chain"] >= memory["flat"]) print "!T3"
      if (publish["chain"] > publish["copy"] || attach["chain"] > attach["copy"]) print "T4 publish=" publish["chain"] " attach=" attach["chain"]
      if (first["chain"] > first["flat"]) print "T5 chain=" first["chain"] " flat=" first["flat"]
    }'
fi

# --- agent count, prefix length, model, workload and long generation (gates
# S1-S4 of run_engine_kvscale.sh) ----------------------------------------------------
verify_scale() {  # runner-name label
  campaign "$1"
  [[ -n "$dir" ]] || return 0
  check_hashes "$dir"
  awk -f "$script_dir/summarize_engine_kvshare.awk" "$dir/raw.log" >"$scratch/$2.csv"
  same "$scratch/$2.csv" "$dir/engine_kvscale_summary.csv"
  awk -v table=publish -f "$script_dir/summarize_engine_kvshare.awk" "$dir/raw.log" \
    >"$scratch/$2-publish.csv"
  same "$scratch/$2-publish.csv" "$dir/engine_kvscale_publish.csv"
  gates "$2" "$dir/engine_kvscale_summary.csv" '
    {
      rows++; mode = $2; agents = $3
      if ($4 != 6 || $5 != 0 || $8 != agents * $4) print "!S1 " mode " " agents
      if (mode == "extent_lazy") {
        if ($7 != $8) print "!S2 " agents " " $7 "/" $8
        extent[agents] = $19
        if ($16 < 0.92) print "S4 agents=" agents " ratio=" $16 " lower_ci95=" $17
      } else if (mode == "restore") restore[agents] = $19
      counts[agents] = 1
    }
    END {
      previous = 0
      for (agents = 1; agents <= 64; ++agents) {
        if (!(agents in counts)) continue
        if (previous && extent[agents] - extent[previous] >= restore[agents] - restore[previous]) print "!S3 " agents
        previous = agents
      }
    }'
  awk -F, 'NR > 1 { rows++; if ($4 != 0) bad++ } END { exit (bad || rows != 2) ? 1 : 0 }' \
    "$dir/engine_kvscale_publish.csv" || fail "publishing failed in $2"
}
verify_scale engine-kvscale kvscale
verify_scale engine-kvscale-p20 kvscale-p20
verify_scale engine-kvscale-p600 kvscale-p600
verify_scale engine-kvscale-agent kvscale-agent
verify_scale engine-kvscale-llama8b kvscale-llama8b
verify_scale engine-kvscale-qwen14b kvscale-qwen14b
verify_scale engine-kvscale-gen2k kvscale-gen2k

# --- the stock server published at a token boundary (V1-V5) ----------------------
campaign engine-kvserver2
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -f "$script_dir/summarize_engine_kvserver.awk" "$dir/raw.log" >"$scratch/kvserver2.csv"
  same "$scratch/kvserver2.csv" "$dir/engine_kvserver_summary.csv"
  gates kvserver2 "$dir/engine_kvserver_summary.csv" '
    {
      rows++; agents = $2; mode = $1; cell = agents
      if ($3 != 6 || $4 != 0 || $17 != agents * $3 || $12 > 256) print "!V1 " mode " " agents
      if ($18 != $17 || $20 != $17 || $19 != 64) print "!V2 " mode " " agents
      save[cell, mode] = $7; server_save[cell, mode] = $8; restore[cell, mode] = $10; memory[cell, mode] = $15
      cells[cell] = 1
    }
    END {
      if (rows != 4) print "!rows " rows
      for (cell in cells) {
        if (memory[cell, "extent"] >= memory[cell, "copy"]) print "!V4 " cell
        if (restore[cell, "extent"] > restore[cell, "copy"] || save[cell, "extent"] > save[cell, "copy"] ||
            server_save[cell, "extent"] > server_save[cell, "copy"]) print "V5 " cell
      }
    }'
fi

# --- per-agent host memory limits (L1-L4) -------------------------------------------
campaign engine-kvlimit
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -f "$script_dir/summarize_engine_kvlimit.awk" "$dir/raw.log" >"$scratch/kvlimit.csv"
  same "$scratch/kvlimit.csv" "$dir/engine_kvlimit_summary.csv"
  gates kvlimit "$dir/engine_kvlimit_summary.csv" '
    {
      rows++; key = $1 SUBSEP $2
      if ($3 != 6 || $10 != $9 || $11 != 0) print "!L4 " $1 " " $2
      if ($1 == "account" && ($5 != $3 || $6 != 0)) print "!L4 account " $2
      limit[key] = $4; done[key] = $5; kills[key] = $6; peak[key] = $7; equal[key] = $13; others[key] = $9
    }
    END {
      if (rows != 4) print "!rows " rows
      r = "limit" SUBSEP "restore"; e = "limit" SUBSEP "extent_lazy"
      if (peak[e] < limit[e] - 16 || peak[e] > limit[e] + 16 || peak[r] >= limit[r]) print "!L1-L2"
      if (done[e] != 0 || kills[e] != 6 || equal[e] != others[e]) print "!L3 extent"
      if (done[r] != 6 || kills[r] != 0) print "!L3 restore"
    }'
fi

# --- read path, VMM attach granularity, protection --------------------------------
campaign read-path
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -f "$script_dir/summarize_read_path.awk" "$dir/raw.log" >"$scratch/readpath.csv"
  same "$scratch/readpath.csv" "$dir/read_path_summary.csv"
  gates read-path "$dir/read_path_summary.csv" '{ rows++; if ($4 != 6 || $5 != 0) print "!probe " $0 } END { if (rows != 12) print "!rows " rows }'
fi
campaign vmm-attach
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -f "$script_dir/summarize_vmm_attach.awk" "$dir/raw.log" >"$scratch/vmmattach.csv"
  same "$scratch/vmmattach.csv" "$dir/vmm_attach_summary.csv"
  gates vmm-attach "$dir/vmm_attach_summary.csv" '{ rows++; if ($3 != 6 || $4 != 0 || $8 != 0) print "!probe " $1 } END { if (rows != 9) print "!rows " rows }'
fi
campaign protect
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -f "$script_dir/summarize_protect.awk" "$dir/raw.log" >"$scratch/protect.csv"
  same "$scratch/protect.csv" "$dir/protect_summary.csv"
  awk -F, '
    $1 == "host" { host++; if ($2 != 12 || $3 != $2 || $4 != $2 || $6 != $2) bad++ }
    $1 == "vmm" { vmm++; if ($2 != 6) bad++ }
    END { exit (bad || host != 1 || vmm != 1) ? 1 : 0 }
  ' "$dir/protect_summary.csv" || fail "the host protection expectation P1 does not hold"
fi

# --- Ollama as one and as two servers (O1, O2; the service of the machine stays) ----
campaign ollama-agents
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -f "$script_dir/summarize_ollama_agents.awk" "$dir/raw.log" >"$scratch/ollama.csv"
  same "$scratch/ollama.csv" "$dir/ollama_agents_summary.csv"
  gates ollama "$dir/ollama_agents_summary.csv" '
    { rows++; expected = $1 == "one" ? 1 : 2
      if ($3 != 6 || $4 != 48 || $5 != $4 || $14 <= 0 || $11 != expected || $12 != expected || $13 <= 0) print "!O1-O2 " $1 " " $2 }
    END { if (rows != 4) print "!rows " rows }'
  before=$(awk -F= '$1 == "system_ollama_pid_before" { print $2 }' "$dir/metadata.txt")
  read -r after alive < <(awk '{ for (i=1;i<=NF;i++) { split($i,a,"=");
    if (a[1] == "pid_after") pid=a[2]; if (a[1] == "alive") live=a[2] } }
    END { print pid, live }' "$dir/raw.log")
  [[ -n "$before" && "$before" == "$after" && "$alive" == 1 ]] ||
    fail "the machine Ollama service did not remain unchanged"
fi

# --- scattered reads and writes by the kind of memory; the SMs of the GPU -------------
campaign tlb-probe
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -f "$script_dir/summarize_tlb_probe.awk" "$dir/raw.log" >"$scratch/tlb.csv"
  same "$scratch/tlb.csv" "$dir/tlb_probe_summary.csv"
  gates tlb "$dir/tlb_probe_summary.csv" '{ if ($c["failed_probes"] != 0 || $c["runs"] != 6) print "!probe " $c["sms"] " " $c["kind"] }'
fi
campaign sm-probe
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -f "$script_dir/summarize_sm_probe.awk" "$dir/raw.log" >"$scratch/sm.csv"
  same "$scratch/sm.csv" "$dir/sm_probe_summary.csv"
  gates sm "$dir/sm_probe_summary.csv" '{ if ($c["runs"] % 6 != 0 || $c["reported_sms"] != $c["blocks_in_parallel_min"] || $c["reported_sms"] != $c["blocks_in_parallel_max"]) print "!sm " $c["instance"] }'
fi

# --- a pipeline of agents on a public workload (P1-P5) --------------------------------
campaign engine-kvpipe-huge
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -f "$script_dir/summarize_engine_kvpipe.awk" "$dir/raw.log" >"$scratch/kvpipe.csv"
  same "$scratch/kvpipe.csv" "$dir/engine_kvpipe_summary.csv"
  gates kvpipe "$dir/engine_kvpipe_summary.csv" '
    {
      mode = $c["mode"]
      if ($c["failed_processes"] != 0 || $c["timed_out"] != 0 || $c["workers_finished"] != $c["workers_started"]) print "!P1 " mode
      if ($c["worker_texts_equal_to_stock"] != $c["worker_texts_compared"] || $c["worker_texts_compared"] != $c["workers_finished"])
        print "!P2 " mode " " $c["worker_texts_equal_to_stock"] "/" $c["workers_finished"]
      memory[mode] = $c["memory_mib"] + $c["files_mib"]; wall[mode] = $c["wall_vs_stock"]; energy[mode] = $c["vin_vs_stock"]
    }
    END {
      if (!(memory["chain"] < memory["copy"] && memory["copy"] < memory["stock"]))
        print "!P3 chain=" memory["chain"] " copy=" memory["copy"] " stock=" memory["stock"]
      if (!(wall["chain"] <= 1.10)) print "P4 wall_vs_stock=" wall["chain"]
      if (!(energy["chain"] <= 1.10)) print "P5 energy_vs_stock=" energy["chain"]
    }'
fi

# --- deeper and wider trees (D1-D5) ------------------------------------------------------
verify_deep() {  # runner-name label
  campaign "$1"
  [[ -n "$dir" ]] || return 0
  check_hashes "$dir"
  awk -f "$script_dir/summarize_engine_kvdeep.awk" "$dir/raw.log" >"$scratch/$2.csv"
  same "$scratch/$2.csv" "$dir/engine_kvdeep_summary.csv"
  gates "$2" "$dir/engine_kvdeep_summary.csv" '
    {
      mode = $c["mode"]
      if ($c["failed_processes"] != 0 || $c["timed_out"] != 0) print "!D1 " mode
      if ($c["leaf_texts_equal_to_copy"] != $c["leaf_texts_compared"] || $c["leaf_texts_compared"] != $c["leaves_finished"])
        print "D2 " mode " " $c["leaf_texts_equal_to_copy"] "/" $c["leaves_finished"]
      memory[mode] = $c["tree_memory_mib"] + $c["files_attached_mib"]; first[mode] = $c["leaf_first_token_ms"]
      if (mode == "chain") { all = $c["files_all_alive_mib"]; left = $c["files_first_subtree_left_mib"]; end = $c["files_end_mib"] }
    }
    END {
      if (!(memory["chain"] < memory["copy"])) print "!D3 chain=" memory["chain"] " copy=" memory["copy"]
      if (!(first["chain"] <= first["copy"])) print "D4 chain=" first["chain"] " copy=" first["copy"]
      if (end != 0 || (left >= 0 && !(left < all))) print "!D5 all=" all " left=" left " end=" end
    }'
}
verify_deep engine-kvdeep kvdeep
verify_deep engine-kvdeep-wide kvdeep-wide

# --- an external weight-sharing library and vLLM ----------------------------------
campaign ext-weightshare
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -f "$script_dir/summarize_ext_weightshare.awk" "$dir/raw.log" >"$scratch/wshare.csv"
  same "$scratch/wshare.csv" "$dir/ext_weightshare_summary.csv"
  gates wshare "$dir/ext_weightshare_summary.csv" '
    { stack = $c["stack"]
      if ($c["agents_finished"] != $c["agents_started"] || $c["timed_out"] != 0) print "!W1 " $c["placement"] " " stack
      if ($c["texts_equal_to_stock"] != $c["texts_compared"]) print "!W2 " $c["placement"] " " stack }'
fi
campaign ext-vllm
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -f "$script_dir/summarize_ext_vllm.awk" "$dir/raw.log" >"$scratch/vllm.csv"
  same "$scratch/vllm.csv" "$dir/ext_vllm_summary.csv"
  gates vllm "$dir/ext_vllm_summary.csv" '{ if ($c["servers_ready"] != $c["runs"] || $c["agents_finished"] != $c["agents_started"]) print "!X1 " $c["round"] }'
fi
campaign vllm-share
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -f "$script_dir/summarize_vllm_share.awk" "$dir/raw.log" >"$scratch/vshare.csv"
  same "$scratch/vshare.csv" "$dir/vllm_share_summary.csv"
  gates vshare "$dir/vllm_share_summary.csv" '
    {
      mode = $c["mode"]; seen[mode] = 1
      if ($c["failed_cases"] != 0 || $c["servers_ready"] != $c["servers"] * $c["runs"] || $c["requests_finished"] != $c["requests_started"])
        print "!V1 " mode
      if ($c["alone1_texts_equal"] != $c["runs"] || $c["alone1_texts_compared"] != $c["runs"] ||
          $c["alone2_texts_equal"] != $c["runs"] || $c["alone2_texts_compared"] != $c["runs"])
        print "V6 " mode " alone1=" $c["alone1_texts_equal"] "/" $c["alone1_texts_compared"] " alone2=" $c["alone2_texts_equal"] "/" $c["alone2_texts_compared"]
      if (mode == "stator" && $c["peak_memory_mib"] - $c["memory_mib"] > 1024)
        print "!V7 peak_minus_steady_mib=" ($c["peak_memory_mib"] - $c["memory_mib"])
      attachers = $c["servers"] - 1
      first[mode] = $c["first_token_ms"]; cached[mode] = $c["first_cached_lowest"]
      uncached[mode] = $c["first_cached_lowest_of_any_run"]
      memory[mode] = $c["memory_mib"]; mapped[mode] = $c["shared_weights_mib"] + $c["shared_kv_mib"]
      ratio[mode] = $c["both_tps_vs_vllm"]
    }
    END {
      if (!("vllm" in seen)) print "!V1 no vllm row"
      if ("stator" in seen) {
        if (!(cached["stator"] >= 0.90)) print "!V2 stator cached=" cached["stator"]
        if (!(uncached["vllm"] == 0)) print "!V2 vllm cached=" uncached["vllm"]
        if (!(first["stator"] <= 0.5 * first["vllm"])) print "V3 first_token_ms " first["stator"] " vs " first["vllm"]
        if (!(memory["vllm"] - memory["stator"] >= 0.8 * attachers * mapped["stator"]))
          print "V4 saved_mib=" (memory["vllm"] - memory["stator"]) " mapped_mib=" attachers * mapped["stator"]
        if (!(ratio["stator"] >= 0.90)) print "V5 tps_vs_vllm=" ratio["stator"]
      }
    }'
fi

# --- the pages that the kernel faults in on behalf of the GPU (C1-C7) ------------------
campaign engine-kvfaultcount
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -f "$script_dir/summarize_engine_kvfaultcount.awk" "$dir/raw.log" >"$scratch/kvfaultcount.csv"
  same "$scratch/kvfaultcount.csv" "$dir/engine_kvfaultcount_summary.csv"
  gates kvfaultcount "$dir/engine_kvfaultcount_summary.csv" '
    {
      kind = $c["kind"]; mode = $c["mode"]
      if ($c["runs"] != 6 || $c["failed"] != 0) print "!C1 " mode
      if (kind == "agents" && ($c["texts_equal_to_copy"] != $c["texts_compared"] || $c["texts_compared"] == 0))
        print "!C2 " mode " " $c["texts_equal_to_copy"] "/" $c["texts_compared"]
      low[mode] = $c["fault_pages_min"]; high[mode] = $c["fault_pages_max"]
      pages[mode] = $c["fault_pages"]; pause[mode] = $c["publish_ms"]; next_ms[mode] = $c["next_task_ms"]
    }
    END {
      if (high["copy"] != 0 || high["extent"] != 0) print "!C3 copy=" high["copy"] " extent=" high["extent"]
      split("no_read no_populate cow cow_noread", faulting, " ")
      for (m = 1; m <= 4; ++m) if (!(low[faulting[m]] > 0)) print "!C4 " faulting[m] "=" low[faulting[m]]
      if (!(low["cow_noread"] > high["cow"])) print "!C4 cow_noread=" low["cow_noread"] " cow=" high["cow"]
      if (high["fork_copy"] != 0 || high["fork_refill"] != 0) print "!C5 fork_copy=" high["fork_copy"] " fork_refill=" high["fork_refill"]
      if (!(pause["fork_extent"] < pause["fork_copy"])) print "!C6 extent=" pause["fork_extent"] " copy=" pause["fork_copy"]
      if (high["fork_extent"] != 0)
        print "C7 fork_extent fault_pages=" pages["fork_extent"] " next_task_ms=" next_ms["fork_extent"] " copy_next_task_ms=" next_ms["fork_copy"]
    }'
fi

# --- a fork of a running parent (F1-F5) and batching servers (N1-N6) ----------------
campaign engine-kvfork
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -f "$script_dir/summarize_engine_kvfork.awk" "$dir/raw.log" >"$scratch/kvfork.csv"
  same "$scratch/kvfork.csv" "$dir/engine_kvfork_summary.csv"
  gates kvfork "$dir/engine_kvfork_summary.csv" '
    {
      mode = $c["mode"]
      if ($c["failed_processes"] != 0 || $c["parent_texts_equal"] != $c["runs"] ||
          $c["children_started"] != $c["runs"] * $c["children"] ||
          $c["child_texts_equal_to_copy"] != $c["children_started"] ||
          $c["own_instance_equal_to_alone"] != $c["own_instance_compared"]) print "!F1 " mode
      publish[mode] = $c["publish_ms"]; state[mode] = $c["state_mib"]; attach[mode] = $c["child_attach_ms"]
      memory[mode] = $c["children_memory_mib"]; pss[mode] = $c["prefix_pss_mib"]; children = $c["children"]
    }
    END {
      if (!(publish["extent"] < publish["copy"])) print "!F2 extent=" publish["extent"] " copy=" publish["copy"]
      if (!(attach["extent"] <= attach["copy"])) print "!F3 extent=" attach["extent"] " copy=" attach["copy"]
      if (memory["copy"] - memory["extent"] < 0.8 * children * state["copy"]) print "!F4 copy=" memory["copy"] " extent=" memory["extent"]
      if (pss["extent"] > 1.15 * state["copy"]) print "F5 prefix_pss_mib=" pss["extent"] " state_mib=" state["copy"]
    }'
fi
campaign engine-kvbatch
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -f "$script_dir/summarize_engine_kvbatch.awk" "$dir/raw.log" >"$scratch/kvbatch.csv"
  same "$scratch/kvbatch.csv" "$dir/engine_kvbatch_summary.csv"
  gates kvbatch "$dir/engine_kvbatch_summary.csv" '
    {
      config = $c["config"]
      if ($c["failed_servers"] != 0) print "!N1 " config
      if (config == "two_servers_extent" && ($c["texts_equal_to_copy"] != $c["texts_compared"] || $c["texts_compared"] != $c["runs"] * 8))
        print "!N2 " $c["texts_equal_to_copy"] "/" $c["texts_compared"]
      tps[config] = $c["generation_tps_total"]; ready[config] = $c["ready_all_servers_ms"]; memory[config] = $c["memory_mib"]
    }
    END {
      if (tps["two_servers_extent"] < 1.25 * tps["one_server"]) print "N3 two=" tps["two_servers_extent"] " one=" tps["one_server"]
      if (memory["two_servers_extent"] > 0.6 * memory["two_servers_copy"] || memory["two_servers_extent"] > 1.1 * memory["one_server"])
        print "N4 extent=" memory["two_servers_extent"] " copy=" memory["two_servers_copy"] " one=" memory["one_server"]
      if (tps["two_servers_extent"] < 0.95 * tps["two_servers_copy"]) print "N5 extent=" tps["two_servers_extent"] " copy=" tps["two_servers_copy"]
      if (ready["two_servers_extent"] >= ready["two_servers_copy"] || ready["two_servers_copy"] >= ready["two_servers_compute"])
        print "N6 extent=" ready["two_servers_extent"] " copy=" ready["two_servers_copy"] " compute=" ready["two_servers_compute"]
    }'
fi

# --- resident agents against released agents (R1-R4) -------------------------------------
campaign engine-kvresident
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -f "$script_dir/summarize_engine_kvresident.awk" "$dir/raw.log" >"$scratch/kvresident.csv"
  same "$scratch/kvresident.csv" "$dir/engine_kvresident_summary.csv"
  reserve=$(awk -F= '$1 == "reserve_gib" { print $2 * 1024 }' "$dir/metadata.txt")
  gates kvresident "$dir/engine_kvresident_summary.csv" '
    {
      n = $c["residents"]
      if ($c["failed_processes"] != 0 || $c["skipped_runs"] != 0 || $c["guard_runs"] != 0 || $c["timed_out_runs"] != 0) print "!R1 " n
      if ($c["texts_equal_to_smallest"] != $c["texts_compared"] || $c["texts_compared"] != $c["runs"] * $c["active"])
        print "!R2 " n " " $c["texts_equal_to_smallest"] "/" $c["texts_compared"]
      if ($c["remaining_without_model_min_mib"] < '"$reserve"') print "!R4 " n " " $c["remaining_without_model_min_mib"]
      last = $c["vs_smallest"]; largest = n
    }
    END { if (last < 0.95) print "R3 " largest " " last }'
fi

# --- a pipeline of agents in which one process fails (Q1-Q4; time slicing is the
# isolated configuration on the whole GPU) -------------------------------------------------
campaign engine-kvpipefault
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -f "$script_dir/summarize_engine_kvpipefault.awk" "$dir/raw.log" >"$scratch/kvpipefault.csv"
  same "$scratch/kvpipefault.csv" "$dir/engine_kvpipefault_summary.csv"
  gates kvpipefault "$dir/engine_kvpipefault_summary.csv" '
    {
      config = $c["config"]; fault = $c["fault"]; isolated = (config == "timeslice")
      if (fault == "none" && $c["others_completed"] != $c["others_started"]) print "!Q1 " config
      if (fault == "write" && $c["writes_refused"] != $c["writes_injected"]) print "!Q2 " config
      if (isolated && fault != "none" && ($c["others_completed"] != $c["others_started"] ||
          $c["workers_with_reference_text"] != $c["workers_started"]))
        print "Q3 " fault " " $c["others_completed"] "/" $c["others_started"] " texts " $c["workers_with_reference_text"] "/" $c["workers_started"]
      if (isolated && fault != "none" && $c["wall_vs_none"] > 1.10) print "Q4 " fault " " $c["wall_vs_none"]
      if (!isolated && fault != "none" && $c["others_completed"] != $c["others_started"])
        print "counted " config " " fault " others_completed=" $c["others_completed"] "/" $c["others_started"]
    }'
fi

# --- the first artifact's campaigns on the whole GPU (verify_llm_share_artifact.sh) -----
model_mib=$(awk -F= '$1 == "model_bytes" { printf "%d", $2 / 1048576 }' \
  "$results/20261005-engine-agents-v1/metadata.txt" 2>/dev/null || echo 4466)
campaign engine-agents
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -f "$script_dir/summarize_engine_agents.awk" "$dir/raw.log" >"$scratch/agents.csv"
  same "$scratch/agents.csv" "$dir/engine_agents_summary.csv"
  gates agents "$dir/engine_agents_summary.csv" '
    { rows++; total[$1, $2, $3] = $12; tps[$1, $2, $3] = $7; configs[$1] = 1; if ($5 != 0) print "!run " $1 " " $2 " " $3 }
    END {
      if (rows != 16) print "!rows " rows
      for (c in configs) {
        if (total[c, "copy", 8] - total[c, "inplace", 8] < 5 * '"$model_mib"') print "!memory " c
        if (tps[c, "inplace", 8] < 0.85 * tps[c, "copy", 8]) print "speed " c " ratio=" tps[c, "inplace", 8] / tps[c, "copy", 8]
      }
    }'
fi
campaign engine-pages
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -f "$script_dir/summarize_engine_pages.awk" "$dir/raw.log" >"$scratch/pages.csv"
  same "$scratch/pages.csv" "$dir/engine_pages_summary.csv"
  gates pages "$dir/engine_pages_summary.csv" '
    { rows++; ratio[$1] = $7; if ($3 != 0 || $4 != $2) print "!run " $1 }
    END {
      if (rows != 4) print "!rows " rows
      if (ratio["inplace_thp"] < ratio["inplace_4k"] || ratio["inplace_hugetlb"] < ratio["inplace_4k"]) print "page_size thp=" ratio["inplace_thp"] " 4k=" ratio["inplace_4k"]
      if (ratio["inplace_hugetlb"] < 0.95) print "hugetlb=" ratio["inplace_hugetlb"]
    }'
fi
campaign engine-adapters
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -f "$script_dir/summarize_engine_adapters.awk" "$dir/raw.log" >"$scratch/adapters.csv"
  same "$scratch/adapters.csv" "$dir/engine_adapters_summary.csv"
  gates adapters "$dir/engine_adapters_summary.csv" '
    { rows++; total[$1] = $9; if ($3 != 0 || $4 != $2 || $5 != $6) print "!run " $1 }
    END { if (rows != 2) print "!rows " rows; if (total["copy"] - total["inplace"] < 3 * '"$model_mib"') print "!memory" }'
fi
verify_speed() {  # runner-name label rows low high
  campaign "$1"
  [[ -n "$dir" ]] || return 0
  check_hashes "$dir"
  awk -f "$script_dir/summarize_engine_kvspeed.awk" "$dir/raw.log" >"$scratch/$2.csv"
  same "$scratch/$2.csv" "$dir/engine_kvspeed_summary.csv"
  gates "$2" "$dir/engine_kvspeed_summary.csv" '
    { rows++; if ($4 != 0) print "!run " $2 " " $1
      if ($2 == "restore" && ($7 < 0.99 || $7 > 1.01)) print "restore " $1 " ratio=" $7
      if (($2 == "anon" || $2 == "extent" || $2 == "extent_lazy") && ($7 < '"$4"' || $7 > '"$5"')) print "host " $2 " " $1 " ratio=" $7 }
    END { if (rows != '"$3"') print "!rows " rows }'
}
verify_speed engine-kvspeed kvspeed 12 0.99 1.02
verify_speed engine-kvspeed-long kvspeed-long 6 0.99 1.02
campaign engine-kvcow
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -f "$script_dir/summarize_engine_kvcow.awk" "$dir/raw.log" >"$scratch/kvcow.csv"
  same "$scratch/kvcow.csv" "$dir/engine_kvcow_summary.csv"
  gates kvcow "$dir/engine_kvcow_summary.csv" '
    { rows++; first[$3, $2] = $11
      if ($5 != 0 || $7 != $8 || $8 != $4 * $3) print "!run " $2 " " $3
      if ($2 ~ /_noread$/ && $17 < 0.9 * $18) print "!copies " $2 " " $3
      if (($2 == "cow" || $2 == "cow_small") && $17 > 0.3 * $18) print "copies " $2 " " $3 " private=" $17 }
    END {
      if (rows != 12) print "!rows " rows
      for (key in first) { split(key, part, SUBSEP)
        if (part[2] ~ /^cow/ && first[key] < first[part[1], "extent_lazy"]) print "first " part[2] " " part[1]
        if (part[2] ~ /_noread$/ && first[key] < 1.5 * first[part[1], "extent_lazy"]) print "first_noread " part[2] " " part[1] }
    }'
fi
campaign engine-kvshare
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -f "$script_dir/summarize_engine_kvshare.awk" "$dir/raw.log" >"$scratch/kvshare.csv"
  same "$scratch/kvshare.csv" "$dir/engine_kvshare_summary.csv"
  awk -v table=publish -f "$script_dir/summarize_engine_kvshare.awk" "$dir/raw.log" >"$scratch/kvshare_publish.csv"
  same "$scratch/kvshare_publish.csv" "$dir/engine_kvshare_publish.csv"
  gates kvshare "$dir/engine_kvshare_summary.csv" '
    { rows++; key = $1 SUBSEP $3; attach[key, $2] = $13
      if ($5 != 0 || $8 != $4 * $3 || $9 != $10) print "!run " $1 " " $2 " " $3
      if ($2 != "recompute" && $7 != $8) print "!texts " $1 " " $2 " " $3 " " $7 "/" $8
      if ($2 ~ /^extent/) {
        if ($20 < 0.8 * $3 * $21) print "!memory " $1 " " $2 " " $3
        if ($22 > 1.15 * $21 || $23 < 0.75 * $3 * $21) print "shared " $1 " " $2 " " $3 " pss=" $22 " prefix=" $21
      }
      if ($2 == "extent_lazy") {
        if ($16 < 0.95) print "speed " $1 " " $3 " ratio=" $16
        if ($13 > attach[key, "restore"]) print "attach " $1 " " $3
      } }
    END { if (rows == 0 || rows % 7 != 0) print "!rows " rows }'
  gates kvshare-publish "$dir/engine_kvshare_publish.csv" '
    { rows++; publish[$1, $2] = $7; state[$1, $2] = $8; sizes[$1] = 1; if ($4 != 0) print "!publish " $1 " " $2 }
    END { for (size in sizes) {
        if (publish[size, "huge"] >= publish[size, "device"] || publish[size, "small"] >= publish[size, "device"]) print "!faster " size
        if (state[size, "huge"] > 1 || state[size, "device"] < 100) print "!state " size }
      if (rows == 0 || rows % 3 != 0) print "!rows " rows }'
fi
campaign engine-kvvmm
if [[ -n "$dir" ]]; then
  check_hashes "$dir"
  awk -f "$script_dir/summarize_engine_kvvmm.awk" "$dir/raw.log" >"$scratch/kvvmm.csv"
  same "$scratch/kvvmm.csv" "$dir/engine_kvvmm_summary.csv"
  gates kvvmm "$dir/engine_kvvmm_summary.csv" '
    { rows++; key = $1 SUBSEP $2; cells[key] = 1
      publish[key, $3] = $6; attach[key, $3] = $13; first[key, $3] = $14; memory[key, $3] = $16
      if (($3 == "copy" || $3 == "extent" || $3 == "vmm") && ($9 != $4 || $10 != $4 * 4 || $11 != $10 || $12 != $11)) print "!run " $3
      if ($3 ~ /_cross$/ && ($10 != $4 || $11 != $10)) print "counted " $3 " finished=" $11 "/" $10 }
    END {
      for (key in cells) {
        if (attach[key, "extent"] > 0.25 * attach[key, "vmm"] || first[key, "extent"] >= first[key, "vmm"]) print "attach_first " key
        if (memory[key, "extent"] >= memory[key, "vmm"] || memory[key, "vmm"] >= memory[key, "copy"]) print "memory " key
        if (publish[key, "extent"] >= publish[key, "vmm"] || publish[key, "vmm"] >= publish[key, "copy"]) print "publish " key
      } }'
fi

# --- the base measurements of Section 2.3 on the whole GPU (../thor_hostmm, gates of
# verify_hostmm_artifact.sh) ---------------------------------------------------------------
hostmm=${HOSTMM_DIR:-"$script_dir/../thor_hostmm"}
if [[ -d "$hostmm/results" ]]; then
  check_hostmm() {
    (cd "$hostmm" && sha256sum -c "$1/source_hashes.txt" >/dev/null) || fail "pinned sources of $1 do not match ../thor_hostmm"
  }
  found=$(find "$hostmm/results" -maxdepth 1 -type d -name "*-af-block-full-v1" | sort | tail -n 1)
  if [[ -n "$found" && -f "$found/source_hashes.txt" ]]; then
    verified+=("hostmm/$(basename "$found")")
    check_hostmm "$found"
    awk -f "$hostmm/summarize_af_block.awk" "$found/raw.log" >"$scratch/af.csv"
    same "$scratch/af.csv" "$found/af_block_summary.csv"
    ! grep -q '^PROBE_FAILED' "$found/raw.log" || fail "an accessed-flag probe failed"
    gates af-block "$found/af_block_summary.csv" '
      { rows++; if ($3 != $4) print "!runs " $1 " " $2
        if ($1 == "writable" && $2 == "none" && $9 != 0) print "!none copied " $9
        if ($1 == "writable" && ($2 == "one_page" || $2 == "one_per_8_blocks" || $2 == "one_per_block") && $10 != 512) print "!block " $2 " " $10
        if ($1 == "writable" && ($2 == "one_per_block" || $2 == "all") && $9 != $5) print "!all " $2
        if ($1 == "readonly" && $9 != 0) print "!readonly " $9 }
      END { if (rows != 7) print "!rows " rows }'
  else
    absent+=("hostmm/af-block")
  fi
  found=$(find "$hostmm/results" -maxdepth 1 -type d -name "*-mm-tax-full-v1" | sort | tail -n 1)
  if [[ -n "$found" && -f "$found/source_hashes.txt" ]]; then
    verified+=("hostmm/$(basename "$found")")
    check_hostmm "$found"
    awk -f "$hostmm/summarize_mm_tax.awk" "$found/raw.log" >"$scratch/tax.csv"
    same "$scratch/tax.csv" "$found/mm_tax_summary.csv"
    awk -f "$hostmm/summarize_fork_guard.awk" "$found/raw.log" >"$scratch/guard.csv"
    same "$scratch/guard.csv" "$found/fork_guard_summary.csv"
    ! grep -q '^PROBE_FAILED' "$found/raw.log" || fail "a memory-management tax probe failed"
    gates mm-tax "$found/mm_tax_summary.csv" '
      { rows++
        if (($1 == "cow_break" || $1 == "reuse_fault" || $1 == "mprotect_page" || $1 == "dontneed_page") && $11 <= 2) print "slowdown " $1 " lower=" $11
        if (($1 == "mprotect_range" || $1 == "dontneed_range" || $1 == "munmap_range" || $1 == "fork") && $6 - $5 > 20) print "range " $1 " " ($6 - $5) }
      END { if (rows != 17) print "!rows " rows }'
    gates fork-guard "$found/fork_guard_summary.csv" '
      { rows++; first[$1] = $6; if ($2 != $3) print "!runs " $1 }
      END { if (rows != 4) print "!rows " rows
        if (first["fork"] < 100 * first["dontfork"] || first["fork"] < 100 * first["posix_spawn"] || first["fork"] < 100 * first["repair"])
          print "hazard fork=" first["fork"] " posix_spawn=" first["posix_spawn"] }'
  else
    absent+=("hostmm/mm-tax")
  fi
fi

echo "stator_full_verification=PASS"
echo "campaigns_verified=${#verified[@]} ${verified[*]:-}"
echo "campaigns_absent=${#absent[@]} ${absent[*]:-}"
echo "gate_exceptions=${#exceptions[@]} ${exceptions[*]:-}"
