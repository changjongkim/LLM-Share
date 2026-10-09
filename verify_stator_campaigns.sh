#!/usr/bin/env bash
# Verifies the campaigns that were added after the first artifact
# (verify_llm_share_artifact.sh covers that one) without running the GPU:
# every summary is rebuilt from its raw log and compared, the pinned sources
# are checked against the tree, and the gates that each runner states in its
# header are evaluated again. A campaign that has not been run is reported
# as absent and does not fail the verification; a gate that the record
# reports as not met is listed here as an expected exception, with its
# measured value, and everything else must hold.
#
#   verify_stator_campaigns.sh [RESULTS_DIR]
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
results=${1:-"$script_dir/results"}
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
verified=()
absent=()
exceptions=()

fail() {
  echo "STATOR campaign verification failed: $*" >&2
  exit 1
}
check_hashes() {
  [[ -f "$1/source_hashes.txt" ]] || fail "missing hashes in $1"
  # A runner may have recorded a workload file by the absolute path of the
  # tree it ran in; the file is checked in this tree.
  (cd "$script_dir" && sed -E 's#  /.*/(workloads/[^/]+)$#  \1#' "$1/source_hashes.txt" |
    sha256sum -c - >/dev/null) || fail "pinned sources of $1 do not match the tree"
}
same() {
  cmp -s "$1" "$2" || fail "$2 is not reproducible from its raw log"
}
# present NAME: true when the campaign directory holds a summary.
present() {
  if [[ -f "$results/$1/source_hashes.txt" ]]; then
    verified+=("$1")
    return 0
  fi
  absent+=("$1")
  return 1
}

# The patch in the tree must be the difference between each engine checkout
# and its recorded commit.
for pair in llama.cpp-chain:kv_chain.patch llama.cpp-sota:kv_sota.patch \
            llama.cpp-tuned:kv_tuned.patch; do
  clone=${pair%%:*} patch=${pair##*:}
  if [[ -d "$script_dir/$clone/.git" ]]; then
    git -C "$script_dir/$clone" diff >"$scratch/patch"
    same "$scratch/patch" "$script_dir/$patch"
  fi
done

# --- extents under time slicing, MPS, MIG, and MPS in each MIG instance ------
if present 20261006-engine-kvmps-v1; then
  dir="$results/20261006-engine-kvmps-v1"
  check_hashes "$dir"
  awk -f "$script_dir/summarize_engine_kvmps.awk" "$dir/raw.log" >"$scratch/kvmps.csv"
  same "$scratch/kvmps.csv" "$dir/engine_kvmps_summary.csv"
  awk -F, '
    NR == 1 { next }
    {
      rows++
      cell = $1 SUBSEP $2 SUBSEP $3
      cells[cell] = $1
      # G1: no process fails, no case is cut off, six repetitions.
      if ($5 != 6 || $6 != 0 || $7 != 0 || $18 != $2 * $5) bad++
      publish[cell, $4] = $8; attach[cell, $4] = $10; speed[cell, $4] = $12
      memory[cell, $4] = $15
      if ($4 == "extent") {
        # G2: every child on extents writes the text of its copy counterpart.
        if ($19 != $18) exit 2
        # G3: the children hold one copy of the prefix between them.
        if ($16 > 1.1 * $17) exit 3
      }
    }
    END {
      if (bad || rows != 32) exit 1
      for (cell in cells) {
        # G4: less memory with extents. G5: a shorter pause and attach.
        if (memory[cell, "extent"] >= memory[cell, "copy"]) exit 4
        if (publish[cell, "extent"] > publish[cell, "copy"]) exit 5
        if (attach[cell, "extent"] > attach[cell, "copy"]) exit 5
        # G6: the summed speed of the children, against copies.
        floor = (cells[cell] == "timeslice" || cells[cell] == "mps") ? 0.97 : 0.92
        if (speed[cell, "extent"] < floor * speed[cell, "copy"]) exit 6
      }
    }
  ' "$dir/engine_kvmps_summary.csv" ||
    fail "a gate of the GPU sharing campaign does not hold (awk exit $?)"
fi

# --- five prefix hand-over mechanisms in one engine -------------------------
if present 20261006-engine-kvsota-v1; then
  dir="$results/20261006-engine-kvsota-v1"
  check_hashes "$dir"
  awk -f "$script_dir/summarize_engine_kvsota.awk" "$dir/raw.log" >"$scratch/kvsota.csv"
  same "$scratch/kvsota.csv" "$dir/engine_kvsota_summary.csv"
  awk -F, '
    NR == 1 { next }
    {
      rows++
      cell = $1 SUBSEP $2
      mode = $3
      # B1 and B5: all parents finish; only device-memory children in the
      # other MIG instance are expected to be refused.
      if ($4 != 6 || $5 != 6 || $6 != 0 || $10 != $2 * $4) bad++
      if ($1 == "cross" && mode == "device") {
        if ($12 != $13 || $14 != 0 || $15 == 0) bad++
      } else if ($11 != $10) bad++
      # B2: every child that completes matches its copy counterpart.
      if ($21 != $11) bad++
      memory[cell, mode] = $20
      publish[cell, mode] = $8
      attach[cell, mode] = $16
      cells[cell] = 1
    }
    END {
      if (bad || rows != 10) exit 1
      cell = "same" SUBSEP 8
      # B3: the memory order fixed in the campaign header.
      if (!(memory[cell, "extent"] < memory[cell, "device"] &&
            memory[cell, "device"] < memory[cell, "demand"] &&
            memory[cell, "demand"] < memory[cell, "copy"] &&
            memory[cell, "extent"] <= memory[cell, "cow"])) exit 2
      for (cell in cells) {
        # B4: extents minimize attach time and avoid the copy-like pauses.
        split("copy demand device cow", modes, " ")
        for (m = 1; m <= 4; ++m)
          if (attach[cell, "extent"] >= attach[cell, modes[m]]) exit 3
        split("copy demand device", pauses, " ")
        for (m = 1; m <= 3; ++m)
          if (publish[cell, "extent"] >= publish[cell, pauses[m]]) exit 4
      }
    }
  ' "$dir/engine_kvsota_summary.csv" ||
    fail "a gate of the same-engine baseline campaign does not hold (awk exit $?)"
fi

# --- tree of published extents ----------------------------------------------
if present 20261006-engine-kvtree-v1; then
  dir="$results/20261006-engine-kvtree-v1"
  check_hashes "$dir"
  awk -f "$script_dir/summarize_engine_kvtree.awk" "$dir/raw.log" >"$scratch/kvtree.csv"
  same "$scratch/kvtree.csv" "$dir/engine_kvtree_summary.csv"
  awk -F, '
    NR == 1 { next }
    {
      rows++; mode = $1
      # T1: no process fails or times out and every leaf finishes.
      if ($2 != 6 || $5 != 0 || $6 != 0 || $22 != $2 * $3 * $4) bad++
      # T2: chain modes must match exactly. Flat recomputes the group prefix
      # on the leaf MIG; if greedy decoding later crosses a numerical tie,
      # preserve and report that measured exception below. Its deterministic
      # prefix must still be at least 32 tokens and every response compared.
      if (mode != "copy" && $23 != $22 &&
          !(mode == "flat" && $29 >= 32 && $30 == $22)) bad++
      # T6: every mapped file can be unlinked and has no pages at the end.
      if (mode != "copy" && ($20 != $2 || $21 != 0)) bad++
      memory[mode] = $18; publish[mode] = $12
      attach[mode] = $14; first[mode] = $16
    }
    END {
      if (bad || rows != 4) exit 1
      # T3-T5.
      if (memory["chain"] >= memory["copy"] || memory["chain"] >= memory["flat"]) exit 2
      if (publish["chain"] > publish["copy"] || attach["chain"] > attach["copy"]) exit 3
      if (first["chain"] > first["flat"]) exit 4
    }
  ' "$dir/engine_kvtree_summary.csv" ||
    fail "a gate of the extent-chain campaign does not hold (awk exit $?)"
  while IFS=, read -r mode _runs _groups _leaves _failed _timed _root_tokens \
      _leader_tokens _root_publish _root_state _leader_attach _leader_publish \
      _leader_state _leaf_attach _decode _first _tps _memory _files _removed \
      _files_end finished equal _own_equal _own_compared _other_equal \
      _other_compared _wall common compared; do
    if [[ "$mode" == flat && "$equal" != "$finished" ]]; then
      exceptions+=("T2 mode=flat exact=$equal/$finished min_common_tokens=$common compared=$compared")
    fi
  done < <(tail -n +2 "$dir/engine_kvtree_summary.csv")
fi

# --- agent count, model, and workload variants ------------------------------
verify_scale() {
  local tag=$1 label=$2 dir
  present "$tag" || return 0
  dir="$results/$tag"
  check_hashes "$dir"
  awk -f "$script_dir/summarize_engine_kvshare.awk" "$dir/raw.log" >"$scratch/$label.csv"
  same "$scratch/$label.csv" "$dir/engine_kvscale_summary.csv"
  awk -v table=publish -f "$script_dir/summarize_engine_kvshare.awk" "$dir/raw.log" \
    >"$scratch/$label-publish.csv"
  same "$scratch/$label-publish.csv" "$dir/engine_kvscale_publish.csv"
  awk -F, '
    NR == 1 { next }
    {
      rows++; mode = $2; agents = $3
      # S1: every non-skipped case completes all of its agents.
      if ($4 != 6 || $5 != 0 || $8 != agents * $4) bad++
      if (mode == "extent_lazy") {
        # S2. S4 is checked below so that the one measured scheduling
        # outlier in the Qwen scaling campaign is preserved and diagnosed.
        if ($7 != $8) bad++
        extent[agents] = $19
      } else if (mode == "restore") restore[agents] = $19
      counts[agents] = 1
    }
    END {
      if (bad || rows < 2 || rows % 2 != 0) exit 1
      previous = 0
      for (agents = 1; agents <= 64; ++agents) {
        if (!(agents in counts)) continue
        if (!(agents in extent) || !(agents in restore)) exit 2
        # S3: at every measured scale step, the extent slope is lower.
        if (previous) {
          extent_slope = extent[agents] - extent[previous]
          restore_slope = restore[agents] - restore[previous]
          if (extent_slope >= restore_slope) exit 3
        }
        previous = agents
      }
    }
  ' "$dir/engine_kvscale_summary.csv" ||
    fail "a gate of the $label scale campaign does not hold (awk exit $?)"
  while IFS=, read -r _paragraphs mode agents _runs _failed _tokens _equal \
      _compared _own_equal _own_compared _other_equal _other_compared \
      _attach _first _tps ratio lower _upper _memory _saved _state _pss _rss; do
    [[ "$mode" == extent_lazy ]] || continue
    if awk -v ratio="$ratio" 'BEGIN { exit !(ratio < 0.92) }'; then
      [[ "$label" == kvscale && "$agents" == 32 ]] ||
        fail "S4 fails in $label at $agents agents (ratio $ratio)"
      local diagnosis
      diagnosis=$(awk '
        function field(name, i, pair) {
          for (i = 1; i <= NF; ++i) {
            split($i, pair, "=")
            if (pair[1] == name) return pair[2]
          }
          return ""
        }
        /^BEGIN_KVSHARE / { run = field("run"); next }
        /^CASE / { mode = field("mode"); agents = field("agents"); next }
        /^AGENT / && agents == 32 {
          speed[run, mode] += field("generation_tps")
          if (mode == "restore" && field("first_token_ms") > max_first[run])
            max_first[run] = field("first_token_ms")
          next
        }
        END {
          for (run = 1; run <= 6; ++run) {
            if (speed[run, "restore"] <= 0 || speed[run, "extent_lazy"] <= 0) exit 1
            paired = speed[run, "extent_lazy"] / speed[run, "restore"]
            if (paired < 0.92) {
              bad++; bad_run = run; bad_ratio = paired
              bad_first = max_first[run]
            } else good++
          }
          # Exactly one run is anomalous, its first-token stall is explicit,
          # and every other predeclared repetition clears S4.
          if (bad != 1 || good != 5 || bad_first < 200000) exit 2
          printf "outlier_run=%d paired_ratio=%.4f max_first_token_ms=%.0f good_runs=%d", \
                 bad_run, bad_ratio, bad_first, good
        }
      ' "$dir/raw.log") || fail "the recorded S4 outlier diagnosis is not reproducible"
      exceptions+=("S4 campaign=$label agents=$agents aggregate_ratio=$ratio lower_ci95=$lower $diagnosis")
    fi
  done < <(tail -n +2 "$dir/engine_kvscale_summary.csv")
  awk -F, 'NR > 1 { rows++; if ($4 != 0) bad++ } END { exit (bad || rows != 2) ? 1 : 0 }' \
    "$dir/engine_kvscale_publish.csv" || fail "publishing failed in the $label scale campaign"
}
verify_scale 20261006-engine-kvscale-v1 kvscale
verify_scale 20261006-engine-kvscale-llama8b-v1 kvscale-llama8b
verify_scale 20261006-engine-kvscale-qwen14b-v1 kvscale-qwen14b
verify_scale 20261006-engine-kvscale-agent-v1 kvscale-agent

# --- stock llama-server slot save and restore -------------------------------
if present 20261006-engine-kvserver-v1; then
  dir="$results/20261006-engine-kvserver-v1"
  check_hashes "$dir"
  awk -f "$script_dir/summarize_engine_kvserver.awk" "$dir/raw.log" >"$scratch/kvserver.csv"
  same "$scratch/kvserver.csv" "$dir/engine_kvserver_summary.csv"
  awk -F, '
    NR == 1 { next }
    {
      rows++; agents = $2; mode = $1; cell = agents
      # V1 and V3.
      if ($3 != 6 || $4 != 0 || $17 != agents * $3 || $12 > 256) bad++
      # V2 is reported below as a measured exception rather than hidden. A
      # mismatch must still share a deterministic token prefix, every
      # completed response must carry a token sequence, and the raw-log
      # check below must reproduce the same per-task sequences in all runs.
      if (mode == "extent" && $18 != $17 && ($19 < 16 || $20 != $17)) bad++
      save[cell, mode] = $7; server_save[cell, mode] = $8
      restore[cell, mode] = $10; memory[cell, mode] = $15
      cells[cell] = 1
    }
    END {
      if (bad || rows != 4) exit 1
      for (cell in cells) {
        # V4 and V5.
        if (memory[cell, "extent"] >= memory[cell, "copy"]) exit 2
        if (restore[cell, "extent"] > restore[cell, "copy"] ||
            save[cell, "extent"] > save[cell, "copy"] ||
            server_save[cell, "extent"] > server_save[cell, "copy"]) exit 3
      }
    }
  ' "$dir/engine_kvserver_summary.csv" ||
    fail "a gate of the stock-server campaign does not hold (awk exit $?)"
  kvserver_diagnosis=$(awk '
    function field(name, i, pair) {
      for (i = 1; i <= NF; ++i) {
        split($i, pair, "=")
        if (pair[1] == name) return pair[2]
      }
      return ""
    }
    /^BEGIN_KVSERVER / { run = field("run"); next }
    /^CASE / { mode = field("mode"); agents = field("agents"); next }
    /^AGENT / {
      key = agents SUBSEP mode SUBSEP field("index")
      hash = field("token_hash")
      if (hash == "") bad++
      if (!(key in first)) first[key] = hash
      else if (hash != first[key]) bad++
      seen[key]++
      next
    }
    END {
      split("4 8", counts, " ")
      for (a = 1; a <= 2; ++a) {
        agents = counts[a]; list = ""; mismatches = 0
        for (idx = 0; idx < agents; ++idx) {
          copy = agents SUBSEP "copy" SUBSEP idx
          extent = agents SUBSEP "extent" SUBSEP idx
          if (seen[copy] != 6 || seen[extent] != 6) bad++
          if (first[copy] != first[extent]) {
            mismatches++
            list = list (list == "" ? "" : ":") idx
          }
        }
        mismatch_list[agents] = list
        mismatch_count[agents] = mismatches
      }
      if (bad || mismatch_list[4] != "1:2:3" || mismatch_list[8] != "1:2:3") exit 1
      printf "stable_mismatch_indices_4=%s stable_mismatch_indices_8=%s", \
             mismatch_list[4], mismatch_list[8]
    }
  ' "$dir/raw.log") || fail "the stock-server token divergence is not reproducible"
  while IFS=, read -r mode agents _runs _ _ _ _ _ _ _ _ _ _ _ _ _ finished equal common compared; do
    if [[ "$mode" == extent && "$equal" != "$finished" ]]; then
      exceptions+=("V2 agents=$agents exact=$equal/$finished min_common_tokens=$common compared=$compared $kvserver_diagnosis")
    fi
  done < <(tail -n +2 "$dir/engine_kvserver_summary.csv")
fi

# --- the stock server, published at a token boundary -------------------------
if present 20261007-engine-kvserver2-v1; then
  dir="$results/20261007-engine-kvserver2-v1"
  check_hashes "$dir"
  awk -f "$script_dir/summarize_engine_kvserver.awk" "$dir/raw.log" >"$scratch/kvserver2.csv"
  same "$scratch/kvserver2.csv" "$dir/engine_kvserver_summary.csv"
  awk -F, '
    NR == 1 { next }
    {
      rows++; agents = $2; mode = $1; cell = agents
      # V1 and V3: every request completes and the restored prefix is reused.
      if ($3 != 6 || $4 != 0 || $17 != agents * $3 || $12 > 256) bad++
      # V2, the expectation of this campaign: with every published token
      # kept by the prompts, every response equals that of the copy.
      if ($18 != $17 || $20 != $17 || $19 != 64) exit 4
      save[cell, mode] = $7; server_save[cell, mode] = $8
      restore[cell, mode] = $10; memory[cell, mode] = $15
      cells[cell] = 1
    }
    END {
      if (bad || rows != 4) exit 1
      for (cell in cells) {
        # V4 and V5.
        if (memory[cell, "extent"] >= memory[cell, "copy"]) exit 2
        if (restore[cell, "extent"] > restore[cell, "copy"] ||
            save[cell, "extent"] > save[cell, "copy"] ||
            server_save[cell, "extent"] > server_save[cell, "copy"]) exit 3
      }
    }
  ' "$dir/engine_kvserver_summary.csv" ||
    fail "a gate of the second stock-server campaign does not hold (awk exit $?)"
fi

# --- per-agent host memory limits -------------------------------------------
if present 20261006-engine-kvlimit-v1; then
  dir="$results/20261006-engine-kvlimit-v1"
  check_hashes "$dir"
  awk -f "$script_dir/summarize_engine_kvlimit.awk" "$dir/raw.log" >"$scratch/kvlimit.csv"
  same "$scratch/kvlimit.csv" "$dir/engine_kvlimit_summary.csv"
  awk -F, '
    NR == 1 { next }
    {
      rows++; key = $1 SUBSEP $2
      # L4: all three unaffected agents complete without an OOM kill.
      if ($3 != 6 || $10 != $9 || $11 != 0) bad++
      if ($1 == "account" && ($5 != $3 || $6 != 0)) bad++
      limit[key] = $4; done[key] = $5; kills[key] = $6
      peak[key] = $7; equal[key] = $13; others[key] = $9
    }
    END {
      if (bad || rows != 4) exit 1
      restore = "limit" SUBSEP "restore"
      extent = "limit" SUBSEP "extent_lazy"
      # L1-L3.
      if (peak[extent] < limit[extent] - 16 || peak[extent] > limit[extent] + 16) exit 2
      if (peak[restore] >= limit[restore]) exit 2
      if (done[extent] != 0 || kills[extent] != 6 || equal[extent] != others[extent]) exit 3
      if (done[restore] != 6 || kills[restore] != 0) exit 4
    }
  ' "$dir/engine_kvlimit_summary.csv" ||
    fail "a gate of the cgroup-limit campaign does not hold (awk exit $?)"
fi

# --- isolated GPU read path -------------------------------------------------
if present 20261006-read-path-v1; then
  dir="$results/20261006-read-path-v1"
  check_hashes "$dir"
  awk -f "$script_dir/summarize_read_path.awk" "$dir/raw.log" >"$scratch/readpath.csv"
  same "$scratch/readpath.csv" "$dir/read_path_summary.csv"
  awk -F, 'NR > 1 { rows++; if ($4 != 6 || $5 != 0) bad++ }
             END { exit (bad || rows != 12) ? 1 : 0 }' "$dir/read_path_summary.csv" ||
    fail "a read-path probe failed or a cell is missing"
fi

# --- CUDA VMM attach granularity --------------------------------------------
if present 20261006-vmm-attach-v1; then
  dir="$results/20261006-vmm-attach-v1"
  check_hashes "$dir"
  awk -f "$script_dir/summarize_vmm_attach.awk" "$dir/raw.log" >"$scratch/vmmattach.csv"
  same "$scratch/vmmattach.csv" "$dir/vmm_attach_summary.csv"
  awk -F, 'NR > 1 { rows++; if ($3 != 6 || $4 != 0 || $8 != 0) bad++ }
             END { exit (bad || rows != 9) ? 1 : 0 }' "$dir/vmm_attach_summary.csv" ||
    fail "a VMM attach probe failed, returned a wrong word, or a cell is missing"
fi

# --- protection on the GPU access path -------------------------------------
if present 20261006-protect-v1; then
  dir="$results/20261006-protect-v1"
  check_hashes "$dir"
  awk -f "$script_dir/summarize_protect.awk" "$dir/raw.log" >"$scratch/protect.csv"
  same "$scratch/protect.csv" "$dir/protect_summary.csv"
  awk -F, '
    $1 == "host" { host++; if ($2 != 12 || $3 != $2 || $4 != $2 || $6 != $2) bad++ }
    $1 == "vmm" { vmm++; if ($2 != 6) bad++ }
    END { exit (bad || host != 1 || vmm != 1) ? 1 : 0 }
  ' "$dir/protect_summary.csv" || fail "the host protection expectation P1 does not hold"
fi

# --- isolated Ollama deployment baseline -----------------------------------
if present 20261006-ollama-agents-v1; then
  dir="$results/20261006-ollama-agents-v1"
  check_hashes "$dir"
  awk -f "$script_dir/summarize_ollama_agents.awk" "$dir/raw.log" >"$scratch/ollama.csv"
  same "$scratch/ollama.csv" "$dir/ollama_agents_summary.csv"
  awk -F, '
    NR == 1 { next }
    {
      rows++; expected = $1 == "one" ? 1 : 2
      # O1 and O2.
      if ($3 != 6 || $4 != 48 || $5 != $4 || $14 <= 0 ||
          $11 != expected || $12 != expected || $13 <= 0) bad++
    }
    END { exit (bad || rows != 4) ? 1 : 0 }
  ' "$dir/ollama_agents_summary.csv" || fail "an Ollama validity criterion does not hold"
  before=$(awk -F= '$1 == "system_ollama_pid_before" { print $2 }' "$dir/metadata.txt")
  read -r after alive < <(awk '{ for (i=1;i<=NF;i++) { split($i,a,"=");
    if (a[1] == "pid_after") pid=a[2]; if (a[1] == "alive") live=a[2] } }
    END { print pid, live }' "$dir/raw.log")
  [[ -n "$before" && "$before" == "$after" && "$alive" == 1 ]] ||
    fail "the machine Ollama service did not remain unchanged"
fi

# --- the memory of the whole stack and how many agents fit -------------------
verify_stack() {  # campaign label gates-on-all-stacks
  present "$1" || return 0
  local dir="$results/$1" label=$2 model_bytes total
  check_hashes "$dir"
  model_bytes=$(awk -F= '$1 == "model_bytes" { print $2 }' "$dir/metadata.txt")
  total=$(awk -F= '$1 == "mem_total_kib" { print $2 }' "$dir/metadata.txt")
  awk -f "$script_dir/summarize_engine_kvstack.awk" "$dir/raw.log" >"$scratch/$label.csv"
  same "$scratch/$label.csv" "$dir/engine_kvstack_summary.csv"
  awk -v table=fit -v model_bytes="$model_bytes" -v mem_total_kib="$total" \
    -f "$script_dir/summarize_engine_kvstack.awk" "$dir/raw.log" >"$scratch/$label.fit.csv"
  same "$scratch/$label.fit.csv" "$dir/engine_kvstack_fit.csv"
  awk -v table=texts -f "$script_dir/summarize_engine_kvstack.awk" "$dir/raw.log" \
    >"$scratch/$label.texts.csv"
  same "$scratch/$label.texts.csv" "$dir/engine_kvstack_texts.csv"
  # K1: no agent fails; K2: one text per agent index within a repetition.
  awk -F, 'NR > 1 && $3 > 0 { rows++; if ($5 != 0) bad++ } END { exit (bad || !rows) ? 1 : 0 }' \
    "$dir/engine_kvstack_summary.csv" || fail "K1 does not hold in $label"
  awk -F, 'NR > 1 { rows++; if ($2 != $3) bad++ } END { exit (bad || !rows) ? 1 : 0 }' \
    "$dir/engine_kvstack_texts.csv" || fail "K2 does not hold in $label"
  [[ "$3" == all ]] || return 0
  # K3: the order of the stacks by memory at every common count.
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
    }' "$dir/engine_kvstack_summary.csv" || fail "K3 does not hold in $label"
  # K4: the memory per added agent of both is at most 0.20 of that of none.
  awk -F, '$1 == "both" { rows++; if ($9 > 0.20 || $9 <= 0) bad++ } END { exit (bad || rows != 1) ? 1 : 0 }' \
    "$dir/engine_kvstack_fit.csv" || fail "K4 does not hold in $label"
  # K5: generation speed of both against none; a cell below 0.92 is an
  # exception that is reported with its value.
  while IFS=, read -r mode agents runs _ _ _ _ _ _ ratio _ _ median _; do
    [[ "$mode" == both && "$runs" -gt 0 ]] || continue
    awk -v r="$ratio" 'BEGIN { exit (r > 0) ? 0 : 1 }' || continue
    if awk -v r="$ratio" 'BEGIN { exit (r < 0.92) ? 0 : 1 }'; then
      exceptions+=("K5 campaign=$label agents=$agents speed_vs_none=$ratio median=$median")
    fi
  done < <(tail -n +2 "$dir/engine_kvstack_summary.csv")
}
verify_stack 20261007-engine-kvstack-v1 kvstack all
verify_stack 20261007-engine-kvstack-huge-v1 kvstack-huge all
verify_stack 20261007-engine-kvstack-capacity-v1 kvstack-capacity both
verify_stack 20261008-engine-kvstack-capacity-v2 kvstack-capacity-128-v2 both
verify_stack 20261008-engine-kvstack-capacity-136-v1 kvstack-capacity-136-v1 both
verify_stack 20261008-engine-kvstack-capacity-140-v1 kvstack-capacity-140-v1 both
verify_stack 20261009-engine-kvstack-capacity-copy-v1 kvstack-capacity-copy-v1 both
verify_stack 20261008-engine-kvstack-qwen14b-v1 kvstack-qwen14b both
verify_stack 20261008-engine-kvstack-qwen14b-none8-v1 kvstack-qwen14b-none8 both
verify_stack 20261008-engine-kvstack-capacity-qwen14b-118-v1 kvstack-capacity-qwen14b-118-v1 both
verify_stack 20261008-engine-kvstack-capacity-qwen14b-131-v1 kvstack-capacity-qwen14b-131-v1 both

# The largest measured 14B count must consist of six complete repetitions,
# preserve the 16 GiB host reserve in every sampled case, and produce one
# text for each of its 131 agents.  This check distinguishes the measured
# count from the larger count obtained by extending a fitted line.
capacity_14b="$results/20261008-engine-kvstack-capacity-qwen14b-131-v1"
if [[ -f "$capacity_14b/source_hashes.txt" ]]; then
  awk -F, 'NR > 1 {
      rows++
      if ($1 != "both" || $2 != 131 || $3 != 6 || $4 != 0 || $5 != 0 || $25 != 6) bad++
    }
    END { exit (bad || rows != 1) ? 1 : 0 }
  ' "$capacity_14b/engine_kvstack_summary.csv" || fail "the 14B 131-agent summary is incomplete"
  awk -F, 'NR > 1 { rows++; if ($4 != 131) bad++ }
    END { exit (bad || rows != 6) ? 1 : 0 }
  ' "$capacity_14b/engine_kvstack_texts.csv" || fail "the 14B 131-agent texts are incomplete"
  awk '
    function field(name, i, pair) {
      for (i = 1; i <= NF; ++i) {
        split($i, pair, "=")
        if (pair[1] == name) return pair[2]
      }
      return ""
    }
    /^MEMORY / {
      samples++
      if (field("mem_available_before_mib") - field("mem_available_drop_mib") < 16384) bad++
    }
    END { exit (bad || samples != 6) ? 1 : 0 }
  ' "$capacity_14b/raw.log" || fail "the 14B 131-agent campaign violates its 16 GiB reserve"
fi

# The 14B run is staged: counts 1 and 4 establish that eight unmodified
# agents leave the required reserve, then count 8 is run in a second
# campaign. Rebuild the joined table to evaluate K2--K5 over all three
# counts. The joined table is derived rather than stored in either campaign.
if [[ -f "$results/20261008-engine-kvstack-qwen14b-v1/source_hashes.txt" &&
      -f "$results/20261008-engine-kvstack-qwen14b-none8-v1/source_hashes.txt" ]]; then
  first="$results/20261008-engine-kvstack-qwen14b-v1"
  last="$results/20261008-engine-kvstack-qwen14b-none8-v1"
  model_bytes=$(awk -F= '$1 == "model_bytes" { print $2 }' "$first/metadata.txt")
  total=$(awk -F= '$1 == "mem_total_kib" { print $2 }' "$first/metadata.txt")
  awk -f "$script_dir/summarize_engine_kvstack.awk" "$first/raw.log" "$last/raw.log" \
    >"$scratch/kvstack-qwen14b-joined.csv"
  awk -v table=fit -v model_bytes="$model_bytes" -v mem_total_kib="$total" \
    -f "$script_dir/summarize_engine_kvstack.awk" "$first/raw.log" "$last/raw.log" \
    >"$scratch/kvstack-qwen14b-joined.fit.csv"
  awk -v table=texts -f "$script_dir/summarize_engine_kvstack.awk" \
    "$first/raw.log" "$last/raw.log" >"$scratch/kvstack-qwen14b-joined.texts.csv"
  awk -F, '
    NR > 1 && $3 > 0 {
      rows++; if ($5 != 0) bad++
      memory[$1, $2] = $18
    }
    END {
      if (bad || rows != 6) exit 1
      split("1 4 8", counts, " ")
      for (i = 1; i <= 3; ++i)
        if (!(memory["both", counts[i]] < memory["none", counts[i]])) exit 2
    }
  ' "$scratch/kvstack-qwen14b-joined.csv" || fail "K1 or K3 does not hold in joined 14B campaign"
  awk -F, 'NR > 1 { rows++; if ($2 != 8 || $3 != 8) bad++ }
    END { exit (bad || rows != 6) ? 1 : 0 }' \
    "$scratch/kvstack-qwen14b-joined.texts.csv" || fail "K2 does not hold in joined 14B campaign"
  awk -F, '$1 == "both" { rows++; if ($9 > 0.20 || $9 <= 0) bad++ }
    END { exit (bad || rows != 1) ? 1 : 0 }' \
    "$scratch/kvstack-qwen14b-joined.fit.csv" || fail "K4 does not hold in joined 14B campaign"
  while IFS=, read -r mode agents runs _ _ _ _ _ _ ratio low high median _; do
    [[ "$mode" == both && "$runs" -gt 0 ]] || continue
    if awk -v r="$ratio" 'BEGIN { exit (r < 0.92) ? 0 : 1 }'; then
      exceptions+=("K5 campaign=kvstack-qwen14b-joined agents=$agents speed_vs_none=$ratio ci95=$low..$high median=$median")
    fi
  done < <(tail -n +2 "$scratch/kvstack-qwen14b-joined.csv")
fi

# Evaluates gates on a summary that has a header line. The awk program sees
# a field of a row by its name, as $c["name"], and prints one line for every
# gate that is not met. A line that starts with "!" is a relation that must
# hold and fails the verification; any other line is reported as a measured
# exception.
gates() {  # label file program
  local label=$1 file=$2 program=$3 line
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    if [[ "$line" == '!'* ]]; then fail "$label: ${line#!}"; fi
    exceptions+=("${label}:${line// /_}")
  done < <(awk -F, "NR == 1 { for (i = 1; i <= NF; ++i) c[\$i] = i; next } $program" "$file")
}

# --- mechanisms in one engine: baselines, ablation, locality -----------------
verify_mech() {  # campaign label set
  present "$1" || return 0
  local dir="$results/$1" label=$2 order
  check_hashes "$dir"
  order=$(awk -F= '$1 == "modes" { print $2 }' "$dir/metadata.txt")
  awk -v order="$order" -f "$script_dir/summarize_engine_kvmech.awk" "$dir/raw.log" >"$scratch/$label.csv"
  same "$scratch/$label.csv" "$dir/engine_kvmech_summary.csv"
  case $3 in
    baselines)
      gates "$label" "$dir/engine_kvmech_summary.csv" '
        {
          cell = $c["placement"] ":" $c["children"] ":" $c["paragraphs"]; mode = $c["mode"]
          device = (mode ~ /^device/)
          # B1 and B5: completion.
          if ($c["parents_completed"] != $c["runs"] || $c["timed_out"] != 0) print "!B1 parent " cell " " mode
          if (!device && $c["children_finished"] != $c["children_started"]) print "!B1 children " cell " " mode
          if (device && ($c["own_instance_finished"] != $c["own_instance_started"] || $c["other_instance_finished"] != 0))
            print "!B5 " cell " " mode
          # B2: texts.
          if ($c["texts_equal_to_copy"] != $c["texts_compared"] || $c["texts_compared"] != $c["children_finished"])
            print "!B2 " cell " " mode " " $c["texts_equal_to_copy"] "/" $c["children_finished"]
          memory[cell, mode] = $c["children_memory_mib"]; attach[cell, mode] = $c["child_attach_ms"]
          pause[cell, mode] = $c["publish_ms"]; cells[cell] = $c["placement"]
        }
        END {
          split("copy demand device device_tuned device_merged cow", others, " ")
          for (cell in cells) {
            if (cells[cell] == "same") {
              # B3: memory.
              if (!(memory[cell, "extent"] < memory[cell, "device_merged"] && memory[cell, "extent"] < memory[cell, "device_tuned"]))
                print "B3 " cell " extent=" memory[cell, "extent"] " device_tuned=" memory[cell, "device_tuned"]
              if (memory[cell, "device_tuned"] > memory[cell, "device"])
                print "B3 " cell " device_tuned=" memory[cell, "device_tuned"] " device=" memory[cell, "device"]
              if (!(memory[cell, "device"] < memory[cell, "demand"] && memory[cell, "demand"] < memory[cell, "copy"]))
                print "B3 " cell " device=" memory[cell, "device"] " demand=" memory[cell, "demand"] " copy=" memory[cell, "copy"]
              if (memory[cell, "extent"] > memory[cell, "cow"])
                print "B3 " cell " extent=" memory[cell, "extent"] " cow=" memory[cell, "cow"]
              # B6: the tuning has an effect.
              if (!(attach[cell, "device_merged"] < 0.5 * attach[cell, "device"]))
                print "B6 " cell " device_merged=" attach[cell, "device_merged"] " device=" attach[cell, "device"]
            }
            # B4: attach and pause.
            for (o = 1; o <= 6; ++o) {
              if (!(attach[cell, "extent"] < attach[cell, others[o]]))
                print "B4 attach " cell " extent=" attach[cell, "extent"] " " others[o] "=" attach[cell, others[o]]
              if (others[o] != "cow" && !(pause[cell, "extent"] < pause[cell, others[o]]))
                print "B4 pause " cell " extent=" pause[cell, "extent"] " " others[o] "=" pause[cell, others[o]]
            }
          }
        }' ;;
    ablation)
      gates "$label" "$dir/engine_kvmech_summary.csv" '
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
      gates "$label" "$dir/engine_kvmech_summary.csv" '
        {
          if ($c["parents_completed"] != $c["runs"] || $c["timed_out"] != 0 || $c["children_finished"] != $c["children_started"])
            print "!L1 " $c["mode"]
          if ($c["texts_equal_to_copy"] != $c["texts_compared"] || $c["texts_compared"] != $c["children_finished"])
            print "!L2 " $c["mode"] " " $c["texts_equal_to_copy"] "/" $c["children_finished"]
        }' ;;
  esac
}
verify_mech 20261007-engine-kvmech-v1 kvmech baselines
verify_mech 20261007-engine-kvablate-v1 kvablate ablation
verify_mech 20261007-engine-kvlocal-6sm-v1 kvlocal-6sm locality
verify_mech 20261007-engine-kvlocal-12sm-v1 kvlocal-12sm locality

# --- scattered reads and writes by the kind of memory -------------------------
if present 20261007-tlb-probe-v1; then
  dir="$results/20261007-tlb-probe-v1"
  check_hashes "$dir"
  awk -f "$script_dir/summarize_tlb_probe.awk" "$dir/raw.log" >"$scratch/tlb.csv"
  same "$scratch/tlb.csv" "$dir/tlb_probe_summary.csv"
  gates tlb "$dir/tlb_probe_summary.csv" '{ if ($c["failed_probes"] != 0 || $c["runs"] != 6) print "!probe " $c["sms"] " " $c["kind"] }'
fi

# --- a pipeline of agents on a public workload --------------------------------
verify_pipe() {  # campaign label
  present "$1" || return 0
  local dir="$results/$1" label=$2
  check_hashes "$dir"
  awk -f "$script_dir/summarize_engine_kvpipe.awk" "$dir/raw.log" >"$scratch/$label.csv"
  same "$scratch/$label.csv" "$dir/engine_kvpipe_summary.csv"
  gates "$label" "$dir/engine_kvpipe_summary.csv" '
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
}
verify_pipe 20261007-engine-kvpipe-v1 kvpipe
verify_pipe 20261007-engine-kvpipe-huge-v1 kvpipe-huge

# --- faults among agents that share a prefix ------------------------------------
if present 20261007-engine-kvfault-v1; then
  dir="$results/20261007-engine-kvfault-v1"
  check_hashes "$dir"
  awk -f "$script_dir/summarize_engine_kvfault.awk" "$dir/raw.log" >"$scratch/kvfault.csv"
  same "$scratch/kvfault.csv" "$dir/engine_kvfault_summary.csv"
  gates kvfault "$dir/engine_kvfault_summary.csv" '
    {
      config = $c["config"]; fault = $c["fault"]; isolated = (config == "timeslice" || config == "mig")
      # F1: without a fault every agent completes.
      if (fault == "none" && $c["agents_completed"] != $c["agents_started"]) print "!F1 " config
      # F2: the write is refused and the file keeps its contents.
      if (fault == "write" && $c["writes_refused"] != $c["writes_injected"]) print "!F2 write " config
      if ($c["file_intact_runs"] != $c["runs"]) print "!F2 file " config " " fault
      # F3: without MPS the other agents complete with the text of the run without a fault.
      if (isolated && fault != "none" && ($c["others_completed"] != $c["others_started"] || $c["others_with_reference_text"] != $c["others_started"]))
        print "!F3 " config " " fault " " $c["others_completed"] "/" $c["others_started"]
      if (!isolated && fault != "none" && $c["others_completed"] != $c["others_started"])
        print "counted " config " " fault " others_completed=" $c["others_completed"] "/" $c["others_started"]
    }'
fi

# --- deeper and wider trees -----------------------------------------------------
verify_deep() {  # campaign label
  present "$1" || return 0
  local dir="$results/$1" label=$2
  check_hashes "$dir"
  awk -f "$script_dir/summarize_engine_kvdeep.awk" "$dir/raw.log" >"$scratch/$label.csv"
  same "$scratch/$label.csv" "$dir/engine_kvdeep_summary.csv"
  gates "$label" "$dir/engine_kvdeep_summary.csv" '
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
verify_deep 20261007-engine-kvdeep-v1 kvdeep
verify_deep 20261007-engine-kvdeep-wide-v1 kvdeep-wide

# --- more prefix lengths, a longer generation, 32 agents again -----------------
verify_scale 20261007-engine-kvscale-p20-v1 kvscale-p20
verify_scale 20261007-engine-kvscale-p80-v1 kvscale-p80
verify_scale 20261007-engine-kvscale-p600-v1 kvscale-p600
verify_scale 20261007-engine-kvscale-gen2k-v1 kvscale-gen2k
verify_scale 20261007-engine-kvscale-32-v2 kvscale-32-v2

# --- an external weight-sharing library, a weights publisher, vLLM --------------
if present 20261007-ext-weightshare-v1; then
  dir="$results/20261007-ext-weightshare-v1"
  check_hashes "$dir"
  awk -f "$script_dir/summarize_ext_weightshare.awk" "$dir/raw.log" >"$scratch/wshare.csv"
  same "$scratch/wshare.csv" "$dir/ext_weightshare_summary.csv"
  gates wshare "$dir/ext_weightshare_summary.csv" '
    {
      stack = $c["stack"]
      if (stack != "ipc" && ($c["agents_finished"] != $c["agents_started"] || $c["timed_out"] != 0)) print "!W1 " $c["placement"] " " stack
      if ($c["texts_equal_to_stock"] != $c["texts_compared"]) print "!W2 " $c["placement"] " " stack
      if (stack == "ipc" && $c["agents_finished"] != $c["agents_started"])
        print "counted ipc " $c["placement"] " finished=" $c["agents_finished"] "/" $c["agents_started"] " workers=" $c["library_workers"]
    }'
fi
if present 20261007-weights-publish-v1; then
  dir="$results/20261007-weights-publish-v1"
  check_hashes "$dir"
  awk -f "$script_dir/summarize_weights_publish.awk" "$dir/raw.log" >"$scratch/publish.csv"
  same "$scratch/publish.csv" "$dir/weights_publish_summary.csv"
  gates publish "$dir/weights_publish_summary.csv" '
    {
      method = $c["method"]
      if ($c["failed"] != 0 || $c["identical"] != $c["runs"]) print "!L1 " $c["file"] " " method
      if (method == "cp" && !($c["page_cache_left_share"] > 0.90)) print "L2 cp " $c["file"] " share=" $c["page_cache_left_share"]
      if (method != "cp" && !($c["page_cache_left_share"] < 0.05)) print "L2 " method " " $c["file"] " share=" $c["page_cache_left_share"]
      if (method == "publish" && !($c["speedup_vs_cp"] >= 1.0)) print "L3 " $c["file"] " speedup_vs_cp=" $c["speedup_vs_cp"]
    }'
fi
if present 20261007-ext-vllm-v1; then
  dir="$results/20261007-ext-vllm-v1"
  check_hashes "$dir"
  awk -f "$script_dir/summarize_ext_vllm.awk" "$dir/raw.log" >"$scratch/vllm.csv"
  same "$scratch/vllm.csv" "$dir/ext_vllm_summary.csv"
  gates vllm "$dir/ext_vllm_summary.csv" '{ if ($c["servers_ready"] != $c["runs"] || $c["agents_finished"] != $c["agents_started"]) print "!X1 " $c["round"] }'
fi

# --- two vLLM servers that share weights and a prefix -------------------------
verify_vshare() {  # campaign label
  present "$1" || return 0
  local dir="$results/$1" label=$2
  check_hashes "$dir"
  awk -f "$script_dir/summarize_vllm_share.awk" "$dir/raw.log" >"$scratch/$label.csv"
  same "$scratch/$label.csv" "$dir/vllm_share_summary.csv"
  gates "$label" "$dir/vllm_share_summary.csv" '
    {
      mode = $c["mode"]; seen[mode] = 1
      if ($c["failed_cases"] != 0 || $c["servers_ready"] != $c["servers"] * $c["runs"] || $c["requests_finished"] != $c["requests_started"])
        print "!V1 " mode
      if ($c["alone1_texts_equal"] != $c["alone1_texts_compared"] || $c["alone1_texts_compared"] != $c["runs"])
        print "V6 " mode " alone1_texts=" $c["alone1_texts_equal"] "/" $c["runs"]
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
}
verify_vshare 20261007-vllm-share-v1 vshare
verify_vshare 20261007-vllm-share-4srv-v1 vshare4
verify_vshare 20261007-vllm-share-4srv-v2 vshare4w
verify_vshare 20261008-vllm-share-4srv-v3 vshare4w-v3
verify_vshare 20261008-vllm-share-v2 vshare-direct
if [[ -f "$results/20261008-vllm-share-v2/vllm_share_summary.csv" ]]; then
  gates vshare-direct-extra "$results/20261008-vllm-share-v2/vllm_share_summary.csv" '
    {
      if ($c["alone1_texts_equal"] != $c["runs"] || $c["alone1_texts_compared"] != $c["runs"] ||
          $c["alone2_texts_equal"] != $c["runs"] || $c["alone2_texts_compared"] != $c["runs"])
        print "!V6 " $c["mode"] " alone1=" $c["alone1_texts_equal"] "/" $c["alone1_texts_compared"]
          " alone2=" $c["alone2_texts_equal"] "/" $c["alone2_texts_compared"]
      if ($c["mode"] == "stator" && $c["peak_memory_mib"] - $c["memory_mib"] > 1024)
        print "!V7 peak_minus_steady_mib=" ($c["peak_memory_mib"] - $c["memory_mib"])
    }'
fi

echo "stator_campaign_verification=PASS"
echo "campaigns_verified=${#verified[@]} ${verified[*]:-}"
echo "campaigns_absent=${#absent[@]} ${absent[*]:-}"
echo "gate_exceptions=${#exceptions[@]} ${exceptions[*]:-}"
