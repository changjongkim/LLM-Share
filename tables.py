#!/usr/bin/env python3
"""Renders the markdown tables of the LLM-serving research record.

usage: tables.py LLM_SHARE_RESULTS HOSTMM_RESULTS OUTPUT_DIR
"""
import csv
import os
import sys

engine_root, hostmm_root, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)


def rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write(name, text):
    with open(os.path.join(out, name), "w") as handle:
        handle.write(text)


CONFIG_NAMES = {
    "mig": "one MIG instance each (alternating)",
    "timeslice": "one MIG instance, time-sliced",
    "mps": "one MIG instance, one MPS server",
    "mig_mps": "two MIG instances, an MPS server in each",
}
CONFIG_ORDER = ["timeslice", "mig", "mps", "mig_mps"]

# --- one serving process -------------------------------------------------------
path = os.path.join(engine_root, "20261005-engine-single-v1", "engine_single_summary.csv")
if os.path.exists(path):
    names = {"copy": "device copy (upstream)",
             "inplace": "in place, CPU reads every page at load",
             "inplace_noread": "in place, no CPU read"}
    text = ("| Weights | Runs (failed) | Identical text | Prompt (tokens/s) | "
            "Generation (tokens/s) | Generation against device copy | Load (ms) | "
            "Memory outside the model file (MiB) | Model file mapped (MiB) |\n"
            "|---|---:|---:|---:|---:|---|---:|---:|---:|\n")
    for r in rows(path):
        text += (f"| {names[r['mode']]} | {r['runs']} ({r['failed_runs']}) | "
                 f"{r['identical_text_runs']}/{r['runs']} | {float(r['prompt_tps']):.0f} | "
                 f"{float(r['generation_tps']):.2f} | {float(r['generation_vs_copy']):.3f}x "
                 f"[{float(r['lower_ci95']):.3f}, {float(r['upper_ci95']):.3f}] | "
                 f"{float(r['load_ms']):.0f} | {r['device_memory_mib']} | "
                 f"{r['model_mapped_mib']} |\n")
    write("engine_single.md", text)

# --- page size of the mapped model ---------------------------------------------
path = os.path.join(engine_root, "20261005-engine-pages-v1", "engine_pages_summary.csv")
if os.path.exists(path):
    names = {"copy": "device copy (upstream)",
             "inplace_4k": "in place, ext4 page cache (4 KiB)",
             "inplace_thp": "in place, tmpfs with huge pages (2 MiB)",
             "inplace_hugetlb": "in place, hugetlbfs (2 MiB)"}
    text = ("| Weights | Runs (failed) | Identical text | Prompt (tokens/s) | "
            "Generation (tokens/s) | Generation against device copy | Load (ms) |\n"
            "|---|---:|---:|---:|---:|---|---:|\n")
    for r in rows(path):
        text += (f"| {names[r['mode']]} | {r['runs']} ({r['failed_runs']}) | "
                 f"{r['identical_text_runs']}/{r['runs']} | {float(r['prompt_tps']):.0f} | "
                 f"{float(r['generation_tps']):.2f} | {float(r['generation_vs_copy']):.3f}x "
                 f"[{float(r['lower_ci95']):.3f}, {float(r['upper_ci95']):.3f}] | "
                 f"{float(r['load_ms']):.0f} |\n")
    write("engine_pages.md", text)

# --- N serving processes ---------------------------------------------------------
path = os.path.join(engine_root, "20261005-engine-agents-v1", "engine_agents_summary.csv")
if os.path.exists(path):
    data = {(r["config"], r["mode"], r["agents"]): r for r in rows(path)}
    text = ("| Compute shared by | Agents | Generation, device copy (tokens/s, sum) | "
            "Generation, in place (tokens/s, sum) | In place / device copy | "
            "Memory, device copy (MiB) | Memory, in place (MiB) |\n"
            "|---|---:|---:|---:|---:|---:|---:|\n")
    for config in CONFIG_ORDER:
        for agents in ["1", "2", "4", "8"]:
            copy = data.get((config, "copy", agents))
            inplace = data.get((config, "inplace", agents))
            if not copy or not inplace:
                continue
            text += (f"| {CONFIG_NAMES[config]} | {agents} | "
                     f"{float(copy['generation_tps_total']):.1f} +/- {float(copy['generation_total_ci95']):.1f} | "
                     f"{float(inplace['generation_tps_total']):.1f} +/- {float(inplace['generation_total_ci95']):.1f} | "
                     f"{float(inplace['generation_tps_total']) / float(copy['generation_tps_total']):.3f}x | "
                     f"{float(copy['memory_total_mib']):.0f} | {float(inplace['memory_total_mib']):.0f} |\n")
    write("engine_agents.md", text)

# --- one process, N sequences ----------------------------------------------------
path = os.path.join(engine_root, "20261005-engine-batched-v1", "engine_batched_summary.csv")
if os.path.exists(path):
    data = [r for r in rows(path) if r["mode"] in ("copy", "inplace")]
    table = {(r["mode"], r["sequences"]): r for r in data}
    text = ("| Sequences in one process | Generation, device copy (tokens/s, sum) | "
            "Generation, in place (tokens/s, sum) | Per sequence, in place (tokens/s) |\n"
            "|---:|---:|---:|---:|\n")
    for sequences in sorted({r["sequences"] for r in data}, key=int):
        copy = table.get(("copy", sequences))
        inplace = table.get(("inplace", sequences))
        if not copy or not inplace:
            continue
        text += (f"| {sequences} | {float(copy['generation_tps_total']):.1f} | "
                 f"{float(inplace['generation_tps_total']):.1f} | "
                 f"{float(inplace['generation_tps_per_sequence']):.1f} |\n")
    write("engine_batched.md", text)

# --- a batching server per MIG instance ------------------------------------------
path = os.path.join(engine_root, "20261005-engine-groups-v1", "engine_groups_summary.csv")
if os.path.exists(path):
    names = {"copy": "device copy", "inplace": "in place"}
    text = ("| Weights | Sequences per server | Runs (failed) | Generation, both servers (tokens/s) | "
            "12-SM instance | 6-SM instance | Memory outside the model file (MiB) | "
            "Model file mapped (MiB) | Total (MiB) |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows(path):
        text += (f"| {names[r['mode']]} | {r['sequences_per_server']} | "
                 f"{r['runs']} ({r['failed_runs']}) | "
                 f"{float(r['generation_tps_total']):.1f} +/- {float(r['generation_tps_sd']):.1f} | "
                 f"{float(r['instance_a_tps']):.1f} | {float(r['instance_b_tps']):.1f} | "
                 f"{r['device_memory_mib']} | {r['model_mapped_mib']} | "
                 f"{r['memory_total_mib']} |\n")
    write("engine_groups.md", text)

# --- prompt prefix ----------------------------------------------------------------
path = os.path.join(engine_root, "20261005-engine-prefix-v1", "engine_prefix_summary.csv")
if os.path.exists(path):
    text = ("| Prefix (tokens) | Process that recomputes it (ms) | of which prompt evaluation (ms) | "
            "Process that restores it from the cache file (ms) | Cache file (MiB) | "
            "KiB per token |\n|---:|---:|---:|---:|---:|---:|\n")
    for r in rows(path):
        text += (f"| {r['prefix_tokens']} | {r['recompute_wall_ms']} | {r['prefill_ms']} | "
                 f"{r['restore_wall_ms']} | {r['cache_mib']} | {r['cache_kib_per_token']} |\n")
    write("engine_prefix.md", text)

# --- agents that start from one computed prefix ----------------------------------
KV_MODES = {
    "recompute": "recompute",
    "restore": "copy from the state file (upstream)",
    "cow": "copy-on-write mapping, 2 MiB file",
    "cow_small": "copy-on-write mapping, 4 KiB file",
    "extent": "extent, whole tail allocated",
    "extent_lazy": "extent, tail follows use",
    "extent_lazy_small": "extent, tail follows use, 4 KiB file",
}
kv_dir = os.path.join(engine_root, "20261006-engine-kvshare-v1")
path = os.path.join(kv_dir, "engine_kvshare_publish.csv")
if os.path.exists(path):
    names = {"device": "save the state file (upstream)",
             "huge": "publish, cache file with 2 MiB pages",
             "small": "publish, cache file with 4 KiB pages"}
    text = ("| Prefix (tokens) | Way | Runs (failed) | Compute the prefix (ms) | "
            "Hand it over (ms) | State file (MiB) | Cache file in use (MiB) |\n"
            "|---:|---|---:|---:|---:|---:|---:|\n")
    for r in rows(path):
        used = "-" if r["store"] == "device" else r["file_used_mib"]
        text += (f"| {r['prefix_tokens']} | {names[r['store']]} | {r['runs']} ({r['failed_runs']}) | "
                 f"{r['prefix_ms']} | {float(r['publish_ms']):.1f} | "
                 f"{float(r['state_mib']):.2f} | {used} |\n")
    write("kv_publish.md", text)
path = os.path.join(kv_dir, "engine_kvshare_summary.csv")
if os.path.exists(path):
    data = rows(path)
    for size in sorted({r["paragraphs"] for r in data}, key=int):
        text = ("| Agents | Way to obtain the prefix | Texts equal to copy | "
                "Texts equal to recompute: publisher's instance, other instance | Attach (ms) | "
                "First token after process start (ms) | Generation, sum (tokens/s) | "
                "Generation against copy | Memory of the agents (MiB) | "
                "Prefix pages the agents map: resident sum / proportional sum (MiB) |\n"
                "|---:|---|---:|---:|---:|---:|---:|---|---:|---:|\n")
        for r in data:
            if r["paragraphs"] != size:
                continue
            ratio = "-"
            if r["mode"] != "restore":
                ratio = (f"{float(r['generation_vs_restore']):.3f}x "
                         f"[{float(r['lower_ci95']):.3f}, {float(r['upper_ci95']):.3f}]")
            mapped = "-"
            if r["mode"] not in ("recompute", "restore"):
                mapped = f"{r['prefix_rss_mib']} / {r['prefix_pss_mib']}"
            other = "-"
            if int(r["other_instance_compared"]) > 0:
                other = f"{r['other_instance_equal_to_recompute']}/{r['other_instance_compared']}"
            copy = "-" if r["mode"] == "recompute" else f"{r['texts_equal_to_restore']}/{r['texts_compared']}"
            text += (f"| {r['agents']} | {KV_MODES[r['mode']]} | {copy} | "
                     f"{r['own_instance_equal_to_recompute']}/{r['own_instance_compared']}, {other} | "
                     f"{float(r['attach_ms']):.0f} | {r['first_token_ms']} | "
                     f"{float(r['generation_tps_total']):.1f} | {ratio} | "
                     f"{r['memory_mib']} | {mapped} |\n")
        tokens = next(r["prefix_tokens"] for r in data if r["paragraphs"] == size)
        write(f"kv_agents_{tokens}.md", text)

# --- a parent that forks its state -----------------------------------------------
path = os.path.join(engine_root, "20261006-engine-kvfork-v1", "engine_kvfork_summary.csv")
if os.path.exists(path):
    names = {"copy": "copy (state file with the rows)", "extent": "extent (freeze and map)"}
    text = ("| Hand-over | Runs (failed processes) | Pause of the parent (ms) | State file (MiB) | "
            "Parent text equal to alone | Child texts equal to copy | "
            "Child texts equal to alone: parent's instance, other instance | Child attach (ms) | "
            "Child first token (ms) | Memory of the children (MiB) | "
            "Prefix pages the children map: resident sum / proportional sum (MiB) |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows(path):
        mapped = "-" if r["mode"] == "copy" else f"{r['prefix_rss_mib']} / {r['prefix_pss_mib']}"
        text += (f"| {names[r['mode']]} | {r['runs']} ({r['failed_processes']}) | "
                 f"{float(r['publish_ms']):.1f} | {float(r['state_mib']):.2f} | "
                 f"{r['parent_texts_equal']}/{r['runs']} | "
                 f"{r['child_texts_equal_to_copy']}/{r['children_started']} | "
                 f"{r['own_instance_equal_to_alone']}/{r['own_instance_compared']}, "
                 f"{r['other_instance_equal_to_alone']}/{r['other_instance_compared']} | "
                 f"{float(r['child_attach_ms']):.0f} | {r['child_first_token_ms']} | "
                 f"{r['children_memory_mib']} | {mapped} |\n")
    write("kv_fork.md", text)

# --- the cache of a prefix, bit for bit, across repetitions and instances ----------
path = os.path.join(engine_root, "20261006-engine-kvdet-v1", "engine_kvdet_summary.csv")
if os.path.exists(path):
    text = ("| Prefix (tokens) | Runs (failed) | Distinct caches, 12-SM instance | "
            "Distinct caches, 6-SM instance | Runs with the same bits in both instances | "
            "16-bit values that differ (mean) | Largest difference | First tensor that differs |\n"
            "|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows(path):
        text += (f"| {r['prefix_tokens']} | {r['runs']} ({r['failed']}) | "
                 f"{r['distinct_caches_instance_a']} | {r['distinct_caches_instance_b']} | "
                 f"{r['runs_with_equal_caches_across_instances']}/{r['runs']} | "
                 f"{r['values_that_differ_mean']} | {r['largest_difference']} | "
                 f"{r['first_tensor_that_differs']} |\n")
    write("kv_det.md", text)

# --- composing shared and private device memory ----------------------------------
path = os.path.join(engine_root, "20261006-vmm-routes-v1", "vmm_routes_summary.csv")
if os.path.exists(path):
    placements = {"same_instance": "two processes in one MIG instance",
                  "across_instances": "one process in each MIG instance",
                  "same_mps_server": "two clients of one MPS server"}
    text = ("| Placement | Shared and private device memory composed | Refusal | "
            "Read-only access to the shared part | Producer's data unchanged |\n"
            "|---|---:|---|---|---:|\n")
    for r in rows(path):
        refusal = "-" if r["refusal"] == "-" else f"`{r['refusal']}`"
        text += (f"| {placements[r['placement']]} | {r['works']}/{r['runs']} | {refusal} | "
                 f"{r['read_only_access']} | {r['producer_intact']}/{r['works']} |\n")
    write("vmm_routes.md", text)

# --- copy-on-write with and without the remedies ----------------------------------
path = os.path.join(engine_root, "20261006-engine-kvcow-v1", "engine_kvcow_summary.csv")
if os.path.exists(path):
    names = {"restore": "copy from the state file (upstream)",
             "cow": "copy-on-write, 2 MiB file, CPU read pass",
             "cow_noread": "copy-on-write, 2 MiB file, no read pass",
             "cow_small": "copy-on-write, 4 KiB file, CPU read pass",
             "cow_small_noread": "copy-on-write, 4 KiB file, no read pass",
             "extent_lazy": "extent, tail follows use"}
    text = ("| Agents | Way to obtain the prefix | Runs (failed agents) | Texts equal to copy | "
            "Decode of the agent's own task (ms) | First token after process start (ms) | "
            "Generation against copy | Memory of the agents (MiB) | "
            "Per agent above extents (MiB) | "
            "Prefix pages the agents map: resident sum / proportional sum (MiB) |\n"
            "|---:|---|---:|---:|---:|---:|---|---:|---:|---:|\n")
    for r in rows(path):
        ratio = "-"
        if r["mode"] != "restore":
            ratio = (f"{float(r['generation_vs_restore']):.3f}x "
                     f"[{float(r['lower_ci95']):.3f}, {float(r['upper_ci95']):.3f}]")
        mapped = "-" if r["mode"] == "restore" else f"{r['prefix_rss_mib']} / {r['prefix_pss_mib']}"
        text += (f"| {r['agents']} | {names[r['mode']]} | {r['runs']} ({r['failed_agents']}) | "
                 f"{r['texts_equal_to_restore']}/{r['texts_compared']} | {r['suffix_ms']} | "
                 f"{r['first_token_ms']} | {ratio} | {r['memory_mib']} | "
                 f"{r['memory_per_agent_above_extent_mib']} | {mapped} |\n")
    write("kv_cow.md", text)

# --- where the generation speed goes -----------------------------------------------
speed_runs = [("20261006-engine-kvspeed-v1", "12-SM"), ("20261006-engine-kvspeed-long-v1", "12-SM"),
              ("20261006-engine-kvspeed-6sm-v1", "6-SM"), ("20261006-engine-kvspeed-6sm-long-v1", "6-SM")]
names = {"device": "device memory, prefix computed (upstream)",
         "anon": "private host memory, 2 MiB pages, prefix computed",
         "anon_lazy": "private host memory that follows use, 4 KiB pages, prefix computed",
         "restore": "device memory, prefix copied from the state file",
         "extent": "mapped prefix, private tail on 2 MiB pages",
         "extent_lazy": "mapped prefix, private tail that follows use"}
text = ("| MIG instance | Prefix (tokens) | Tokens generated | Processes | Cache | Runs (failed) | "
        "Generation, sum (tokens/s) | Against device memory |\n|---|---:|---:|---:|---|---:|---:|---|\n")
found = False
for tag, instance in speed_runs:
    path = os.path.join(engine_root, tag, "engine_kvspeed_summary.csv")
    if not os.path.exists(path):
        continue
    found = True
    generated = "?"
    with open(os.path.join(engine_root, tag, "metadata.txt")) as handle:
        for line in handle:
            if line.startswith("n_gen="):
                generated = line.strip().split("=", 1)[1]
    for r in rows(path):
        ratio = "-"
        if r["mode"] != "device":
            ratio = (f"{float(r['generation_vs_device']):.3f}x "
                     f"[{float(r['lower_ci95']):.3f}, {float(r['upper_ci95']):.3f}]")
        text += (f"| {instance} | {r['prefix_tokens']} | {generated} | {r['processes']} | "
                 f"{names[r['mode']]} | {r['runs']} ({r['failed_agents']}) | "
                 f"{float(r['generation_tps_total']):.2f} | {ratio} |\n")
if found:
    write("kv_speed.md", text)

# --- agents that may share a process ------------------------------------------------
path = os.path.join(engine_root, "20261006-engine-kvbatch-v1", "engine_kvbatch_summary.csv")
if os.path.exists(path):
    names = {"one_server": "one server in the 12-SM instance, 8 sequences",
             "two_servers_compute": "a server in each instance, 4 sequences each; each computes the prefix",
             "two_servers_copy": "a server in each instance; the second copies the state file",
             "two_servers_extent": "a server in each instance; the second maps the published prefix"}
    text = ("| Configuration | Runs (failed servers) | Generation, all agents (tokens/s) | "
            "12-SM server | 6-SM server | First server ready (s) | All agents ready (s) | "
            "Hand-over (ms) | State file (MiB) | Memory of all servers (MiB) | "
            "Texts equal to the copy configuration |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows(path):
        second = "-" if r["config"] == "one_server" else f"{float(r['generation_tps_6sm']):.1f}"
        hand = "-" if float(r["publish_ms"]) == 0 else f"{float(r['publish_ms']):.1f}"
        state = "-" if float(r["state_mib"]) == 0 else f"{float(r['state_mib']):.2f}"
        text += (f"| {names[r['config']]} | {r['runs']} ({r['failed_servers']}) | "
                 f"{float(r['generation_tps_total']):.1f} | {float(r['generation_tps_12sm']):.1f} | "
                 f"{second} | {float(r['ready_first_server_ms']) / 1000:.1f} | "
                 f"{float(r['ready_all_servers_ms']) / 1000:.1f} | {hand} | {state} | "
                 f"{r['memory_mib']} | {r['texts_equal_to_copy']}/{r['texts_compared']} |\n")
    write("kv_batch.md", text)

# --- the device-memory counterpart, in the engine ------------------------------------
path = os.path.join(engine_root, "20261006-engine-kvvmm-v1", "engine_kvvmm_summary.csv")
if os.path.exists(path):
    instances = {"fafc828a": "12-SM", "31ffbfe4": "6-SM"}
    names = {"copy": "copy (state file with the rows)",
             "extent": "host memory: mapped file, private tail",
             "vmm": "device memory: shared allocations, private tail",
             "extent_cross": "host memory, one child in the other instance",
             "vmm_cross": "device memory, one child in the other instance"}
    text = ("| Parent's instance | Prefix (tokens) | Hand-over | Runs | Pause of the parent (ms) | "
            "State file (MiB) | Children that ran | Child texts equal to copy | "
            "Child attach (ms) | Child first token (ms) | Child generation (tokens/s, each) | "
            "Parent generation (tokens/s) | Memory of the children (MiB) |\n"
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows(path):
        ran = f"{r['children_finished']}/{r['children_started']}"
        if int(r["children_finished"]) == 0:
            cells = "- | - | - | -"
        else:
            cells = (f"{r['child_texts_equal_to_copy']}/{r['children_finished']} | "
                     f"{float(r['child_attach_ms']):.0f} | {r['child_first_token_ms']} | "
                     f"{float(r['child_generation_tps']):.2f}")
        text += (f"| {instances.get(r['parent_mig'], r['parent_mig'])} | {r['prefix_tokens']} | "
                 f"{names[r['mode']]} | {r['runs']} | {float(r['publish_ms']):.1f} | "
                 f"{float(r['state_mib']):.2f} | {ran} | {cells} | "
                 f"{float(r['parent_generation_tps']):.2f} | {r['children_memory_mib']} |\n")
    write("kv_vmm.md", text)

# --- adapters --------------------------------------------------------------------
path = os.path.join(engine_root, "20261005-engine-adapters-v1", "engine_adapters_summary.csv")
if os.path.exists(path):
    names = {"copy": "device copy (upstream)", "inplace": "in place"}
    text = ("| Base weights | Runs (failed) | Runs with five distinct texts | "
            "Texts equal to the device-copy text | Memory outside the model file (MiB) | "
            "Model file mapped (MiB) | Total (MiB) |\n|---|---:|---:|---:|---:|---:|---:|\n")
    for r in rows(path):
        text += (f"| {names[r['mode']]} | {r['runs']} ({r['failed_runs']}) | "
                 f"{r['runs_with_five_distinct_texts']}/{r['runs']} | "
                 f"{r['texts_equal_to_copy']}/{r['texts_compared']} | "
                 f"{r['device_memory_mib']} | {r['model_mapped_mib']} | "
                 f"{r['memory_total_mib']} |\n")
    write("engine_adapters.md", text)

# --- sharing routes --------------------------------------------------------------
path = os.path.join(engine_root, "20261005-sharing-routes-v2", "sharing_routes_summary.csv")
if os.path.exists(path):
    data = {(r["route"], r["placement"]): r for r in rows(path)}
    placements = [("same_instance", "two processes in one MIG instance"),
                  ("across_instances", "one process in each MIG instance"),
                  ("same_mps_server", "two clients of one MPS server")]
    text = ("| Placement | CUDA IPC | Host page table |\n|---|---|---|\n")
    for key, label in placements:
        cells = []
        for route in ["cuda_ipc", "host_page_table"]:
            r = data[(route, key)]
            cell = f"works {r['works']}/{r['runs']}"
            if r["refusal"] != "-":
                cell += f" (`{r['refusal']}`)"
            cells.append(cell)
        text += f"| {label} | {cells[0]} | {cells[1]} |\n"
    write("sharing_routes.md", text)

# --- compute-sharing matrix of the sealed base -----------------------------------
modes_dir = os.path.join(hostmm_root, "20261005-sharing-modes-v1")
path = os.path.join(modes_dir, "sharing_perf_summary.csv")
if os.path.exists(path):
    data = {(r["config"], r["mode"], r["tenants"]): r for r in rows(path)}
    text = ("| Compute shared by | Tenants | Read at the same time, copies (ms) | "
            "Read at the same time, shared base (ms) | Slowdown against a tenant alone, shared base | "
            "Memory, copies (MiB) | Memory, shared base (MiB) |\n"
            "|---|---:|---:|---:|---:|---:|---:|\n")
    for config in CONFIG_ORDER:
        for tenants in ["2", "4", "8"]:
            copy = data[(config, "copy", tenants)]
            shared = data[(config, "ro_preread", tenants)]
            text += (f"| {CONFIG_NAMES[config]} | {tenants} | "
                     f"{float(copy['concurrent_read_ms']):.1f} | "
                     f"{float(shared['concurrent_read_ms']):.1f} | "
                     f"{float(shared['slowdown_when_concurrent']):.2f}x | "
                     f"{float(copy['memory_mib']):.0f} | {float(shared['memory_mib']):.0f} |\n")
    write("modes_perf.md", text)
    fault = rows(os.path.join(modes_dir, "sharing_fault_summary.csv"))
    relation = {"same_mps_server": "client of the same MPS server",
                "same_instance": "same MIG instance, time-sliced",
                "other_instance": "other MIG instance"}
    names = {"timeslice": "no MPS", "mps": "MPS server in the faulting tenant's instance",
             "mig_mps": "MPS server in each instance"}
    text = ("| Configuration | Victim is | Victims | Finished the read in progress | "
            "Read again afterwards | Base intact |\n|---|---|---:|---:|---:|---:|\n")
    for config in ["timeslice", "mps", "mig_mps"]:
        for r in fault:
            if r["config"] != config:
                continue
            who = relation[r["relation"]]
            if r["relation"] == "other_instance" and r["victim_is_mps_client"] == "1":
                who += ", client of its own MPS server"
            text += (f"| {names[config]} | {who} | {r['victims']} | "
                     f"{r['survived_during']}/{r['victims']} | "
                     f"{r['survived_after']}/{r['victims']} | "
                     f"{r['base_intact']}/{r['victims']} |\n")
    write("modes_fault.md", text)

print("tables written to", out)
