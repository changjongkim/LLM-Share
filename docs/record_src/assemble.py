#!/usr/bin/env python3
"""Assembles docs/RESEARCH_LLM_SHARE_2026-10-05.md from the sections and tables here.

The tables are written by ../../tables.py into this directory:
  python3 tables.py results ../thor_hostmm/results docs/record_src
  python3 docs/record_src/assemble.py
"""
import os
import sys

docs = os.path.dirname(os.path.abspath(__file__))
record = os.path.join(docs, "..", "RESEARCH_LLM_SHARE_2026-10-05.md")
# In the working tree this repository sits in, a copy of the record is kept
# one directory above the repository; it is refreshed when it exists.
workspace_copy = os.path.join(docs, "..", "..", "..", "RESEARCH_LLM_SHARE_2026-10-05.md")


def read(name):
    with open(os.path.join(docs, name)) as handle:
        return handle.read().rstrip("\n")


text = "\n\n".join(read(f"s{n}.md") for n in [1, 3, 4, 5, 6, 7, 8, 9])
text = text.replace("@PATCH_STAT@", "59 lines added, 6 removed")
tables = {
    "@ENGINE_SINGLE@": "engine_single.md", "@ENGINE_PAGES@": "engine_pages.md",
    "@ENGINE_ADAPTERS@": "engine_adapters.md", "@ENGINE_AGENTS@": "engine_agents.md",
    "@ENGINE_BATCHED@": "engine_batched.md", "@MODES_PERF@": "modes_perf.md",
    "@MODES_FAULT@": "modes_fault.md", "@ROUTES@": "sharing_routes.md",
    "@ENGINE_GROUPS@": "engine_groups.md", "@ENGINE_PREFIX@": "engine_prefix.md",
    "@KV_PUBLISH@": "kv_publish.md", "@KV_AGENTS_4081@": "kv_agents_4081.md",
    "@KV_AGENTS_16321@": "kv_agents_16321.md", "@KV_FORK@": "kv_fork.md",
    "@KV_DET@": "kv_det.md", "@VMM_ROUTES@": "vmm_routes.md",
    "@KV_SPEED@": "kv_speed.md", "@KV_COW@": "kv_cow.md",
    "@KV_BATCH@": "kv_batch.md", "@KV_VMM@": "kv_vmm.md",
    "@KV_MPS@": "kv_mps.md",
    "@KV_SOTA@": "kv_sota.md", "@KV_TREE@": "kv_tree.md",
    "@KV_SCALE@": "kv_scale.md", "@KV_SERVER@": "kv_server.md", "@KV_SERVER2@": "kv_server2.md",
    "@KV_LIMIT@": "kv_limit.md", "@READ_PATH@": "read_path.md",
    "@VMM_ATTACH@": "vmm_attach.md", "@PROTECT@": "protect.md",
    "@OLLAMA_AGENTS@": "ollama_agents.md",
}
for marker, name in tables.items():
    path = os.path.join(docs, name)
    if marker in text and os.path.exists(path):
        text = text.replace(marker, read(name))
leftover = sorted({word for word in text.split() if word.startswith("@") and word.endswith("@")})
if leftover and "--allow-missing" not in sys.argv:
    sys.exit("unfilled placeholders: " + ", ".join(leftover))
with open(record, "w") as handle:
    handle.write(text + "\n")
if os.path.exists(workspace_copy):
    with open(workspace_copy, "w") as handle:
        handle.write(text + "\n")
print("assembled", record, len(text.split("\n")), "lines", "missing:", leftover)
