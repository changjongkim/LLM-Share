#!/usr/bin/env bash
# The substrate record and the prior-art audits are written in the working
# tree one directory above this repository, next to the substrate project.
# Copies them here when they are newer. The record of this repository,
# RESEARCH_LLM_SHARE_2026-10-05.md, is built by record_src/assemble.py.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for name in RESEARCH_HOSTMM_2026-10-05.md SOTA_HOSTMM_2026-10-05.md; do
  if [[ -f "$here/../../$name" ]]; then cp -u "$here/../../$name" "$here/$name"; fi
done
