#!/usr/bin/env bash
# Copies the figures that the paper includes into the working tree of the
# paper (a separate repository): every PDF named by an \includegraphics of
# PAPER_DIR/*.tex that exists here goes to PAPER_DIR/Figures/. A figure
# whose campaign has not been run yet is listed as missing and left as it is
# in the paper tree.
#
#   figures/sync_to_paper.sh [PAPER_DIR]     default: ../LLM-Share-Paper
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
paper=${1:-"$here/../../LLM-Share-Paper"}
[[ -f "$paper/main.tex" ]] || { echo "no main.tex in $paper" >&2; exit 2; }
mkdir -p "$paper/Figures"
copied=0
missing=()
while read -r name; do
  if [[ -f "$here/$name" ]]; then
    cp "$here/$name" "$paper/Figures/$name"
    copied=$((copied + 1))
  else
    missing+=("$name")
  fi
done < <(grep -oh 'includegraphics\[[^]]*\]{[^}]*}' "$paper"/*.tex |
  sed 's/.*{\(.*\)}/\1/' | sort -u)
printf 'copied=%s missing=%s %s\n' "$copied" "${#missing[@]}" "${missing[*]:-}"
