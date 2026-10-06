#!/usr/bin/env bash
# Builds the figures: the TikZ sources in src/ into PDF (for a paper) and PNG
# (for the README), then the evaluation graphs from the packaged summaries.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT
for source in "$here"/src/*.tex; do
  name=$(basename "$source" .tex)
  [[ "$name" == llmshare_style ]] && continue
  (cd "$here/src" && pdflatex -interaction=nonstopmode -halt-on-error \
    -output-directory "$build" "$name.tex" >"$build/$name.log" 2>&1) ||
    { tail -n 20 "$build/$name.log" >&2; exit 1; }
  cp "$build/$name.pdf" "$here/$name.pdf"
  pdftoppm -r 220 -png -singlefile "$here/$name.pdf" "$here/$name"
done
if [[ -f "$here/make_eval_figures.py" ]]; then python3 "$here/make_eval_figures.py"; fi
