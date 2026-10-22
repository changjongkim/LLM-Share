#!/usr/bin/env python3
"""Writes BFCL prefixes of an exact number of tokens, for the mechanism
experiments that ran on a repeated paragraph of 4,081 and 16,321 tokens
(review of 2026-10-22: the same lengths with the text of a public benchmark).

  make_workload_bfcl_tokens.py SRC_DIR OUT_DIR TOKENIZER_JSON LLAMA_TOKENIZE MODEL TOKENS...

Every prefix is the system prompt of make_workload_bfcl.py and whole tool
lines of BFCL v4 simple_python, taken in the stride order of that script.
Lines are added while the prefix stays within TOKENS. The last line is then
replaced by an unused line, or the last two lines by two unused lines, whose
tokens fill the prefix to exactly TOKENS. The count is that of the engine
(llama-tokenize with its default special tokens), which the drivers print as
prefix_tokens. Line i of the tasks file is the request of the entry whose
function is line i of the tools. The same files as make_workload_bfcl.py are
read and checked against their pinned sha256; run with python3 -I.

Outputs per TOKENS: bfcl_prefix_TOKENS.txt, bfcl_tasks_TOKENS.txt,
bfcl_source_TOKENS.txt.
"""
import importlib.util
import itertools
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("make_workload_bfcl", os.path.join(HERE, "make_workload_bfcl.py"))
bfcl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bfcl)


def engine_count(llama_tokenize, model, text, scratch, special=True):
    """The tokens of text for the engine; special=False as the drivers tokenize the task of an agent."""
    with open(scratch, "w", encoding="ascii", newline="\n") as handle:
        handle.write(text)
    # --no-escape: the drivers read the files as they are, without turning a written \\n into a newline
    out = subprocess.run([llama_tokenize, "-m", model, "-f", scratch, "--ids", "--log-disable", "--no-escape"]
                         + ([] if special else ["--no-parse-special"]),
                         env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True,
                         check=True).stdout.strip()
    return out.count(",") + 1 if out.startswith("[") and out != "[]" else 0


def fit(ordered, head, target, count):
    """The lines of a prefix of exactly target tokens, or None."""
    chosen = []
    for item in ordered:
        if count(head + "".join(line + "\n" for _, _, line, _ in chosen + [item])) > target:
            break
        chosen.append(item)
    unused = [item for item in ordered if item not in chosen]
    for drop in (1, 2):
        base = chosen[:-drop]
        for extra in itertools.combinations(unused[:120], drop):
            lines = base + list(extra)
            if count(head + "".join(line + "\n" for _, _, line, _ in lines)) == target:
                return lines
    return None


def main():
    if len(sys.argv) < 7:
        print(__doc__, file=sys.stderr)
        return 2
    src, out_dir, tokenizer_json, llama_tokenize, model = sys.argv[1:6]
    targets = [int(value) for value in sys.argv[6:]]
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_json)
    # the engine adds no BOS for this model; the fast count is checked against the engine below
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)  # noqa: E731
    data = bfcl.read_sources(src)
    usable = bfcl.usable_entries(data)
    ordered = [usable[(number * bfcl.STRIDE) % len(usable)] for number in range(len(usable))]
    head = bfcl.SYSTEM + "\n# Tools\n\n"
    os.makedirs(out_dir, exist_ok=True)
    scratch = os.path.join(out_dir, ".count.txt")
    for target in targets:
        lines = fit(ordered, head, target, count)
        if lines is None:
            raise SystemExit(f"no set of whole tool lines has {target} tokens")
        prefix = head + "".join(line + "\n" for _, _, line, _ in lines)
        engine = engine_count(llama_tokenize, model, prefix, scratch)
        if engine != target:
            raise SystemExit(f"{target}: the engine counts {engine} tokens")
        source = (f"Source of bfcl_prefix_{target}.txt and bfcl_tasks_{target}.txt "
                  "(written by make_workload_bfcl_tokens.py)\n\n"
                  "dataset: Berkeley Function Calling Leaderboard (BFCL v4), category simple_python\n"
                  f"repository: {bfcl.REPOSITORY}\nrevision: {bfcl.REVISION}\n"
                  f"license: {bfcl.LICENSE} ({bfcl.RAW}LICENSE)\n"
                  f"prefix: {len(prefix)} bytes, {target} tokens of the engine, {len(lines)} functions, "
                  f"{len(lines)} tasks\n\n# Raw files: sha256, bytes, name in SRC_DIR, URL\n\n")
        for name, path, pinned in bfcl.SOURCES:
            source += f"{pinned} {len(data[name])} {name} {bfcl.RAW}{path}\n"
        source += "\n# Tasks: line of the tasks file, dataset entry, function in the prefix\n\n"
        for number, (entry, name, _, _) in enumerate(lines, 1):
            source += f"{number} {entry} {name}\n"
        for name, text in ((f"bfcl_prefix_{target}.txt", prefix),
                           (f"bfcl_tasks_{target}.txt", "".join(request + "\n" for _, _, _, request in lines)),
                           (f"bfcl_source_{target}.txt", source)):
            with open(os.path.join(out_dir, name), "w", encoding="ascii", newline="\n") as handle:
                handle.write(text)
        print(f"{target} tokens: {len(prefix)} bytes, {len(lines)} functions")
    os.remove(scratch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
