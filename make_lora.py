#!/usr/bin/env python3
"""Writes a synthetic LoRA adapter for a GGUF model.

The adapter has rank-R factors with small random values for the query and
value projections of every block. It stands in for a fine-tuned adapter in
experiments that only need distinct adapters over one base model.

usage: make_lora.py BASE.gguf OUT.gguf SEED [RANK] [SCALE]
"""
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0] + "/llama.cpp/gguf-py")
import gguf  # noqa: E402
import numpy as np  # noqa: E402


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__, file=sys.stderr)
        return 2
    base, out, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    rank = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    scale = float(sys.argv[5]) if len(sys.argv) > 5 else 0.02
    reader = gguf.GGUFReader(base)
    architecture = bytes(
        reader.fields["general.architecture"].parts[-1]).decode()
    rng = np.random.default_rng(seed)
    writer = gguf.GGUFWriter(out, architecture)
    writer.add_type("adapter")
    writer.add_string("adapter.type", "lora")
    writer.add_float32("adapter.lora.alpha", float(rank))
    count = 0
    for tensor in reader.tensors:
        if not (tensor.name.endswith("attn_q.weight") or
                tensor.name.endswith("attn_v.weight")):
            continue
        # GGUF lists dimensions innermost first: [inputs, outputs].
        inputs, outputs = int(tensor.shape[0]), int(tensor.shape[1])
        writer.add_tensor(tensor.name + ".lora_a",
                          rng.normal(0.0, scale, (rank, inputs)).astype(np.float32))
        writer.add_tensor(tensor.name + ".lora_b",
                          rng.normal(0.0, scale, (outputs, rank)).astype(np.float32))
        count += 1
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"adapter={out} seed={seed} rank={rank} scale={scale} matrices={count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
