#!/usr/bin/env python3
"""! @file inspect_onnx_graph.py
@brief Inspect an ONNX graph and print likely output/truncate candidates.
@details This helper is primarily used to understand output contracts before
adding new preprocess, decode, or quantization flows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx


def main() -> int:
    """Run the graph inspection CLI.

    Returns:
        int: Process exit code, zero on success.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to the ONNX model to inspect.")
    parser.add_argument("--tail", type=int, default=30, help="Number of tail nodes to print.")
    args = parser.parse_args()
    # Load the full graph once and then print stable JSON snippets for the
    # inputs, outputs, and tail nodes that most often matter for decode work.
    model = onnx.load(args.model)
    graph = model.graph
    print("MODEL", Path(args.model).resolve())
    print("INPUTS")
    for value in graph.input:
        dims = [d.dim_value if d.dim_value else d.dim_param for d in value.type.tensor_type.shape.dim]
        print(json.dumps({"name": value.name, "shape": dims}))
    print("OUTPUTS")
    for value in graph.output:
        dims = [d.dim_value if d.dim_value else d.dim_param for d in value.type.tensor_type.shape.dim]
        print(json.dumps({"name": value.name, "shape": dims}))
    print("TAIL_NODES")
    for node in graph.node[-args.tail:]:
        print(json.dumps({
            "name": node.name,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
