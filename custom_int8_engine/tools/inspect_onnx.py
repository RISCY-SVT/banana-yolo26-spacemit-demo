#!/usr/bin/env python3
"""Small ONNX graph inventory helper for the Stage 0 custom INT8 engine."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import onnx


def _shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [dim.dim_value if dim.dim_value else dim.dim_param for dim in value.type.tensor_type.shape.dim]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--tail", type=int, default=24)
    args = parser.parse_args()

    model = onnx.load(args.model)
    graph = model.graph
    out = {
        "path": str(args.model.resolve()),
        "producer": model.producer_name,
        "opset": [{"domain": ops.domain, "version": ops.version} for ops in model.opset_import],
        "inputs": [{"name": value.name, "shape": _shape(value)} for value in graph.input],
        "outputs": [{"name": value.name, "shape": _shape(value)} for value in graph.output],
        "op_counts": dict(sorted(Counter(node.op_type for node in graph.node).items())),
        "tail_nodes": [
            {"name": node.name, "op_type": node.op_type, "inputs": list(node.input), "outputs": list(node.output)}
            for node in graph.node[-args.tail :]
        ],
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
