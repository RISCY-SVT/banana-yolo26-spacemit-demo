#!/usr/bin/env python3
"""Dump Q/DQ scale and zero-point metadata from an ONNX model."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, numpy_helper


def _tensor_summary(model: onnx.ModelProto, name: str) -> tuple[str, str]:
    init = {tensor.name: tensor for tensor in model.graph.initializer}
    tensor = init.get(name)
    if tensor is None:
        return "missing", "missing"
    array = numpy_helper.to_array(tensor)
    dtype = TensorProto.DataType.Name(tensor.data_type)
    if array.size == 1:
        return dtype, str(array.reshape(-1)[0].item())
    flat = array.reshape(-1)
    return dtype, f"shape={list(array.shape)} min={flat.min().item()} max={flat.max().item()}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    args = parser.parse_args()

    model = onnx.load(args.model)
    print("node\top_type\tscale_name\tscale_dtype\tscale_summary\tzero_point_name\tzero_point_dtype\tzero_point_summary")
    for node in model.graph.node:
        if node.op_type not in {"QuantizeLinear", "DequantizeLinear"}:
            continue
        scale_name = node.input[1] if len(node.input) > 1 else ""
        zero_point_name = node.input[2] if len(node.input) > 2 else ""
        scale_dtype, scale_summary = _tensor_summary(model, scale_name)
        zp_dtype, zp_summary = _tensor_summary(model, zero_point_name)
        print(
            "\t".join(
                [node.name, node.op_type, scale_name, scale_dtype, scale_summary, zero_point_name, zp_dtype, zp_summary]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
