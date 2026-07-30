#!/usr/bin/env python3
"""Generate small redistribution-safe ONNX controls for Stage64."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(model: onnx.ModelProto, path: Path) -> None:
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)


def tensor(name: str, array: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(array, name)


def reduce_models(output: Path) -> list[Path]:
    paths: list[Path] = []
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 1])
    axes = tensor("axes", np.array([2], dtype=np.int64))
    graph = helper.make_graph(
        [helper.make_node("ReduceMax", ["x", "axes"], ["y"], keepdims=1)],
        "reducemax_two_input_opset18",
        [x],
        [y],
        [axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    path = output / "reducemax_two_input_opset18.onnx"
    save(model, path)
    paths.append(path)

    graph = helper.make_graph(
        [helper.make_node("ReduceMax", ["x"], ["y"], axes=[2], keepdims=1)],
        "reducemax_attr_opset13",
        [x],
        [y],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path = output / "reducemax_attr_opset13.onnx"
    save(model, path)
    paths.append(path)

    cx = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 5, 5])
    cy = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 1, 5])
    weight = tensor("weight", np.ones((1, 1, 3, 3), dtype=np.float32))
    caxes = tensor("axes", np.array([2], dtype=np.int64))
    nodes = [
        helper.make_node(
            "Conv",
            ["x", "weight"],
            ["conv"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        ),
        helper.make_node("ReduceMax", ["conv", "axes"], ["y"], keepdims=1),
    ]
    graph = helper.make_graph(
        nodes, "conv_reducemax_two_input_opset18", [cx], [cy], [weight, caxes]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    path = output / "conv_reducemax_two_input_opset18.onnx"
    save(model, path)
    paths.append(path)
    return paths


def qdq_conv(
    output: Path,
    name: str,
    activation_dtype: np.dtype,
    activation_zero_point: int,
    per_channel_weight: bool,
    explicit_kernel: bool,
) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 5, 5])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 5, 5])
    zp_dtype = np.dtype(activation_dtype)
    wq = np.array(
        [
            [[[1, -2, 1], [0, 3, 0], [-1, 2, -1]]],
            [[[-2, 1, 0], [1, -1, 2], [0, 1, -2]]],
        ],
        dtype=np.int8,
    )
    if per_channel_weight:
        w_scale = np.array([0.125, 0.25], dtype=np.float32)
        w_zp = np.zeros(2, dtype=np.int8)
        weight_axis = 0
    else:
        w_scale = np.array(0.125, dtype=np.float32)
        w_zp = np.array(0, dtype=np.int8)
        weight_axis = None
    initializers = [
        tensor("x_scale", np.array(0.0625, dtype=np.float32)),
        tensor("x_zp", np.array(activation_zero_point, dtype=zp_dtype)),
        tensor("w_q", wq),
        tensor("w_scale", w_scale),
        tensor("w_zp", w_zp),
        tensor("y_scale", np.array(0.125, dtype=np.float32)),
        tensor("y_zp", np.array(0, dtype=np.int8)),
    ]
    weight_attributes = {} if weight_axis is None else {"axis": weight_axis}
    conv_attributes = {
        "pads": [1, 1, 1, 1],
        "strides": [1, 1],
    }
    if explicit_kernel:
        conv_attributes["kernel_shape"] = [3, 3]
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "x_scale", "x_zp"], ["x_q"]),
        helper.make_node(
            "DequantizeLinear", ["x_q", "x_scale", "x_zp"], ["x_dq"]
        ),
        helper.make_node(
            "DequantizeLinear",
            ["w_q", "w_scale", "w_zp"],
            ["w_dq"],
            **weight_attributes,
        ),
        helper.make_node("Conv", ["x_dq", "w_dq"], ["conv"], **conv_attributes),
        helper.make_node(
            "QuantizeLinear", ["conv", "y_scale", "y_zp"], ["y_q"]
        ),
        helper.make_node(
            "DequantizeLinear", ["y_q", "y_scale", "y_zp"], ["y"]
        ),
    ]
    graph = helper.make_graph(nodes, name, [x], [y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    path = output / f"{name}.onnx"
    save(model, path)
    return path


def qdq_matmul(
    output: Path,
    name: str,
    activation_dtype: np.dtype,
    activation_zero_point: int,
) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    zp_dtype = np.dtype(activation_dtype)
    wq = np.array(
        [[1, -2, 3], [4, 1, -1], [-2, 2, 1], [1, -3, 2]], dtype=np.int8
    )
    initializers = [
        tensor("x_scale", np.array(0.0625, dtype=np.float32)),
        tensor("x_zp", np.array(activation_zero_point, dtype=zp_dtype)),
        tensor("w_q", wq),
        tensor("w_scale", np.array(0.125, dtype=np.float32)),
        tensor("w_zp", np.array(0, dtype=np.int8)),
        tensor("y_scale", np.array(0.125, dtype=np.float32)),
        tensor("y_zp", np.array(0, dtype=np.int8)),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "x_scale", "x_zp"], ["x_q"]),
        helper.make_node(
            "DequantizeLinear", ["x_q", "x_scale", "x_zp"], ["x_dq"]
        ),
        helper.make_node(
            "DequantizeLinear", ["w_q", "w_scale", "w_zp"], ["w_dq"]
        ),
        helper.make_node("MatMul", ["x_dq", "w_dq"], ["matmul"]),
        helper.make_node(
            "QuantizeLinear", ["matmul", "y_scale", "y_zp"], ["y_q"]
        ),
        helper.make_node(
            "DequantizeLinear", ["y_q", "y_scale", "y_zp"], ["y"]
        ),
    ]
    graph = helper.make_graph(nodes, name, [x], [y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    path = output / f"{name}.onnx"
    save(model, path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    options = parser.parse_args()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    paths = reduce_models(options.output_dir)
    paths.extend(
        [
            qdq_conv(options.output_dir, "c1_s8_conv_pc_explicit", np.int8, 0, True, True),
            qdq_conv(options.output_dir, "c2_s8_conv_pc_nonzero_zp", np.int8, -3, True, True),
            qdq_conv(options.output_dir, "c3_s8_conv_pc_no_kernel", np.int8, 0, True, False),
            qdq_conv(options.output_dir, "c4_u8_conv_pc_explicit", np.uint8, 123, True, True),
            qdq_conv(options.output_dir, "c5_s8_conv_pt_explicit", np.int8, 0, False, True),
            qdq_matmul(options.output_dir, "m1_s8_matmul", np.int8, 0),
            qdq_matmul(options.output_dir, "m2_s8_matmul_nonzero_zp", np.int8, -3),
            qdq_matmul(options.output_dir, "m3_u8_matmul", np.uint8, 123),
        ]
    )
    with (options.output_dir / "fixture_manifest.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(["filename", "bytes", "sha256"])
        for path in sorted(paths):
            writer.writerow([path.name, path.stat().st_size, sha256(path)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
