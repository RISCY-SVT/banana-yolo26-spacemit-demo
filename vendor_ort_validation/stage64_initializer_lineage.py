#!/usr/bin/env python3
"""Audit quantized-weight lineage and exact FP32-tail initializer retention."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--tail", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    result = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    for node in model.graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        value = attribute(node, "value")
        if isinstance(value, onnx.TensorProto):
            result[node.output[0]] = numpy_helper.to_array(value)
    return result


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes(order="C")
    ).hexdigest()


def shape(array: np.ndarray) -> str:
    return "x".join(map(str, array.shape)) or "scalar"


def attribute(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    for item in node.attribute:
        if item.name == name:
            return onnx.helper.get_attribute_value(item)
    return default


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write an empty audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def dequantize(
    quantized: np.ndarray,
    scale: np.ndarray,
    zero_point: np.ndarray,
    axis: int,
) -> np.ndarray:
    values = quantized.astype(np.float64)
    zeros = zero_point.astype(np.float64)
    scales = scale.astype(np.float64)
    if scales.ndim == 0:
        return (values - zeros) * scales
    broadcast = [1] * values.ndim
    broadcast[axis] = scales.size
    return (values - zeros.reshape(broadcast)) * scales.reshape(broadcast)


def main() -> int:
    options = parse_args()
    source_model = onnx.load(options.source, load_external_data=True)
    candidate_model = onnx.load(options.candidate, load_external_data=True)
    tail_model = onnx.load(options.tail, load_external_data=True)
    source_arrays = arrays(source_model)
    candidate_arrays = arrays(candidate_model)
    tail_arrays = arrays(tail_model)

    source_nodes = {
        node.name: node for node in source_model.graph.node if node.name
    }
    candidate_producers = {
        output: node
        for node in candidate_model.graph.node
        for output in node.output
    }

    tail_rows: list[dict[str, Any]] = []
    for name, tail_value in sorted(tail_arrays.items()):
        source_value = source_arrays.get(name)
        candidate_value = candidate_arrays.get(name)
        tail_rows.append(
            {
                "initializer": name,
                "tail_shape": shape(tail_value),
                "tail_dtype": str(tail_value.dtype),
                "tail_sha256": digest(tail_value),
                "source_present": int(source_value is not None),
                "source_sha256": (
                    digest(source_value) if source_value is not None else ""
                ),
                "candidate_present": int(candidate_value is not None),
                "candidate_sha256": (
                    digest(candidate_value) if candidate_value is not None else ""
                ),
                "source_exact": int(
                    source_value is not None
                    and source_value.dtype == tail_value.dtype
                    and source_value.shape == tail_value.shape
                    and np.array_equal(source_value, tail_value)
                ),
                "candidate_exact": int(
                    candidate_value is not None
                    and candidate_value.dtype == tail_value.dtype
                    and candidate_value.shape == tail_value.shape
                    and np.array_equal(candidate_value, tail_value)
                ),
            }
        )

    weight_rows: list[dict[str, Any]] = []
    weighted_ops = {"Conv", "Gemm", "MatMul"}
    for candidate_node in candidate_model.graph.node:
        if (
            candidate_node.op_type not in weighted_ops
            or len(candidate_node.input) < 2
            or not candidate_node.name
        ):
            continue
        source_node = source_nodes.get(candidate_node.name)
        if source_node is None or len(source_node.input) < 2:
            continue
        source_weight = source_arrays.get(source_node.input[1])
        weight_producer = candidate_producers.get(candidate_node.input[1])
        if (
            source_weight is None
            or weight_producer is None
            or weight_producer.op_type != "DequantizeLinear"
            or len(weight_producer.input) < 3
        ):
            continue
        quantized = candidate_arrays.get(weight_producer.input[0])
        scale = candidate_arrays.get(weight_producer.input[1])
        zero_point = candidate_arrays.get(weight_producer.input[2])
        if quantized is None or scale is None or zero_point is None:
            continue
        axis = int(attribute(weight_producer, "axis", 1))
        reconstructed = dequantize(quantized, scale, zero_point, axis)
        same_shape = source_weight.shape == reconstructed.shape
        if same_shape:
            difference = np.abs(source_weight.astype(np.float64) - reconstructed)
            mae = float(difference.mean())
            max_abs = float(difference.max())
            denominator = float(
                np.linalg.norm(source_weight.astype(np.float64).reshape(-1))
                * np.linalg.norm(reconstructed.reshape(-1))
            )
            cosine = (
                float(
                    np.dot(
                        source_weight.astype(np.float64).reshape(-1),
                        reconstructed.reshape(-1),
                    )
                    / denominator
                )
                if denominator
                else float("nan")
            )
        else:
            mae = float("nan")
            max_abs = float("nan")
            cosine = float("nan")
        weight_rows.append(
            {
                "node_name": candidate_node.name,
                "op_type": candidate_node.op_type,
                "source_weight": source_node.input[1],
                "quantized_weight": weight_producer.input[0],
                "source_shape": shape(source_weight),
                "quantized_shape": shape(quantized),
                "source_dtype": str(source_weight.dtype),
                "quantized_dtype": str(quantized.dtype),
                "zero_point_dtype": str(zero_point.dtype),
                "zero_point_min": int(zero_point.min()),
                "zero_point_max": int(zero_point.max()),
                "scale_shape": shape(scale),
                "axis": axis,
                "shape_match": int(same_shape),
                "mae_after_dequantize": mae,
                "max_abs_after_dequantize": max_abs,
                "cosine_after_dequantize": cosine,
                "source_sha256": digest(source_weight),
                "quantized_sha256": digest(quantized),
            }
        )

    write_tsv(options.output_dir / "tail_initializer_identity.tsv", tail_rows)
    write_tsv(options.output_dir / "quantized_weight_lineage.tsv", weight_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
