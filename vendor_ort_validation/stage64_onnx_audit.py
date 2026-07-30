#!/usr/bin/env python3
"""Audit Stage64 ONNX representation, quantization and split contracts."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
from onnx import numpy_helper


SPLIT_NAMES = [
    "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0",
    "/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0",
    "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0",
    "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0",
    "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0",
    "/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def attribute(node: onnx.NodeProto, name: str) -> Any:
    for item in node.attribute:
        if item.name == name:
            return onnx.helper.get_attribute_value(item)
    return None


def array_summary(array: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(array).reshape(-1)
    if flat.size == 0:
        return {
            "shape": "x".join(map(str, array.shape)),
            "dtype": str(array.dtype),
            "minimum": "",
            "maximum": "",
            "unique_count": 0,
            "values": "",
        }
    unique = np.unique(flat)
    values = ",".join(map(str, unique.tolist())) if unique.size <= 16 else ""
    return {
        "shape": "x".join(map(str, array.shape)) or "scalar",
        "dtype": str(array.dtype),
        "minimum": flat.min().item(),
        "maximum": flat.max().item(),
        "unique_count": unique.size,
        "values": values,
    }


def classify_consumer_role(
    output_name: str,
    source_name: str,
    initializers: dict[str, np.ndarray],
    producers: dict[str, onnx.NodeProto],
    consumers: dict[str, list[onnx.NodeProto]],
) -> str:
    source_producer = producers.get(source_name)
    origin_name = (
        source_producer.input[0]
        if source_producer
        and source_producer.op_type == "QuantizeLinear"
        and source_producer.input
        else source_name
    )
    is_parameter = origin_name in initializers
    downstream = consumers.get(output_name, [])
    if downstream and all(node.op_type == "DequantizeLinear" for node in downstream):
        downstream = [
            consumer
            for dequantize in downstream
            for consumer in consumers.get(dequantize.output[0], [])
        ]

    roles: set[str] = set()
    parameter_ops = {"Conv", "ConvTranspose", "Gemm", "MatMul", "BatchMatMul"}
    for consumer in downstream:
        for input_index, input_name in enumerate(consumer.input):
            if input_name != output_name and not (
                input_name in {
                    dequantize.output[0]
                    for dequantize in consumers.get(output_name, [])
                    if dequantize.op_type == "DequantizeLinear"
                }
            ):
                continue
            if (
                consumer.op_type in parameter_ops
                and input_index == 1
                and is_parameter
            ):
                roles.add("weight")
            elif consumer.op_type in {"Conv", "ConvTranspose", "Gemm"} and input_index == 2:
                roles.add("bias")
            else:
                roles.add("activation")

    if len(roles) == 1:
        return next(iter(roles))
    if roles:
        return "mixed:" + ",".join(sorted(roles))
    return "constant" if source_name in initializers else "activation"


def audit(path: Path, label: str) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    model = onnx.load(path, load_external_data=True)
    checker_status = "pass"
    checker_error = ""
    shape_inference_status = "pass"
    shape_inference_error = ""
    try:
        onnx.checker.check_model(model)
    except Exception as exc:  # noqa: BLE001 - evidence must preserve checker text
        checker_status = "fail"
        checker_error = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    try:
        onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except Exception as exc:  # noqa: BLE001 - evidence must preserve checker text
        shape_inference_status = "fail"
        shape_inference_error = f"{type(exc).__name__}: {exc}".replace("\n", " ")

    initializers = {
        item.name: numpy_helper.to_array(item) for item in model.graph.initializer
    }
    producers: dict[str, onnx.NodeProto] = {}
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for name in node.output:
            producers[name] = node
        for name in node.input:
            consumers[name].append(node)

    operations = Counter(node.op_type for node in model.graph.node)
    initializer_payload = hashlib.sha256()
    for name in sorted(initializers):
        initializer_payload.update(name.encode())
        initializer_payload.update(initializers[name].tobytes(order="C"))

    result["manifest"].append(
        {
            "model": label,
            "path": str(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "ir_version": model.ir_version,
            "opsets": ",".join(
                f"{item.domain or 'ai.onnx'}:{item.version}"
                for item in model.opset_import
            ),
            "nodes": len(model.graph.node),
            "initializers": len(model.graph.initializer),
            "initializer_payload_sha256": initializer_payload.hexdigest(),
            "inputs": ",".join(item.name for item in model.graph.input),
            "outputs": ",".join(item.name for item in model.graph.output),
            "functions": len(model.functions),
        }
    )
    result["checker"].append(
        {
            "model": label,
            "checker_status": checker_status,
            "checker_error": checker_error,
            "shape_inference": shape_inference_status,
            "shape_inference_error": shape_inference_error,
            "split_tensor_count": sum(name in producers for name in SPLIT_NAMES),
        }
    )
    for op_type, count in sorted(operations.items()):
        result["operators"].append(
            {"model": label, "op_type": op_type, "count": count}
        )
    for op_type, count in sorted(operations.items()):
        if op_type.startswith("QLinear"):
            result["qlinear"].append(
                {"model": label, "op_type": op_type, "count": count}
            )
    if not result["qlinear"]:
        result["qlinear"].append({"model": label, "op_type": "none", "count": 0})

    for index, node in enumerate(model.graph.node):
        if node.op_type not in {"QuantizeLinear", "DequantizeLinear"}:
            continue
        scale_name = node.input[1] if len(node.input) > 1 else ""
        zero_name = node.input[2] if len(node.input) > 2 else ""
        scale = initializers.get(scale_name)
        zero = initializers.get(zero_name)
        scale_info = array_summary(scale) if scale is not None else {}
        zero_info = array_summary(zero) if zero is not None else {}
        axis = attribute(node, "axis")
        classification = classify_consumer_role(
            node.output[0],
            node.input[0] if node.input else "",
            initializers,
            producers,
            consumers,
        )
        result["qdq"].append(
            {
                "model": label,
                "index": index,
                "node_name": node.name,
                "op_type": node.op_type,
                "source": node.input[0] if node.input else "",
                "output": node.output[0] if node.output else "",
                "classification": classification,
                "axis": "" if axis is None else axis,
                "scale_name": scale_name,
                "scale_shape": scale_info.get("shape", "not-constant"),
                "scale_min": scale_info.get("minimum", ""),
                "scale_max": scale_info.get("maximum", ""),
                "zero_point_name": zero_name,
                "zero_point_dtype": zero_info.get("dtype", "implicit-uint8"),
                "zero_point_shape": zero_info.get("shape", "implicit"),
                "zero_point_min": zero_info.get("minimum", 0),
                "zero_point_max": zero_info.get("maximum", 0),
                "zero_point_unique_count": zero_info.get("unique_count", 1),
                "zero_point_values": zero_info.get("values", "0"),
            }
        )

    for index, node in enumerate(model.graph.node):
        if node.op_type != "Conv":
            continue
        weight_name = node.input[1] if len(node.input) > 1 else ""
        weight_shape = ""
        derived = ""
        weight_source = producers.get(weight_name)
        raw_weight_name = weight_name
        if weight_source and weight_source.op_type == "DequantizeLinear":
            raw_weight_name = weight_source.input[0]
        if raw_weight_name in initializers:
            weight = initializers[raw_weight_name]
            weight_shape = "x".join(map(str, weight.shape))
            if weight.ndim >= 4:
                derived = ",".join(map(str, weight.shape[-2:]))
        kernel = attribute(node, "kernel_shape")
        explicit = "" if kernel is None else ",".join(map(str, kernel))
        result["conv"].append(
            {
                "model": label,
                "index": index,
                "node_name": node.name,
                "weight_tensor": raw_weight_name,
                "weight_shape": weight_shape,
                "explicit_kernel_shape": explicit,
                "derived_kernel_shape": derived,
                "status": (
                    "pass"
                    if explicit and explicit == derived
                    else "missing"
                    if not explicit
                    else "mismatch"
                ),
            }
        )

    for split_name in SPLIT_NAMES:
        producer = producers.get(split_name)
        result["split"].append(
            {
                "model": label,
                "tensor": split_name,
                "present": int(producer is not None),
                "producer_name": producer.name if producer else "",
                "producer_op": producer.op_type if producer else "",
                "consumer_count": len(consumers.get(split_name, [])),
            }
        )
    return result


def combine(
    audits: Iterable[dict[str, list[dict[str, Any]]]]
) -> dict[str, list[dict[str, Any]]]:
    combined: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in audits:
        for key, rows in item.items():
            combined[key].extend(rows)
    return combined


def main() -> int:
    options = parse_args()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    audits = combine(
        audit(path.resolve(), path.stem) for path in sorted(options.models)
    )
    outputs = {
        "generated_model_manifest.tsv": (
            "manifest",
            [
                "model",
                "path",
                "sha256",
                "bytes",
                "ir_version",
                "opsets",
                "nodes",
                "initializers",
                "initializer_payload_sha256",
                "inputs",
                "outputs",
                "functions",
            ],
        ),
        "generated_model_checker.tsv": (
            "checker",
            [
                "model",
                "checker_status",
                "checker_error",
                "shape_inference",
                "shape_inference_error",
                "split_tensor_count",
            ],
        ),
        "operator_census.tsv": ("operators", ["model", "op_type", "count"]),
        "qlinear_operator_census.tsv": (
            "qlinear",
            ["model", "op_type", "count"],
        ),
        "qdq_schema_census.tsv": (
            "qdq",
            [
                "model",
                "index",
                "node_name",
                "op_type",
                "source",
                "output",
                "classification",
                "axis",
                "scale_name",
                "scale_shape",
                "scale_min",
                "scale_max",
                "zero_point_name",
                "zero_point_dtype",
                "zero_point_shape",
                "zero_point_min",
                "zero_point_max",
                "zero_point_unique_count",
                "zero_point_values",
            ],
        ),
        "zero_point_dtype_value_census.tsv": (
            "qdq",
            [
                "model",
                "index",
                "node_name",
                "op_type",
                "classification",
                "zero_point_name",
                "zero_point_dtype",
                "zero_point_shape",
                "zero_point_min",
                "zero_point_max",
                "zero_point_unique_count",
                "zero_point_values",
            ],
        ),
        "scale_granularity_census.tsv": (
            "qdq",
            [
                "model",
                "index",
                "node_name",
                "op_type",
                "classification",
                "axis",
                "scale_name",
                "scale_shape",
                "scale_min",
                "scale_max",
            ],
        ),
        "conv_kernel_shape_audit.tsv": (
            "conv",
            [
                "model",
                "index",
                "node_name",
                "weight_tensor",
                "weight_shape",
                "explicit_kernel_shape",
                "derived_kernel_shape",
                "status",
            ],
        ),
        "source_model_split_tensor_audit.tsv": (
            "split",
            [
                "model",
                "tensor",
                "present",
                "producer_name",
                "producer_op",
                "consumer_count",
            ],
        ),
    }
    for filename, (key, fields) in outputs.items():
        write_tsv(options.output_dir / filename, audits.get(key, []), fields)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
