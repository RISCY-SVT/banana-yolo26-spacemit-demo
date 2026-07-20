#!/usr/bin/env python3
"""Create an exact static-resolution derivative of the accepted YOLO26 Q/DQ graph."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
from onnx import checker, helper, numpy_helper, shape_inference


SOURCE_MODEL_SHA256 = "30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c"
MANDATORY_RESOLUTIONS = (640, 512, 448, 416, 384, 352, 320, 256)
OPTIONAL_RESOLUTIONS = (288,)
ATTENTION_BASES = (
    "/model.10/m/m.0/attn",
    "/model.22/m.0/m.0.1/attn",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def set_constant(node: onnx.NodeProto, value: np.ndarray) -> None:
    del node.attribute[:]
    node.attribute.extend(
        [helper.make_attribute("value", numpy_helper.from_array(np.ascontiguousarray(value), name="value"))]
    )


def head_geometry(resolution: int) -> tuple[np.ndarray, np.ndarray]:
    anchors: list[np.ndarray] = []
    strides: list[np.ndarray] = []
    for stride in (8, 16, 32):
        side = resolution // stride
        y, x = np.meshgrid(
            np.arange(side, dtype=np.float32) + np.float32(0.5),
            np.arange(side, dtype=np.float32) + np.float32(0.5),
            indexing="ij",
        )
        anchors.append(np.stack((x, y), axis=0).reshape(2, -1))
        strides.append(np.full(side * side, np.float32(stride), dtype=np.float32))
    return np.concatenate(anchors, axis=1)[None, :, :], np.concatenate(strides)[None, :]


def tensor_shape(value: onnx.ValueInfoProto) -> str:
    dims: list[str] = []
    for dim in value.type.tensor_type.shape.dim:
        dims.append(str(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?")
    return "x".join(dims)


def transform(args: argparse.Namespace) -> None:
    source = args.source_model.resolve()
    output = args.output.resolve()
    if sha256_file(source) != SOURCE_MODEL_SHA256:
        raise ValueError("accepted source-model SHA-256 mismatch")
    allowed = set(MANDATORY_RESOLUTIONS + OPTIONAL_RESOLUTIONS)
    if args.resolution not in allowed or args.resolution % 32 != 0:
        raise ValueError(f"unsupported Stage60 static resolution: {args.resolution}")

    model = onnx.load(source, load_external_data=True)
    source_nodes = [(node.name, node.op_type, node.domain, tuple(node.input), tuple(node.output))
                    for node in model.graph.node]
    source_initializers = {
        item.name: sha256_bytes(np.ascontiguousarray(numpy_helper.to_array(item)).tobytes())
        for item in model.graph.initializer
    }

    input_value = model.graph.input[0]
    if input_value.name != "images" or len(input_value.type.tensor_type.shape.dim) != 4:
        raise ValueError("unexpected accepted graph input")
    for dim, value in zip(input_value.type.tensor_type.shape.dim,
                          (1, 3, args.resolution, args.resolution), strict=True):
        dim.ClearField("dim_param")
        dim.dim_value = value

    constants = {node.name: node for node in model.graph.node if node.op_type == "Constant"}
    attention_side = args.resolution // 32
    for base in ATTENTION_BASES:
        set_constant(constants[f"{base}/Constant"],
                     np.asarray([1, 2, 128, attention_side * attention_side], dtype=np.int64))
        shape = np.asarray([1, 128, attention_side, attention_side], dtype=np.int64)
        set_constant(constants[f"{base}/Constant_2"], shape)
        set_constant(constants[f"{base}/Constant_3"], shape)

    anchors, strides = head_geometry(args.resolution)
    set_constant(constants["/model.23/Constant_12"], anchors)
    set_constant(constants["/model.23/Constant_13"], anchors.copy())
    set_constant(constants["/model.23/Constant_14"], strides)

    del model.graph.value_info[:]
    metadata = {item.key: item.value for item in model.metadata_props}
    metadata.update({
        "y26_stage60_parent_model_sha256": SOURCE_MODEL_SHA256,
        "y26_stage60_resolution": str(args.resolution),
        "y26_stage60_transform": "static-input-attention-and-head-geometry-v1",
    })
    del model.metadata_props[:]
    for key in sorted(metadata):
        item = model.metadata_props.add()
        item.key = key
        item.value = metadata[key]

    try:
        inferred = shape_inference.infer_shapes(model, data_prop=True)
    except Exception:
        inferred = shape_inference.infer_shapes(model)
    checker.check_model(inferred, full_check=True)

    candidate_nodes = [(node.name, node.op_type, node.domain, tuple(node.input), tuple(node.output))
                       for node in inferred.graph.node]
    if candidate_nodes != source_nodes:
        raise ValueError("static resolution transform changed graph topology")
    candidate_initializers = {
        item.name: sha256_bytes(np.ascontiguousarray(numpy_helper.to_array(item)).tobytes())
        for item in inferred.graph.initializer
    }
    if candidate_initializers != source_initializers:
        raise ValueError("static resolution transform changed weights or initializer values")

    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(inferred, output)
    output_sha = sha256_file(output)
    graph_output_shape = tensor_shape(inferred.graph.output[0])
    if graph_output_shape != "1x300x6":
        raise ValueError(f"unexpected detector output shape: {graph_output_shape}")

    op_counts = Counter(node.op_type for node in inferred.graph.node)
    write_tsv(args.operator_inventory, (
        {
            "index": index,
            "name": node.name,
            "op_type": node.op_type,
            "domain": node.domain,
            "input_count": len(node.input),
            "output_count": len(node.output),
        }
        for index, node in enumerate(inferred.graph.node)
    ), ["index", "name", "op_type", "domain", "input_count", "output_count"])
    values = [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
    write_tsv(args.tensor_inventory, (
        {
            "name": value.name,
            "elem_type": value.type.tensor_type.elem_type,
            "shape": tensor_shape(value),
        }
        for value in values
    ), ["name", "elem_type", "shape"])
    write_tsv(args.identity, [{
        "resolution": args.resolution,
        "source_model": str(source),
        "source_model_sha256": SOURCE_MODEL_SHA256,
        "static_model": str(output),
        "static_model_sha256": output_sha,
        "node_count": len(inferred.graph.node),
        "initializer_count": len(inferred.graph.initializer),
        "initializer_identity": "byte-identical",
        "topology_identity": "node-sequence-identical",
        "input_shape": f"1x3x{args.resolution}x{args.resolution}",
        "output_shape": graph_output_shape,
        "operator_classes": ",".join(f"{key}:{op_counts[key]}" for key in sorted(op_counts)),
    }], [
        "resolution", "source_model", "source_model_sha256", "static_model",
        "static_model_sha256", "node_count", "initializer_count", "initializer_identity",
        "topology_identity", "input_shape", "output_shape", "operator_classes",
    ])
    print(f"resolution={args.resolution}")
    print(f"static_model_sha256={output_sha}")
    print(f"nodes={len(inferred.graph.node)}")
    print("initializer_identity=byte-identical")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--operator-inventory", type=Path, required=True)
    parser.add_argument("--tensor-inventory", type=Path, required=True)
    transform(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
