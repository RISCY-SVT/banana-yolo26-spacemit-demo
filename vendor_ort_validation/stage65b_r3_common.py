#!/usr/bin/env python3
"""Shared deterministic helpers for Stage65B-R3 host localization."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Iterable

import onnx
from onnx import ModelProto, TensorProto, ValueInfoProto, shape_inference


MAX_CSV_FIELD_SIZE = 16 * 1024 * 1024
csv.field_size_limit(MAX_CSV_FIELD_SIZE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def shape_text(info: ValueInfoProto) -> str:
    tensor = info.type.tensor_type
    if not tensor.HasField("shape"):
        return "unresolved"
    dimensions: list[str] = []
    for dimension in tensor.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(str(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append("?")
    return "x".join(dimensions)


def dtype_text(info: ValueInfoProto) -> str:
    return TensorProto.DataType.Name(info.type.tensor_type.elem_type)


def static_tensor(info: ValueInfoProto) -> bool:
    tensor = info.type.tensor_type
    if tensor.elem_type == TensorProto.UNDEFINED or not tensor.HasField("shape"):
        return False
    return all(dimension.HasField("dim_value") for dimension in tensor.shape.dim)


def inferred_model(path: Path) -> ModelProto:
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    inferred = shape_inference.infer_shapes(model)
    onnx.checker.check_model(inferred)
    return inferred


def value_info_map(model: ModelProto) -> dict[str, ValueInfoProto]:
    result: dict[str, ValueInfoProto] = {}
    for item in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        if item.name in result:
            old = result[item.name]
            if (
                old.type.tensor_type.elem_type != item.type.tensor_type.elem_type
                or shape_text(old) != shape_text(item)
            ):
                raise ValueError(f"conflicting value-info declarations: {item.name}")
        result[item.name] = item
    return result


def producers(model: ModelProto) -> dict[str, onnx.NodeProto]:
    result: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for output in node.output:
            if not output:
                continue
            if output in result:
                raise ValueError(f"duplicate tensor producer: {output}")
            result[output] = node
    return result


def consumers(model: ModelProto) -> dict[str, list[onnx.NodeProto]]:
    result: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for value in node.input:
            if value:
                result.setdefault(value, []).append(node)
    return result


def node_name_map(model: ModelProto) -> dict[str, list[onnx.NodeProto]]:
    result: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        result.setdefault(node.name, []).append(node)
    return result


def copy_model_metadata(source: ModelProto, target: ModelProto) -> None:
    target.ir_version = source.ir_version
    target.model_version = source.model_version
    target.domain = source.domain
    target.doc_string = source.doc_string
    target.graph.doc_string = source.graph.doc_string
    del target.metadata_props[:]
    target.metadata_props.extend(source.metadata_props)


def validate_extraction_contract(source: ModelProto, target: ModelProto) -> None:
    source_opsets = [(item.domain, item.version) for item in source.opset_import]
    target_opsets = [(item.domain, item.version) for item in target.opset_import]
    if target_opsets != source_opsets:
        raise ValueError("opset imports changed during extraction")
    source_functions = {(item.domain, item.name) for item in source.functions}
    target_functions = {(item.domain, item.name) for item in target.functions}
    if not target_functions.issubset(source_functions):
        raise ValueError("extraction introduced an unknown FunctionProto")
    required_functions = {
        (node.domain, node.op_type)
        for node in target.graph.node
        if (node.domain, node.op_type) in source_functions
    }
    missing_functions = required_functions - target_functions
    if missing_functions:
        raise ValueError(f"extraction lost required FunctionProto: {sorted(missing_functions)}")


def extract_model(
    source: ModelProto, input_names: list[str], output_names: list[str]
) -> ModelProto:
    if len(input_names) != len(set(input_names)):
        raise ValueError("duplicate extraction input name")
    if len(output_names) != len(set(output_names)):
        raise ValueError("duplicate extraction output name")
    extracted = onnx.utils.Extractor(source).extract_model(input_names, output_names)
    copy_model_metadata(source, extracted)
    onnx.checker.check_model(extracted)
    inferred = shape_inference.infer_shapes(extracted)
    copy_model_metadata(source, inferred)
    onnx.checker.check_model(inferred)
    validate_extraction_contract(source, inferred)
    return inferred


def model_manifest(path: Path, model: ModelProto) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "nodes": len(model.graph.node),
        "inputs": len(model.graph.input),
        "outputs": len(model.graph.output),
        "initializers": len(model.graph.initializer),
        "functions": len(model.functions),
        "opsets": ",".join(
            f"{item.domain or 'ai.onnx'}:{item.version}" for item in model.opset_import
        ),
        "external_initializers": sum(
            item.data_location == TensorProto.EXTERNAL
            for item in model.graph.initializer
        ),
    }


def hash_rows(rows: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row)
    return digest.hexdigest()
