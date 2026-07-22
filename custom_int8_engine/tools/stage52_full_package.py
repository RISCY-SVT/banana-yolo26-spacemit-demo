#!/usr/bin/env python3
"""Build the deterministic K1X_INT8_V1 full YOLO26 package.

The source ONNX graph is used only offline.  This tool collapses every Q/DQ
island into a static integer operation and emits no executable graph format.
Runtime code consumes numeric tensor and operation IDs from the generated TSV
tables and never loads ONNX.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import struct
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

from stage49_slice_package import (
    build_unary_lut,
    encode_multiplier,
    f32_bits,
    fraction_from_f32_bits,
    round_fraction_even,
)


CONTRACT_ID = "K1X_INT8_V1"
PROFILE_ID = "K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001"
LAYOUT_FEATURE = "NCHWc8_SPATIAL_INNER_V1"
LAYOUT_LINEAR = "ROW_MAJOR_U8_V1"
SCHEMA_VERSION = 2
EXPECTED_MODEL_SHA256 = "30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c"
ALIGNMENT = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value.strip("/"))


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    materialized = list(rows)
    if fields is None:
        fields = []
        seen: set[str] = set()
        for row in materialized:
            for key in row:
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        if not fields:
            return
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def copy_runtime_package(source: Path, destination: Path) -> str:
    """Copy only runtime assets from a verified prior package.

    Stage51 packages also carry large oracle tensors.  Those remain raw evidence;
    the full runtime package embeds only files named by its reduced manifest.
    """
    source = source.resolve()
    manifest_path = source / "asset_hashes.tsv"
    with manifest_path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    selected = [row for row in rows if not row["path"].startswith("oracles/")]
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    for row in selected:
        relative = Path(row["path"])
        source_path = source / relative
        target_path = destination / relative
        if source_path.is_symlink() or not source_path.is_file():
            raise ValueError(f"invalid optimized-core asset: {relative}")
        if source_path.stat().st_size != int(row["bytes"]) or sha256_file(source_path) != row["sha256"]:
            raise ValueError(f"optimized-core asset mismatch: {relative}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        copied.append({"path": str(relative), "bytes": int(row["bytes"]), "sha256": row["sha256"]})
    write_tsv(destination / "asset_hashes.tsv", copied, ["path", "bytes", "sha256"])
    return sha256_file(destination / "asset_hashes.tsv")


def attrs(node: onnx.NodeProto) -> dict[str, Any]:
    return {item.name: helper.get_attribute_value(item) for item in node.attribute}


def product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


@dataclass(frozen=True)
class QSpec:
    name: str
    shape: tuple[int, ...]
    scale: np.float32
    zero_point: int
    dtype: int


class Graph:
    def __init__(self, model: onnx.ModelProto) -> None:
        try:
            inferred = shape_inference.infer_shapes(model, data_prop=True)
        except Exception:
            inferred = shape_inference.infer_shapes(model)
        self.model = model
        self.initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
        self.producer = {output: node for node in model.graph.node for output in node.output}
        self.consumers: dict[str, list[onnx.NodeProto]] = {}
        for node in model.graph.node:
            for name in node.input:
                self.consumers.setdefault(name, []).append(node)
        self.meta: dict[str, tuple[int, tuple[int, ...]]] = {}
        for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
            tensor = value.type.tensor_type
            shape: list[int] = []
            if not tensor.HasField("shape"):
                continue
            for dim in tensor.shape.dim:
                if not dim.HasField("dim_value"):
                    break
                shape.append(int(dim.dim_value))
            else:
                self.meta[value.name] = (int(tensor.elem_type), tuple(shape))
        self.quant_by_source: dict[str, onnx.NodeProto] = {
            node.input[0]: node for node in model.graph.node if node.op_type == "QuantizeLinear"
        }
        self.quant_nodes = [node for node in model.graph.node if node.op_type == "QuantizeLinear"]

    def scalar(self, name: str) -> Any:
        value = np.asarray(self.initializers[name]).reshape(-1)
        if value.size != 1:
            raise ValueError(f"expected scalar initializer: {name}: {value.shape}")
        return value[0].item()

    def qspec(self, output: str) -> QSpec:
        node = self.producer[output]
        if node.op_type != "QuantizeLinear":
            raise ValueError(f"not a quantized tensor: {output}")
        dtype, shape = self.meta[output]
        return QSpec(output, shape, np.float32(self.scalar(node.input[1])), int(self.scalar(node.input[2])), dtype)

    def dq_root(self, float_name: str) -> str:
        node = self.producer.get(float_name)
        if node is None or node.op_type != "DequantizeLinear":
            raise ValueError(f"expected DequantizeLinear root: {float_name}")
        return node.input[0]

    def q_for_source(self, float_name: str) -> str:
        node = self.quant_by_source.get(float_name)
        if node is None:
            raise ValueError(f"source has no quantized boundary: {float_name}")
        return node.output[0]

    def constant(self, name: str) -> np.ndarray:
        if name in self.initializers:
            return np.asarray(self.initializers[name])
        node = self.producer.get(name)
        if node is None or node.op_type != "Constant":
            raise ValueError(f"not a constant: {name}")
        value = attrs(node).get("value")
        if value is None:
            raise ValueError(f"unsupported Constant: {name}")
        return np.asarray(numpy_helper.to_array(value))


def is_attention_linear(name: str, source: onnx.NodeProto, shape: tuple[int, ...]) -> bool:
    if len(shape) != 4 or "/attn/" not in name:
        return False
    if source.op_type in ("Conv", "Add") and shape[1] >= 8:
        return False
    return source.op_type in ("Split", "Transpose", "Reshape", "MatMul") or "Transpose_1" in name


class PackageBuilder:
    def __init__(self, graph: Graph, output: Path) -> None:
        self.graph = graph
        self.output = output
        self.assets = output / "assets"
        self.assets.mkdir(parents=True, exist_ok=True)
        self.tensors: list[dict[str, Any]] = []
        self.tensor_id: dict[str, int] = {}
        self.operations: list[dict[str, Any]] = []
        for node in graph.quant_nodes:
            spec = graph.qspec(node.output[0])
            source = graph.producer[node.input[0]] if node.input[0] in graph.producer else node
            layout = LAYOUT_LINEAR if is_attention_linear(spec.name, source, spec.shape) else LAYOUT_FEATURE
            storage_shape = list(spec.shape)
            if layout == LAYOUT_FEATURE and len(storage_shape) == 4:
                storage_shape[1] = align_up(storage_shape[1], 8)
            storage_bytes = product(storage_shape)
            tensor_id = len(self.tensors)
            self.tensor_id[spec.name] = tensor_id
            self.tensors.append({
                "id": tensor_id,
                "name": spec.name,
                "rank": len(spec.shape),
                "shape": "x".join(map(str, spec.shape)),
                "dim0": spec.shape[0] if len(spec.shape) > 0 else 1,
                "dim1": spec.shape[1] if len(spec.shape) > 1 else 1,
                "dim2": spec.shape[2] if len(spec.shape) > 2 else 1,
                "dim3": spec.shape[3] if len(spec.shape) > 3 else 1,
                "logical_elements": product(spec.shape),
                "storage_bytes": storage_bytes,
                "layout": layout,
                "scale": f"{float(spec.scale):.17g}",
                "scale_bits": f"0x{f32_bits(spec.scale):08x}",
                "zero_point": spec.zero_point,
                "dtype": "uint8" if spec.dtype == TensorProto.UINT8 else "int8",
                "first_op": -1,
                "last_op": -1,
                "arena_offset": 0,
            })

    def tid(self, name: str) -> int:
        return self.tensor_id[name]

    def tensor(self, name: str) -> dict[str, Any]:
        return self.tensors[self.tid(name)]

    def add_internal_tensor(self, name: str, source_name: str, target: str) -> None:
        if name in self.tensor_id:
            return
        dtype, shape = self.graph.meta[source_name]
        target_spec = self.graph.qspec(target)
        layout = LAYOUT_FEATURE if len(shape) == 4 else LAYOUT_LINEAR
        storage_shape = list(shape)
        if layout == LAYOUT_FEATURE:
            storage_shape[1] = align_up(storage_shape[1], 8)
        tensor_id = len(self.tensors)
        self.tensor_id[name] = tensor_id
        self.tensors.append({
            "id": tensor_id, "name": name, "rank": len(shape),
            "shape": "x".join(map(str, shape)),
            "dim0": shape[0] if len(shape) > 0 else 1,
            "dim1": shape[1] if len(shape) > 1 else 1,
            "dim2": shape[2] if len(shape) > 2 else 1,
            "dim3": shape[3] if len(shape) > 3 else 1,
            "logical_elements": product(shape), "storage_bytes": product(storage_shape),
            "layout": layout, "scale": f"{float(target_spec.scale):.17g}",
            "scale_bits": f"0x{f32_bits(target_spec.scale):08x}",
            "zero_point": target_spec.zero_point,
            "dtype": "uint8" if dtype == TensorProto.UINT8 else "uint8",
            "first_op": -1, "last_op": -1, "arena_offset": 0,
        })

    def asset_path(self, op_index: int, suffix: str) -> Path:
        path = self.assets / f"op_{op_index:03d}_{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def add(self, kind: str, name: str, output: str, inputs: list[str], **params: Any) -> int:
        row: dict[str, Any] = {
            "index": len(self.operations), "kind": kind, "name": name,
            "output": self.tid(output), "inputs": ",".join(str(self.tid(item)) for item in inputs),
        }
        row.update(params)
        self.operations.append(row)
        return int(row["index"])

    def root_q(self, name: str) -> str:
        producer = self.graph.producer.get(name)
        if producer is not None and producer.op_type == "DequantizeLinear":
            return producer.input[0]
        return self.graph.q_for_source(name)

    def scalar_expression_roots(self, name: str) -> tuple[list[str], Any]:
        roots: list[str] = []

        def build(value_name: str) -> Any:
            node = self.graph.producer.get(value_name)
            if node is None:
                raise ValueError(f"unsupported scalar expression root: {value_name}")
            if node.op_type == "DequantizeLinear":
                root = node.input[0]
                if root not in roots:
                    roots.append(root)
                return ("x", roots.index(root))
            if node.op_type == "Sigmoid":
                return ("sigmoid", build(node.input[0]))
            if node.op_type == "Reshape":
                return build(node.input[0])
            if node.op_type in ("Add", "Mul"):
                return (node.op_type.lower(), build(node.input[0]), build(node.input[1]))
            raise ValueError(f"unsupported scalar expression node: {node.op_type}: {node.name}")

        return roots, build(name)

    @staticmethod
    def eval_expression(expr: Any, values: list[np.float32]) -> np.float32:
        if expr[0] == "x":
            return values[int(expr[1])]
        if expr[0] == "sigmoid":
            x = PackageBuilder.eval_expression(expr[1], values)
            with np.errstate(over="ignore"):
                return np.float32(np.float32(1.0) /
                                  np.float32(np.float32(1.0) + np.exp(np.float32(-x))))
        left = PackageBuilder.eval_expression(expr[1], values)
        right = PackageBuilder.eval_expression(expr[2], values)
        return np.float32(left + right) if expr[0] == "add" else np.float32(left * right)

    def expression_lut(self, source_name: str, output: str) -> tuple[list[str], Path]:
        roots, expression = self.scalar_expression_roots(source_name)
        if len(roots) not in (1, 2):
            raise ValueError(f"scalar LUT needs one or two roots: {source_name}: {roots}")
        output_spec = self.graph.qspec(output)
        shape = (256,) if len(roots) == 1 else (256, 256)
        table = np.empty(shape, dtype=np.int8)
        root_specs = [self.graph.qspec(root) for root in roots]
        for index in np.ndindex(shape):
            values = [np.float32((code - spec.zero_point) * spec.scale) for code, spec in zip(index, root_specs)]
            value = self.eval_expression(expression, values)
            quantized = round_fraction_even(
                Fraction.from_float(float(value)) /
                fraction_from_f32_bits(f32_bits(output_spec.scale))
            ) + output_spec.zero_point
            table[index] = np.int8(min(255, max(0, quantized)) - 128)
        path = self.asset_path(len(self.operations), f"expression_{len(roots)}d_s8.bin")
        table.tofile(path)
        return roots, path

    def add_input(self, node: onnx.NodeProto) -> None:
        output = node.output[0]
        self.add("input_quant", node.name or "images_QuantizeLinear", output, [],
                 source="images")

    def add_conv(self, qnode: onnx.NodeProto, conv: onnx.NodeProto) -> None:
        output = qnode.output[0]
        input_q = self.graph.dq_root(conv.input[0])
        weight_dq = self.graph.producer[conv.input[1]]
        weights = np.asarray(self.graph.initializers[weight_dq.input[0]], dtype=np.int8)
        weight_scales = np.asarray(self.graph.initializers[weight_dq.input[1]], dtype=np.float32).reshape(-1)
        weight_zp = np.asarray(self.graph.initializers[weight_dq.input[2]], dtype=np.int8).reshape(-1)
        if np.any(weight_zp != 0):
            raise ValueError(f"nonzero weight zero point: {conv.name}")
        bias_quant_name = weight_dq.input[0].removesuffix(".weight_quantized") + ".bias_quantized"
        if bias_quant_name in self.graph.initializers:
            bias_i32 = np.asarray(self.graph.initializers[bias_quant_name], dtype=np.int32)
        else:
            bias = np.asarray(self.graph.initializers[conv.input[2]], dtype=np.float32)
            input_scale = self.graph.qspec(input_q).scale
            bias_i32 = np.rint(bias / (input_scale * weight_scales)).astype(np.int32)
        a = attrs(conv)
        group = int(a.get("group", 1))
        kernel_h, kernel_w = map(int, a.get("kernel_shape", weights.shape[-2:]))
        stride_h, stride_w = map(int, a.get("strides", [1, 1]))
        pads = list(map(int, a.get("pads", [0, 0, 0, 0])))
        output_spec = self.graph.qspec(output)
        input_spec = self.graph.qspec(input_q)
        multipliers = np.empty(weights.shape[0], dtype="<i8")
        shifts = np.empty(weights.shape[0], dtype="<i4")
        input_fraction = fraction_from_f32_bits(f32_bits(input_spec.scale))
        output_fraction = fraction_from_f32_bits(f32_bits(output_spec.scale))
        for channel, weight_scale in enumerate(weight_scales):
            ratio = input_fraction * fraction_from_f32_bits(f32_bits(weight_scale)) / output_fraction
            multipliers[channel], shifts[channel] = encode_multiplier(ratio)
        index = len(self.operations)
        weight_path = self.asset_path(index, "weights_oihw_s8.bin")
        bias_path = self.asset_path(index, "bias_i32.bin")
        multiplier_path = self.asset_path(index, "multiplier_i64.bin")
        shift_path = self.asset_path(index, "right_shift_i32.bin")
        weights.tofile(weight_path)
        bias_i32.astype("<i4").tofile(bias_path)
        multipliers.tofile(multiplier_path)
        shifts.tofile(shift_path)
        packed_weight_file = "-"
        weight_sum_file = "-"
        corrected_bias_file = "-"
        multiplier_m63_file = "-"
        e2c_compatible = bool(
            group == 1 and int(weights.shape[0]) % 4 == 0 and np.all(shifts == 62) and
            np.all(multipliers > 0) and np.all(multipliers <= np.iinfo(np.int64).max // 2)
        )
        if group == 1:
            input_channels = int(self.graph.qspec(input_q).shape[1])
            input_blocks = (input_channels + 7) // 8
            output_blocks = (int(weights.shape[0]) + 15) // 16
            k_tiles = kernel_h * kernel_w * input_blocks
            packed = np.zeros((output_blocks, k_tiles, 16, 8), dtype=np.int8)
            for output_channel in range(int(weights.shape[0])):
                for kernel_y in range(kernel_h):
                    for kernel_x in range(kernel_w):
                        for input_channel in range(input_channels):
                            tile = ((kernel_y * kernel_w + kernel_x) * input_blocks +
                                    input_channel // 8)
                            packed[output_channel // 16, tile, output_channel % 16,
                                   input_channel % 8] = weights[
                                       output_channel, input_channel, kernel_y, kernel_x]
            weight_sums = np.sum(weights.astype(np.int64), axis=(1, 2, 3), dtype=np.int64)
            corrected_bias = (bias_i32.astype(np.int64) +
                              (128 - input_spec.zero_point) * weight_sums)
            multiplier_m63 = np.zeros_like(multipliers, dtype="<i8")
            if e2c_compatible:
                multiplier_m63 = (multipliers * np.int64(2)).astype("<i8")
            packed_path = self.asset_path(index, "weights_packed_n16k8_s8.bin")
            weight_sum_path = self.asset_path(index, "weight_sums_i64.bin")
            corrected_bias_path = self.asset_path(index, "corrected_bias_i64.bin")
            multiplier_m63_path = self.asset_path(index, "multiplier_m63_i64.bin")
            packed.tofile(packed_path)
            weight_sums.astype("<i8").tofile(weight_sum_path)
            corrected_bias.astype("<i8").tofile(corrected_bias_path)
            multiplier_m63.tofile(multiplier_m63_path)
            packed_weight_file = str(packed_path.relative_to(self.output))
            weight_sum_file = str(weight_sum_path.relative_to(self.output))
            corrected_bias_file = str(corrected_bias_path.relative_to(self.output))
            multiplier_m63_file = str(multiplier_m63_path.relative_to(self.output))
        maximum_activation = max(input_spec.zero_point, 255 - input_spec.zero_point)
        maximum_weight = int(np.max(np.abs(weights.astype(np.int16))))
        k = int(weights.shape[1] * kernel_h * kernel_w)
        bound = int(np.max(np.abs(bias_i32.astype(np.int64)))) + k * maximum_activation * maximum_weight
        if bound > np.iinfo(np.int32).max:
            raise ValueError(f"int32 accumulator unsafe: {conv.name}: {bound}")
        self.add(
            "conv_dense" if group == 1 else "conv_grouped", conv.name, output, [input_q],
            output_c=int(weights.shape[0]), input_c=int(self.graph.qspec(input_q).shape[1]),
            group=group, kernel_h=kernel_h, kernel_w=kernel_w,
            stride_h=stride_h, stride_w=stride_w,
            pad_top=pads[0], pad_left=pads[1], pad_bottom=pads[2], pad_right=pads[3],
            weight_file=str(weight_path.relative_to(self.output)),
            bias_file=str(bias_path.relative_to(self.output)),
            multiplier_file=str(multiplier_path.relative_to(self.output)),
            shift_file=str(shift_path.relative_to(self.output)),
            packed_weight_file=packed_weight_file,
            weight_sum_file=weight_sum_file,
            corrected_bias_file=corrected_bias_file,
            multiplier_m63_file=multiplier_m63_file,
            accumulator_bound=bound,
            e2c_compatible=int(e2c_compatible),
        )

    def add_expression(self, qnode: onnx.NodeProto) -> None:
        roots, path = self.expression_lut(qnode.input[0], qnode.output[0])
        self.add(f"lut{len(roots)}", qnode.name, qnode.output[0], roots,
                 lut_file=str(path.relative_to(self.output)))

    def branch_descriptor(self, name: str, target: str) -> dict[str, Any]:
        node = self.graph.producer.get(name)
        if node is None:
            raise ValueError(f"missing branch producer: {name}")
        if node.op_type == "DequantizeLinear":
            root = node.input[0]
            source = self.graph.qspec(root)
            target_spec = self.graph.qspec(target)
            lut = build_unary_lut(source.scale, source.zero_point, target_spec.scale, target_spec.zero_point, "none")
            return {"root": root, "transform": "copy", "lut": lut}
        if node.op_type in ("Add", "Mul"):
            roots, path = self.expression_lut(name, target)
            if len(roots) != 1:
                synthetic = f"__stage52_internal_{safe_name(name)}_at_{safe_name(target)}"
                self.add_internal_tensor(synthetic, name, target)
                self.add("lut2", f"{name}@{target}", synthetic, roots,
                         lut_file=str(path.relative_to(self.output)))
                identity = np.arange(256, dtype=np.int16)
                identity = (identity - 128).astype(np.int8)
                return {"root": synthetic, "transform": "copy", "lut": identity}
            return {"root": roots[0], "transform": "copy", "lut": np.fromfile(path, dtype=np.int8)}
        if node.op_type == "Split":
            split_input = node.input[0]
            source_node = self.graph.producer.get(split_input)
            output_index = list(node.output).index(name)
            axis = int(attrs(node).get("axis", 0))
            source_shape = self.graph.meta[split_input][1]
            if axis < 0:
                axis += len(source_shape)
            split_offset = sum(self.graph.meta[output][1][axis]
                               for output in list(node.output)[:output_index])
            if source_node is not None and source_node.op_type in ("Mul", "Add"):
                roots, path = self.expression_lut(split_input, target)
                if len(roots) != 1:
                    raise ValueError(f"split scalar source has multiple roots: {node.name}")
                lut = np.fromfile(path, dtype=np.int8)
                root = roots[0]
                transform = "split"
            elif source_node is not None and source_node.op_type == "Reshape":
                root = self.graph.dq_root(source_node.input[0])
                source = self.graph.qspec(root)
                target_spec = self.graph.qspec(target)
                lut = build_unary_lut(source.scale, source.zero_point,
                                      target_spec.scale, target_spec.zero_point, "none")
                transform = "reshape_split"
            else:
                root = self.root_q(split_input)
                source = self.graph.qspec(root)
                target_spec = self.graph.qspec(target)
                lut = build_unary_lut(source.scale, source.zero_point, target_spec.scale, target_spec.zero_point, "none")
                transform = "split"
            return {"root": root, "transform": transform, "axis": axis,
                    "part": output_index, "parts": len(node.output),
                    "source_shape": "x".join(map(str, source_shape)),
                    "split_offset": split_offset, "lut": lut}
        if node.op_type == "Resize":
            inner = self.branch_descriptor(node.input[0], target)
            inner["transform"] = "resize"
            inner["resize_mode"] = attrs(node).get("mode", b"nearest").decode()
            return inner
        raise ValueError(f"unsupported branch expression: {node.op_type}: {node.name}")

    def add_split(self, qnode: onnx.NodeProto, split: onnx.NodeProto) -> None:
        descriptor = self.branch_descriptor(qnode.input[0], qnode.output[0])
        path = self.asset_path(len(self.operations), "split_lut_s8.bin")
        np.asarray(descriptor.pop("lut"), dtype=np.int8).tofile(path)
        root = descriptor.pop("root")
        self.add("split", split.name, qnode.output[0], [root],
                 lut_file=str(path.relative_to(self.output)), **descriptor)

    def add_concat(self, qnode: onnx.NodeProto, concat: onnx.NodeProto) -> None:
        descriptors: list[dict[str, Any]] = []
        roots: list[str] = []
        if concat.name == "/model.9/Concat":
            base_descriptor = self.branch_descriptor(concat.input[0], qnode.output[0])
            base = base_descriptor["root"]
            lut = np.asarray(base_descriptor["lut"], dtype=np.int8)
            roots = [base]
            descriptors = [
                {"root": base, "transform": "pool0", "lut": lut},
                {"root": base, "transform": "pool1", "lut": lut},
                {"root": base, "transform": "pool2", "lut": lut},
                {"root": base, "transform": "pool3", "lut": lut},
            ]
        else:
            for input_name in concat.input:
                descriptor = self.branch_descriptor(input_name, qnode.output[0])
                if descriptor["root"] not in roots:
                    roots.append(descriptor["root"])
                descriptors.append(descriptor)
        params: dict[str, Any] = {"axis": int(attrs(concat).get("axis", 1)), "branch_count": len(descriptors)}
        for index, descriptor in enumerate(descriptors):
            root = descriptor.pop("root")
            params[f"branch{index}_input_slot"] = roots.index(root)
            lut = descriptor.pop("lut", None)
            if lut is not None:
                path = self.asset_path(len(self.operations), f"concat{index}_lut_s8.bin")
                np.asarray(lut, dtype=np.int8).tofile(path)
                params[f"branch{index}_lut_file"] = str(path.relative_to(self.output))
            else:
                params[f"branch{index}_lut_file"] = "-"
            for key, value in descriptor.items():
                params[f"branch{index}_{key}"] = value
        self.add("concat", concat.name, qnode.output[0], roots, **params)

    def add_transform(self, qnode: onnx.NodeProto, node: onnx.NodeProto) -> None:
        source_name = node.input[0]
        producer = self.graph.producer.get(source_name)
        descriptor: dict[str, Any] = {}
        if producer is not None and producer.op_type == "DequantizeLinear":
            source_q = producer.input[0]
        elif source_name in self.graph.quant_by_source:
            source_q = self.graph.q_for_source(source_name)
        else:
            descriptor = self.branch_descriptor(source_name, qnode.output[0])
            source_q = descriptor.pop("root")
        source_spec = self.graph.qspec(source_q) if source_q in self.graph.producer else self.graph.qspec(qnode.output[0])
        target_spec = self.graph.qspec(qnode.output[0])
        lut = descriptor.pop("lut", None)
        if lut is None:
            lut = build_unary_lut(source_spec.scale, source_spec.zero_point,
                                  target_spec.scale, target_spec.zero_point, "none")
        path = self.asset_path(len(self.operations), "transform_lut_s8.bin")
        lut.tofile(path)
        params: dict[str, Any] = {"lut_file": str(path.relative_to(self.output)), **descriptor}
        if node.op_type == "Transpose":
            params["perm"] = ",".join(map(str, attrs(node)["perm"]))
        elif node.op_type == "Reshape":
            params["target_shape"] = "x".join(map(str, target_spec.shape))
        kind = node.op_type.lower() if not descriptor else f"{descriptor.get('transform', 'view')}_{node.op_type.lower()}"
        self.add(kind, node.name, qnode.output[0], [source_q], **params)

    def add_matmul(self, qnode: onnx.NodeProto, node: onnx.NodeProto) -> None:
        left = self.graph.dq_root(node.input[0])
        right = self.graph.dq_root(node.input[1])
        left_spec, right_spec, output_spec = self.graph.qspec(left), self.graph.qspec(right), self.graph.qspec(qnode.output[0])
        ratio = (fraction_from_f32_bits(f32_bits(left_spec.scale)) *
                 fraction_from_f32_bits(f32_bits(right_spec.scale)) /
                 fraction_from_f32_bits(f32_bits(output_spec.scale)))
        multiplier, shift = encode_multiplier(ratio)
        self.add("matmul", node.name, qnode.output[0], [left, right],
                 multiplier=multiplier, right_shift=shift,
                 left_zero_point=left_spec.zero_point, right_zero_point=right_spec.zero_point,
                 output_zero_point=output_spec.zero_point)

    def add_softmax(self, qnode: onnx.NodeProto, transpose: onnx.NodeProto) -> None:
        softmax = self.graph.producer[transpose.input[0]]
        multiply = self.graph.producer[softmax.input[0]]
        source_q = self.graph.dq_root(multiply.input[0])
        factor = np.float32(self.graph.constant(multiply.input[1]).reshape(-1)[0])
        source_spec = self.graph.qspec(source_q)
        target_spec = self.graph.qspec(qnode.output[0])
        exp_q48 = np.empty(256, dtype="<u8")
        for difference in range(256):
            value = math.exp(-float(np.float32(difference * source_spec.scale * factor)))
            exp_q48[difference] = max(1, round_fraction_even(Fraction.from_float(value) * (1 << 48)))
        exp_path = self.asset_path(len(self.operations), "softmax_exp_q48_u64.bin")
        exp_q48.tofile(exp_path)
        self.add("softmax_transpose", transpose.name, qnode.output[0], [source_q],
                 axis=int(attrs(softmax).get("axis", -1)),
                 perm=",".join(map(str, attrs(transpose)["perm"])),
                 exp_file=str(exp_path.relative_to(self.output)),
                 output_zero_point=target_spec.zero_point,
                 output_scale=f"{float(target_spec.scale):.17g}",
                 output_reciprocal_q32=round_fraction_even(
                     Fraction(1 << 32, 1) /
                     fraction_from_f32_bits(f32_bits(target_spec.scale))
                 ))

    def build_operations(self) -> None:
        for qnode in self.graph.quant_nodes:
            source = self.graph.producer.get(qnode.input[0])
            if source is None:
                self.add_input(qnode)
            elif source.op_type == "Conv":
                self.add_conv(qnode, source)
            elif source.op_type in ("Mul", "Add"):
                self.add_expression(qnode)
            elif source.op_type == "Split":
                self.add_split(qnode, source)
            elif source.op_type == "Concat":
                self.add_concat(qnode, source)
            elif source.op_type in ("Reshape", "Transpose"):
                self.add_transform(qnode, source)
            elif source.op_type == "MatMul":
                self.add_matmul(qnode, source)
            elif source.op_type == "Resize":
                descriptor = self.branch_descriptor(qnode.input[0], qnode.output[0])
                path = self.asset_path(len(self.operations), "resize_lut_s8.bin")
                np.asarray(descriptor.pop("lut"), dtype=np.int8).tofile(path)
                root = descriptor.pop("root")
                self.add("resize", source.name, qnode.output[0], [root],
                         lut_file=str(path.relative_to(self.output)), **descriptor)
            elif source.op_type == "Transpose" and self.graph.producer[source.input[0]].op_type == "Softmax":
                self.add_softmax(qnode, source)
            else:
                # Softmax+transpose reaches this branch through Transpose above only when
                # tested before the generic transform.
                raise ValueError(f"unsupported quantized source: {source.op_type}: {source.name}")

    def fix_softmax_order(self) -> None:
        # Replace the two generic Transpose rows whose input is Softmax with the
        # fixed-point softmax descriptor.  Kept separate to make the dispatch
        # order obvious in generated evidence.
        for qnode in self.graph.quant_nodes:
            source = self.graph.producer.get(qnode.input[0])
            if source is None or source.op_type != "Transpose":
                continue
            parent = self.graph.producer.get(source.input[0])
            if parent is None or parent.op_type != "Softmax":
                continue
            index = next(i for i, row in enumerate(self.operations) if row["output"] == self.tid(qnode.output[0]))
            self.operations.pop(index)
            for row_index, row in enumerate(self.operations):
                row["index"] = row_index
            # Asset names remain stable enough for reproducibility even though
            # this replacement is appended; schedule is topologically sorted below.
            self.add_softmax(qnode, source)
        output_to_row = {int(row["output"]): row for row in self.operations}
        pending = list(self.operations)
        ordered: list[dict[str, Any]] = []
        ready_tensors: set[int] = set()
        while pending:
            progress = False
            for row in list(pending):
                inputs = [int(value) for value in str(row["inputs"]).split(",") if value != ""]
                if row["kind"] == "input_quant" or all(value in ready_tensors for value in inputs):
                    pending.remove(row)
                    ordered.append(row)
                    ready_tensors.add(int(row["output"]))
                    progress = True
            if not progress:
                missing = [(row["name"], row["inputs"]) for row in pending[:5]]
                raise ValueError(f"operation DAG cannot be ordered: {missing}")
        self.operations = ordered
        for index, row in enumerate(self.operations):
            row["index"] = index

    def finalize_arena(self) -> int:
        for tensor in self.tensors:
            tensor["first_op"] = len(self.operations)
            tensor["last_op"] = -1
        for row in self.operations:
            output = int(row["output"])
            self.tensors[output]["first_op"] = min(int(self.tensors[output]["first_op"]), int(row["index"]))
            for value in str(row["inputs"]).split(","):
                if value:
                    tensor_id = int(value)
                    self.tensors[tensor_id]["last_op"] = max(int(self.tensors[tensor_id]["last_op"]), int(row["index"]))
        for tensor in self.tensors:
            if int(tensor["last_op"]) < int(tensor["first_op"]):
                tensor["last_op"] = len(self.operations)
        active: list[tuple[int, int, int]] = []
        free: list[tuple[int, int]] = []
        arena_end = 0
        for tensor in sorted(self.tensors, key=lambda item: (int(item["first_op"]), int(item["id"]))):
            remaining: list[tuple[int, int, int]] = []
            for last_op, offset, size in active:
                if last_op < int(tensor["first_op"]):
                    free.append((offset, size))
                else:
                    remaining.append((last_op, offset, size))
            active = remaining
            size = align_up(int(tensor["storage_bytes"]))
            candidate = next((item for item in sorted(free, key=lambda item: (item[1], item[0])) if item[1] >= size), None)
            if candidate is None:
                offset = arena_end
                arena_end += size
            else:
                free.remove(candidate)
                offset = candidate[0]
                if candidate[1] > size:
                    free.append((offset + size, candidate[1] - size))
            tensor["arena_offset"] = offset
            active.append((int(tensor["last_op"]), offset, size))
        return arena_end


def generate_head_assets(graph: Graph, builder: PackageBuilder,
                         input_resolution: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for scale_index in range(3):
        reg_suffix = f"/model.23/one2one_cv2.{scale_index}/one2one_cv2.{scale_index}.2/Conv_output_0_QuantizeLinear_Output"
        cls_suffix = f"/model.23/one2one_cv3.{scale_index}/one2one_cv3.{scale_index}.2/Conv_output_0_QuantizeLinear_Output"
        reg = graph.qspec(reg_suffix)
        cls = graph.qspec(cls_suffix)
        if len(reg.shape) != 4 or reg.shape[-2] != reg.shape[-1] or cls.shape[-2:] != reg.shape[-2:]:
            raise ValueError(f"unexpected head tensor geometry at scale {scale_index}")
        resolution = reg.shape[-1]
        if input_resolution % resolution != 0:
            raise ValueError(f"non-integral head stride at scale {scale_index}")
        reg_lut = np.empty((4, 256), dtype="<i4")
        for channel in range(4):
            for code in range(256):
                reg_lut[channel, code] = round_fraction_even(
                    Fraction(code - reg.zero_point) * fraction_from_f32_bits(f32_bits(reg.scale)) * (1 << 16)
                )
        cls_lut = np.empty(256, dtype="<u4")
        for code in range(256):
            value = np.float32((code - cls.zero_point) * cls.scale)
            with np.errstate(over="ignore"):
                probability = float(np.float32(1.0) /
                                    np.float32(np.float32(1.0) + np.exp(np.float32(-value))))
            cls_lut[code] = min((1 << 24) - 1, max(0, round_fraction_even(Fraction.from_float(probability) * (1 << 24))))
        reg_path = builder.assets / f"head_scale{scale_index}_reg_q16_i32.bin"
        cls_path = builder.assets / f"head_scale{scale_index}_sigmoid_q24_u32.bin"
        reg_lut.tofile(reg_path)
        cls_lut.tofile(cls_path)
        rows.append({
            "scale_index": scale_index, "resolution": resolution,
            "stride": input_resolution // resolution,
            "reg_tensor": builder.tid(reg_suffix), "cls_tensor": builder.tid(cls_suffix),
            "reg_lut_file": str(reg_path.relative_to(builder.output)),
            "cls_lut_file": str(cls_path.relative_to(builder.output)),
        })
    write_tsv(builder.output / "head_assets.tsv", rows)
    return {"scales": rows, "output_shape": [1, 300, 6], "tie_policy": "score-descending-index-ascending"}


def generate(args: argparse.Namespace) -> None:
    model_path = args.model.resolve()
    model_sha256 = sha256_file(model_path)
    source_model_sha256 = EXPECTED_MODEL_SHA256
    model = onnx.load(model_path, load_external_data=True)
    metadata = {item.key: item.value for item in model.metadata_props}
    graph_input = model.graph.input[0].type.tensor_type.shape.dim
    if len(graph_input) != 4 or any(not dim.HasField("dim_value") for dim in graph_input):
        raise ValueError("full package requires a fully static 1x3xRxR input")
    input_shape = tuple(int(dim.dim_value) for dim in graph_input)
    if input_shape[0] != 1 or input_shape[1] != 3 or input_shape[2] != input_shape[3]:
        raise ValueError(f"unsupported full-graph input shape: {input_shape}")
    input_resolution = input_shape[2]
    if model_sha256 == EXPECTED_MODEL_SHA256:
        if input_resolution != 640:
            raise ValueError("accepted source model must retain its 640x640 input")
    else:
        if not args.allow_stage60_static_derivative:
            raise ValueError("accepted model SHA-256 mismatch")
        if metadata.get("y26_stage60_parent_model_sha256") != EXPECTED_MODEL_SHA256:
            raise ValueError("Stage60 derivative parent-model identity mismatch")
        if metadata.get("y26_stage60_resolution") != str(input_resolution):
            raise ValueError("Stage60 derivative resolution metadata mismatch")
        if metadata.get("y26_stage60_transform") != "static-input-attention-and-head-geometry-v1":
            raise ValueError("unsupported Stage60 static transform")
    profile_id = args.profile_id or (
        PROFILE_ID if input_resolution == 640
        else f"K1X_INT8_V1_YOLO26N_{input_resolution}_FULL_GRAPH_001"
    )
    expected_profile = f"K1X_INT8_V1_YOLO26N_{input_resolution}_FULL_GRAPH_001"
    if profile_id != expected_profile:
        raise ValueError(f"profile ID does not match static input: {profile_id}")
    output = args.out_dir.resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    graph = Graph(model)
    builder = PackageBuilder(graph, output)
    # Softmax transpose must be classified before generic transpose.
    original_add_transform = builder.add_transform
    def guarded_transform(qnode: onnx.NodeProto, node: onnx.NodeProto) -> None:
        parent = graph.producer.get(node.input[0])
        if node.op_type == "Transpose" and parent is not None and parent.op_type == "Softmax":
            builder.add_softmax(qnode, node)
        else:
            original_add_transform(qnode, node)
    builder.add_transform = guarded_transform  # type: ignore[method-assign]
    builder.build_operations()
    arena_bytes = builder.finalize_arena()
    head = generate_head_assets(graph, builder, input_resolution)
    optimized_core_manifest = ""
    if args.optimized_core_package is not None:
        optimized_core_manifest = copy_runtime_package(
            args.optimized_core_package, output / "optimized_core")
    write_tsv(output / "tensors.tsv", builder.tensors)
    write_tsv(output / "operations.tsv", builder.operations)
    package = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "profile_id": profile_id,
        "model_sha256": model_sha256,
        "source_lineage_id": (
            f"accepted-yolo26-qdq:{source_model_sha256}:stage52-full"
            if model_sha256 == source_model_sha256 else
            f"accepted-yolo26-qdq:{source_model_sha256}:stage60-static-r{input_resolution}"
        ),
        "byte_order": "little-endian",
        "layout_id": LAYOUT_FEATURE,
        "feature_layout": LAYOUT_FEATURE,
        "linear_layout": LAYOUT_LINEAR,
        "input_name": "images",
        "input_dim0": 1,
        "input_dim1": 3,
        "input_dim2": input_resolution,
        "input_dim3": input_resolution,
        "input_tensor_id": builder.tid("images_QuantizeLinear_Output"),
        "output_name": "output0",
        "output_dim0": 1,
        "output_dim1": 300,
        "output_dim2": 6,
        "tensor_count": len(builder.tensors),
        "operation_count": len(builder.operations),
        "integer_boundary_count": len(builder.tensors),
        "arena_bytes": arena_bytes,
        "head_scale_count": len(head["scales"]),
        "head_tie_policy": head["tie_policy"],
        "runtime_graph_dispatch": "numeric-static-operation-ids",
        "optimized_core_manifest_sha256": optimized_core_manifest,
    }
    if model_sha256 != source_model_sha256:
        package["source_model_sha256"] = source_model_sha256
        package["static_input_resolution"] = input_resolution
    (output / "package.json").write_text(json.dumps(package, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    root_manifest = output / "asset_hashes.tsv"
    files = [path for path in sorted(output.rglob("*")) if path.is_file() and path != root_manifest]
    hashes = [{"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
               "sha256": sha256_file(path)} for path in files]
    write_tsv(output / "asset_hashes.tsv", hashes, ["path", "bytes", "sha256"])
    print(json.dumps({
        "package": str(output), "manifest_sha256": sha256_file(output / "asset_hashes.tsv"),
        "model_sha256": model_sha256, "source_model_sha256": source_model_sha256,
        "profile_id": profile_id, "resolution": input_resolution,
        "tensors": len(builder.tensors),
        "operations": len(builder.operations), "arena_bytes": arena_bytes,
    }, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--optimized-core-package", type=Path)
    parser.add_argument("--profile-id")
    parser.add_argument("--allow-stage60-static-derivative", action="store_true")
    generate(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
