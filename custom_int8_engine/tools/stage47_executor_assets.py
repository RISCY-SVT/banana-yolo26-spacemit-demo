#!/usr/bin/env python3
"""Generate Stage47 graph census, integrated-kernel cases, and model4-8 AOT assets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
import onnx.utils
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper, shape_inference


MODEL4_PREACT = "/model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output"
MODEL4_POSTACT = "/model.4/cv2/act/Mul_output_0_QuantizeLinear_Output"
MODEL5_OUTPUT = "/model.5/act/Mul_output_0_QuantizeLinear_Output"
MODEL6_OUTPUT = "/model.6/cv2/act/Mul_output_0_QuantizeLinear_Output"
MODEL7_OUTPUT = "/model.7/act/Mul_output_0_QuantizeLinear_Output"
MODEL8_OUTPUT = "/model.8/cv2/act/Mul_output_0_QuantizeLinear_Output"
SLICE_OUTPUT_TO_KEY = {
    MODEL4_POSTACT: "model4.postact",
    MODEL5_OUTPUT: "model.5.output",
    "/model.6/Split_output_1_QuantizeLinear_Output": "model.6.split1",
    "/model.6/m.0/cv1/act/Mul_output_0_QuantizeLinear_Output": "model.6.m0_cv1_act",
    "/model.6/m.0/m/m.0/cv1/act/Mul_output_0_QuantizeLinear_Output": "model.6.r0_cv1_act",
    "/model.6/m.0/m/m.0/cv2/conv/Conv_output_0_QuantizeLinear_Output": "model.6.r0_cv2_preact",
    "/model.6/m.0/m/m.0/Add_output_0_QuantizeLinear_Output": "model.6.add0",
    "/model.6/m.0/m/m.1/cv1/act/Mul_output_0_QuantizeLinear_Output": "model.6.r1_cv1_act",
    "/model.6/m.0/m/m.1/cv2/conv/Conv_output_0_QuantizeLinear_Output": "model.6.r1_cv2_preact",
    "/model.6/m.0/Concat_output_0_QuantizeLinear_Output": "model.6.inner_concat",
    "/model.6/Concat_output_0_QuantizeLinear_Output": "model.6.outer_concat",
    MODEL6_OUTPUT: "model.6.output",
    MODEL7_OUTPUT: "model.7.output",
    "/model.8/Split_output_1_QuantizeLinear_Output": "model.8.split1",
    "/model.8/m.0/cv1/act/Mul_output_0_QuantizeLinear_Output": "model.8.m0_cv1_act",
    "/model.8/m.0/m/m.0/cv1/act/Mul_output_0_QuantizeLinear_Output": "model.8.r0_cv1_act",
    "/model.8/m.0/m/m.0/cv2/conv/Conv_output_0_QuantizeLinear_Output": "model.8.r0_cv2_preact",
    "/model.8/m.0/m/m.0/Add_output_0_QuantizeLinear_Output": "model.8.add0",
    "/model.8/m.0/m/m.1/cv1/act/Mul_output_0_QuantizeLinear_Output": "model.8.r1_cv1_act",
    "/model.8/m.0/m/m.1/cv2/conv/Conv_output_0_QuantizeLinear_Output": "model.8.r1_cv2_preact",
    "/model.8/m.0/Concat_output_0_QuantizeLinear_Output": "model.8.inner_concat",
    "/model.8/Concat_output_0_QuantizeLinear_Output": "model.8.outer_concat",
    MODEL8_OUTPUT: "model.8.output",
}
SLICE_OUTPUTS = list(SLICE_OUTPUT_TO_KEY)
ALIGNMENT = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", name.strip("/"))


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in materialized:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        if not fieldnames:
            return
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def shape_count(shape: list[int]) -> int:
    return math.prod(shape)


def element_bytes(dtype: int) -> int:
    return {
        TensorProto.FLOAT: 4,
        TensorProto.UINT8: 1,
        TensorProto.INT8: 1,
        TensorProto.UINT16: 2,
        TensorProto.INT16: 2,
        TensorProto.INT32: 4,
        TensorProto.INT64: 8,
        TensorProto.FLOAT16: 2,
    }.get(dtype, 0)


@dataclass(frozen=True)
class TensorMeta:
    dtype: int
    shape: list[int]


@dataclass(frozen=True)
class QuantSpec:
    tensor_name: str
    scale: float
    zero_point: int
    shape: list[int]

    @property
    def h(self) -> int:
        return self.shape[-2]

    @property
    def w(self) -> int:
        return self.shape[-1]

    @property
    def c(self) -> int:
        return self.shape[1]


class GraphIndex:
    def __init__(self, model: onnx.ModelProto) -> None:
        self.model = model
        try:
            inferred = shape_inference.infer_shapes(model, data_prop=True)
        except Exception:
            inferred = shape_inference.infer_shapes(model)
        self.metadata: dict[str, TensorMeta] = {}
        for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
            tensor = value.type.tensor_type
            dims: list[int] = []
            concrete = True
            for dim in tensor.shape.dim:
                if not dim.HasField("dim_value"):
                    concrete = False
                    break
                dims.append(int(dim.dim_value))
            if concrete:
                self.metadata[value.name] = TensorMeta(int(tensor.elem_type), dims)
        self.initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
        self.producer = {output: node for node in model.graph.node for output in node.output}
        self.consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
        for node in model.graph.node:
            for input_name in node.input:
                self.consumers[input_name].append(node)
        self.nodes_by_name = {node.name: node for node in model.graph.node}

    def scalar(self, name: str) -> float | int:
        array = np.asarray(self.initializers[name]).reshape(-1)
        if array.size != 1:
            raise ValueError(f"expected scalar initializer {name}, got {array.shape}")
        return array[0].item()

    def qspec(self, tensor_name: str) -> QuantSpec:
        node = self.producer.get(tensor_name)
        if node is None or node.op_type != "QuantizeLinear" or len(node.input) < 3:
            raise ValueError(f"not a QuantizeLinear output: {tensor_name}")
        meta = self.metadata.get(tensor_name)
        if meta is None or meta.dtype not in (TensorProto.UINT8, TensorProto.INT8) or len(meta.shape) != 4:
            raise ValueError(f"unsupported quantized tensor metadata: {tensor_name}: {meta}")
        scale = float(self.scalar(node.input[1]))
        zero_point = int(self.scalar(node.input[2]))
        return QuantSpec(tensor_name, scale, zero_point, meta.shape)

    def input_quant_tensor(self, node: onnx.NodeProto) -> str:
        producer = self.producer.get(node.input[0])
        if producer is None or producer.op_type != "DequantizeLinear":
            raise ValueError(f"Conv input is not a Q/DQ boundary: {node.name}")
        return producer.input[0]

    def conv_output_quant_tensor(self, node: onnx.NodeProto) -> str:
        candidates = [consumer for consumer in self.consumers[node.output[0]] if consumer.op_type == "QuantizeLinear"]
        if len(candidates) != 1:
            raise ValueError(f"expected one Conv output QuantizeLinear: {node.name}: {len(candidates)}")
        return candidates[0].output[0]

    def postact_quant_tensor(self, node: onnx.NodeProto) -> str | None:
        conv_q = self.conv_output_quant_tensor(node)
        conv_dq = next((item for item in self.consumers[conv_q] if item.op_type == "DequantizeLinear"), None)
        if conv_dq is None:
            return None
        sigmoid = next((item for item in self.consumers[conv_dq.output[0]] if item.op_type == "Sigmoid"), None)
        if sigmoid is None:
            return None
        mul = next(
            (
                item
                for item in self.consumers[conv_dq.output[0]]
                if item.op_type == "Mul" and sigmoid.output[0] in item.input
            ),
            None,
        )
        if mul is None:
            return None
        quant = next((item for item in self.consumers[mul.output[0]] if item.op_type == "QuantizeLinear"), None)
        return None if quant is None else quant.output[0]

    def conv_assets(self, node: onnx.NodeProto) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        weight_dq = self.producer.get(node.input[1])
        if weight_dq is None or weight_dq.op_type != "DequantizeLinear":
            raise ValueError(f"Conv weight is not Q/DQ: {node.name}")
        weight_name = weight_dq.input[0]
        scale_name = weight_dq.input[1]
        weights_oihw = np.asarray(self.initializers[weight_name], dtype=np.int8)
        scales = np.asarray(self.initializers[scale_name], dtype=np.float32)
        stem = weight_name.removesuffix(".weight_quantized")
        bias_name = f"{stem}.bias_quantized"
        if bias_name not in self.initializers:
            raise ValueError(f"missing quantized bias {bias_name} for {node.name}")
        bias = np.asarray(self.initializers[bias_name], dtype=np.int32)
        return np.ascontiguousarray(np.transpose(weights_oihw, (0, 2, 3, 1))), scales, bias, weight_name


def node_attributes(node: onnx.NodeProto) -> dict[str, Any]:
    return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}


def block_name(node_name: str) -> str:
    match = re.search(r"/model\.(\d+)(?:/|$)", node_name)
    return f"model.{match.group(1)}" if match else "graph"


def shape_class(op_type: str, output_h: int, output_w: int, kernel_h: int, kernel_w: int, stride: int, n: int, k: int, group: int) -> str:
    if op_type in ("MatMul", "Gemm"):
        return "matmul_attention"
    if group != 1:
        return "grouped_or_depthwise_conv"
    if n < 16:
        return "small_n_head_conv"
    if kernel_h == 1 and kernel_w == 1:
        resolution = "high" if output_h * output_w >= 6400 else "low"
        return f"1x1_{resolution}_resolution"
    if kernel_h == 3 and kernel_w == 3 and stride == 2:
        return "3x3_stride2"
    if kernel_h == 3 and kernel_w == 3:
        return "3x3_stride1"
    if k >= 1024:
        return "high_k_other_conv"
    return "other_conv"


def graph_shape_census(index: GraphIndex, out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    class_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"nodes": 0, "macs": 0, "weights": 0, "activations": 0})
    total_macs = 0
    for graph_index, node in enumerate(index.model.graph.node):
        if node.op_type not in ("Conv", "MatMul", "Gemm"):
            continue
        attrs = node_attributes(node)
        input_meta = index.metadata.get(node.input[0])
        output_meta = index.metadata.get(node.output[0]) if node.output else None
        if input_meta is None or output_meta is None:
            continue
        group = int(attrs.get("group", 1))
        kernel_h = kernel_w = stride_h = stride_w = pad_h = pad_w = 0
        weight_bytes = 0
        if node.op_type == "Conv":
            weight_dq = index.producer.get(node.input[1])
            weight = None if weight_dq is None else index.initializers.get(weight_dq.input[0])
            if weight is None or len(input_meta.shape) != 4 or len(output_meta.shape) != 4:
                continue
            kernel_h, kernel_w = map(int, attrs.get("kernel_shape", weight.shape[-2:]))
            strides = attrs.get("strides", [1, 1])
            pads = attrs.get("pads", [0, 0, 0, 0])
            stride_h, stride_w = map(int, strides)
            pad_h, pad_w = int(pads[0]), int(pads[1])
            m = int(output_meta.shape[0] * output_meta.shape[2] * output_meta.shape[3])
            n = int(output_meta.shape[1])
            k = int(kernel_h * kernel_w * input_meta.shape[1] // group)
            weight_bytes = int(weight.nbytes)
            output_h, output_w = output_meta.shape[-2:]
        else:
            rhs_meta = index.metadata.get(node.input[1])
            if rhs_meta is None or len(output_meta.shape) < 2:
                continue
            m = shape_count(output_meta.shape) // int(output_meta.shape[-1])
            n = int(output_meta.shape[-1])
            k = int(input_meta.shape[-1])
            output_h = output_w = 0
            rhs_initializer = index.initializers.get(node.input[1])
            weight_bytes = 0 if rhs_initializer is None else int(rhs_initializer.nbytes)
        macs = int(m * n * k)
        total_macs += macs
        activation_bytes = shape_count(input_meta.shape) * element_bytes(input_meta.dtype) + shape_count(output_meta.shape) * element_bytes(output_meta.dtype)
        category = shape_class(node.op_type, output_h, output_w, kernel_h, kernel_w, stride_h, n, k, group)
        kernel_class = (
            "matmul_conservative"
            if node.op_type != "Conv"
            else "grouped_conservative"
            if group != 1
            else "m12n16_fullshape"
            if n >= 16
            else "m4n16_n_tail"
        )
        row = {
            "graph_index": graph_index,
            "node_name": node.name,
            "block": block_name(node.name),
            "op_type": node.op_type,
            "input_shape": "x".join(map(str, input_meta.shape)),
            "output_shape": "x".join(map(str, output_meta.shape)),
            "kernel": f"{kernel_h}x{kernel_w}" if node.op_type == "Conv" else "not-applicable",
            "stride": f"{stride_h}x{stride_w}" if node.op_type == "Conv" else "not-applicable",
            "padding": f"{pad_h}x{pad_w}" if node.op_type == "Conv" else "not-applicable",
            "group": group,
            "M": m,
            "N": n,
            "K": k,
            "MACs": macs,
            "weight_bytes": weight_bytes,
            "activation_bytes": activation_bytes,
            "M_mod_4": m % 4,
            "M_mod_8": m % 8,
            "M_mod_12": m % 12,
            "N_mod_16": n % 16,
            "N_mod_32": n % 32,
            "K_padding_to_8": align_up(k, 8) - k,
            "candidate_layout": "resident_NHWC16_signed_s8",
            "candidate_kernel_class": kernel_class,
            "shape_class": category,
        }
        rows.append(row)
        class_totals[category]["nodes"] += 1
        class_totals[category]["macs"] += macs
        class_totals[category]["weights"] += weight_bytes
        class_totals[category]["activations"] += activation_bytes
    write_tsv(out_dir / "graph_shape_census.tsv", rows)
    coverage_rows = []
    for category, values in sorted(class_totals.items()):
        coverage_rows.append(
            {
                "shape_class": category,
                "node_count": values["nodes"],
                "macs": values["macs"],
                "mac_coverage_pct": 0.0 if total_macs == 0 else 100.0 * values["macs"] / total_macs,
                "weight_bytes": values["weights"],
                "activation_bytes_contract": values["activations"],
            }
        )
    write_tsv(out_dir / "graph_shape_coverage.tsv", coverage_rows)
    summary = {
        "compute_node_count": len(rows),
        "conv_node_count": sum(row["op_type"] == "Conv" for row in rows),
        "matmul_gemm_node_count": sum(row["op_type"] != "Conv" for row in rows),
        "total_macs": total_macs,
        "shape_classes": len(coverage_rows),
    }
    report = [
        "# Graph Shape Class Report",
        "",
        f"- Compute nodes: `{summary['compute_node_count']}`",
        f"- Conv nodes: `{summary['conv_node_count']}`",
        f"- MatMul/Gemm nodes: `{summary['matmul_gemm_node_count']}`",
        f"- Static MACs: `{summary['total_macs']}`",
        "- Shapes, divisibility, groups, and bytes are derived from the accepted ONNX bytes.",
        "- `m12n16_fullshape` is a candidate mapping, not measured proof until board results are joined.",
        "",
        "| class | nodes | MACs | MAC share |",
        "|---|---:|---:|---:|",
    ]
    for row in coverage_rows:
        report.append(f"| {row['shape_class']} | {row['node_count']} | {row['macs']} | {row['mac_coverage_pct']:.6f}% |")
    (out_dir / "graph_shape_class_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return rows, summary


class ScheduleBuilder:
    def __init__(self, index: GraphIndex, package: Path) -> None:
        self.index = index
        self.package = package
        self.assets = package / "assets"
        self.assets.mkdir(parents=True, exist_ok=True)
        self.tensors: list[dict[str, Any]] = []
        self.ops: list[dict[str, Any]] = []
        self.tensor_by_key: dict[str, int] = {}

    def tensor(self, key: str, qspec: QuantSpec, channels: int | None = None, name: str | None = None) -> int:
        if key in self.tensor_by_key:
            return self.tensor_by_key[key]
        tensor_id = len(self.tensors)
        channel_count = qspec.c if channels is None else channels
        self.tensors.append(
            {
                "id": tensor_id,
                "key": key,
                "logical_name": qspec.tensor_name if name is None else name,
                "h": qspec.h,
                "w": qspec.w,
                "c": channel_count,
                "scale": f"{qspec.scale:.17g}",
                "zero_point": qspec.zero_point,
                "first_op": -1,
                "last_op": -1,
                "arena_offset": 0,
                "bytes": qspec.h * qspec.w * channel_count,
                "physical_layout": "NHWC_signed_s8",
            }
        )
        self.tensor_by_key[key] = tensor_id
        return tensor_id

    def add_op(self, kind: str, name: str, **values: Any) -> int:
        row = {
            "index": len(self.ops),
            "kind": kind,
            "name": name,
            "input0": -1,
            "input1": -1,
            "input2": -1,
            "input3": -1,
            "output0": -1,
            "output1": -1,
            "weights_file": "",
            "weight_scales_file": "",
            "bias_file": "",
            "weight_count": 0,
            "kernel_h": 0,
            "kernel_w": 0,
            "stride_h": 0,
            "stride_w": 0,
            "pad_h": 0,
            "pad_w": 0,
            "group": 1,
            "conv_output_scale": 0,
            "conv_output_zero_point": 0,
            "segment0_channel_begin": 0,
            "segment0_channel_count": 0,
            "segment0_activation": "none",
            "segment1_channel_begin": 0,
            "segment1_channel_count": 0,
            "segment1_activation": "none",
        }
        row.update(values)
        self.ops.append(row)
        return int(row["index"])

    def add_conv(
        self,
        node_name: str,
        input_id: int,
        output0: int,
        activation0: str,
        output1: int = -1,
        activation1: str = "none",
        segment0_begin: int = 0,
        segment0_count: int | None = None,
        segment1_begin: int = 0,
        segment1_count: int = 0,
    ) -> int:
        node = self.index.nodes_by_name[node_name]
        attrs = node_attributes(node)
        weights, scales, bias, source_weight_name = self.index.conv_assets(node)
        conv_q = self.index.qspec(self.index.conv_output_quant_tensor(node))
        prefix = f"conv_{len([op for op in self.ops if op['kind'] == 'conv']):03d}_{safe_name(node_name)}"
        paths = {
            "weights_file": f"assets/{prefix}.weights_ohwi_s8.bin",
            "weight_scales_file": f"assets/{prefix}.weight_scales_f32.bin",
            "bias_file": f"assets/{prefix}.bias_i32.bin",
        }
        weights.tofile(self.package / paths["weights_file"])
        scales.tofile(self.package / paths["weight_scales_file"])
        bias.tofile(self.package / paths["bias_file"])
        output_c = int(weights.shape[0])
        if segment0_count is None:
            segment0_count = output_c
        strides = attrs.get("strides", [1, 1])
        pads = attrs.get("pads", [0, 0, 0, 0])
        return self.add_op(
            "conv",
            node_name,
            input0=input_id,
            output0=output0,
            output1=output1,
            **paths,
            weight_count=int(weights.size),
            kernel_h=int(weights.shape[1]),
            kernel_w=int(weights.shape[2]),
            stride_h=int(strides[0]),
            stride_w=int(strides[1]),
            pad_h=int(pads[0]),
            pad_w=int(pads[1]),
            group=int(attrs.get("group", 1)),
            conv_output_scale=f"{conv_q.scale:.17g}",
            conv_output_zero_point=conv_q.zero_point,
            segment0_channel_begin=segment0_begin,
            segment0_channel_count=segment0_count,
            segment0_activation=activation0,
            segment1_channel_begin=segment1_begin,
            segment1_channel_count=segment1_count,
            segment1_activation=activation1,
            source_weight_name=source_weight_name,
            weights_sha256=sha256_array(weights),
            weight_scales_sha256=sha256_array(scales),
            bias_sha256=sha256_array(bias),
        )

    def add_lut(self, name: str, input_id: int, output_id: int, activation: str) -> None:
        self.add_op("lut", name, input0=input_id, output0=output_id, segment0_activation=activation)

    def add_add_silu(self, name: str, lhs_id: int, rhs_preact_id: int, output_id: int) -> None:
        self.add_op("add_silu", name, input0=lhs_id, input1=rhs_preact_id, output0=output_id)

    def add_concat(self, name: str, inputs: list[int], output_id: int) -> None:
        if len(inputs) > 4:
            raise ValueError("resident executor Concat supports at most four inputs")
        padded = inputs + [-1] * (4 - len(inputs))
        self.add_op(
            "concat", name, input0=padded[0], input1=padded[1], input2=padded[2],
            input3=padded[3], output0=output_id,
        )

    def add_maxpool(self, name: str, input_id: int, output_id: int) -> None:
        node = self.index.nodes_by_name[name]
        attrs = node_attributes(node)
        kernel = attrs.get("kernel_shape", [1, 1])
        strides = attrs.get("strides", [1, 1])
        pads = attrs.get("pads", [0, 0, 0, 0])
        self.add_op(
            "maxpool", name, input0=input_id, output0=output_id,
            kernel_h=int(kernel[0]), kernel_w=int(kernel[1]),
            stride_h=int(strides[0]), stride_w=int(strides[1]),
            pad_h=int(pads[0]), pad_w=int(pads[1]),
        )

    def simple_conv_block(self, prefix: str, input_id: int, output_q_name: str) -> int:
        output_spec = self.index.qspec(output_q_name)
        output_id = self.tensor(f"{prefix}.output", output_spec)
        self.add_conv(f"/{prefix}/conv/Conv", input_id, output_id, "silu")
        return output_id

    def c2f_cib_block(self, prefix: str, input_id: int, output_q_name: str) -> int:
        outer_q_name = f"/{prefix}/Concat_output_0_QuantizeLinear_Output"
        inner_q_name = f"/{prefix}/m.0/Concat_output_0_QuantizeLinear_Output"
        split1_q_name = f"/{prefix}/Split_output_1_QuantizeLinear_Output"
        add0_q_name = f"/{prefix}/m.0/m/m.0/Add_output_0_QuantizeLinear_Output"
        output_spec = self.index.qspec(output_q_name)
        outer_spec = self.index.qspec(outer_q_name)
        inner_spec = self.index.qspec(inner_q_name)
        split1_spec = self.index.qspec(split1_q_name)
        add0_spec = self.index.qspec(add0_q_name)
        cv1_node = self.index.nodes_by_name[f"/{prefix}/cv1/conv/Conv"]
        cv1_conv_q = self.index.qspec(self.index.conv_output_quant_tensor(cv1_node))
        hidden = cv1_conv_q.c // 2
        split0_id = self.tensor(f"{prefix}.split0_outer", outer_spec, hidden, f"/{prefix}/Split_output_0@outer_scale")
        split1_id = self.tensor(f"{prefix}.split1", split1_spec)
        self.add_conv(
            f"/{prefix}/cv1/conv/Conv",
            input_id,
            split0_id,
            "silu",
            output1=split1_id,
            activation1="silu",
            segment0_begin=0,
            segment0_count=hidden,
            segment1_begin=hidden,
            segment1_count=hidden,
        )

        m0_cv1_act = self.index.qspec(f"/{prefix}/m.0/cv1/act/Mul_output_0_QuantizeLinear_Output")
        m0_cv1_id = self.tensor(f"{prefix}.m0_cv1_act", m0_cv1_act)
        self.add_conv(f"/{prefix}/m.0/cv1/conv/Conv", split1_id, m0_cv1_id, "silu")
        m0_cv2_node = self.index.nodes_by_name[f"/{prefix}/m.0/cv2/conv/Conv"]
        m0_cv2_channels = int(self.index.conv_assets(m0_cv2_node)[0].shape[0])
        m0_cv2_id = self.tensor(
            f"{prefix}.m0_cv2_inner",
            inner_spec,
            m0_cv2_channels,
            f"/{prefix}/m.0/cv2/act@inner_scale",
        )
        self.add_conv(f"/{prefix}/m.0/cv2/conv/Conv", split1_id, m0_cv2_id, "silu")

        r0_cv1_act = self.index.qspec(f"/{prefix}/m.0/m/m.0/cv1/act/Mul_output_0_QuantizeLinear_Output")
        r0_cv1_id = self.tensor(f"{prefix}.r0_cv1_act", r0_cv1_act)
        self.add_conv(f"/{prefix}/m.0/m/m.0/cv1/conv/Conv", m0_cv1_id, r0_cv1_id, "silu")
        r0_cv2_node = self.index.nodes_by_name[f"/{prefix}/m.0/m/m.0/cv2/conv/Conv"]
        r0_cv2_q = self.index.qspec(self.index.conv_output_quant_tensor(r0_cv2_node))
        r0_cv2_id = self.tensor(f"{prefix}.r0_cv2_preact", r0_cv2_q)
        self.add_conv(r0_cv2_node.name, r0_cv1_id, r0_cv2_id, "none")
        add0_id = self.tensor(f"{prefix}.add0", add0_spec)
        self.add_add_silu(f"/{prefix}/m.0/m/m.0/Add", m0_cv1_id, r0_cv2_id, add0_id)

        r1_cv1_act = self.index.qspec(f"/{prefix}/m.0/m/m.1/cv1/act/Mul_output_0_QuantizeLinear_Output")
        r1_cv1_id = self.tensor(f"{prefix}.r1_cv1_act", r1_cv1_act)
        self.add_conv(f"/{prefix}/m.0/m/m.1/cv1/conv/Conv", add0_id, r1_cv1_id, "silu")
        r1_cv2_node = self.index.nodes_by_name[f"/{prefix}/m.0/m/m.1/cv2/conv/Conv"]
        r1_cv2_q = self.index.qspec(self.index.conv_output_quant_tensor(r1_cv2_node))
        r1_cv2_id = self.tensor(f"{prefix}.r1_cv2_preact", r1_cv2_q)
        self.add_conv(r1_cv2_node.name, r1_cv1_id, r1_cv2_id, "none")
        add1_id = self.tensor(
            f"{prefix}.add1_inner",
            inner_spec,
            add0_spec.c,
            f"/{prefix}/m.0/m/m.1/Add@inner_scale",
        )
        self.add_add_silu(f"/{prefix}/m.0/m/m.1/Add", add0_id, r1_cv2_id, add1_id)

        inner_id = self.tensor(f"{prefix}.inner_concat", inner_spec)
        self.add_concat(f"/{prefix}/m.0/Concat", [add1_id, m0_cv2_id], inner_id)
        cv3_node = self.index.nodes_by_name[f"/{prefix}/m.0/cv3/conv/Conv"]
        cv3_channels = int(self.index.conv_assets(cv3_node)[0].shape[0])
        cv3_id = self.tensor(
            f"{prefix}.cv3_outer",
            outer_spec,
            cv3_channels,
            f"/{prefix}/m.0/cv3/act@outer_scale",
        )
        self.add_conv(f"/{prefix}/m.0/cv3/conv/Conv", inner_id, cv3_id, "silu")
        outer_id = self.tensor(f"{prefix}.outer_concat", outer_spec)
        self.add_concat(f"/{prefix}/Concat", [split0_id, split1_id, cv3_id], outer_id)
        output_id = self.tensor(f"{prefix}.output", output_spec)
        self.add_conv(f"/{prefix}/cv2/conv/Conv", outer_id, output_id, "silu")
        return output_id

    def finalize(self) -> None:
        for tensor in self.tensors:
            tensor["first_op"] = -1 if tensor["id"] == 0 else len(self.ops)
            tensor["last_op"] = -1
        for op in self.ops:
            for key in ("input0", "input1", "input2", "input3"):
                tensor_id = int(op[key])
                if tensor_id >= 0:
                    self.tensors[tensor_id]["last_op"] = max(self.tensors[tensor_id]["last_op"], op["index"])
            for key in ("output0", "output1"):
                tensor_id = int(op[key])
                if tensor_id >= 0:
                    self.tensors[tensor_id]["first_op"] = min(self.tensors[tensor_id]["first_op"], op["index"])
        self.tensors[-1]["last_op"] = len(self.ops)

        active: list[tuple[int, int, int]] = []
        free_blocks: list[tuple[int, int]] = []
        arena_end = 0
        for tensor in sorted(self.tensors, key=lambda row: (row["first_op"], row["id"])):
            still_active = []
            for last_op, offset, size in active:
                if last_op < tensor["first_op"]:
                    free_blocks.append((offset, size))
                else:
                    still_active.append((last_op, offset, size))
            active = still_active
            size = align_up(int(tensor["bytes"]))
            free_blocks.sort(key=lambda item: (item[1], item[0]))
            selected = next((item for item in free_blocks if item[1] >= size), None)
            if selected is None:
                offset = arena_end
                arena_end += size
            else:
                free_blocks.remove(selected)
                offset = selected[0]
                remainder = selected[1] - size
                if remainder:
                    free_blocks.append((offset + size, remainder))
            tensor["arena_offset"] = offset
            active.append((int(tensor["last_op"]), offset, size))

        write_tsv(self.package / "tensors.tsv", self.tensors)
        write_tsv(self.package / "ops.tsv", self.ops)
        package_meta = {
            "format": "y26-stage47-aot-tsv-v1",
            "model_sha256": sha256_file(Path(self.index.model.graph.doc_string)) if self.index.model.graph.doc_string else "recorded-by-caller",
            "tensor_count": len(self.tensors),
            "operation_count": len(self.ops),
            "arena_bytes": arena_end,
            "input_tensor_id": 0,
            "output_tensor_id": len(self.tensors) - 1,
            "physical_layout": "NHWC_signed_s8",
        }
        (self.package / "package.json").write_text(json.dumps(package_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_session(model_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.enable_mem_pattern = True
    options.enable_cpu_mem_arena = True
    return ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])


def generate_slice_oracles(args: argparse.Namespace, package: Path, builder: ScheduleBuilder) -> list[dict[str, Any]]:
    cuts = package / "cuts"
    cuts.mkdir(parents=True, exist_ok=True)
    island_cut = cuts / "model4_preact_to_model8_disable_all.onnx"
    onnx.utils.extract_model(str(args.model), str(island_cut), [MODEL4_PREACT], SLICE_OUTPUTS)
    session = make_session(island_cut)
    stage43_root = args.stage43_oracle_root.resolve()
    manifest_path = stage43_root / "model5_8_oracle_manifest.tsv"
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream, delimiter="\t"))
    rows: list[dict[str, Any]] = []
    boundary_to_tensor = {MODEL4_PREACT: 0}
    boundary_to_tensor.update(
        {name: builder.tensor_by_key[key] for name, key in SLICE_OUTPUT_TO_KEY.items()}
    )
    for fixture_index in range(8):
        fixture_id = f"F{fixture_index}"
        source = next(
            row
            for row in manifest
            if row["fixture_id"] == fixture_id and row["tensor_name"] == MODEL4_PREACT
        )
        input_array = np.fromfile(source["raw_path"], dtype=np.uint8).reshape((1, 128, 80, 80))
        outputs = session.run(SLICE_OUTPUTS, {MODEL4_PREACT: input_array})
        values = {MODEL4_PREACT: input_array, **dict(zip(SLICE_OUTPUTS, outputs, strict=True))}
        fixture_dir = package / "oracles" / fixture_id
        fixture_dir.mkdir(parents=True, exist_ok=True)
        for boundary, array in values.items():
            tensor_id = boundary_to_tensor[boundary]
            path = fixture_dir / f"tensor_{tensor_id:03d}_nchw_u8.bin"
            contiguous = np.ascontiguousarray(array, dtype=np.uint8)
            contiguous.tofile(path)
            rows.append(
                {
                    "fixture_id": fixture_id,
                    "tensor_id": tensor_id,
                    "tensor_name": boundary,
                    "path": str(path.relative_to(package)),
                    "sha256": sha256_file(path),
                    "dtype": "uint8",
                    "shape": "x".join(map(str, contiguous.shape)),
                    "element_count": contiguous.size,
                    "min": int(contiguous.min()),
                    "max": int(contiguous.max()),
                    "sum": int(contiguous.astype(np.uint64).sum()),
                    "oracle": "host_ort_1.27_cpu_disable_all",
                }
            )
    write_tsv(package / "oracle_manifest.tsv", rows)
    return rows


def build_slice_package(args: argparse.Namespace, index: GraphIndex, out_dir: Path) -> tuple[ScheduleBuilder, list[dict[str, Any]]]:
    package = out_dir / "model4_8_aot_package"
    package.mkdir(parents=True, exist_ok=True)
    builder = ScheduleBuilder(index, package)
    preact = index.qspec(MODEL4_PREACT)
    input_id = builder.tensor("model4.preact", preact)
    postact = index.qspec(MODEL4_POSTACT)
    postact_id = builder.tensor("model4.postact", postact)
    builder.add_lut("/model.4/cv2/final_silu", input_id, postact_id, "silu")
    model5_id = builder.simple_conv_block("model.5", postact_id, MODEL5_OUTPUT)
    model6_id = builder.c2f_cib_block("model.6", model5_id, MODEL6_OUTPUT)
    model7_id = builder.simple_conv_block("model.7", model6_id, MODEL7_OUTPUT)
    builder.c2f_cib_block("model.8", model7_id, MODEL8_OUTPUT)
    builder.finalize()
    package_json = json.loads((package / "package.json").read_text(encoding="utf-8"))
    package_json["model_path"] = str(args.model.resolve())
    package_json["model_sha256"] = sha256_file(args.model)
    package_json["ort_version"] = ort.__version__
    package_json["oracle_session"] = "CPUExecutionProvider;ORT_DISABLE_ALL;sequential;intra=1;inter=1"
    (package / "package.json").write_text(json.dumps(package_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    oracle_rows = generate_slice_oracles(args, package, builder)
    return builder, oracle_rows


KERNEL_CASE_NODES = [
    "/model.0/conv/Conv",
    "/model.3/conv/Conv",
    "/model.4/cv2/conv/Conv",
    "/model.4/m.0/cv1/conv/Conv",
    "/model.5/conv/Conv",
    "/model.7/conv/Conv",
    "/model.8/cv2/conv/Conv",
    "/model.10/m/m.0/attn/proj/conv/Conv",
    "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv",
]


def generate_kernel_cases(args: argparse.Namespace, index: GraphIndex, out_dir: Path) -> list[dict[str, Any]]:
    root = out_dir / "integrated_kernel_cases"
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case_id, node_name in enumerate(KERNEL_CASE_NODES):
        node = index.nodes_by_name[node_name]
        attrs = node_attributes(node)
        input_q_name = index.input_quant_tensor(node)
        input_spec = index.qspec(input_q_name)
        conv_q_name = index.conv_output_quant_tensor(node)
        output_q_name = index.postact_quant_tensor(node) or conv_q_name
        output_spec = index.qspec(output_q_name)
        activation = "silu" if output_q_name != conv_q_name else "none"
        weights, scales, bias, source_weight = index.conv_assets(node)
        case_dir = root / f"case_{case_id:02d}_{safe_name(node_name)}"
        case_dir.mkdir(parents=True, exist_ok=True)
        weight_path = case_dir / "weights_ohwi_s8.bin"
        scale_path = case_dir / "weight_scales_f32.bin"
        bias_path = case_dir / "bias_i32.bin"
        weights.tofile(weight_path)
        scales.tofile(scale_path)
        bias.tofile(bias_path)
        values = np.arange(shape_count(input_spec.shape), dtype=np.uint64).reshape(input_spec.shape)
        input_array = np.asarray(
            np.clip(input_spec.zero_point + ((values * 17 + case_id * 13) % 31).astype(np.int16) - 15, 0, 255),
            dtype=np.uint8,
        )
        input_path = case_dir / "input_nchw_u8.bin"
        input_array.tofile(input_path)
        cut_path = case_dir / "cut.onnx"
        onnx.utils.extract_model(str(args.model), str(cut_path), [input_q_name], [output_q_name])
        expected = np.asarray(make_session(cut_path).run([output_q_name], {input_q_name: input_array})[0], dtype=np.uint8)
        expected_path = case_dir / "expected_nchw_u8.bin"
        expected.tofile(expected_path)
        pads = attrs.get("pads", [0, 0, 0, 0])
        strides = attrs.get("strides", [1, 1])
        rows.append(
            {
                "case_id": case_id,
                "node_name": node_name,
                "input_tensor": input_q_name,
                "output_tensor": output_q_name,
                "input_h": input_spec.h,
                "input_w": input_spec.w,
                "input_c": input_spec.c,
                "output_h": output_spec.h,
                "output_w": output_spec.w,
                "output_c": output_spec.c,
                "kernel_h": int(weights.shape[1]),
                "kernel_w": int(weights.shape[2]),
                "stride_h": int(strides[0]),
                "stride_w": int(strides[1]),
                "pad_h": int(pads[0]),
                "pad_w": int(pads[1]),
                "group": int(attrs.get("group", 1)),
                "input_scale": f"{input_spec.scale:.17g}",
                "input_zero_point": input_spec.zero_point,
                "conv_output_scale": f"{index.qspec(conv_q_name).scale:.17g}",
                "conv_output_zero_point": index.qspec(conv_q_name).zero_point,
                "output_scale": f"{output_spec.scale:.17g}",
                "output_zero_point": output_spec.zero_point,
                "activation": activation,
                "weights_file": str(weight_path.relative_to(out_dir)),
                "weight_scales_file": str(scale_path.relative_to(out_dir)),
                "bias_file": str(bias_path.relative_to(out_dir)),
                "input_file": str(input_path.relative_to(out_dir)),
                "expected_file": str(expected_path.relative_to(out_dir)),
                "cut_file": str(cut_path.relative_to(out_dir)),
                "source_weight": source_weight,
                "weights_sha256": sha256_file(weight_path),
                "input_sha256": sha256_file(input_path),
                "expected_sha256": sha256_file(expected_path),
                "cut_sha256": sha256_file(cut_path),
                "macs": output_spec.h * output_spec.w * output_spec.c * weights.shape[1] * weights.shape[2] * input_spec.c // int(attrs.get("group", 1)),
            }
        )
    write_tsv(root / "cases.tsv", rows)
    return rows


def generate(args: argparse.Namespace) -> None:
    args.model = args.model.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(args.model)
    model.graph.doc_string = ""
    index = GraphIndex(model)
    census_rows, census_summary = graph_shape_census(index, out_dir)
    builder, oracle_rows = build_slice_package(args, index, out_dir)
    kernel_cases = generate_kernel_cases(args, index, out_dir)
    summary = {
        "model_path": str(args.model),
        "model_sha256": sha256_file(args.model),
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "numpy_version": np.__version__,
        "graph": census_summary,
        "aot_tensor_count": len(builder.tensors),
        "aot_operation_count": len(builder.ops),
        "oracle_rows": len(oracle_rows),
        "kernel_cases": len(kernel_cases),
        "output_dir": str(out_dir),
    }
    (out_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_tsv(
        out_dir / "asset_hash_manifest.tsv",
        (
            {"path": str(path.relative_to(out_dir)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(out_dir.rglob("*"))
            if path.is_file()
        ),
        ["path", "bytes", "sha256"],
    )
    print(json.dumps(summary, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--stage43-oracle-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
