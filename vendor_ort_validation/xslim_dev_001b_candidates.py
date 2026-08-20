#!/usr/bin/env python3
"""Prepare and build the bounded DEV-001B all-S8 candidate matrix."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import random
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import helper, numpy_helper
from stage65b_r1_evaluate import image_tensor_and_geometry, paths_from_list
from xslim.range_policy import ConstrainedRangeSpec, validate_qparams_contract
from xslim.reconstruction import (
    AdaptiveWeightRounder,
    ReconstructionConfig,
    apply_bias_correction,
    compute_bias_correction,
    reconstruct_block,
    stratified_activation_sample,
)

OUTPUTS = (
    "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0",
    "/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0",
    "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0",
    "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0",
    "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0",
    "/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0",
)
TERMINAL_CONFIDENCE = (OUTPUTS[1], OUTPUTS[3], OUTPUTS[5])
R0_ANCHORS = (
    "/model.0/act/Mul_output_0",
    "/model.1/act/Mul_output_0",
    "/model.2/cv1/act/Mul_output_0",
    "/model.2/cv2/act/Mul_output_0",
)
R7_ANCHORS = (
    "/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.1/act/Mul_output_0",
    "/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1/act/Mul_output_0",
    "/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/act/Mul_output_0",
)
GROUPS = {"R0": R0_ANCHORS, "R7": R7_ANCHORS}
QUANT_MIN = -128
QUANT_MAX = 127
TAIL_DEBUG_OUTPUTS = (
    "/model.23/TopK_output_1",
    "/model.23/TopK_1_output_1",
    "/model.23/TopK_1_output_0",
    "/model.23/Sigmoid_output_0",
    "output0",
)
THRESHOLDS = (0.001, 0.01, 0.05, 0.25, 0.5)
PATCH_SELECTION_POLICY = "xslim-dev-001b-conv-patch-splitmix64-v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_u64(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def splitmix64(values: np.ndarray) -> np.ndarray:
    """Return a deterministic bijective uint64 ranking for integer positions."""

    result = np.asarray(values, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    result = (result ^ (result >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    result = (result ^ (result >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return result ^ (result >> np.uint64(31))


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status\nno-rows\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_npz_deterministic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write a compressed NumPy archive without wall-clock ZIP metadata."""

    with zipfile.ZipFile(path, "w") as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100600 << 16
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def producer_map(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    return {output: node for node in model.graph.node for output in node.output if output}


def initializer_map(model: onnx.ModelProto) -> dict[str, onnx.TensorProto]:
    return {item.name: item for item in model.graph.initializer}


def attributes(node: onnx.NodeProto) -> dict[str, Any]:
    return {item.name: helper.get_attribute_value(item) for item in node.attribute}


def scalar(model: onnx.ModelProto, name: str) -> tuple[float, np.dtype[Any]]:
    item = initializer_map(model).get(name)
    if item is None:
        raise ValueError(f"missing scalar initializer: {name}")
    value = numpy_helper.to_array(item)
    if value.size != 1:
        raise ValueError(f"initializer is not scalar: {name} {value.shape}")
    return float(value.reshape(())), value.dtype


def trace_terminal_qdq(model: onnx.ModelProto, output: str) -> dict[str, Any]:
    producers = producer_map(model)
    dq = producers.get(output)
    wrappers: list[str] = []
    while dq is not None and dq.op_type == "Identity":
        wrappers.append(dq.name)
        dq = producers.get(dq.input[0])
    if dq is None or dq.op_type != "DequantizeLinear":
        raise ValueError(f"terminal output is not DQ-produced: {output}")
    q = producers.get(dq.input[0])
    if q is None or q.op_type != "QuantizeLinear":
        raise ValueError(f"terminal output lacks paired Q: {output}")
    q_scale, q_scale_dtype = scalar(model, q.input[1])
    dq_scale, dq_scale_dtype = scalar(model, dq.input[1])
    q_zp, q_zp_dtype = scalar(model, q.input[2])
    dq_zp, dq_zp_dtype = scalar(model, dq.input[2])
    if q_scale != dq_scale or int(q_zp) != int(dq_zp):
        raise ValueError(f"terminal Q/DQ values differ: {output}")
    if q_scale_dtype != np.float32 or dq_scale_dtype != np.float32:
        raise ValueError(f"terminal scale is not float32: {output}")
    if q_zp_dtype != np.int8 or dq_zp_dtype != np.int8:
        raise ValueError(f"terminal zero point is not signed INT8: {output}")
    return {
        "output": output,
        "q_node": q.name,
        "dq_node": dq.name,
        "wrappers": wrappers,
        "float_source": q.input[0],
        "q_scale": q.input[1],
        "q_zero_point": q.input[2],
        "dq_scale": dq.input[1],
        "dq_zero_point": dq.input[2],
        "scale": q_scale,
        "zero_point": int(q_zp),
    }


def _follow_to_op(
    producers: Mapping[str, onnx.NodeProto], tensor: str, wanted: str
) -> onnx.NodeProto:
    current = tensor
    seen: set[str] = set()
    while current not in seen:
        seen.add(current)
        node = producers.get(current)
        if node is None:
            break
        if node.op_type == wanted:
            return node
        if node.op_type not in {"DequantizeLinear", "QuantizeLinear", "Identity"}:
            break
        current = node.input[0]
    raise ValueError(f"cannot trace {tensor!r} to {wanted}")


def target_descriptor(
    fp32: onnx.ModelProto, b2: onnx.ModelProto, group: str, anchor: str
) -> dict[str, Any]:
    b2_producers = producer_map(b2)
    fp32_nodes = {node.name: node for node in fp32.graph.node}
    anchor_dq = b2_producers.get(anchor)
    if anchor_dq is None or anchor_dq.op_type != "DequantizeLinear":
        raise ValueError(f"anchor is not a B2 DQ output: {anchor}")
    anchor_q = b2_producers.get(anchor_dq.input[0])
    if anchor_q is None or anchor_q.op_type != "QuantizeLinear":
        raise ValueError(f"anchor does not have Q/DQ topology: {anchor}")
    activation = b2_producers.get(anchor_q.input[0])
    if activation is None or activation.op_type != "Mul":
        raise ValueError(f"anchor does not trace to a SiLU Mul: {anchor}")
    conv: onnx.NodeProto | None = None
    for item in activation.input:
        try:
            conv = _follow_to_op(b2_producers, item, "Conv")
            break
        except ValueError:
            continue
    if conv is None:
        raise ValueError(f"anchor does not trace to Conv: {anchor}")
    fp_conv = fp32_nodes.get(conv.name)
    if fp_conv is None or fp_conv.op_type != "Conv":
        raise ValueError(f"FP32 source lacks matching Conv: {conv.name}")
    weight_dq = b2_producers.get(conv.input[1])
    if weight_dq is None or weight_dq.op_type != "DequantizeLinear":
        raise ValueError(f"B2 Conv weight is not DQ-produced: {conv.name}")
    b2_initializers = initializer_map(b2)
    fp32_initializers = initializer_map(fp32)
    weight_name = weight_dq.input[0]
    scale_name = weight_dq.input[1]
    zero_name = weight_dq.input[2]
    if weight_name not in b2_initializers or scale_name not in b2_initializers or zero_name not in b2_initializers:
        raise ValueError(f"B2 Conv weight qparams are dynamic: {conv.name}")
    if fp_conv.input[1] not in fp32_initializers:
        raise ValueError(f"FP32 Conv weight is not static: {conv.name}")
    b2_weight = numpy_helper.to_array(b2_initializers[weight_name])
    weight_scale = numpy_helper.to_array(b2_initializers[scale_name])
    weight_zero = numpy_helper.to_array(b2_initializers[zero_name])
    fp_weight = numpy_helper.to_array(fp32_initializers[fp_conv.input[1]])
    if b2_weight.dtype != np.int8 or weight_zero.dtype != np.int8:
        raise ValueError(f"B2 Conv weight is not signed INT8: {conv.name}")
    if not np.issubdtype(weight_scale.dtype, np.floating) or not np.all(
        np.isfinite(weight_scale) & (weight_scale > 0)
    ):
        raise ValueError(f"B2 Conv weight scale is invalid: {conv.name}")
    if not np.all(weight_zero == 0):
        raise ValueError(f"B2 Conv weight is not symmetric: {conv.name}")
    if fp_weight.shape != b2_weight.shape:
        raise ValueError(f"FP32/B2 Conv weight shape differs: {conv.name}")
    conv_attrs = attributes(conv)
    kernel = tuple(int(item) for item in conv_attrs.get("kernel_shape", b2_weight.shape[-2:]))
    pads = tuple(int(item) for item in conv_attrs.get("pads", (0, 0, 0, 0)))
    strides = tuple(int(item) for item in conv_attrs.get("strides", (1, 1)))
    dilations = tuple(int(item) for item in conv_attrs.get("dilations", (1, 1)))
    groups = int(conv_attrs.get("group", 1))
    bias_name = conv.input[2] if len(conv.input) > 2 else ""
    if not bias_name or bias_name not in b2_initializers:
        raise ValueError(f"target Conv lacks a static existing bias: {conv.name}")
    return {
        "group": group,
        "anchor": anchor,
        "anchor_scale": anchor_q.input[1],
        "anchor_zero_point": anchor_q.input[2],
        "conv_node": conv.name,
        "conv_input": conv.input[0],
        "conv_output": fp_conv.output[0],
        "weight_initializer": weight_name,
        "weight_scale_initializer": scale_name,
        "weight_zero_point_initializer": zero_name,
        "fp32_weight_initializer": fp_conv.input[1],
        "bias_initializer": bias_name,
        "kernel_shape": list(kernel),
        "pads": list(pads),
        "strides": list(strides),
        "dilations": list(dilations),
        "groups": groups,
        "weight_shape": list(b2_weight.shape),
    }


def add_outputs(model: onnx.ModelProto, names: Iterable[str]) -> onnx.ModelProto:
    inferred = onnx.shape_inference.infer_shapes(copy.deepcopy(model))
    values = {
        item.name: item
        for item in list(inferred.graph.input)
        + list(inferred.graph.output)
        + list(inferred.graph.value_info)
    }
    existing = {item.name for item in inferred.graph.output}
    for name in names:
        if name in existing:
            continue
        if name not in values:
            raise ValueError(f"missing static value information for diagnostic output: {name}")
        inferred.graph.output.append(copy.deepcopy(values[name]))
        existing.add(name)
    return inferred


def make_session(path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def run_named(session: ort.InferenceSession, image: np.ndarray) -> dict[str, np.ndarray]:
    return {
        meta.name: value
        for meta, value in zip(
            session.get_outputs(),
            session.run(None, {session.get_inputs()[0].name: image}),
        )
    }


def selected_positions(
    image_name: str, target: str, height: int, width: int, count: int
) -> list[tuple[int, int]]:
    if height <= 0 or width <= 0:
        raise ValueError("patch-selection surface must have positive dimensions")
    if count <= 0:
        return []
    size = height * width
    take = min(count, size)
    indices = np.arange(size, dtype=np.uint64)
    seed = np.uint64(
        stable_u64(PATCH_SELECTION_POLICY, image_name, target, height, width)
    )
    ranks = splitmix64(indices ^ seed)
    if take == size:
        selected = indices
    else:
        selected = indices[np.argpartition(ranks, take - 1)[:take]]
    order = np.lexsort((selected, ranks[selected.astype(np.int64)]))
    selected = selected[order]
    return [(int(item) // width, int(item) % width) for item in selected]


def extract_patches(
    value: np.ndarray,
    positions: Sequence[tuple[int, int]],
    descriptor: Mapping[str, Any],
) -> np.ndarray:
    if value.ndim != 4 or value.shape[0] != 1:
        raise ValueError(f"target Conv input must be static NCHW: {value.shape}")
    kernel_h, kernel_w = map(int, descriptor["kernel_shape"])
    stride_h, stride_w = map(int, descriptor["strides"])
    dilation_h, dilation_w = map(int, descriptor["dilations"])
    pad_top, pad_left, _, _ = map(int, descriptor["pads"])
    _, channels, height, width = value.shape
    patches = np.zeros((len(positions), channels, kernel_h, kernel_w), dtype=np.float32)
    source = value[0]
    for sample, (output_row, output_column) in enumerate(positions):
        for kernel_row in range(kernel_h):
            input_row = output_row * stride_h - pad_top + kernel_row * dilation_h
            if input_row < 0 or input_row >= height:
                continue
            for kernel_column in range(kernel_w):
                input_column = output_column * stride_w - pad_left + kernel_column * dilation_w
                if input_column < 0 or input_column >= width:
                    continue
                patches[sample, :, kernel_row, kernel_column] = source[:, input_row, input_column]
    return patches


def qdq(values: np.ndarray, scale: float, zero_point: int) -> tuple[np.ndarray, np.ndarray]:
    raw = np.rint(values.astype(np.float64) / float(scale) + int(zero_point))
    codes = np.clip(raw, QUANT_MIN, QUANT_MAX).astype(np.int8)
    rebuilt = (codes.astype(np.float64) - int(zero_point)) * float(scale)
    return rebuilt.astype(np.float32), codes


def inversion_count(values: np.ndarray) -> int:
    array = list(map(int, values.tolist()))

    def merge_count(items: list[int]) -> tuple[list[int], int]:
        if len(items) <= 1:
            return items, 0
        middle = len(items) // 2
        left, left_count = merge_count(items[:middle])
        right, right_count = merge_count(items[middle:])
        merged: list[int] = []
        count = left_count + right_count
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                count += len(left) - i
                j += 1
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged, count

    return merge_count(array)[1]


def rank_inversions(teacher_scores: np.ndarray, candidate_scores: np.ndarray, k: int = 600) -> int:
    teacher = teacher_scores.reshape(-1).astype(np.float64)
    candidate = candidate_scores.reshape(-1).astype(np.float64)
    indices = np.lexsort((np.arange(teacher.size), -teacher))[: min(k, teacher.size)]
    candidate_order = np.lexsort((indices, -candidate[indices]))
    ranks = np.empty(candidate_order.size, dtype=np.int64)
    ranks[candidate_order] = np.arange(candidate_order.size, dtype=np.int64)
    return inversion_count(ranks)


def topk_overlap(teacher: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray]) -> int:
    teacher_positions = teacher[TAIL_DEBUG_OUTPUTS[0]].reshape(-1)
    teacher_classes = teacher[TAIL_DEBUG_OUTPUTS[1]].reshape(-1)
    candidate_positions = candidate[TAIL_DEBUG_OUTPUTS[0]].reshape(-1)
    candidate_classes = candidate[TAIL_DEBUG_OUTPUTS[1]].reshape(-1)
    if teacher_positions.shape != teacher_classes.shape:
        raise ValueError("teacher TopK position/class shapes differ")
    if candidate_positions.shape != candidate_classes.shape:
        raise ValueError("candidate TopK position/class shapes differ")
    teacher_pairs = set(zip(map(int, teacher_positions), map(int, teacher_classes)))
    candidate_pairs = set(zip(map(int, candidate_positions), map(int, candidate_classes)))
    return len(teacher_pairs.intersection(candidate_pairs))


def threshold_crossings(teacher: np.ndarray, candidate: np.ndarray, k: int = 600) -> int:
    teacher_flat = teacher.reshape(-1).astype(np.float64)
    candidate_flat = candidate.reshape(-1).astype(np.float64)
    teacher_top = np.lexsort((np.arange(teacher_flat.size), -teacher_flat))[: min(k, teacher_flat.size)]
    candidate_top = np.lexsort((np.arange(candidate_flat.size), -candidate_flat))[: min(k, candidate_flat.size)]
    indices = np.unique(np.concatenate((teacher_top, candidate_top)))
    return sum(
        int(np.count_nonzero((teacher_flat[indices] >= threshold) != (candidate_flat[indices] >= threshold)))
        for threshold in THRESHOLDS
    )


def candidate_scales(values: np.ndarray, positive_codes: int) -> list[float]:
    zero_point = QUANT_MAX - positive_codes
    negative_codes = zero_point - QUANT_MIN
    flat = values.reshape(-1).astype(np.float64)
    positive = flat[flat > 0]
    negative = -flat[flat < 0]
    quantiles = (0.99, 0.995, 0.999, 0.9995, 0.9999, 1.0)
    positive_thresholds = [float(np.quantile(positive, item)) for item in quantiles] if positive.size else [0.0]
    negative_thresholds = [float(np.quantile(negative, item)) for item in quantiles] if negative.size else [0.0]
    scales: set[float] = set()
    for positive_threshold in positive_thresholds:
        for negative_threshold in negative_thresholds:
            base = max(
                positive_threshold / max(positive_codes, 1),
                negative_threshold / max(negative_codes, 1),
                1.0e-12,
            )
            for multiplier in (0.9, 1.0, 1.1, 1.25):
                scales.add(float(np.float32(base * multiplier)))
    return sorted(scale for scale in scales if math.isfinite(scale) and scale > 0)


def tail_inputs(boundaries: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    if len(boundaries) != len(OUTPUTS):
        raise ValueError("tail requires the exact six-boundary contract")
    return dict(zip(OUTPUTS, boundaries))


def run_tail(session: ort.InferenceSession, boundaries: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    return {
        meta.name: value
        for meta, value in zip(session.get_outputs(), session.run(None, tail_inputs(boundaries)))
    }


def qparam_metrics(values: np.ndarray, scale: float, zero_point: int) -> dict[str, float]:
    rebuilt, codes = qdq(values, scale, zero_point)
    original = values.astype(np.float64)
    reconstructed = rebuilt.astype(np.float64)
    error = reconstructed - original
    low = (QUANT_MIN - zero_point) * scale
    high = (QUANT_MAX - zero_point) * scale
    denominator = float(np.linalg.norm(original) * np.linalg.norm(reconstructed))
    return {
        "mse": float(np.mean(error * error)),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "cosine": float(np.dot(original, reconstructed) / denominator) if denominator else 1.0,
        "clipping_fraction": float(np.mean((original < low) | (original > high))),
        "rail_fraction": float(np.mean((codes == QUANT_MIN) | (codes == QUANT_MAX))),
        "representable_min": float(low),
        "representable_max": float(high),
    }


def replace_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    for index, item in enumerate(model.graph.initializer):
        if item.name != name:
            continue
        model.graph.initializer[index].CopyFrom(numpy_helper.from_array(np.asarray(value), name=name))
        return
    raise ValueError(f"initializer not found: {name}")


def conv_outputs(
    patches: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    samples, channels, kernel_h, kernel_w = patches.shape
    output_channels = weight.shape[0]
    if channels % groups or output_channels % groups:
        raise ValueError("Conv group contract is not divisible")
    input_per_group = channels // groups
    output_per_group = output_channels // groups
    patch_matrix = patches.reshape(samples, groups, input_per_group * kernel_h * kernel_w)
    weight_matrix = weight.reshape(groups, output_per_group, input_per_group * kernel_h * kernel_w)
    output = torch.einsum("sgi,goi->sgo", patch_matrix, weight_matrix).reshape(samples, output_channels)
    return output + bias.reshape(1, -1)


def normalized_mse(student: Sequence[np.ndarray], teacher: Sequence[np.ndarray]) -> float:
    losses = []
    for observed, expected in zip(student, teacher):
        denominator = max(float(np.mean(np.square(expected.astype(np.float64)))), 1.0e-12)
        losses.append(float(np.mean(np.square(observed.astype(np.float64) - expected.astype(np.float64)))) / denominator)
    return float(np.mean(losses))


def require_baseline_rounding_identity(
    initial_codes: np.ndarray,
    accepted_codes: np.ndarray,
    target_name: str,
) -> int:
    """Require adaptive-rounding rollback to reconstruct the accepted baseline."""

    if initial_codes.shape != accepted_codes.shape or initial_codes.dtype != accepted_codes.dtype:
        raise RuntimeError(f"baseline rounding shape/dtype differs for {target_name}")
    differences = int(np.count_nonzero(initial_codes != accepted_codes))
    if differences:
        raise RuntimeError(
            "post-equalization FP reference does not reconstruct the accepted B2 "
            f"weight codes for {target_name}: {differences} differences"
        )
    return differences


def require_python_hash_seed(seed: int) -> str:
    """Fail before mutation when hash-backed traversal is not reproducible."""

    expected = str(seed)
    actual = os.environ.get("PYTHONHASHSEED")
    if actual != expected:
        raise RuntimeError(
            "PYTHONHASHSEED must equal the configured seed before process start: "
            f"expected {expected}, got {actual!r}"
        )
    return actual


def canonicalize_transient_export_metadata(model: onnx.ModelProto) -> list[str]:
    """Remove wall-clock metadata from a stage-local diagnostic export."""

    transient_keys = {"xslim_export_time"}
    removed = sorted(item.key for item in model.metadata_props if item.key in transient_keys)
    retained = [copy.deepcopy(item) for item in model.metadata_props if item.key not in transient_keys]
    del model.metadata_props[:]
    model.metadata_props.extend(retained)
    return removed


def export_prequant_reference(args: argparse.Namespace) -> int:
    """Export the deterministic FP graph after XSlim's prequant passes only."""

    require_python_hash_seed(args.seed)
    if args.output.exists() or args.manifest.exists():
        raise RuntimeError("refusing to overwrite prequant reference state")

    import cv2
    from xslim import CalibrationCollect, XSlimDataset
    from xslim.optimizer import GraphLegalized
    from xslim.ppq_decorator import ONNXRUNTIMExporter, TorchExecutor
    from xslim.quantizer import XSlimQuantizer
    from xslim.xslim_pipeline import (
        dispatch_graph,
        parse_xslim_config,
        xslim_load_onnx_graph,
    )
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    cv2.setRNGSeed(args.seed % (2**31))
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(args.threads)

    setting = parse_xslim_config(str(args.base_config))
    model_path = setting.model_parameters.onnx_model
    graph, _tail, _truncate_vars, saved_functions = xslim_load_onnx_graph(
        model_path,
        not setting.model_parameters.skip_onnxsim,
        setting.quantization_parameters.truncate_var_names,
        setting.model_parameters.opset,
    )
    GraphLegalized(graph)()
    graph = dispatch_graph(graph=graph, dispatcher="conservative")
    setting.calibration_parameters.check_input_parameters(graph)
    input_parameters = setting.calibration_parameters.input_parameters
    dataset = XSlimDataset(setting.calibration_parameters)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=dataset.auto_batch_size,
    )
    quantizer = XSlimQuantizer(graph)
    executor = TorchExecutor(
        graph=quantizer._graph,
        device=setting.calibration_parameters.calibration_device,
    )
    collate = CalibrationCollect(
        input_parameters,
        setting.calibration_parameters.calibration_device,
    )
    input_names = list(graph.inputs)
    if len(input_names) != 1:
        raise RuntimeError("prequant reference exporter requires one graph input")
    dummy_input: dict[str, torch.Tensor] | None = None
    for raw_data in dataloader:
        data = collate(raw_data)
        if isinstance(data, torch.Tensor):
            dummy_input = {input_names[0]: data}
        elif isinstance(data, dict):
            dummy_input = data
        else:
            raise TypeError(f"unsupported calibration input type: {type(data)!r}")
    if dummy_input is None:
        raise RuntimeError("prequant reference calibration dataset is empty")

    executor.load_graph(quantizer._graph)
    executor.tracing_operation_meta(inputs=dummy_input)
    pipeline = quantizer.build_prequant_pipeline(setting, executor=executor)
    pipeline.optimize(
        graph=quantizer._graph,
        dataloader=dataloader,
        executor=executor,
        verbose=False,
        calib_steps=setting.calibration_parameters.calibration_step,
        collate_fn=collate,
    )
    model = ONNXRUNTIMExporter().export(quantizer._graph)
    for function_proto in saved_functions:
        if not any(
            item.domain == function_proto.domain and item.name == function_proto.name
            for item in model.functions
        ):
            model.functions.append(function_proto)
    removed_metadata = canonicalize_transient_export_metadata(model)
    onnx.checker.check_model(model)
    onnx.shape_inference.infer_shapes(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, args.output)
    json_dump(
        args.manifest,
        {
            "contract": "xslim-dev-001b-post-equalization-fp-reference-v1",
            "base_config": {
                "path": str(args.base_config),
                "sha256": sha256(args.base_config),
            },
            "output": {
                "path": str(args.output),
                "bytes": args.output.stat().st_size,
                "sha256": sha256(args.output),
            },
            "seed": args.seed,
            "threads": args.threads,
            "removed_transient_metadata": removed_metadata,
            "calibration_images": len(dataset),
            "truncate_outputs": list(setting.quantization_parameters.truncate_var_names),
        },
    )
    return 0


def prepare(args: argparse.Namespace) -> int:
    require_python_hash_seed(args.seed)
    if args.raw_root.exists() or args.report_root.exists():
        raise RuntimeError("refusing to overwrite candidate preparation state")
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(args.threads)
    args.raw_root.mkdir(parents=True)
    args.report_root.mkdir(parents=True)
    diagnostic_root = args.raw_root / "diagnostics"
    diagnostic_root.mkdir()

    fp32 = onnx.load(args.fp32_inference)
    reconstruction_reference = onnx.load(args.prequant_reference)
    b2 = onnx.load(args.b2_inference)
    descriptors = [
        target_descriptor(reconstruction_reference, b2, group, anchor)
        for group, anchors in GROUPS.items()
        for anchor in anchors
    ]
    terminal = [trace_terminal_qdq(b2, output) for output in TERMINAL_CONFIDENCE]
    fp_outputs = list(OUTPUTS)
    reference_outputs = [item["conv_output"] for item in descriptors]
    b2_outputs = (
        list(OUTPUTS)
        + [item["float_source"] for item in terminal]
        + [item["conv_input"] for item in descriptors]
    )
    fp_diagnostic = add_outputs(fp32, fp_outputs)
    reference_diagnostic = add_outputs(reconstruction_reference, reference_outputs)
    b2_diagnostic = add_outputs(b2, b2_outputs)
    fp_path = diagnostic_root / "fp32-diagnostic.onnx"
    reference_path = diagnostic_root / "prequant-reference-diagnostic.onnx"
    b2_path = diagnostic_root / "b2-diagnostic.onnx"
    onnx.save_model(fp_diagnostic, fp_path)
    onnx.save_model(reference_diagnostic, reference_path)
    onnx.save_model(b2_diagnostic, b2_path)
    onnx.checker.check_model(fp_diagnostic)
    onnx.checker.check_model(reference_diagnostic)
    onnx.checker.check_model(b2_diagnostic)

    tail = onnx.load(args.tail)
    tail_debug = add_outputs(tail, TAIL_DEBUG_OUTPUTS)
    tail_path = diagnostic_root / "tail-debug.onnx"
    onnx.save_model(tail_debug, tail_path)
    tail_session = make_session(tail_path, 1)
    fp_session = make_session(fp_path, args.threads)
    reference_session = make_session(reference_path, args.threads)
    b2_session = make_session(b2_path, args.threads)

    b2_initializers = initializer_map(b2)
    fp_initializers = initializer_map(reconstruction_reference)
    reconstruction_data: dict[str, dict[str, list[dict[str, np.ndarray]]]] = {
        item["anchor"]: {"optimization": [], "validation": []} for item in descriptors
    }
    terminal_samples: dict[str, list[np.ndarray]] = {item["output"]: [] for item in terminal}
    validation_cases: list[Path] = []
    validation_root = args.raw_root / "terminal-validation"
    validation_root.mkdir()

    def collect(list_path: Path, split: str) -> None:
        paths = paths_from_list(list_path, 0)
        for image_index, image_path in enumerate(paths):
            image, _ = image_tensor_and_geometry(image_path)
            fp_values = run_named(fp_session, image)
            reference_values = run_named(reference_session, image)
            b2_values = run_named(b2_session, image)
            for descriptor in descriptors:
                teacher = reference_values[descriptor["conv_output"]]
                height, width = map(int, teacher.shape[-2:])
                positions = selected_positions(
                    image_path.name,
                    descriptor["anchor"],
                    height,
                    width,
                    args.patches_per_image,
                )
                patches = extract_patches(b2_values[descriptor["conv_input"]], positions, descriptor)
                rows = np.stack([teacher[0, :, row, column] for row, column in positions]).astype(np.float32)
                reconstruction_data[descriptor["anchor"]][split].append(
                    {"patches": patches, "teacher": rows}
                )
            if split == "optimization":
                for item in terminal:
                    values = b2_values[item["float_source"]]
                    sampled, _, _ = stratified_activation_sample(
                        values,
                        args.terminal_samples_per_image,
                        seed=args.seed,
                        tensor_name=f"{item['output']}:{image_path.name}",
                    )
                    terminal_samples[item["output"]].append(sampled.astype(np.float32))
            else:
                destination = validation_root / f"{image_index:03d}-{image_path.stem}.npz"
                payload: dict[str, np.ndarray] = {"image_name": np.asarray(image_path.name)}
                for output_index, output in enumerate(OUTPUTS):
                    payload[f"fp_boundary_{output_index}"] = fp_values[output]
                    payload[f"b2_boundary_{output_index}"] = b2_values[output]
                for output_index, item in enumerate(terminal):
                    payload[f"b2_terminal_float_{output_index}"] = b2_values[item["float_source"]]
                np.savez_compressed(destination, **payload)
                validation_cases.append(destination)
            if args.log_every and (image_index + 1) % args.log_every == 0:
                print(f"{split}: {image_index + 1}/{len(paths)}", flush=True)

    collect(args.optimization_list, "optimization")
    collect(args.validation_list, "validation")

    terminal_rows: list[dict[str, Any]] = []
    selected_qparams: dict[str, dict[str, Any]] = {}
    for terminal_index, item in enumerate(terminal):
        values = np.concatenate(terminal_samples[item["output"]])
        teacher_cache: list[dict[str, np.ndarray]] = []
        case_cache: list[dict[str, Any]] = []
        for case_path in validation_cases:
            with np.load(case_path, allow_pickle=False) as case:
                fp_boundaries = [case[f"fp_boundary_{index}"] for index in range(6)]
                b2_boundaries = [case[f"b2_boundary_{index}"] for index in range(6)]
                source = case[f"b2_terminal_float_{terminal_index}"]
                teacher_tail = run_tail(tail_session, fp_boundaries)
                teacher_cache.append(teacher_tail)
                case_cache.append({"boundaries": b2_boundaries, "source": source})
        candidates: list[dict[str, Any]] = []
        for budget in (4, 8, 16):
            zero_point = QUANT_MAX - budget
            for scale in candidate_scales(values, budget):
                spec = ConstrainedRangeSpec(
                    objective="constrained-mse",
                    preserve_zero=True,
                    minimum_positive_codes=budget,
                    minimum_negative_codes=QUANT_MAX - budget - QUANT_MIN,
                )
                validate_qparams_contract(scale, zero_point, spec)
                metrics = qparam_metrics(values, scale, zero_point)
                overlap = inversions = crossings = 0
                for teacher_tail, case in zip(teacher_cache, case_cache):
                    boundaries = list(case["boundaries"])
                    boundaries[terminal_index * 2 + 1] = qdq(case["source"], scale, zero_point)[0]
                    candidate_tail = run_tail(tail_session, boundaries)
                    overlap += topk_overlap(teacher_tail, candidate_tail)
                    inversions += rank_inversions(
                        teacher_tail["/model.23/Sigmoid_output_0"],
                        candidate_tail["/model.23/Sigmoid_output_0"],
                    )
                    crossings += threshold_crossings(
                        teacher_tail["/model.23/Sigmoid_output_0"],
                        candidate_tail["/model.23/Sigmoid_output_0"],
                    )
                key = (
                    -overlap,
                    inversions,
                    crossings,
                    metrics["mse"],
                    metrics["clipping_fraction"],
                    metrics["rail_fraction"],
                    scale,
                    zero_point,
                )
                candidates.append(
                    {
                        "terminal": item["output"],
                        "positive_code_budget": budget,
                        "negative_code_count": zero_point - QUANT_MIN,
                        "scale": scale,
                        "zero_point": zero_point,
                        "topk_overlap": overlap,
                        "pairwise_rank_inversions": inversions,
                        "threshold_crossings": crossings,
                        **metrics,
                        "selection_key": json.dumps(key, separators=(",", ":")),
                    }
                )
        selected = min(
            candidates,
            key=lambda row: tuple(json.loads(str(row["selection_key"]))),
        )
        selected_qparams[item["output"]] = {
            **item,
            "selected": selected,
        }
        for row in candidates:
            row["selected"] = int(row is selected)
            terminal_rows.append(row)

    write_tsv(args.report_root / "terminal_rank_qparam_search.tsv", terminal_rows)
    write_tsv(
        args.report_root / "positive_code_budget.tsv",
        [
            {
                "terminal": output,
                "positive_code_budget": value["selected"]["positive_code_budget"],
                "negative_code_count": value["selected"]["negative_code_count"],
                "scale": value["selected"]["scale"],
                "zero_point": value["selected"]["zero_point"],
            }
            for output, value in selected_qparams.items()
        ],
    )
    write_tsv(
        args.report_root / "rank_topk_proxy.tsv",
        [
            {
                "terminal": output,
                **{
                    key: value["selected"][key]
                    for key in (
                        "topk_overlap",
                        "pairwise_rank_inversions",
                        "threshold_crossings",
                        "mse",
                        "mae",
                        "bias",
                        "cosine",
                        "clipping_fraction",
                        "rail_fraction",
                    )
                },
            }
            for output, value in selected_qparams.items()
        ],
    )
    (args.report_root / "terminal_rank_qparam_selection.md").write_text(
        "# Terminal rank qparam selection\n\n"
        "Selection used no COCO labels. For each P3/P4/P5 confidence domain, the "
        "lexicographic decision first maximized exact common-tail TopK overlap with "
        "the FP32 teacher, then minimized teacher-top-2K pair inversions, threshold "
        "crossings, reconstruction error, clipping and rail occupancy.\n",
        encoding="utf-8",
    )

    component_arrays: dict[str, np.ndarray] = {}
    reconstruction_rows: list[dict[str, Any]] = []
    rounding_rows: list[dict[str, Any]] = []
    bias_rows: list[dict[str, Any]] = []
    range_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    group_status: dict[str, dict[str, Any]] = {}
    for group, anchors in GROUPS.items():
        group_improvements: list[float] = []
        for anchor in anchors:
            descriptor = next(item for item in descriptors if item["anchor"] == anchor)
            fp_weight = numpy_helper.to_array(fp_initializers[descriptor["fp32_weight_initializer"]]).astype(np.float32)
            b2_codes = numpy_helper.to_array(b2_initializers[descriptor["weight_initializer"]]).astype(np.int8)
            weight_scale = numpy_helper.to_array(b2_initializers[descriptor["weight_scale_initializer"]]).astype(np.float32)
            weight_zero = numpy_helper.to_array(b2_initializers[descriptor["weight_zero_point_initializer"]]).astype(np.int8)
            bias = numpy_helper.to_array(b2_initializers[descriptor["bias_initializer"]]).astype(np.float32)
            default_rounder = AdaptiveWeightRounder(
                torch.from_numpy(fp_weight),
                torch.from_numpy(weight_scale),
                torch.from_numpy(weight_zero),
                channel_axis=0,
            )
            default_nearest_codes = default_rounder.codes(hard=True).detach().numpy().astype(np.int8)
            default_nearest_diff_from_b2 = int(np.count_nonzero(default_nearest_codes != b2_codes))
            rounder = AdaptiveWeightRounder(
                torch.from_numpy(fp_weight),
                torch.from_numpy(weight_scale),
                torch.from_numpy(weight_zero),
                initial_codes=torch.from_numpy(b2_codes),
                channel_axis=0,
            )
            initial_baseline_codes = rounder.codes(hard=True).detach().numpy().astype(np.int8)
            initial_diff_from_b2 = require_baseline_rounding_identity(
                initial_baseline_codes,
                b2_codes,
                str(descriptor["conv_node"]),
            )
            train_items = [
                (
                    torch.from_numpy(item["patches"]),
                    torch.from_numpy(item["teacher"]),
                )
                for item in reconstruction_data[anchor]["optimization"]
            ]
            validation_items = [
                (
                    torch.from_numpy(item["patches"]),
                    torch.from_numpy(item["teacher"]),
                )
                for item in reconstruction_data[anchor]["validation"]
            ]
            bias_tensor = torch.from_numpy(bias)
            current_weight_name = str(descriptor["weight_initializer"])
            current_groups = int(descriptor["groups"])

            def teacher_forward(item: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
                return item[1]

            def student_forward(
                item: tuple[torch.Tensor, torch.Tensor],
                weights: Mapping[str, torch.Tensor],
                drop_probability: float,
                generator: torch.Generator,
                _weight_name: str = current_weight_name,
                _bias: torch.Tensor = bias_tensor,
                _groups: int = current_groups,
            ) -> torch.Tensor:
                del generator
                output = conv_outputs(
                    item[0],
                    weights[_weight_name],
                    _bias,
                    _groups,
                )
                if drop_probability:
                    output = output * (1.0 - drop_probability) + item[1] * drop_probability
                return output

            config = ReconstructionConfig(
                seed=args.seed,
                max_iterations=args.reconstruction_iterations,
                validation_interval=10,
                patience=5,
                learning_rate=args.reconstruction_learning_rate,
                rounding_regularization=0.01,
                bias_error_weight=0.1,
                rank_loss_weight=0.02 if group == "R7" else 0.0,
                rank_margin=0.0,
                activation_drop_probability=0.0,
                minimum_improvement=1.0e-9,
            )
            result = reconstruct_block(
                {descriptor["weight_initializer"]: rounder},
                train_items,
                validation_items,
                teacher_forward,
                student_forward,
                block_name=descriptor["conv_node"],
                config=config,
            )
            hardened = result.hardened_weights[descriptor["weight_initializer"]]
            weight_shape = [1] * fp_weight.ndim
            weight_shape[0] = weight_scale.size
            dequantized = (
                hardened.astype(np.float32) - weight_zero.reshape(weight_shape).astype(np.float32)
            ) * weight_scale.reshape(weight_shape)
            b2_dequantized = (
                b2_codes.astype(np.float32) - weight_zero.reshape(weight_shape).astype(np.float32)
            ) * weight_scale.reshape(weight_shape)

            def numpy_student(
                items: Sequence[tuple[torch.Tensor, torch.Tensor]],
                candidate_bias: np.ndarray,
                _weight: np.ndarray = dequantized,
                _groups: int = current_groups,
            ) -> list[np.ndarray]:
                outputs = []
                weight_tensor = torch.from_numpy(_weight)
                candidate_bias_tensor = torch.from_numpy(candidate_bias)
                for patches, _ in items:
                    outputs.append(
                        conv_outputs(
                            patches,
                            weight_tensor,
                            candidate_bias_tensor,
                            _groups,
                        )
                        .detach()
                        .numpy()
                    )
                return outputs

            train_teacher = [item[1].numpy() for item in train_items]
            validation_teacher = [item[1].numpy() for item in validation_items]
            b2_validation_student = numpy_student(
                validation_items,
                bias,
                _weight=b2_dequantized,
            )
            b2_validation_loss = normalized_mse(
                b2_validation_student,
                validation_teacher,
            )
            train_student = numpy_student(train_items, bias)
            validation_student = numpy_student(validation_items, bias)
            correction = compute_bias_correction(train_teacher, train_student, channel_axis=1)
            corrected_bias = apply_bias_correction(bias, correction)
            validation_corrected = numpy_student(validation_items, corrected_bias)
            before_bias_loss = normalized_mse(validation_student, validation_teacher)
            after_bias_loss = normalized_mse(validation_corrected, validation_teacher)
            bias_accepted = after_bias_loss < before_bias_loss
            final_bias = corrected_bias if bias_accepted else bias
            final_validation = min(before_bias_loss, after_bias_loss)
            changed_from_b2 = int(np.count_nonzero(hardened != b2_codes))
            component_arrays[f"weight::{descriptor['weight_initializer']}"] = hardened
            component_arrays[f"bias::{descriptor['bias_initializer']}"] = final_bias
            improvement = b2_validation_loss - final_validation
            group_improvements.append(improvement)
            result_manifest = result.manifest()
            reconstruction_rows.append(
                {
                    "group": group,
                    "anchor": anchor,
                    "conv_node": descriptor["conv_node"],
                    "initial_train_loss": result.initial_train_loss,
                    "initial_validation_loss": result.initial_validation_loss,
                    "b2_validation_loss": b2_validation_loss,
                    "final_train_loss": result.final_train_loss,
                    "final_validation_loss_before_bias": before_bias_loss,
                    "final_validation_loss": final_validation,
                    "validation_improvement": improvement,
                    "iterations": result.iterations,
                    "stop_reason": result.stop_reason,
                    "rolled_back": int(result.rolled_back),
                    "bias_correction_accepted": int(bias_accepted),
                    "proxy_pass": int(improvement > 1.0e-9),
                }
            )
            rounding_rows.append(
                {
                    "group": group,
                    "anchor": anchor,
                    "weight_initializer": descriptor["weight_initializer"],
                    "weight_elements": hardened.size,
                    "default_nearest_diff_from_b2": default_nearest_diff_from_b2,
                    "initial_baseline_diff_from_b2": initial_diff_from_b2,
                    "trainable_floor_ceil_elements": int(torch.count_nonzero(rounder.trainable_mask).item()),
                    "frozen_baseline_elements": int(rounder.trainable_mask.numel() - torch.count_nonzero(rounder.trainable_mask).item()),
                    "hardened_diff_from_b2": changed_from_b2,
                    "round_up_fraction": result_manifest["per_weight_round_up_fraction"][descriptor["weight_initializer"]],
                    "final_dtype": str(hardened.dtype),
                    "training_only_variables_exported": 0,
                }
            )
            bias_rows.append(
                {
                    "group": group,
                    "anchor": anchor,
                    "bias_initializer": descriptor["bias_initializer"],
                    "correction_l1_mean": float(np.mean(np.abs(correction))),
                    "correction_max_abs": float(np.max(np.abs(correction))),
                    "validation_loss_before": before_bias_loss,
                    "validation_loss_after": after_bias_loss,
                    "accepted": int(bias_accepted),
                    "dtype": str(final_bias.dtype),
                }
            )
            anchor_scale, _ = scalar(b2, descriptor["anchor_scale"])
            anchor_zp, _ = scalar(b2, descriptor["anchor_zero_point"])
            range_rows.append(
                {
                    "group": group,
                    "anchor": anchor,
                    "scale": anchor_scale,
                    "zero_point": int(anchor_zp),
                    "representable_min": (QUANT_MIN - int(anchor_zp)) * anchor_scale,
                    "representable_max": (QUANT_MAX - int(anchor_zp)) * anchor_scale,
                    "silu_floor": -0.2784645427610738,
                    "silu_floor_preserved": int((QUANT_MIN - int(anchor_zp)) * anchor_scale <= -0.2784645427610738),
                    "positive_codes": QUANT_MAX - int(anchor_zp),
                    "negative_codes": int(anchor_zp) - QUANT_MIN,
                    "qparams_changed": 0,
                }
            )
            seed_rows.append(
                {
                    "group": group,
                    "conv_node": descriptor["conv_node"],
                    "global_seed": config.seed,
                    "derived_seed": stable_u64("xslim-reconstruction-order-v1", config.seed, descriptor["conv_node"]),
                    "sample_order_sha256": result.sample_order_sha256,
                    "patch_selection_policy": PATCH_SELECTION_POLICY,
                    "optimization_images": len(train_items),
                    "validation_images": len(validation_items),
                    "activation_drop_probability": config.activation_drop_probability,
                }
            )
        qualified = bool(group_improvements) and any(item > 1.0e-9 for item in group_improvements) and all(
            item >= -1.0e-9 for item in group_improvements
        )
        group_status[group] = {
            "qualified": qualified,
            "aggregate_validation_improvement": float(sum(group_improvements)),
            "targets": len(group_improvements),
        }

    components_path = args.raw_root / "reconstruction-components.npz"
    save_npz_deterministic(components_path, component_arrays)
    write_tsv(args.report_root / "reconstruction_target_manifest.tsv", descriptors)
    write_tsv(args.report_root / "reconstruction_seed_order.tsv", seed_rows)
    write_tsv(args.report_root / "reconstruction_train_validation_loss.tsv", reconstruction_rows)
    write_tsv(args.report_root / "reconstruction_rounding_audit.tsv", rounding_rows)
    write_tsv(args.report_root / "reconstruction_bias_correction.tsv", bias_rows)
    write_tsv(args.report_root / "reconstruction_activation_range.tsv", range_rows)

    manifest = {
        "contract": "xslim-dev-001b-candidate-components-v1",
        "fp32_inference": {"path": str(args.fp32_inference), "sha256": sha256(args.fp32_inference)},
        "prequant_reference": {
            "path": str(args.prequant_reference),
            "sha256": sha256(args.prequant_reference),
        },
        "b2_inference": {"path": str(args.b2_inference), "sha256": sha256(args.b2_inference)},
        "tail": {"path": str(args.tail), "sha256": sha256(args.tail)},
        "optimization_list": {"path": str(args.optimization_list), "sha256": sha256(args.optimization_list)},
        "validation_list": {"path": str(args.validation_list), "sha256": sha256(args.validation_list)},
        "terminal_qparams": selected_qparams,
        "descriptors": descriptors,
        "group_status": group_status,
        "components": {"path": components_path.name, "sha256": sha256(components_path)},
        "activation_drop_probability": 0.0,
        "seed": args.seed,
        "threads": args.threads,
        "patch_selection_policy": PATCH_SELECTION_POLICY,
    }
    manifest_path = args.raw_root / "candidate-components.json"
    json_dump(manifest_path, manifest)
    return 0


def set_scalar_initializer(model: onnx.ModelProto, name: str, value: float, dtype: np.dtype[Any]) -> None:
    replace_initializer(model, name, np.asarray(value, dtype=dtype))


def apply_terminal_qparams(model: onnx.ModelProto, terminal_qparams: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for output in TERMINAL_CONFIDENCE:
        accepted = terminal_qparams[output]
        selected = accepted["selected"]
        current = trace_terminal_qdq(model, output)
        scale = float(selected["scale"])
        zero_point = int(selected["zero_point"])
        for name in (current["q_scale"], current["dq_scale"]):
            set_scalar_initializer(model, name, scale, np.dtype(np.float32))
        for name in (current["q_zero_point"], current["dq_zero_point"]):
            set_scalar_initializer(model, name, zero_point, np.dtype(np.int8))
        rows.append(
            {
                "tensor": output,
                "before_scale": current["scale"],
                "before_zero_point": current["zero_point"],
                "after_scale": scale,
                "after_zero_point": zero_point,
            }
        )
    return rows


def apply_group(
    model: onnx.ModelProto,
    descriptors: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    group: str,
) -> list[dict[str, Any]]:
    rows = []
    current = initializer_map(model)
    for descriptor in descriptors:
        if descriptor["group"] != group:
            continue
        weight_name = str(descriptor["weight_initializer"])
        bias_name = str(descriptor["bias_initializer"])
        weight_key = f"weight::{weight_name}"
        bias_key = f"bias::{bias_name}"
        before_weight = numpy_helper.to_array(current[weight_name])
        before_bias = numpy_helper.to_array(current[bias_name])
        after_weight = arrays[weight_key]
        after_bias = arrays[bias_key]
        if before_weight.shape != after_weight.shape or before_weight.dtype != after_weight.dtype:
            raise ValueError(f"weight replacement contract differs: {weight_name}")
        if before_bias.shape != after_bias.shape or before_bias.dtype != after_bias.dtype:
            raise ValueError(f"bias replacement contract differs: {bias_name}")
        replace_initializer(model, weight_name, after_weight)
        replace_initializer(model, bias_name, after_bias)
        rows.append(
            {
                "group": group,
                "weight_initializer": weight_name,
                "weight_changed": int(np.count_nonzero(before_weight != after_weight)),
                "bias_initializer": bias_name,
                "bias_changed": int(np.count_nonzero(before_bias != after_bias)),
            }
        )
    return rows


def graph_signature(model: onnx.ModelProto) -> dict[str, Any]:
    return {
        "nodes": [
            {
                "name": node.name,
                "domain": node.domain,
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attributes": [item.SerializeToString().hex() for item in node.attribute],
            }
            for node in model.graph.node
        ],
        "inputs": [item.SerializeToString().hex() for item in model.graph.input],
        "outputs": [item.SerializeToString().hex() for item in model.graph.output],
        "opsets": [item.SerializeToString().hex() for item in model.opset_import],
        "functions": [item.SerializeToString().hex() for item in model.functions],
    }


def candidate_lane_groups(
    group_status: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    """Return the frozen matrix, conditionally adding only the combined lane."""

    missing = sorted(set(GROUPS) - set(group_status))
    if missing:
        raise ValueError("missing reconstruction group status: " + ", ".join(missing))
    lanes: dict[str, tuple[str, ...]] = {
        "C2_T6_RANK_QP": (),
        "C3_R7_BR": ("R7",),
        "C4_R0_BR": ("R0",),
    }
    qualified = tuple(
        group for group in ("R7", "R0") if bool(group_status[group]["qualified"])
    )
    if qualified:
        lanes["C5_COMBINED"] = qualified
    return lanes


def build(args: argparse.Namespace) -> int:
    if args.output_root.exists():
        raise RuntimeError("refusing to overwrite candidate generation output")
    args.output_root.mkdir(parents=True)
    manifest = json.loads(args.components_manifest.read_text(encoding="utf-8"))
    components_path = args.components_manifest.parent / Path(manifest["components"]["path"])
    if sha256(components_path) != manifest["components"]["sha256"]:
        raise RuntimeError("candidate component archive identity differs")
    arrays_file = np.load(components_path, allow_pickle=False)
    arrays = {name: arrays_file[name] for name in arrays_file.files}
    group_status = manifest["group_status"]
    lanes = candidate_lane_groups(group_status)

    identity_rows = []
    qparam_rows = []
    initializer_rows = []
    for lane, groups in lanes.items():
        lane_root = args.output_root / lane
        lane_root.mkdir()
        for model_kind, source in (("deployable", args.b2_deployable), ("inference", args.b2_inference)):
            model = onnx.load(source)
            before_signature = graph_signature(model)
            qparam_changes = apply_terminal_qparams(model, manifest["terminal_qparams"])
            group_changes: list[dict[str, Any]] = []
            for group in groups:
                group_changes.extend(apply_group(model, manifest["descriptors"], arrays, group))
            if graph_signature(model) != before_signature:
                raise RuntimeError(f"candidate topology changed while editing initializers: {lane} {model_kind}")
            onnx.checker.check_model(model)
            onnx.shape_inference.infer_shapes(model)
            destination = lane_root / f"{lane.lower()}.{model_kind}.onnx"
            onnx.save_model(model, destination)
            identity_rows.append(
                {
                    "lane": lane,
                    "run_id": args.run_id,
                    "model_kind": model_kind,
                    "path": str(destination),
                    "bytes": destination.stat().st_size,
                    "sha256": sha256(destination),
                    "source_sha256": sha256(source),
                    "groups": ";".join(groups) or "none",
                    "checker": "pass",
                    "shape_inference": "pass",
                }
            )
            for row in qparam_changes:
                qparam_rows.append({"lane": lane, "run_id": args.run_id, "model_kind": model_kind, **row})
            for row in group_changes:
                initializer_rows.append({"lane": lane, "run_id": args.run_id, "model_kind": model_kind, **row})
    write_tsv(args.output_root / "candidate-identity.tsv", identity_rows)
    write_tsv(args.output_root / "candidate-qparam-diff.tsv", qparam_rows)
    write_tsv(args.output_root / "candidate-initializer-diff.tsv", initializer_rows)
    json_dump(
        args.output_root / "generation-manifest.json",
        {
            "contract": "xslim-dev-001b-candidate-generation-v1",
            "run_id": args.run_id,
            "source_components": {"path": str(args.components_manifest), "sha256": sha256(args.components_manifest)},
            "lanes": {lane: list(groups) for lane, groups in lanes.items()},
            "identity_rows": identity_rows,
        },
    )
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    export_parser = commands.add_parser("export-prequant")
    export_parser.add_argument("--base-config", required=True, type=Path)
    export_parser.add_argument("--output", required=True, type=Path)
    export_parser.add_argument("--manifest", required=True, type=Path)
    export_parser.add_argument("--seed", type=int, default=65001)
    export_parser.add_argument("--threads", type=int, default=4)
    export_parser.set_defaults(function=export_prequant_reference)

    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--fp32-inference", required=True, type=Path)
    prepare_parser.add_argument("--prequant-reference", required=True, type=Path)
    prepare_parser.add_argument("--b2-inference", required=True, type=Path)
    prepare_parser.add_argument("--tail", required=True, type=Path)
    prepare_parser.add_argument("--optimization-list", required=True, type=Path)
    prepare_parser.add_argument("--validation-list", required=True, type=Path)
    prepare_parser.add_argument("--raw-root", required=True, type=Path)
    prepare_parser.add_argument("--report-root", required=True, type=Path)
    prepare_parser.add_argument("--seed", type=int, default=65001)
    prepare_parser.add_argument("--threads", type=int, default=4)
    prepare_parser.add_argument("--patches-per-image", type=int, default=8)
    prepare_parser.add_argument("--terminal-samples-per-image", type=int, default=4096)
    prepare_parser.add_argument("--reconstruction-iterations", type=int, default=200)
    prepare_parser.add_argument("--reconstruction-learning-rate", type=float, default=1.0e-2)
    prepare_parser.add_argument("--log-every", type=int, default=10)
    prepare_parser.set_defaults(function=prepare)

    build_parser = commands.add_parser("build")
    build_parser.add_argument("--components-manifest", required=True, type=Path)
    build_parser.add_argument("--b2-deployable", required=True, type=Path)
    build_parser.add_argument("--b2-inference", required=True, type=Path)
    build_parser.add_argument("--output-root", required=True, type=Path)
    build_parser.add_argument("--run-id", required=True)
    build_parser.set_defaults(function=build)
    return root


def main() -> int:
    args = parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
