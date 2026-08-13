#!/usr/bin/env python3
"""Stream matched FP32/B2 activation diagnostics for selected R3 frontiers."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

from stage65b_r1_evaluate import (
    image_tensor_and_geometry,
    paths_from_list,
    session,
    sha256_array,
)
from stage65b_r3_common import extract_model, inferred_model, read_tsv, sha256, write_tsv


def selected_rows(
    live_paths: list[Path], correspondence_paths: list[Path], frontiers: set[str]
) -> list[dict[str, str]]:
    live = [row for path in live_paths for row in read_tsv(path)]
    correspondence = {
        row["source_tensor"]: row
        for path in correspondence_paths
        for row in read_tsv(path)
        if row.get("source_tensor")
    }
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in live:
        key = (row.get("frontier", ""), row.get("source_tensor", ""))
        if key[0] not in frontiers or not key[1] or key in seen:
            continue
        if key[1] not in correspondence:
            raise ValueError(f"missing QDQ correspondence for {key}")
        result.append({**row, **correspondence[key[1]]})
        seen.add(key)
    missing = sorted(frontiers - {row["frontier"] for row in result})
    if missing:
        raise ValueError(f"selected frontiers have no tensors: {missing}")
    return result


def qlimits(dtype: int) -> tuple[int, int]:
    if dtype == TensorProto.INT8:
        return -128, 127
    if dtype == TensorProto.UINT8:
        return 0, 255
    raise ValueError(f"unsupported activation zero-point dtype: {dtype}")


def broadcast_qparam(values: np.ndarray, shape: tuple[int, ...], axis: str) -> np.ndarray:
    if values.ndim == 0 or values.size == 1:
        return values.reshape((1,) * len(shape))
    if not axis:
        raise ValueError("non-scalar qparam has no axis")
    dimension = int(axis)
    if dimension < 0:
        dimension += len(shape)
    if not 0 <= dimension < len(shape) or values.size != shape[dimension]:
        raise ValueError(f"qparam axis/shape mismatch: {values.shape}, axis={axis}, tensor={shape}")
    target = [1] * len(shape)
    target[dimension] = values.size
    return values.reshape(target)


def percentile_text(values: np.ndarray) -> str:
    if not values.size:
        return "empty"
    points = np.percentile(values.astype(np.float64), [0, 1, 10, 50, 90, 99, 100])
    return ";".join(
        f"p{label}={value:.9g}"
        for label, value in zip((0, 1, 10, 50, 90, 99, 100), points)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--live", required=True, action="append", type=Path)
    parser.add_argument("--correspondence", required=True, action="append", type=Path)
    parser.add_argument("--frontier", required=True, action="append")
    parser.add_argument("--image-list", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--sample-per-image", type=int, default=1024)
    parser.add_argument("--log-every", type=int, default=25)
    options = parser.parse_args()
    if options.output_dir.exists():
        raise RuntimeError(f"refusing to reuse output directory: {options.output_dir}")
    options.output_dir.mkdir(parents=True)

    rows = selected_rows(
        options.live, options.correspondence, set(options.frontier)
    )
    tensors: list[str] = []
    tensor_rows: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        tensor = row["source_tensor"]
        tensor_rows.setdefault(tensor, []).append(row)
        if tensor not in tensors:
            tensors.append(tensor)

    fp32 = inferred_model(options.fp32)
    candidate = inferred_model(options.candidate)
    image_inputs = [item.name for item in fp32.graph.input]
    if image_inputs != [item.name for item in candidate.graph.input]:
        raise ValueError("FP32/B2 graph-input contracts differ")
    fp_diag = extract_model(fp32, image_inputs, tensors)
    b2_diag = extract_model(candidate, image_inputs, tensors)
    fp_path = options.output_dir / "models" / "selected-fp32.onnx"
    b2_path = options.output_dir / "models" / "selected-b2.onnx"
    fp_path.parent.mkdir(parents=True)
    onnx.save_model(fp_diag, fp_path)
    onnx.save_model(b2_diag, b2_path)
    inferred_model(fp_path)
    inferred_model(b2_path)

    initializers = {item.name: item for item in candidate.graph.initializer}
    qparams: dict[str, dict[str, Any]] = {}
    qparam_rows: list[dict[str, Any]] = []
    for tensor in tensors:
        row = tensor_rows[tensor][0]
        scale_proto = initializers[row["scale_initializer"]]
        zp_proto = initializers[row["zero_point_initializer"]]
        scale = numpy_helper.to_array(scale_proto).astype(np.float64, copy=False)
        zp = numpy_helper.to_array(zp_proto).astype(np.float64, copy=False)
        qmin, qmax = qlimits(zp_proto.data_type)
        low = (qmin - zp) * scale
        high = (qmax - zp) * scale
        qparams[tensor] = {
            "scale": scale,
            "zp": zp,
            "low": low,
            "high": high,
            "axis": row.get("axis", ""),
        }
        qparam_rows.append(
            {
                "frontiers": ";".join(item["frontier"] for item in tensor_rows[tensor]),
                "tensor": tensor,
                "source_op": row["source_op"],
                "shape": row["shape"],
                "dtype": row["dtype"],
                "scale_initializer": row["scale_initializer"],
                "scale_shape": "scalar" if scale.ndim == 0 else "x".join(map(str, scale.shape)),
                "scale_min": float(scale.min()),
                "scale_max": float(scale.max()),
                "zero_point_initializer": row["zero_point_initializer"],
                "zero_point_dtype": TensorProto.DataType.Name(zp_proto.data_type),
                "zero_point_min": float(zp.min()),
                "zero_point_max": float(zp.max()),
                "axis": row.get("axis", ""),
                "representable_min": float(low.min()),
                "representable_max": float(high.max()),
            }
        )

    aggregates: dict[str, dict[str, Any]] = {}
    for tensor in tensors:
        aggregates[tensor] = {
            "count": 0,
            "fp_min": float("inf"),
            "fp_max": float("-inf"),
            "b2_min": float("inf"),
            "b2_max": float("-inf"),
            "below": 0,
            "above": 0,
            "rail_low": 0,
            "rail_high": 0,
            "sum_diff": 0.0,
            "sum_abs": 0.0,
            "sum_abs_fp": 0.0,
            "dot": 0.0,
            "norm_fp": 0.0,
            "norm_b2": 0.0,
            "hash": hashlib.sha256(),
            "fp_samples": [],
            "b2_samples": [],
        }

    fp_session = session(fp_path, options.threads)
    b2_session = session(b2_path, options.threads)
    images = paths_from_list(options.image_list, 0)
    for image_index, path in enumerate(images):
        tensor, _ = image_tensor_and_geometry(path)
        feed = {fp_session.get_inputs()[0].name: tensor}
        fp_values = fp_session.run(None, feed)
        feed = {b2_session.get_inputs()[0].name: tensor}
        b2_values = b2_session.run(None, feed)
        for name, fp_value, b2_value in zip(tensors, fp_values, b2_values):
            if fp_value.shape != b2_value.shape or fp_value.dtype != b2_value.dtype:
                raise ValueError(f"activation shape/dtype mismatch at {name}")
            if not np.isfinite(fp_value).all() or not np.isfinite(b2_value).all():
                raise ValueError(f"non-finite activation at {name}, image {path}")
            state = aggregates[name]
            qp = qparams[name]
            low = np.broadcast_to(
                broadcast_qparam(qp["low"], fp_value.shape, qp["axis"]), fp_value.shape
            )
            high = np.broadcast_to(
                broadcast_qparam(qp["high"], fp_value.shape, qp["axis"]), fp_value.shape
            )
            fp64 = fp_value.astype(np.float64, copy=False)
            b264 = b2_value.astype(np.float64, copy=False)
            delta = b264 - fp64
            state["count"] += fp_value.size
            state["fp_min"] = min(state["fp_min"], float(fp_value.min()))
            state["fp_max"] = max(state["fp_max"], float(fp_value.max()))
            state["b2_min"] = min(state["b2_min"], float(b2_value.min()))
            state["b2_max"] = max(state["b2_max"], float(b2_value.max()))
            state["below"] += int(np.count_nonzero(fp64 < low))
            state["above"] += int(np.count_nonzero(fp64 > high))
            tolerance = np.maximum(np.abs(high - low) * 1.0e-7, 1.0e-12)
            state["rail_low"] += int(np.count_nonzero(np.abs(b264 - low) <= tolerance))
            state["rail_high"] += int(np.count_nonzero(np.abs(b264 - high) <= tolerance))
            state["sum_diff"] += float(delta.sum())
            state["sum_abs"] += float(np.abs(delta).sum())
            state["sum_abs_fp"] += float(np.abs(fp64).sum())
            state["dot"] += float(np.sum(fp64 * b264))
            state["norm_fp"] += float(np.sum(fp64 * fp64))
            state["norm_b2"] += float(np.sum(b264 * b264))
            state["hash"].update(path.name.encode("utf-8") + b"\0")
            state["hash"].update(sha256_array(fp_value).encode("ascii") + b"\0")
            state["hash"].update(sha256_array(b2_value).encode("ascii") + b"\0")
            flat_fp = fp_value.reshape(-1)
            flat_b2 = b2_value.reshape(-1)
            stride = max(1, flat_fp.size // options.sample_per_image)
            state["fp_samples"].append(flat_fp[::stride][: options.sample_per_image].copy())
            state["b2_samples"].append(flat_b2[::stride][: options.sample_per_image].copy())
        if options.log_every and (image_index + 1) % options.log_every == 0:
            print(f"activation audit: {image_index + 1}/{len(images)}", flush=True)

    output_rows: list[dict[str, Any]] = []
    for tensor in tensors:
        state = aggregates[tensor]
        count = state["count"]
        mae = state["sum_abs"] / count
        mean_abs_fp = state["sum_abs_fp"] / count
        denominator = np.sqrt(state["norm_fp"] * state["norm_b2"])
        output_rows.append(
            {
                "frontiers": ";".join(item["frontier"] for item in tensor_rows[tensor]),
                "tensor": tensor,
                "shape": tensor_rows[tensor][0]["shape"],
                "values": count,
                "fp32_min": state["fp_min"],
                "fp32_max": state["fp_max"],
                "b2_dequantized_min": state["b2_min"],
                "b2_dequantized_max": state["b2_max"],
                "fp32_below_range_count": state["below"],
                "fp32_below_range_fraction": state["below"] / count,
                "fp32_above_range_count": state["above"],
                "fp32_above_range_fraction": state["above"] / count,
                "b2_low_rail_count": state["rail_low"],
                "b2_low_rail_fraction": state["rail_low"] / count,
                "b2_high_rail_count": state["rail_high"],
                "b2_high_rail_fraction": state["rail_high"] / count,
                "mean_bias_b2_minus_fp32": state["sum_diff"] / count,
                "mae": mae,
                "normalized_mae_over_mean_abs_fp32": mae / max(mean_abs_fp, 1.0e-12),
                "cosine": state["dot"] / denominator if denominator else float("nan"),
                "fp32_histogram_sample": percentile_text(np.concatenate(state["fp_samples"])),
                "b2_histogram_sample": percentile_text(np.concatenate(state["b2_samples"])),
                "per_image_pair_hash": state["hash"].hexdigest(),
            }
        )
    write_tsv(options.output_dir / "selected_region_activation_error.tsv", output_rows)
    write_tsv(options.output_dir / "selected_region_qparams.tsv", qparam_rows)
    write_tsv(
        options.output_dir / "model_identity.tsv",
        [
            {"role": "source_fp32", "path": str(options.fp32), "sha256": sha256(options.fp32)},
            {"role": "candidate_b2", "path": str(options.candidate), "sha256": sha256(options.candidate)},
            {"role": "diagnostic_fp32", "path": str(fp_path), "sha256": sha256(fp_path)},
            {"role": "diagnostic_b2", "path": str(b2_path), "sha256": sha256(b2_path)},
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
