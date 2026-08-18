#!/usr/bin/env python3
"""Compare frozen A1/B2 CPU and SpaceMIT EP six-boundary outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper

LABELS = ("P3_bbox", "P3_confidence", "P4_bbox", "P4_confidence", "P5_bbox", "P5_confidence")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_tsv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise ValueError(f"refusing empty TSV: {path}")
    fields = list(values[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def tensor_shapes(tail: Path) -> list[tuple[str, tuple[int, ...]]]:
    model = onnx.load(tail)
    result = []
    for value in model.graph.input:
        shape = tuple(int(item.dim_value) for item in value.type.tensor_type.shape.dim)
        if not shape or any(item <= 0 for item in shape):
            raise ValueError(f"dynamic tail input: {value.name}")
        result.append((value.name, shape))
    if len(result) != 6:
        raise ValueError("tail does not expose six inputs")
    return result


def qparams(model_path: Path, names: list[str]) -> dict[str, dict[str, Any]]:
    model = onnx.load(model_path)
    producer = {output: node for node in model.graph.node for output in node.output}
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    result = {}
    for name in names:
        dq = producer[name]
        if dq.op_type != "DequantizeLinear":
            raise ValueError(f"graph output is not DequantizeLinear: {name}")
        q = producer[dq.input[0]]
        if q.op_type != "QuantizeLinear":
            raise ValueError(f"graph output does not have final Q/DQ: {name}")
        scale = np.asarray(initializers[dq.input[1]]).reshape(-1)
        zero_point = np.asarray(initializers[dq.input[2]]).reshape(-1)
        if scale.size != 1 or zero_point.size != 1:
            raise ValueError(f"terminal activation qparams are not per-tensor: {name}")
        value_scale = float(scale[0])
        value_zp = int(zero_point[0])
        result[name] = {
            "scale": value_scale,
            "zero_point": value_zp,
            "minimum": (-128 - value_zp) * value_scale,
            "maximum": (127 - value_zp) * value_scale,
            "q_node": q.name,
            "dq_node": dq.name,
        }
    return result


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    l = left.astype(np.float64, copy=False).ravel()
    r = right.astype(np.float64, copy=False).ravel()
    denominator = float(np.linalg.norm(l) * np.linalg.norm(r))
    return float(np.dot(l, r) / denominator) if denominator else (1.0 if np.array_equal(l, r) else 0.0)


def top_rank_summary(left: np.ndarray, right: np.ndarray, count: int = 100) -> tuple[float, float]:
    left_order = np.argsort(-left.ravel(), kind="stable")[:count]
    right_order = np.argsort(-right.ravel(), kind="stable")[:count]
    overlap = set(map(int, left_order)) & set(map(int, right_order))
    jaccard = len(overlap) / len(set(map(int, left_order)) | set(map(int, right_order)))
    if not overlap:
        return jaccard, float("nan")
    left_rank = {int(value): rank for rank, value in enumerate(left_order)}
    right_rank = {int(value): rank for rank, value in enumerate(right_order)}
    displacement = np.mean([abs(left_rank[value] - right_rank[value]) for value in overlap])
    return jaccard, float(displacement)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--boundary-root", required=True, type=Path)
    parser.add_argument("--b2-model", required=True, type=Path)
    parser.add_argument("--a1-model", required=True, type=Path)
    parser.add_argument("--tail", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    options = parser.parse_args()
    if options.output_dir.exists():
        raise RuntimeError(f"refusing existing output directory: {options.output_dir}")
    options.output_dir.mkdir(parents=True)
    selection = list(csv.DictReader(options.selection.open(encoding="utf-8"), delimiter="\t"))
    shapes = tensor_shapes(options.tail)
    names = [name for name, _ in shapes]
    model_qparams = {
        "B2": qparams(options.b2_model, names),
        "A1": qparams(options.a1_model, names),
    }
    qparam_rows = []
    for model, values in model_qparams.items():
        for index, name in enumerate(names):
            qparam_rows.append({"model": model, "boundary_index": index, "boundary_label": LABELS[index], "boundary_name": name, **values[name]})
    write_tsv(options.output_dir / "boundary_qparams.tsv", qparam_rows)

    identity_rows = []
    numeric_rows = []
    qnorm_rows = []
    crossing_rows = []
    interaction_values: dict[int, list[dict[str, float]]] = {index: [] for index in range(6)}
    for case in selection:
        image_id = int(case["image_id"])
        arrays: dict[tuple[str, str, int], np.ndarray] = {}
        for model in ("B2", "A1"):
            for provider in ("cpu", "spacemit"):
                surface = options.boundary_root / str(image_id) / f"{model}-{provider}" / "boundaries"
                for index, (name, shape) in enumerate(shapes):
                    path = surface / f"boundary-{index}.bin"
                    expected = int(np.prod(shape))
                    value = np.fromfile(path, dtype=np.float32)
                    if value.size != expected:
                        raise ValueError(f"boundary size mismatch: {path}")
                    value = value.reshape(shape)
                    arrays[(model, provider, index)] = value
                    qp = model_qparams[model][name]
                    lower = np.isclose(value, qp["minimum"], atol=qp["scale"] * 1e-5)
                    upper = np.isclose(value, qp["maximum"], atol=qp["scale"] * 1e-5)
                    identity_rows.append(
                        {
                            "selection_group": case["selection_group"], "image_id": image_id,
                            "model": model, "provider": provider, "boundary_index": index,
                            "boundary_label": LABELS[index], "boundary_name": name,
                            "shape": "x".join(map(str, shape)), "dtype": "float32",
                            "bytes": path.stat().st_size, "sha256": sha256(path),
                            "minimum": float(value.min()), "maximum": float(value.max()),
                            "mean": float(value.mean(dtype=np.float64)), "stddev": float(value.std(dtype=np.float64)),
                            "non_finite": int(value.size - np.count_nonzero(np.isfinite(value))),
                            "lower_rail_count": int(np.count_nonzero(lower)), "upper_rail_count": int(np.count_nonzero(upper)),
                        }
                    )

        for model in ("B2", "A1"):
            for index, (name, shape) in enumerate(shapes):
                cpu = arrays[(model, "cpu", index)]
                ep = arrays[(model, "spacemit", index)]
                delta = ep.astype(np.float64) - cpu.astype(np.float64)
                absolute = np.abs(delta)
                qp = model_qparams[model][name]
                channel_rms = np.sqrt(np.mean(np.square(delta), axis=(0, 2, 3)))
                spatial_rms = np.sqrt(np.mean(np.square(delta), axis=(0, 1)))
                channel = int(np.argmax(channel_rms))
                spatial = np.unravel_index(int(np.argmax(spatial_rms)), spatial_rms.shape)
                jaccard, displacement = top_rank_summary(cpu, ep) if "confidence" in LABELS[index] else (float("nan"), float("nan"))
                numeric_rows.append(
                    {
                        "selection_group": case["selection_group"], "image_id": image_id,
                        "model": model, "boundary_index": index, "boundary_label": LABELS[index],
                        "max_abs_difference": float(absolute.max()), "mean_abs_difference": float(absolute.mean()),
                        "rms_difference": float(np.sqrt(np.mean(np.square(delta)))), "cosine": cosine(cpu, ep),
                        "largest_channel": channel, "largest_channel_rms": float(channel_rms[channel]),
                        "largest_spatial_y": spatial[0], "largest_spatial_x": spatial[1],
                        "largest_spatial_rms": float(spatial_rms[spatial]),
                        "top100_jaccard": jaccard, "top100_mean_rank_displacement": displacement,
                    }
                )
                qnorm_rows.append(
                    {
                        "selection_group": case["selection_group"], "image_id": image_id,
                        "model": model, "boundary_index": index, "boundary_label": LABELS[index],
                        "scale": qp["scale"], "zero_point": qp["zero_point"],
                        "representable_min": qp["minimum"], "representable_max": qp["maximum"],
                        "mean_abs_difference_qsteps": float(absolute.mean() / qp["scale"]),
                        "rms_difference_qsteps": float(np.sqrt(np.mean(np.square(delta))) / qp["scale"]),
                        "max_abs_difference_qsteps": float(absolute.max() / qp["scale"]),
                    }
                )
                cpu_lower = np.isclose(cpu, qp["minimum"], atol=qp["scale"] * 1e-5)
                ep_lower = np.isclose(ep, qp["minimum"], atol=qp["scale"] * 1e-5)
                cpu_upper = np.isclose(cpu, qp["maximum"], atol=qp["scale"] * 1e-5)
                ep_upper = np.isclose(ep, qp["maximum"], atol=qp["scale"] * 1e-5)
                crossing_rows.append(
                    {
                        "selection_group": case["selection_group"], "image_id": image_id,
                        "model": model, "boundary_index": index, "boundary_label": LABELS[index],
                        "sign_crossings": int(np.count_nonzero(np.signbit(cpu) != np.signbit(ep))),
                        "zero_crossings": int(np.count_nonzero((cpu == 0) != (ep == 0))),
                        "cpu_lower_ep_not": int(np.count_nonzero(cpu_lower & ~ep_lower)),
                        "ep_lower_cpu_not": int(np.count_nonzero(ep_lower & ~cpu_lower)),
                        "cpu_upper_ep_not": int(np.count_nonzero(cpu_upper & ~ep_upper)),
                        "ep_upper_cpu_not": int(np.count_nonzero(ep_upper & ~cpu_upper)),
                    }
                )

        for index, (_name, _shape) in enumerate(shapes):
            a1_delta = arrays[("A1", "spacemit", index)].astype(np.float64) - arrays[("A1", "cpu", index)].astype(np.float64)
            b2_delta = arrays[("B2", "spacemit", index)].astype(np.float64) - arrays[("B2", "cpu", index)].astype(np.float64)
            interaction = a1_delta - b2_delta
            scale = model_qparams["A1"][names[index]]["scale"]
            interaction_values[index].append(
                {
                    "rms": float(np.sqrt(np.mean(np.square(interaction)))),
                    "mean_abs": float(np.mean(np.abs(interaction))),
                    "max_abs": float(np.max(np.abs(interaction))),
                    "rms_qsteps": float(np.sqrt(np.mean(np.square(interaction))) / scale),
                    "nonzero_fraction": float(np.mean(interaction != 0)),
                }
            )

    write_tsv(options.output_dir / "boundary_output_identity.tsv", identity_rows)
    write_tsv(options.output_dir / "boundary_numeric_comparison.tsv", numeric_rows)
    write_tsv(options.output_dir / "boundary_qparam_normalized_difference.tsv", qnorm_rows)
    write_tsv(options.output_dir / "boundary_sign_and_headroom_crossings.tsv", crossing_rows)
    ranking = []
    for index, values in interaction_values.items():
        ranking.append(
            {
                "boundary_index": index, "boundary_label": LABELS[index], "boundary_name": names[index],
                "selected_cases": len(values),
                "mean_interaction_rms": float(np.mean([row["rms"] for row in values])),
                "mean_interaction_mean_abs": float(np.mean([row["mean_abs"] for row in values])),
                "maximum_interaction_abs": float(np.max([row["max_abs"] for row in values])),
                "mean_interaction_rms_qsteps": float(np.mean([row["rms_qsteps"] for row in values])),
                "mean_nonzero_fraction": float(np.mean([row["nonzero_fraction"] for row in values])),
            }
        )
    ranking.sort(key=lambda row: (-row["mean_interaction_rms_qsteps"], row["boundary_index"]))
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank
    write_tsv(options.output_dir / "boundary_interaction_ranking.tsv", ranking)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
