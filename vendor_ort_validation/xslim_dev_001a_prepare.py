#!/usr/bin/env python3
"""Freeze DEV-001A targets, regenerate H500 samples, and screen observers."""

from __future__ import annotations

import argparse
import csv
import json
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
from stage65b_r3_common import (
    dtype_text,
    extract_model,
    inferred_model,
    producers,
    sha256,
    shape_text,
    value_info_map,
    write_tsv,
)
from xslim.range_policy import (
    ConstrainedRangeSpec,
    quantize_dequantize,
    select_qparams,
)

R0 = (
    "/model.0/act/Mul_output_0",
    "/model.1/act/Mul_output_0",
    "/model.2/cv1/act/Mul_output_0",
    "/model.2/cv2/act/Mul_output_0",
)
R7_CONFIDENCE = (
    ("/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.1/act/Mul_output_0"),
    ("/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1/act/Mul_output_0"),
    ("/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/act/Mul_output_0"),
)
R7_BBOX = (
    "/model.23/one2one_cv2.0/one2one_cv2.0.1/act/Mul_output_0",
    "/model.23/one2one_cv2.1/one2one_cv2.1.1/act/Mul_output_0",
    "/model.23/one2one_cv2.2/one2one_cv2.2.1/act/Mul_output_0",
)
T6 = (
    "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0",
    "/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0",
    "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0",
    "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0",
    "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0",
    "/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0",
)
TARGET_SETS = {
    "R0": R0,
    "R7-confidence": R7_CONFIDENCE,
    "R7-all-six": (*R7_BBOX, *R7_CONFIDENCE),
    "T6": T6,
}
METHODS = (
    "default",
    "minmax",
    "percentile-0.999",
    "percentile-0.9995",
    "percentile-0.9999",
    "mse",
    "kl",
    "constrained-mse",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-inference", required=True, type=Path)
    parser.add_argument("--b2-inference", required=True, type=Path)
    parser.add_argument("--h500-list", required=True, type=Path)
    parser.add_argument("--b2-config", required=True, type=Path)
    parser.add_argument("--r3-hashes", required=True, type=Path)
    parser.add_argument("--r1-boundary-audit", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--sample-per-image", type=int, default=2048)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def stable_name(index: int) -> str:
    return f"target-{index:02d}.npy"


def scalar_initializer(model: onnx.ModelProto, name: str) -> tuple[float, int]:
    initializers = {item.name: item for item in model.graph.initializer}
    if name not in initializers:
        raise ValueError(f"missing qparam initializer: {name}")
    value = numpy_helper.to_array(initializers[name])
    if value.size != 1:
        raise ValueError(f"activation qparam is not scalar: {name} {value.shape}")
    return float(value.reshape(-1)[0]), initializers[name].data_type


def qdq_record(model: onnx.ModelProto, tensor: str) -> dict[str, Any]:
    by_output = producers(model)
    terminal = by_output.get(tensor)
    wrappers: list[str] = []
    while terminal is not None and terminal.op_type == "Identity":
        wrappers.append(terminal.name)
        terminal = by_output.get(terminal.input[0])
    if terminal is None or terminal.op_type != "DequantizeLinear":
        raise ValueError(f"{tensor} is not produced by DequantizeLinear")
    quantize = by_output.get(terminal.input[0])
    if quantize is None or quantize.op_type != "QuantizeLinear":
        raise ValueError(f"{tensor} does not have terminal Q->DQ topology")
    q_scale, q_scale_dtype = scalar_initializer(model, quantize.input[1])
    q_zero_point_value, q_zero_point_dtype = scalar_initializer(
        model, quantize.input[2]
    )
    dq_scale, dq_scale_dtype = scalar_initializer(model, terminal.input[1])
    dq_zero_point_value, dq_zero_point_dtype = scalar_initializer(
        model, terminal.input[2]
    )
    if (
        q_scale_dtype != TensorProto.FLOAT
        or dq_scale_dtype != TensorProto.FLOAT
        or q_zero_point_dtype != TensorProto.INT8
        or dq_zero_point_dtype != TensorProto.INT8
    ):
        raise ValueError(f"{tensor} is not FLOAT-scale/S8-zero-point QDQ")
    if q_scale != dq_scale or q_zero_point_value != dq_zero_point_value:
        raise ValueError(f"{tensor} Q/DQ qparam values differ")
    zero_point = int(q_zero_point_value)
    return {
        "quantize_node": quantize.name,
        "dequantize_node": terminal.name,
        "identity_wrappers": ";".join(wrappers),
        "float_source": quantize.input[0],
        "quantize_scale_initializer": quantize.input[1],
        "quantize_zero_point_initializer": quantize.input[2],
        "dequantize_scale_initializer": terminal.input[1],
        "dequantize_zero_point_initializer": terminal.input[2],
        "scale": q_scale,
        "zero_point": zero_point,
        "representable_min": (-128 - zero_point) * q_scale,
        "representable_max": (127 - zero_point) * q_scale,
    }


def source_record(model: onnx.ModelProto, tensor: str) -> dict[str, Any]:
    by_output = producers(model)
    producer = by_output.get(tensor)
    if producer is None:
        raise ValueError(f"missing FP32 producer for {tensor}")
    infos = value_info_map(model)
    if tensor not in infos:
        raise ValueError(f"missing FP32 shape/dtype for {tensor}")
    return {
        "source_producer": producer.name,
        "source_op": producer.op_type,
        "source_output_index": list(producer.output).index(tensor),
        "shape": shape_text(infos[tensor]),
        "dtype": dtype_text(infos[tensor]),
    }


def read_r3_hashes(path: Path) -> dict[tuple[str, str], str]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream, delimiter="\t")
        return {
            (row["tensor"], row["image"]): row["fp32_activation_sha256"] for row in rows
        }


def read_r1_hashes(path: Path) -> dict[tuple[str, str], str]:
    csv.field_size_limit(16 * 1024 * 1024)
    result: dict[tuple[str, str], str] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            for item in json.loads(row["per_image_hashes"]):
                result[(row["float_tensor"], item["image"])] = item["float"]
    return result


def default_metrics(
    values: np.ndarray, scale: float, zero_point: int
) -> dict[str, Any]:
    reconstructed, quantized = quantize_dequantize(values, scale, zero_point)
    original = values.astype(np.float64)
    rebuilt = reconstructed.astype(np.float64)
    error = rebuilt - original
    low = (-128 - zero_point) * scale
    high = (127 - zero_point) * scale
    denominator = float(np.linalg.norm(original) * np.linalg.norm(rebuilt))
    mae = float(np.mean(np.abs(error)))
    return {
        "scale": scale,
        "zero_point": zero_point,
        "representable_min": low,
        "representable_max": high,
        "objective": "default",
        "objective_value": "",
        "observed_min": float(values.min()),
        "observed_max": float(values.max()),
        "clipping_fraction": float(np.mean((original < low) | (original > high))),
        "rail_fraction": float(np.mean((quantized == -128) | (quantized == 127))),
        "bias": float(np.mean(error)),
        "mae": mae,
        "normalized_mae": mae / max(float(np.mean(np.abs(original))), 1.0e-12),
        "cosine": float(np.dot(original, rebuilt) / denominator)
        if denominator
        else 1.0,
        "constraint_margins": {},
    }


def method_spec(method: str, tensor: str, values: np.ndarray) -> ConstrainedRangeSpec:
    is_silu = tensor not in T6
    required_min = (
        float(np.quantile(values.astype(np.float64), 0.0001)) if tensor in T6 else None
    )
    required_max = float(np.quantile(values.astype(np.float64), 0.9999))
    common: dict[str, Any] = {
        "preserve_zero": True,
        "required_real_min": required_min,
        "required_real_max": required_max,
        "semantic_floor": "silu" if is_silu else None,
        "search_steps": 32,
    }
    if method == "minmax":
        return ConstrainedRangeSpec(objective="minmax", **common)
    if method.startswith("percentile-"):
        return ConstrainedRangeSpec(
            objective="percentile",
            percentile=float(method.split("-", 1)[1]),
            **common,
        )
    if method in {"mse", "kl"}:
        return ConstrainedRangeSpec(objective=method, **common)
    if method != "constrained-mse":
        raise ValueError(f"unsupported method: {method}")
    return ConstrainedRangeSpec(objective="constrained-mse", **common)


def constraint_status(
    tensor: str, values: np.ndarray, result: dict[str, Any]
) -> tuple[bool, float, float]:
    required_min = (
        float(np.quantile(values.astype(np.float64), 0.0001))
        if tensor in T6
        else -0.2784645427610738
    )
    required_max = float(np.quantile(values.astype(np.float64), 0.9999))
    scale = float(result["scale"])
    tolerance = max(1.0e-12, scale * 1.0e-6)
    passed = (
        float(result["representable_min"]) <= required_min + tolerance
        and float(result["representable_max"]) >= required_max - tolerance
    )
    return passed, required_min, required_max


def candidate_configs(
    b2_config: Path,
    selected: list[dict[str, Any]],
    destination: Path,
) -> None:
    selected_by_group: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        selected_by_group.setdefault(str(row["target_set"]), []).append(row)
    lane_groups = {
        "A1": ("T6",),
        "A2": ("R0",),
        "A3": ("R7-confidence",),
        "A4": ("R7-all-six",),
        "A5": ("T6", "R0", "R7-confidence"),
        "A6": ("T6", "R0", "R7-all-six"),
    }
    base = json.loads(b2_config.read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=True)
    for lane, groups in lane_groups.items():
        config = json.loads(json.dumps(base))
        config["model_parameters"]["output_prefix"] = (
            f"xslim_dev_001a_{lane.lower()}_split_s8_qdq"
        )
        config["model_parameters"]["working_dir"] = "__RUN_WORKING_DIR__"
        custom: list[dict[str, Any]] = []
        seen: set[str] = set()
        for group in groups:
            for row in selected_by_group[group]:
                tensor = str(row["tensor"])
                if tensor in seen:
                    continue
                seen.add(tensor)
                custom.append(
                    {
                        "name": f"{lane}-{group}-{len(custom):02d}",
                        "tensor_names": [tensor],
                        "range_policy": {
                            "enabled": True,
                            "strict": True,
                            "lock_qparams": True,
                            **json.loads(str(row["selected_spec_json"])),
                        },
                    }
                )
        config["quantization_parameters"]["custom_setting"] = custom
        config["quantization_parameters"]["range_policy_manifest_path"] = (
            "__RUN_RANGE_POLICY_MANIFEST__"
        )
        (destination / f"{lane}.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def main() -> int:
    options = parse_args()
    if options.raw_dir.exists():
        raise RuntimeError("refusing to reuse preparation raw output root")
    expected_report = options.report_dir / "policy_target_sets.tsv"
    if expected_report.exists():
        raise RuntimeError(
            f"refusing to overwrite preparation report: {expected_report}"
        )
    options.report_dir.mkdir(parents=True, exist_ok=True)
    options.raw_dir.mkdir(parents=True)
    sample_dir = options.raw_dir / "samples"
    sample_dir.mkdir()

    fp32 = inferred_model(options.fp32_inference)
    b2 = inferred_model(options.b2_inference)
    all_targets = sorted(
        {tensor for values in TARGET_SETS.values() for tensor in values}
    )
    fp_records = {tensor: source_record(fp32, tensor) for tensor in all_targets}
    qdq_records = {tensor: qdq_record(b2, tensor) for tensor in all_targets}

    target_rows: list[dict[str, Any]] = []
    for target_set, tensors in TARGET_SETS.items():
        for tensor in tensors:
            target_rows.append(
                {
                    "target_set": target_set,
                    "tensor": tensor,
                    **fp_records[tensor],
                    **qdq_records[tensor],
                    "selector_rule": "exact-tensor-name-strict",
                    "source_evidence": "accepted-stage65b-r3-source-qdq-map",
                }
            )
    target_path = options.report_dir / "policy_target_sets.tsv"
    write_tsv(target_path, target_rows)
    (options.report_dir / "policy_target_manifest.sha256").write_text(
        f"{sha256(target_path)}  policy_target_sets.tsv\n", encoding="utf-8"
    )
    output_contract = {
        "profile": "spacemit_k1x_s8_qdq_split_v1",
        "outputs": [
            {
                "name": tensor,
                "shape": [int(item) for item in fp_records[tensor]["shape"].split("x")],
                "dtype": str(fp_records[tensor]["dtype"]).lower(),
            }
            for tensor in T6
        ],
    }
    (options.report_dir / "spacemit_output_contract.json").write_text(
        json.dumps(output_contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    diagnostic = extract_model(
        fp32, [item.name for item in fp32.graph.input], all_targets
    )
    diagnostic_path = options.raw_dir / "fp32-target-diagnostic.onnx"
    onnx.save_model(diagnostic, diagnostic_path)
    onnx.checker.check_model(onnx.load(diagnostic_path))
    runtime = session(diagnostic_path, options.threads)
    r1_audit_runtime = runtime if options.threads == 1 else session(diagnostic_path, 1)
    images = paths_from_list(options.h500_list, 0)
    accepted_r3_hashes = read_r3_hashes(options.r3_hashes)
    accepted_r1_hashes = read_r1_hashes(options.r1_boundary_audit)
    samples: dict[str, list[np.ndarray]] = {tensor: [] for tensor in all_targets}
    hash_rows: list[dict[str, Any]] = []
    for index, image_path in enumerate(images):
        image, _ = image_tensor_and_geometry(image_path)
        values = {
            meta.name: value
            for meta, value in zip(
                runtime.get_outputs(),
                runtime.run(None, {runtime.get_inputs()[0].name: image}),
            )
        }
        r1_values = {
            meta.name: value
            for meta, value in zip(
                r1_audit_runtime.get_outputs(),
                r1_audit_runtime.run(
                    None, {r1_audit_runtime.get_inputs()[0].name: image}
                ),
            )
        }
        if set(values) != set(all_targets) or set(r1_values) != set(all_targets):
            raise ValueError("diagnostic runtime output-name contract mismatch")
        for tensor in all_targets:
            value = values[tensor]
            if not np.isfinite(value).all():
                raise ValueError(f"non-finite FP32 activation: {tensor} {image_path}")
            observed_hash = sha256_array(value)
            r1_observed_hash = sha256_array(r1_values[tensor])
            expected_r3_hash = accepted_r3_hashes.get((tensor, image_path.name), "")
            expected_r1_hash = accepted_r1_hashes.get((tensor, image_path.name), "")
            if expected_r3_hash and observed_hash != expected_r3_hash:
                raise ValueError(
                    f"accepted R3 activation hash mismatch: {tensor} {image_path.name}"
                )
            if expected_r1_hash and r1_observed_hash != expected_r1_hash:
                raise ValueError(
                    f"accepted R1 activation hash mismatch: {tensor} {image_path.name}"
                )
            flat = value.reshape(-1)
            stride = max(1, flat.size // options.sample_per_image)
            samples[tensor].append(
                flat[::stride][: options.sample_per_image].astype(np.float32, copy=True)
            )
            hash_rows.append(
                {
                    "tensor": tensor,
                    "image_index": index,
                    "image": image_path.name,
                    "input_sha256": sha256_array(image),
                    "sampling_threads": options.threads,
                    "sampling_activation_sha256": observed_hash,
                    "accepted_r3_sha256": expected_r3_hash,
                    "accepted_r3_match": int(
                        not expected_r3_hash or observed_hash == expected_r3_hash
                    ),
                    "r1_audit_threads": 1,
                    "r1_audit_activation_sha256": r1_observed_hash,
                    "accepted_r1_sha256": expected_r1_hash,
                    "accepted_r1_match": int(
                        not expected_r1_hash or r1_observed_hash == expected_r1_hash
                    ),
                    "accepted_match": int(
                        (not expected_r3_hash or observed_hash == expected_r3_hash)
                        and (
                            not expected_r1_hash or r1_observed_hash == expected_r1_hash
                        )
                    ),
                }
            )
        if options.log_every and (index + 1) % options.log_every == 0:
            print(f"target sampling: {index + 1}/{len(images)}", flush=True)
    write_tsv(options.raw_dir / "activation-hashes.tsv", hash_rows)

    sample_paths: dict[str, Path] = {}
    activation_rows: list[dict[str, Any]] = []
    for target_index, tensor in enumerate(all_targets):
        values = np.concatenate(samples[tensor])
        path = sample_dir / stable_name(target_index)
        np.save(path, values, allow_pickle=False)
        sample_paths[tensor] = path
        activation_rows.append(
            {
                "tensor": tensor,
                "sample_file": str(path),
                "sample_sha256": sha256(path),
                "sample_count": values.size,
                "observed_min": float(values.min()),
                "observed_max": float(values.max()),
                "p0001": float(np.quantile(values.astype(np.float64), 0.0001)),
                "p001": float(np.quantile(values.astype(np.float64), 0.001)),
                "p50": float(np.quantile(values.astype(np.float64), 0.5)),
                "p999": float(np.quantile(values.astype(np.float64), 0.999)),
                "p9999": float(np.quantile(values.astype(np.float64), 0.9999)),
                "full_hash_reconciliation": "pass",
            }
        )
    write_tsv(
        options.report_dir / "policy_target_activation_evidence.tsv", activation_rows
    )
    write_tsv(
        options.report_dir / "policy_target_qparams_before.tsv",
        [{"tensor": tensor, **qdq_records[tensor]} for tensor in all_targets],
    )

    screening_rows: list[dict[str, Any]] = []
    for target_set, tensors in TARGET_SETS.items():
        for tensor in tensors:
            values = np.load(sample_paths[tensor], allow_pickle=False)
            for method in METHODS:
                if method == "default":
                    result = default_metrics(
                        values,
                        float(qdq_records[tensor]["scale"]),
                        int(qdq_records[tensor]["zero_point"]),
                    )
                    spec_json = "{}"
                else:
                    spec = method_spec(method, tensor, values)
                    result = select_qparams(values, spec).to_dict()
                    spec_json = json.dumps(
                        spec.to_dict(), sort_keys=True, separators=(",", ":")
                    )
                constraint_pass, required_min, required_max = constraint_status(
                    tensor, values, result
                )
                screening_rows.append(
                    {
                        "target_set": target_set,
                        "tensor": tensor,
                        "method": method,
                        "spec_json": spec_json,
                        **result,
                        "required_real_min_for_screen": required_min,
                        "required_real_max_for_screen": required_max,
                        "constraint_pass": int(constraint_pass),
                    }
                )
    write_tsv(options.report_dir / "observer_screening_results.tsv", screening_rows)

    selected_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    for target_set, tensors in TARGET_SETS.items():
        aggregate: list[tuple[tuple[Any, ...], str, list[dict[str, Any]]]] = []
        for method in METHODS:
            rows = [
                row
                for row in screening_rows
                if row["target_set"] == target_set and row["method"] == method
            ]
            eligible = len(rows) == len(tensors) and all(
                int(row["constraint_pass"]) for row in rows
            )
            key = (
                0 if eligible else 1,
                float(np.mean([float(row["normalized_mae"]) for row in rows])),
                float(np.mean([float(row["clipping_fraction"]) for row in rows])),
                method,
            )
            aggregate.append((key, method, rows))
            rejection_rows.append(
                {
                    "target_set": target_set,
                    "method": method,
                    "eligible": int(eligible),
                    "mean_normalized_mae": key[1],
                    "mean_clipping_fraction": key[2],
                    "decision": "eligible" if eligible else "rejected-constraint",
                }
            )
        key, method, rows = min(aggregate, key=lambda item: item[0])
        if key[0] != 0 or method == "default":
            raise RuntimeError(
                f"no non-default robust observer qualified for {target_set}: {method}"
            )
        for row in rows:
            selected_rows.append(
                {
                    "target_set": target_set,
                    "tensor": row["tensor"],
                    "selected_method": method,
                    "selected_spec_json": row["spec_json"],
                    "scale": row["scale"],
                    "zero_point": row["zero_point"],
                    "representable_min": row["representable_min"],
                    "representable_max": row["representable_max"],
                    "normalized_mae": row["normalized_mae"],
                    "cosine": row["cosine"],
                    "constraint_pass": row["constraint_pass"],
                    "selection_key": json.dumps(key, separators=(",", ":")),
                }
            )
    write_tsv(options.report_dir / "observer_selected_by_group.tsv", selected_rows)
    write_tsv(options.report_dir / "observer_screening_rejections.tsv", rejection_rows)
    candidate_configs(
        options.b2_config,
        selected_rows,
        options.report_dir / "candidate_effective_configs",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
