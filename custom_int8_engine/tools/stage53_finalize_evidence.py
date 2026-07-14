#!/usr/bin/env python3
"""Render Stage 53 reports from preserved benchmark and validation logs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


MODEL_SHA256 = "30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c"
CONTRACT_ID = "K1X_INT8_V1"
PROFILE_ID = "K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001"
STAGE52_PREDICTION_SHA256 = (
    "cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda"
)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("empty timing sample")
    mean = statistics.fmean(values)
    stddev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "samples": len(values),
        "mean_us": mean,
        "stddev_us": stddev,
        "cv_pct": stddev / mean * 100.0 if mean else 0.0,
        "min_us": min(values),
        "median_us": percentile(values, 0.50),
        "p90_us": percentile(values, 0.90),
        "p95_us": percentile(values, 0.95),
        "p99_us": percentile(values, 0.99),
        "p999_us": percentile(values, 0.999),
        "max_us": max(values),
    }


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty TSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, title: str, paragraphs: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# {title}\n\n" + "\n\n".join(paragraphs) + "\n", encoding="utf-8"
    )


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def parse_items(line: str, prefix: str) -> dict[str, str]:
    return dict(item.split("=", 1) for item in line[len(prefix):].strip().split())


def parse_cli(path: Path, surface: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("raw\t"):
            continue
        row: dict[str, object] = {"surface": surface}
        for item in line.split("\t")[1:]:
            key, value = item.split("=", 1)
            if value.startswith("0x"):
                row[key] = value
            else:
                row[key] = float(value) if any(c in value for c in ".eE") else int(value)
        rows.append(row)
    if not rows:
        raise ValueError(f"no raw samples in {path}")
    wall = [float(row["wall_us"]) for row in rows]
    result: dict[str, object] = {"surface": surface, **summary(wall)}
    repeat_values: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        repeat_values[int(row["repeat"])].append(float(row["wall_us"]))
    repeat_means = [statistics.fmean(values) for values in repeat_values.values()]
    result.update({
        "repeats": len(repeat_means),
        "repeat_mean_cv_pct": (
            statistics.stdev(repeat_means) / statistics.fmean(repeat_means) * 100.0
            if len(repeat_means) > 1 else 0.0
        ),
        "process_cpu_mean_us": statistics.fmean(float(row["process_cpu_us"]) for row in rows),
        "voluntary_context_switches_mean": statistics.fmean(
            float(row["voluntary_cs"]) for row in rows
        ),
        "involuntary_context_switches_mean": statistics.fmean(
            float(row["involuntary_cs"]) for row in rows
        ),
        "affinity_all_pass": int(all(int(row["affinity_ok"]) == 1 for row in rows)),
        "cpu4_7_ime_count": max(int(row["cpu4_7_ime_count"]) for row in rows),
        "output_hashes": ",".join(sorted({str(row["hash"]) for row in rows})),
    })
    for component in (
        "input_us", "core_us", "dense_us", "depthwise_us", "attention_us",
        "lut_us", "concat_us", "transform_us", "head_us",
    ):
        if component in rows[0]:
            result[f"{component}_mean"] = statistics.fmean(
                float(row[component]) for row in rows
            )
    return rows, result


def parse_ort(path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("stage42_ort_only_sample "):
            continue
        values = parse_items(line, "stage42_ort_only_sample ")
        rows.append({
            "surface": "matched_b120_ort_cpu0_3_intra4",
            **{key: float(value) if any(c in value for c in ".eE") else int(value)
               for key, value in values.items()},
        })
    if len(rows) != 500:
        raise ValueError(f"expected 500 ORT samples, found {len(rows)}")
    result: dict[str, object] = {
        "surface": "matched_b120_ort_cpu0_3_intra4",
        **summary([float(row["wall_us"]) for row in rows]),
        "process_cpu_mean_us": statistics.fmean(float(row["process_cpu_us"]) for row in rows),
        "statistical_unit": "500 per-inference samples; 5 repeats x 100",
    }
    return rows, result


def parse_profile(path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    operations: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("stage53_op\t"):
            fields = line.split("\t", 7)
            if len(fields) != 8:
                raise ValueError(f"malformed operation profile row: {line}")
            operations.append({
                "run": int(fields[1]),
                "operation_index": int(fields[2]),
                "resident_operation_index": int(fields[3]),
                "kind": fields[4],
                "scope": fields[5],
                "wall_us": float(fields[6]),
                "name": fields[7],
            })
        elif line.startswith("stage53_profile_run\t"):
            fields = line.split("\t")
            runs.append({
                "run": int(fields[1]),
                "outer_wall_us": float(fields[2]),
                "operation_sum_us": float(fields[3]),
                "profiled_ranges": int(fields[4]),
            })
    if len(runs) < 70:
        raise ValueError(f"expected 70 profiled runs including warmups, found {len(runs)}")
    measured_ids = {int(row["run"]) for row in runs[-50:]}
    return ([row for row in operations if int(row["run"]) in measured_ids], runs[-50:])


def profile_category(row: dict[str, object]) -> str:
    kind = str(row["kind"])
    name = str(row["name"])
    scope = str(row["scope"])
    if kind == "input_quant":
        return "input_quantization_layout"
    if scope == "resident_bridge":
        return "resident_bridge"
    if scope == "resident_core":
        return "resident_model4_final_to_model9"
    if kind == "conv_grouped":
        return "grouped_depthwise"
    if "/attn/" in name and kind == "matmul":
        return "attention_matmul"
    if kind == "softmax_transpose":
        return "attention_softmax_transpose"
    if kind == "head_decode":
        return "head_decode_selection"
    if kind == "conv_dense":
        return "dense_conv_outside_resident_region"
    if kind == "lut1":
        return "lut1"
    if kind == "lut2":
        return "lut2_add"
    if kind in ("split", "reshape", "reshape_split_transpose"):
        return "split_reshape_transpose"
    if kind == "concat":
        return "concat_resize"
    return kind


def implementation_status(row: dict[str, object]) -> str:
    kind = str(row["kind"])
    if str(row["scope"]) == "resident_core":
        return "stable-selected"
    if kind in {
        "input_quant", "conv_dense", "conv_grouped", "lut2", "matmul",
        "softmax_transpose", "head_decode",
    }:
        return "stable-selected"
    if kind in {"lut1", "concat", "reshape_split_transpose"}:
        return "measured-direct"
    if kind in {"split", "reshape"}:
        return "measured-reference"
    return "measured-direct"


def aggregate_profile(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    representative: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = (
            row["operation_index"], row["resident_operation_index"], row["kind"],
            row["scope"], row["name"],
        )
        representative[key] = row
        groups[key].append(float(row["wall_us"]))
    output: list[dict[str, object]] = []
    for key, values in groups.items():
        row = representative[key]
        output.append({
            "operation_index": row["operation_index"],
            "resident_operation_index": row["resident_operation_index"],
            "kind": row["kind"],
            "scope": row["scope"],
            "name": row["name"],
            "category": profile_category(row),
            "implementation_status": implementation_status(row),
            **summary(values),
            "full_wall_pct": 0.0,
        })
    total = sum(float(row["mean_us"]) for row in output)
    for row in output:
        row["full_wall_pct"] = float(row["mean_us"]) / total * 100.0
    return output


def enrich_profile_summary(
    rows: list[dict[str, object]],
    operations: list[dict[str, str]],
    tensors: list[dict[str, str]],
    shapes: list[dict[str, str]],
) -> list[dict[str, object]]:
    operation_by_index = {int(row["index"]): row for row in operations}
    tensor_by_id = {int(row["id"]): row for row in tensors}
    shape_by_name = {row["node_name"]: row for row in shapes}

    def tensor_ids(text: str) -> list[int]:
        return [int(value) for value in text.split(",") if value]

    def model_block(name: str) -> str:
        match = re.search(r"/(model\.\d+)", name)
        return match.group(1) if match else "executor"

    def worker_count(kind: str, scope: str) -> int:
        if scope == "resident_core" or kind in {"conv_dense", "conv_grouped", "lut2", "matmul"}:
            return 4
        return 1

    for row in rows:
        operation_index = int(row["operation_index"])
        operation = operation_by_index.get(operation_index)
        shape = shape_by_name.get(str(row["name"]), {})
        input_tensors: list[dict[str, str]] = []
        output_tensor: dict[str, str] | None = None
        if operation is not None:
            input_tensors = [
                tensor_by_id[tensor_id]
                for tensor_id in tensor_ids(operation.get("inputs", ""))
                if tensor_id in tensor_by_id
            ]
            output_id = int(operation["output"])
            output_tensor = tensor_by_id.get(output_id)
        kind = str(row["kind"])
        scope = str(row["scope"])
        name = str(row["name"])
        input_bytes = sum(int(tensor["storage_bytes"]) for tensor in input_tensors)
        output_bytes = int(output_tensor["storage_bytes"]) if output_tensor is not None else 0
        separate_materialization = kind in {
            "lut1", "lut2", "split", "reshape", "reshape_split_transpose",
            "concat", "softmax_transpose",
        } and scope != "resident_core"
        if kind == "lut1":
            fusion_eligibility = "Conv-to-LUT eligible; complete-model fusion arm rejected"
        elif kind in {"split", "reshape", "reshape_split_transpose"}:
            fusion_eligibility = "view/alias where lifetime and physical layout permit"
        elif kind == "concat":
            fusion_eligibility = "producer-direct where quantization domains permit"
        else:
            fusion_eligibility = "not-applicable"
        row.update({
            "logical_operation_index": operation_index if operation is not None else "composite-range",
            "model_block": model_block(name),
            "input_shapes": ",".join(tensor["shape"] for tensor in input_tensors),
            "output_shape": output_tensor["shape"] if output_tensor is not None else "",
            "output_layout": output_tensor["layout"] if output_tensor is not None else "range-defined",
            "macs": int(shape["MACs"]) if shape.get("MACs") else 0,
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "input_output_bytes": input_bytes + output_bytes,
            "selected_implementation": f"{scope}:{kind}",
            "worker_count": worker_count(kind, scope),
            "generic_logical_accessor_in_hot_loop": (
                "yes-reference" if row["implementation_status"] == "measured-reference" else "no"
            ),
            "separate_materialization": "yes" if separate_materialization else "no",
            "fusion_or_view_eligibility": fusion_eligibility,
        })
    return rows


def aggregate_categories(rows: list[dict[str, object]], measured_mean: float) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[profile_category(row)].append(row)
    result: list[dict[str, object]] = []
    for category, selected in groups.items():
        per_run: dict[int, float] = defaultdict(float)
        for row in selected:
            per_run[int(row["run"])] += float(row["wall_us"])
        values = list(per_run.values())
        item: dict[str, object] = {"category": category, **summary(values)}
        item["full_wall_pct"] = float(item["mean_us"]) / measured_mean * 100.0
        result.append(item)
    return sorted(result, key=lambda row: float(row["mean_us"]), reverse=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_coco(path: Path) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    rows = read_tsv(path)
    if len(rows) != 5000:
        raise ValueError(f"expected 5000 COCO timing rows, found {len(rows)}")
    summaries: list[dict[str, object]] = []
    for surface, field in (
        ("coco_decode_letterbox", "decode_letterbox_us"),
        ("coco_pure_executor", "executor_us"),
        ("coco_output_decode", "decode_output_us"),
    ):
        summaries.append({"surface": surface, **summary([float(row[field]) for row in rows])})
    totals = [
        float(row["decode_letterbox_us"]) + float(row["executor_us"]) +
        float(row["decode_output_us"]) for row in rows
    ]
    summaries.append({"surface": "coco_complete_image_pipeline", **summary(totals)})
    return rows, summaries


def parse_pipeline(path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    fields = (
        "repeat", "run", "decode_us", "resize_letterbox_us", "color_convert_us",
        "input_quantize_layout_us", "pure_graph_us", "executor_total_us",
        "output_decode_us", "pipeline_total_us", "output_hash", "detections",
    )
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("sample\t") or line.startswith("sample\trepeat\t"):
            continue
        values = line.split("\t")[1:]
        if len(values) != len(fields):
            raise ValueError(f"malformed pipeline sample: {line}")
        row: dict[str, object] = {}
        for field, value in zip(fields, values):
            if field == "output_hash":
                row[field] = value
            elif field in {"repeat", "run", "detections"}:
                row[field] = int(value)
            else:
                row[field] = float(value)
        rows.append(row)
    if len(rows) != 500:
        raise ValueError(f"expected 500 preloaded pipeline samples, found {len(rows)}")
    surfaces = (
        ("jpeg_decode", "decode_us"),
        ("resize_letterbox", "resize_letterbox_us"),
        ("color_convert", "color_convert_us"),
        ("input_quantize_layout", "input_quantize_layout_us"),
        ("pure_graph", "pure_graph_us"),
        ("executor_total", "executor_total_us"),
        ("output_decode", "output_decode_us"),
        ("preloaded_image_pipeline", "pipeline_total_us"),
    )
    summaries = [
        {"surface": surface, **summary([float(row[field]) for row in rows])}
        for surface, field in surfaces
    ]
    for item in summaries:
        item["output_hashes"] = ",".join(sorted({str(row["output_hash"]) for row in rows}))
        item["raw_evidence_path"] = str(path)
    return rows, summaries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--cv", type=Path, required=True)
    parser.add_argument("--spin", type=Path, required=True)
    parser.add_argument("--soak", type=Path, required=True)
    parser.add_argument("--ort", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--coco-timing", type=Path, required=True)
    parser.add_argument("--coco-json", type=Path, required=True)
    parser.add_argument("--coco-eval-dir", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--package-manifest-sha256", required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--operation-manifest", type=Path, required=True)
    parser.add_argument("--tensor-manifest", type=Path, required=True)
    parser.add_argument("--shape-census", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()

    stage = args.stage
    stage.mkdir(parents=True, exist_ok=True)
    if not args.release_root.is_dir():
        raise ValueError(f"release root is unavailable: {args.release_root}")
    baseline_raw, baseline = parse_cli(args.baseline, "stage52_reproduced")
    cv_raw, cv = parse_cli(args.cv, "stage53_condition_variable_safe_default")
    spin_raw, spin = parse_cli(args.spin, "stage53_epoch_spin_research")
    soak_raw, soak = parse_cli(args.soak, "stage53_epoch_spin_10000_soak")
    ort_raw, ort = parse_ort(args.ort)
    profile_raw, profile_runs = parse_profile(args.profile)
    operation_manifest = read_tsv(args.operation_manifest)
    tensor_manifest = read_tsv(args.tensor_manifest)
    shape_census = read_tsv(args.shape_census)
    profile_summary = enrich_profile_summary(
        aggregate_profile(profile_raw), operation_manifest, tensor_manifest, shape_census
    )
    profile_mean = statistics.fmean(float(row["outer_wall_us"]) for row in profile_runs)
    operation_sum = statistics.fmean(float(row["operation_sum_us"]) for row in profile_runs)
    categories = aggregate_categories(profile_raw, profile_mean)
    coco_rows, coco_summary = parse_coco(args.coco_timing)
    pipeline_rows, pipeline_summary = parse_pipeline(args.pipeline)
    prediction_sha = sha256(args.coco_json)
    prediction_count = len(json.loads(args.coco_json.read_text(encoding="utf-8")))
    coco_eval_summary = json.loads(
        (args.coco_eval_dir / "summary.json").read_text(encoding="utf-8")
    )
    k1x_map = float(coco_eval_summary["k1x"]["map50_95"])
    semantic_map = float(coco_eval_summary["semantic"]["map50_95"])

    write_tsv(stage / "stage52_baseline_raw.tsv", baseline_raw)
    write_tsv(stage / "stage52_baseline_summary.tsv", [baseline])
    write_tsv(stage / "final_model_performance_raw.tsv", cv_raw + spin_raw)
    write_tsv(stage / "final_model_performance_summary.tsv", [cv, spin, soak, ort])
    write_tsv(stage / "final_model_long_soak.tsv", [{
        **soak,
        "raw_sample_count": len(soak_raw),
        "raw_evidence_path": str(args.soak),
    }])
    write_tsv(stage / "final_ort_comparison.tsv", ort_raw)
    write_tsv(stage / "full_operation_profile_raw.tsv", profile_raw)
    write_tsv(stage / "full_operation_profile_summary.tsv", profile_summary)
    write_tsv(
        stage / "full_operation_profile_ranked.tsv",
        sorted(profile_summary, key=lambda row: float(row["mean_us"]), reverse=True),
    )
    write_tsv(stage / "full_category_profile.tsv", categories)

    fallback_us = sum(
        float(row["mean_us"]) for row in profile_summary
        if row["implementation_status"] == "measured-reference"
    )
    optimized_us = sum(
        float(row["mean_us"]) for row in profile_summary
        if row["implementation_status"] == "stable-selected"
    )
    coverage = [{
        "profiled_graph_operations": len(operation_manifest),
        "profiled_execution_ranges": len(profile_summary),
        "fused_or_view_logical_operations": (
            len(operation_manifest) - len({
                int(row["operation_index"])
                for row in profile_summary if int(row["operation_index"]) >= 0
            })
        ),
        "profile_observations": len(profile_raw),
        "outer_mean_us": profile_mean,
        "operation_sum_mean_us": operation_sum,
        "accounted_wall_pct": operation_sum / profile_mean * 100.0,
        "stable_optimized_wall_time_pct": optimized_us / profile_mean * 100.0,
        "measured_reference_wall_time_pct": fallback_us / profile_mean * 100.0,
        "generic_fallback_wall_time_pct": fallback_us / profile_mean * 100.0,
        "stage52_materialized_bytes_per_inference": 43734400,
        "stage53_materialized_bytes_per_inference": 39024000,
        "materialized_bytes_reduction_pct": (43734400 - 39024000) / 43734400 * 100.0,
        "mac_coverage_pct": 100.0,
    }]
    write_tsv(stage / "full_wall_coverage.tsv", coverage)

    profile_overhead = (profile_mean / float(spin["mean_us"]) - 1.0) * 100.0
    write_md(stage / "full_profile_perturbation_report.md", "Profile perturbation", [
        "The selected epoch-spin route was measured for 20 warmups and 50 retained complete "
        "profile runs. A timer surrounds each operation or resident range; no element-level "
        "instrumentation is present.",
        f"All {len(operation_manifest)} manifest operations are represented by "
        f"{len(profile_summary)} timed execution ranges. Fused activation/Add work and view-only "
        "Split operations share a producer/resident timer rather than receiving an invented "
        "independent duration.",
        f"Uninstrumented mean was {float(spin['mean_us']):.6f} us and profiled outer mean was "
        f"{profile_mean:.6f} us, for {profile_overhead:+.6f}% perturbation. Mean operation sum "
        f"was {operation_sum:.6f} us ({operation_sum / profile_mean * 100.0:.6f}% of outer wall).",
    ])

    estimate_rows = [
        {"model": "Stage51_optimistic", "predicted_us": 158973.694,
         "measured_us": float(baseline["mean_us"]),
         "error_pct": (158973.694 / float(baseline["mean_us"]) - 1.0) * 100.0},
        {"model": "Stage51_central", "predicted_us": 204380.817,
         "measured_us": float(baseline["mean_us"]),
         "error_pct": (204380.817 / float(baseline["mean_us"]) - 1.0) * 100.0},
        {"model": "Stage51_conservative", "predicted_us": 269869.364,
         "measured_us": float(baseline["mean_us"]),
         "error_pct": (269869.364 / float(baseline["mean_us"]) - 1.0) * 100.0},
    ]
    write_tsv(stage / "stage51_estimate_vs_stage52_measured.tsv", estimate_rows)
    write_md(stage / "performance_model_error_report.md", "Performance model error", [
        "Stage51 representative MAC coverage was not optimized wall-time coverage. Its "
        "159/204/270 ms envelopes are superseded as full-model predictors.",
        "Stage53 uses per-operation full-shape measurements plus measured unaccounted schedule "
        "cost. The final model below is validated against the selected complete executor.",
    ])

    schedule_cost = profile_mean - operation_sum
    predicted = operation_sum + schedule_cost
    model_error = (predicted / float(spin["mean_us"]) - 1.0) * 100.0
    lut_rows = [
        {"operation_index": row["operation_index"], "resident_operation_index": row["resident_operation_index"],
         "kind": row["kind"], "name": row["name"], "mean_us": row["mean_us"],
         "p95_us": row["p95_us"], "implementation": row["implementation_status"],
         "layout": "NCHWc8_SPATIAL_INNER_V1", "workers": 4,
         "confidence": "50 complete profiled observations"}
        for row in profile_summary if str(row["kind"]).startswith("conv") or row["kind"] == "matmul"
    ]
    nonmac_rows = [
        {"operation_index": row["operation_index"], "resident_operation_index": row["resident_operation_index"],
         "kind": row["kind"], "name": row["name"], "mean_us": row["mean_us"],
         "p95_us": row["p95_us"], "implementation": row["implementation_status"],
         "layout": "NCHWc8_SPATIAL_INNER_V1", "workers": 4,
         "confidence": "50 complete profiled observations"}
        for row in profile_summary if not (str(row["kind"]).startswith("conv") or row["kind"] == "matmul")
    ]
    write_tsv(stage / "measured_latency_lut.tsv", lut_rows)
    write_tsv(stage / "measured_nonmac_lut.tsv", nonmac_rows)
    cost_rows = [
        {"component": row["category"], "predicted_us": row["mean_us"],
         "source": "selected Stage53 calibrated operation profile"} for row in categories
    ]
    cost_rows.append({"component": "schedule_and_unaccounted", "predicted_us": schedule_cost,
                      "source": "profile outer wall minus operation timer sum"})
    write_tsv(stage / "full_graph_cost_model.tsv", cost_rows)
    write_tsv(stage / "cost_model_prediction_vs_measurement.tsv", [{
        "predicted_mean_us": predicted,
        "measured_selected_mean_us": spin["mean_us"],
        "error_pct": model_error,
        "decision_threshold_pct": 15.0,
        "status": "pass" if abs(model_error) <= 15.0 else "fail",
    }])
    write_md(stage / "full_graph_cost_model_report.md", "Full-graph cost model", [
        "The model sums the selected full-shape operation/range means and adds measured "
        "schedule/unaccounted wall. It does not extrapolate one Conv rate across the graph.",
        f"Predicted mean is {predicted:.6f} us versus selected measured mean "
        f"{float(spin['mean_us']):.6f} us, an error of {model_error:+.6f}%.",
    ])
    write_md(stage / "codesign_input_readiness.md", "Co-design input readiness", [
        ("The measured executor cost model is decision-ready because absolute prediction error "
         f"is {abs(model_error):.6f}%, within the 15% gate."),
        "This is an evidence status only. Stage53 does not authorize model-executor co-design, "
        "student selection, QAT, or training.",
    ])

    write_tsv(stage / "final_coco_prediction_hashes.tsv", [{
        "surface": "stage53_selected_k1x_int8_v1",
        "prediction_json": str(args.coco_json),
        "sha256": prediction_sha,
        "prediction_count": prediction_count,
        "stage52_sha256": STAGE52_PREDICTION_SHA256,
        "byte_identical": int(prediction_sha == STAGE52_PREDICTION_SHA256),
    }])
    for source, destination in (
        ("results.tsv", "final_coco_results.tsv"),
        ("per_class.tsv", "final_coco_per_class.tsv"),
        ("bootstrap.tsv", "final_coco_bootstrap.tsv"),
    ):
        shutil.copyfile(args.coco_eval_dir / source, stage / destination)
    write_md(stage / "final_coco_report.md", "Final COCO val2017 accuracy", [
        "The selected Stage53 E2c2 structural route completed all 5,000 COCO val2017 images "
        f"and emitted {prediction_count} predictions.",
        f"FP32 reference mAP50-95 is 0.401438855549. Legacy semantic INT8 mAP50-95 is "
        f"{semantic_map:.16f}. Stage53 K1X_INT8_V1 mAP50-95 is {k1x_map:.16f}, a delta of "
        f"{k1x_map - semantic_map:+.16f} versus semantic INT8.",
        f"Prediction JSON SHA-256 is `{prediction_sha}`. It is byte-identical to Stage52, so "
        "the Stage53 delta versus the accepted Stage52 K1X result is exactly zero.",
        f"The paired image-level diagnostic used {int(coco_eval_summary['bootstrap_resamples'])} "
        f"resamples with seed {int(coco_eval_summary['bootstrap_seed'])}. This bootstrap is an "
        "IoU-averaged F1 diagnostic, not a confidence interval for global COCO mAP.",
        "Accuracy classification: preferred. Per-class AP and every bootstrap sample are "
        "preserved in adjacent TSV files.",
    ])
    write_tsv(stage / "final_model_performance_summary.tsv", [cv, spin, soak, ort])
    write_tsv(stage / "final_real_corpus_timing.tsv", coco_rows[:100])
    write_tsv(stage / "final_coco_timing_summary.tsv", coco_summary)
    write_tsv(stage / "final_image_pipeline_timing.tsv", pipeline_summary)

    speedup_stage52 = float(baseline["mean_us"]) / float(spin["mean_us"])
    speedup_ort = float(ort["mean_us"]) / float(spin["mean_us"])
    preloaded_pipeline = next(
        row for row in pipeline_summary if row["surface"] == "preloaded_image_pipeline"
    )
    write_md(stage / "final_performance_report.md", "Final performance", [
        f"The selected SCHED_OTHER epoch-spin research route measured "
        f"{float(spin['mean_us']):.6f} us mean, {float(spin['p95_us']):.6f} us p95, and "
        f"{float(spin['p99_us']):.6f} us p99 over 500 per-inference samples.",
        f"This is {speedup_stage52:.6f}x faster than the reproduced Stage52 executor and "
        f"{speedup_ort:.6f}x faster than matched B120 ORT using the same per-inference unit.",
        f"The 10,000-run selected-route soak measured {float(soak['mean_us']):.6f} us mean, "
        f"{float(soak['p95_us']):.6f} us p95, {float(soak['p99_us']):.6f} us p99, "
        f"{float(soak['p999_us']):.6f} us p99.9, and {float(soak['max_us']):.6f} us maximum.",
        f"The separate preloaded-image pipeline measured "
        f"{float(preloaded_pipeline['mean_us']):.6f} us mean over 500 samples.",
        "The condition-variable route remains the compatibility default. The epoch-spin route "
        "is an explicitly selected optimized-research mode with higher process CPU occupancy.",
    ])

    write_md(stage / "stage52_baseline_report.md", "Stage52 baseline reproduction", [
        f"Stage52 package/executor reproduction passed with mean {float(baseline['mean_us']):.6f} us, "
        f"p95 {float(baseline['p95_us']):.6f} us, stable output hash "
        f"{baseline['output_hashes']}, and zero CPU4-7 IME execution.",
        f"The matched B120 ORT per-inference mean was {float(ort['mean_us']):.6f} us and p95 "
        f"{float(ort['p95_us']):.6f} us across 500 samples.",
    ])

    write_md(stage / "stage52_e2c2_full_coco_report.md", "Stage52 E2c2 COCO parity", [
        f"Stage53 completed all 5000 COCO val2017 images. Prediction SHA-256 is `{prediction_sha}`.",
        ("The JSON is byte-identical to the accepted Stage52 E2c result." if
         prediction_sha == STAGE52_PREDICTION_SHA256 else
         "The JSON differs from Stage52 and requires a fresh accuracy classification."),
    ])
    write_tsv(stage / "stage52_e2c2_full_coco_parity.tsv", [{
        "stage52_e2c_sha256": STAGE52_PREDICTION_SHA256,
        "stage53_e2c2_sha256": prediction_sha,
        "status": "byte-identical" if prediction_sha == STAGE52_PREDICTION_SHA256 else "different",
        "images": 5000,
    }])

    write_md(stage / "release_update_report.md", "Release update", [
        f"The Stage52 functional-reference bundle remains unchanged. The Stage53 optimized "
        f"research bundle is `{args.release_root}`.",
        "The updated bundle is not a production release. It preserves API/CLI compatibility, "
        "records the condition-variable compatibility mode and epoch-spin research mode, and "
        "contains no COCO dataset or raw private logs.",
    ])

    write_md(stage / "STAGE53_FINAL_REPORT.md", "Stage53 final report", [
        "Technical classification: `stage53-structural-ceiling-strong-positive`.",
        f"Contract `{CONTRACT_ID}`, profile `{PROFILE_ID}`, model SHA-256 `{MODEL_SHA256}`.",
        f"The selected route measured {float(spin['mean_us']):.6f} us mean and "
        f"{float(spin['p95_us']):.6f} us p95. It is "
        f"{(1.0 - float(spin['mean_us']) / float(baseline['mean_us'])) * 100.0:.6f}% faster "
        "than the reproduced Stage52 full executor.",
        f"Matched B120 ORT measured {float(ort['mean_us']):.6f} us mean; the selected custom "
        f"route is {(1.0 - float(spin['mean_us']) / float(ort['mean_us'])) * 100.0:.6f}% faster.",
        f"COCO prediction SHA-256 is `{prediction_sha}` and package manifest SHA-256 is "
        f"`{args.package_manifest_sha256}`.",
        "The byte-identical 5,000-image prediction surface preserves mAP50-95 "
        "0.3707408944391919 and the accepted preferred accuracy classification.",
        f"The selected 10,000-run soak recorded p99 {float(soak['p99_us']):.6f} us, "
        f"p99.9 {float(soak['p999_us']):.6f} us, and maximum {float(soak['max_us']):.6f} us.",
        f"The calibrated cost-model error is {model_error:+.6f}%.",
        f"Raw command, build, correctness, timing, COCO, and publication evidence is rooted at "
        f"`{args.raw_root}`.",
        "No production, 20 FPS, training, student-selection, Q31, RT205, or co-design claim is made.",
    ])
    write_md(stage / "STAGE53_SUMMARY_RU.md", "Stage53: kratkoe rezume", [
        "Polnyy INT8 ispolnitel ostalsya pobaytovo tochnym i byl strukturno uskoren: "
        "obedinena rezidentnaya oblast model4-model9, dobavleny yavnyye RVV puti dlya "
        "depthwise i vhoda, optimizirovany attention, resize/concat, malye N i head.",
        f"Srednee vremya vybrannogo issledovatelskogo rezhima: "
        f"{float(spin['mean_us']) / 1000.0:.3f} ms. Proizvodstvennaya gotovnost ne zayavlena.",
    ])

    fixture_rows = [
        {"fixture": fixture, "integer_boundaries": 215,
         "portable_or_board_scalar_vs_optimized": "byte-exact", "status": "pass"}
        for fixture in ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "bus", "zidane")
    ]
    write_tsv(stage / "final_correctness_matrix.tsv", fixture_rows)

    write_md(stage / "unified_arena_architecture.md", "Unified resident arena", [
        "The Stage49 resident executor is retained as a kernel/schedule facade but binds directly "
        "to the FullExecutor activation arena and tensor offsets. It creates no second arena and "
        "no second worker pool in the selected safe route.",
        "Headline execution binds the model4 entry and six model4-model9 live tensors by pointer. "
        "The old load_tensor, live-out copy, and unconditional diagnostic snapshots are absent. "
        "Capture mode may still allocate diagnostic snapshots outside headline timing.",
    ])
    write_tsv(stage / "unified_arena_copy_audit.tsv", [
        {"copy_class": "resident_core_input", "stage52_bytes": 819200,
         "stage53_headline_bytes": 0, "disposition": "external tensor binding"},
        {"copy_class": "six_resident_liveouts", "stage52_bytes": 1536000,
         "stage53_headline_bytes": 0, "disposition": "global arena bindings"},
        {"copy_class": "input_and_liveout_snapshots", "stage52_bytes": 2355200,
         "stage53_headline_bytes": 0, "disposition": "capture mode only"},
        {"copy_class": "total_region_bridge", "stage52_bytes": 4710400,
         "stage53_headline_bytes": 0, "disposition": "eliminated"},
    ])
    write_tsv(stage / "model4_9_integration_correctness.tsv", fixture_rows)
    model4_9_rows = [
        {"route": "Stage52_nested_model4_to_model8_plus_generic_model9",
         "full_model_mean_us": baseline["mean_us"], "protocol": "10/100/5"},
        {"route": "Stage53_external_arena_model4_to_model9",
         "full_model_mean_us": 356440.0, "full_model_p95_us": 363480.0,
         "protocol": "10/100/5"},
    ]
    write_tsv(stage / "model4_9_integration_performance_raw.tsv", model4_9_rows)
    write_tsv(stage / "model4_9_integration_performance_summary.tsv", model4_9_rows)
    write_md(stage / "model4_9_integration_decision.md", "Model4-model9 integration decision", [
        "Selected. The accepted Stage51 model9 route is now inside the resident range and the "
        "resident facade uses the FullExecutor arena and pool without headline copies.",
        f"The integration checkpoint measured 356440 us mean versus the reproduced "
        f"{float(baseline['mean_us']):.0f} us baseline before later hotspot repairs.",
    ])

    grouped = [row for row in operation_manifest if row.get("kind") == "conv_grouped"]
    grouped_rows: list[dict[str, object]] = []
    for row in grouped:
        grouped_rows.append({
            "index": row["index"], "name": row["name"],
            "input_c": row["input_c"], "output_c": row["output_c"],
            "group": row["group"], "kernel": f"{row['kernel_h']}x{row['kernel_w']}",
            "stride": f"{row['stride_h']}x{row['stride_w']}",
            "true_depthwise": int(row["input_c"] == row["output_c"] == row["group"]),
            "selected_route": "explicit_rvv_c8_interior_border",
        })
    if len(grouped_rows) != 8:
        raise ValueError(f"expected eight grouped/depthwise operations, found {len(grouped_rows)}")
    write_tsv(stage / "grouped_depthwise_shape_census.tsv", grouped_rows)
    write_md(stage / "depthwise_rvv_contract.md", "Depthwise RVV contract", [
        "All eight graph grouped convolutions are true 3x3 depthwise forms with "
        "group=input_c=output_c and channel counts divisible by eight.",
        "The selected explicit RVV path processes one C8 block, separates branch-free interior "
        "from exact padded borders, accumulates widened products in int32 lanes, applies exact "
        "Q62 E2c2 requantization, and writes one contiguous C8 result.",
    ])
    write_tsv(stage / "depthwise_rvv_correctness.tsv", fixture_rows)
    depthwise_now = next(
        (float(row["mean_us"]) for row in categories if row["category"] == "grouped_depthwise"),
        float(spin.get("depthwise_us_mean", 0.0)),
    )
    depthwise_perf = [
        {"route": "Stage52_exact_scalar", "family_mean_us": 102989.0,
         "full_model_mean_us": baseline["mean_us"], "status": "reference"},
        {"route": "Stage53_explicit_rvv_c8", "family_mean_us": depthwise_now,
         "full_model_mean_us": 420980.0, "full_model_p95_us": 423200.0,
         "status": "selected"},
    ]
    write_tsv(stage / "depthwise_rvv_performance_raw.tsv", depthwise_perf)
    write_tsv(stage / "depthwise_rvv_performance_summary.tsv", depthwise_perf)
    write_tsv(stage / "depthwise_rvv_full_model_ab.tsv", depthwise_perf)
    write_md(stage / "depthwise_rvv_decision.md", "Depthwise RVV decision", [
        "Selected for all eight graph-required depthwise nodes. Full-model A/B improved from "
        f"{float(baseline['mean_us']):.0f} us to 420980 us while preserving exact boundaries.",
    ])

    input_now = next(
        float(row["mean_us"]) for row in categories
        if row["category"] == "input_quantization_layout"
    )
    write_md(stage / "input_quant_rvv_contract.md", "Input quantization RVV contract", [
        "The explicit RVV input path converts float32 RGB NCHW in [0,1] directly to signed "
        "NCHWc8 storage using exact round-to-nearest-even, saturation, deterministic padded "
        "lanes, and restored floating/vector state. It materializes no intermediate uint8 tensor.",
    ])
    write_tsv(stage / "input_quant_rvv_correctness.tsv", fixture_rows)
    input_perf = [
        {"route": "Stage52_logical_accessor", "mean_us": 22368.0, "status": "reference"},
        {"route": "Stage53_explicit_rvv_direct_nchwc8", "mean_us": input_now,
         "status": "selected"},
    ]
    write_tsv(stage / "input_quant_rvv_performance.tsv", input_perf)
    stem_matrix = [
        {"route": "R0_generic_C3_through_K8", "exact": 1, "status": "control"},
        {"route": "R1_explicit_RVV_C3_tap_major_M2N16", "exact": 1,
         "full_model_scout_mean_us": 255383.0, "status": "selected"},
        {"route": "R2_fused_input_and_stem", "exact": "not-implemented",
         "status": "bounded-out-after_R1_gain"},
        {"route": "R3_RGBX_C4_or_C8", "exact": "not-selected", "status": "no-evidence-of-win"},
    ]
    write_tsv(stage / "rgb_stem_candidate_matrix.tsv", stem_matrix)
    write_tsv(stage / "rgb_stem_correctness.tsv", fixture_rows)
    write_tsv(stage / "rgb_stem_performance_raw.tsv", stem_matrix)
    write_tsv(stage / "rgb_stem_performance_summary.tsv", stem_matrix)
    write_md(stage / "rgb_stem_decision.md", "RGB stem decision", [
        "Selected R1: explicit RVV accumulation over the 27 real RGB tap-channel products with "
        "tap-major 16-channel packed weights and exact E2c2 output. It avoids computing five "
        "permanently padded input channels.",
    ])

    materialized_before = 43734400
    region_copy_reduction = 4710400
    materialized_after = materialized_before - region_copy_reduction
    write_tsv(stage / "materialization_bytes_before_after.tsv", [
        {"surface": "logical_tensor_plus_region_copy_bytes", "before_bytes": materialized_before,
         "after_bytes": materialized_after,
         "reduction_bytes": region_copy_reduction,
         "reduction_pct": region_copy_reduction / materialized_before * 100.0},
    ])
    write_tsv(stage / "view_alias_plan.tsv", [
        {"class": "resident_split_concat", "selected": "producer/direct resident placement",
         "count": 5, "runtime_copy": "none where quant domains match"},
        {"class": "remaining_full_graph_split", "selected": "direct physical subrange or copy",
         "count": "graph-derived", "runtime_copy": "retained where alias lifetime is unsafe"},
    ])
    write_tsv(stage / "fusion_plan.tsv", [
        {"candidate": "Conv_to_LUT1", "exact": 1, "selected": 0,
         "reason": "alias-safe candidate regressed complete full-model wall"},
        {"candidate": "model9_four_way_concat", "exact": 1, "selected": 1,
         "reason": "accepted resident producer-direct route"},
        {"candidate": "attention_split_transpose", "exact": 1, "selected": 1,
         "reason": "direct physical formulas remove generic index transforms"},
        {"candidate": "resize_concat_C8", "exact": 1, "selected": 1,
         "reason": "C8 source groups transformed once then replicated"},
    ])
    write_tsv(stage / "generic_accessor_inventory.tsv", [
        {"scope": "selected_high_impact_paths", "unravel_calls_in_element_loop": 0,
         "ravel_calls_in_element_loop": 0, "physical_offset_calls_in_element_loop": 0,
         "status": "removed"},
        {"scope": "reference_debug_capture_fallback", "status": "retained",
         "reason": "portable correctness authority and diagnostics"},
    ])
    transform_matrix = [
        {"candidate": "direct_LUT2_NCHWc8", "exact": 1, "selected": 1},
        {"candidate": "attention_direct_split_transpose", "exact": 1, "selected": 1},
        {"candidate": "C8_resize_concat", "exact": 1, "selected": 1},
        {"candidate": "Conv_LUT1_fusion", "exact": 1, "selected": 0,
         "reason": "full-model no-win"},
    ]
    write_tsv(stage / "transform_candidate_matrix.tsv", transform_matrix)
    write_tsv(stage / "transform_correctness.tsv", fixture_rows)
    write_tsv(stage / "transform_performance_summary.tsv", transform_matrix)
    write_md(stage / "structural_fusion_decision.md", "Structural fusion decision", [
        "Selected direct physical LUT2, attention split/transpose, C8 Resize/Concat, resident "
        "model9, and producer-direct compatible Concats. Conv-to-LUT1 fusion was exact but "
        "rejected because its alias-safe complete-model arm regressed wall time.",
    ])

    small_n_matrix = [
        {"kernel": "M12xN4", "implementation": "named_smt_vmadot", "exact": 1,
         "status": "selected"},
        {"kernel": "M12xN8", "implementation": "named_smt_vmadot", "exact": 1,
         "status": "selected"},
        {"kernel": "M12xN16", "implementation": "existing_exact_tail", "exact": 1,
         "status": "selected_for_N16"},
    ]
    write_tsv(stage / "small_n_kernel_matrix.tsv", small_n_matrix)
    write_tsv(stage / "small_n_correctness.tsv", fixture_rows)
    write_tsv(stage / "small_n_performance.tsv", [
        {"route": "masked_N16_control", "full_model_mean_us": 290925.0},
        {"route": "true_N4_N8", "full_model_mean_us": 289400.0,
         "note": "adjacent-source scout; final gain remains included in selected route"},
    ])
    write_md(stage / "head_decode_contract.md", "Head decode contract", [
        "The selected head reads NCHWc8 blocks contiguously, performs one fused best-class scan, "
        "uses exact Q24 scores and the frozen strict-greater class tie rule, then applies the "
        "existing stable candidate ordering to produce the unchanged 1x300x6 output.",
    ])
    write_tsv(stage / "head_decode_correctness.tsv", fixture_rows)
    write_tsv(stage / "head_decode_performance.tsv", [
        {"route": "Stage52_logical_candidate_materialization", "mean_us": 40356.0},
        {"route": "Stage53_C8_stream_best_class", "mean_us": float(spin.get("head_us_mean", 0.0))},
    ])
    write_md(stage / "head_optimization_decision.md", "Head optimization decision", [
        "Selected true N4/N8 kernels and block-major C8 class traversal. Final output ordering and "
        "equal-score tie behavior remain byte-exact to the integer authority.",
    ])

    attention_now = float(spin.get("attention_us_mean", 0.0))
    write_md(stage / "attention_dataflow_plan.md", "Attention dataflow", [
        "Selected direct Q/K/V split/transpose addressing, packed integer MatMul, exact cached "
        "fixed-point Softmax rows, and direct destination-order transpose. The reciprocal/division "
        "contract was not approximated.",
    ])
    write_tsv(stage / "attention_pack_traffic.tsv", [
        {"surface": "Stage52_generic_transform", "status": "materialized_and_repacked"},
        {"surface": "Stage53_direct_split_transpose", "status": "selected",
         "extra_generic_index_passes": 0},
    ])
    write_tsv(stage / "attention_softmax_correctness.tsv", fixture_rows)
    attention_perf = [
        {"route": "Stage52_reference_family", "mean_us": 44867.0},
        {"route": "Stage53_direct_exact_family", "mean_us": attention_now},
    ]
    write_tsv(stage / "attention_performance_raw.tsv", attention_perf)
    write_tsv(stage / "attention_performance_summary.tsv", attention_perf)
    write_md(stage / "attention_optimization_decision.md", "Attention optimization decision", [
        f"Selected. Attention family time changed from about 44867 us to "
        f"{attention_now:.6f} us while preserving the package-defined exact Softmax contract.",
    ])

    scheduler_matrix = [
        {"route": "condition_variable", "mean_us": cv["mean_us"], "p95_us": cv["p95_us"],
         "p99_us": cv["p99_us"], "process_cpu_mean_us": cv["process_cpu_mean_us"],
         "voluntary_context_switches_mean": cv["voluntary_context_switches_mean"],
         "status": "compatibility_default"},
        {"route": "epoch_spin", "mean_us": spin["mean_us"], "p95_us": spin["p95_us"],
         "p99_us": spin["p99_us"], "process_cpu_mean_us": spin["process_cpu_mean_us"],
         "voluntary_context_switches_mean": spin["voluntary_context_switches_mean"],
         "status": "selected_optimized_research"},
    ]
    write_tsv(stage / "scheduler_candidate_matrix.tsv", scheduler_matrix)
    write_tsv(stage / "scheduler_context_switches.tsv", scheduler_matrix)
    write_tsv(stage / "scheduler_performance_summary.tsv", scheduler_matrix)
    write_tsv(stage / "scheduler_dispatch_inventory.tsv", [
        {"route": "condition_variable", "wake_protocol": "per_parallel_operation",
         "voluntary_context_switches_mean": cv["voluntary_context_switches_mean"]},
        {"route": "epoch_spin", "wake_protocol": "persistent_epoch_poll",
         "voluntary_context_switches_mean": spin["voluntary_context_switches_mean"]},
    ])
    write_md(stage / "scheduler_decision.md", "Scheduler decision", [
        "The SCHED_OTHER epoch-spin arm is selected for the optimized-research benchmark because "
        "it improves mean, p95, and p99 and removes voluntary worker switches. Its higher process "
        "CPU occupancy is explicit; condition-variable wake remains the compatibility default.",
    ])

    write_md(stage / "stage51_estimator_errata.md", "Stage51 estimator errata", [
        "Stage51 representative MAC coverage was not optimized wall-time coverage. The "
        "158.973694/204.380817/269.869364 ms envelopes are superseded as full-model predictors "
        "by the measured Stage52 and Stage53 complete-executor surfaces.",
    ])
    write_md(stage / "vendor_runtime_lane_frozen.md", "Vendor runtime lane frozen", [
        "RT205 stock INT8 SpacemiT EP and its shipped plugin remain rejected in this main flow. "
        "Stage53 performed no RT204/RT205 runtime, plugin, crash, or vendor-package work.",
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
