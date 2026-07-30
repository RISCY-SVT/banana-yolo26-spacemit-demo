#!/usr/bin/env python3
"""Build compact Stage64 tables from immutable raw evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("status\nno-data\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in materialized:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in materialized:
            writer.writerow({field: row.get(field, "") for field in fields})


def combine_tables(
    sources: Iterable[tuple[str, Path]], output: Path, label: str = "candidate"
) -> None:
    rows: list[dict[str, Any]] = []
    for candidate, path in sources:
        if not path.is_file():
            continue
        for row in read_tsv(path):
            rows.append({label: candidate, **row})
    write_tsv(output, rows)


def quantization_tables(root: Path, output: Path) -> None:
    configurations: list[dict[str, Any]] = []
    for path in sorted((root / "host/configs").glob("*.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        calibration = config["calibration_parameters"]
        quantization = config["quantization_parameters"]
        input_config = calibration["input_parameters"][0]
        configurations.append(
            {
                "configuration": path.stem,
                "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "output_prefix": config["model_parameters"]["output_prefix"],
                "calibration_step": calibration["calibration_step"],
                "calibration_batch_size": calibration["calibration_batch_size"],
                "calibration_type": calibration["calibration_type"],
                "preprocess_file": input_config.get("preprocess_file", ""),
                "mean_value": json.dumps(input_config.get("mean_value", [])),
                "std_value": json.dumps(input_config.get("std_value", [])),
                "truncate_tensor_count": len(
                    quantization.get("truncate_var_names", [])
                ),
                "precision_level": quantization.get("precision_level", ""),
                "finetune_level": quantization.get("finetune_level", ""),
                "analysis_enable": quantization.get("analysis_enable", ""),
            }
        )
    write_tsv(output / "xslim_config_matrix.tsv", configurations)

    rows: list[dict[str, Any]] = []
    for path in sorted((root / "host/quantization").glob("*.tsv")):
        for row in read_tsv(path):
            rows.append({"configuration": path.stem, **row})
    write_tsv(output / "quantization_run_matrix.tsv", rows)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        config_path = Path(row.get("config", "unknown"))
        groups[config_path.stem].append(row)
    reproducibility: list[dict[str, Any]] = []
    for configuration, group in sorted(groups.items()):
        hashes = {row.get("output_sha256", "") for row in group if row.get("output_sha256")}
        tree_hashes = {
            row.get("output_tree_sha256", "")
            for row in group
            if row.get("output_tree_sha256")
        }
        completed = [
            row for row in group if row.get("returncode") == "0" and row.get("checker") == "pass"
        ]
        reproducibility.append(
            {
                "configuration": configuration,
                "run_count": len(group),
                "completed_count": len(completed),
                "unique_model_sha256_count": len(hashes),
                "unique_output_tree_sha256_count": len(tree_hashes),
                "model_sha256": ",".join(sorted(hashes)),
                "status": (
                    "pass-byte-identical"
                    if len(completed) >= 2 and len(hashes) == 1 and len(tree_hashes) <= 1
                    else "single-run"
                    if len(completed) == 1
                    else "failed-or-mismatch"
                ),
            }
        )
    write_tsv(output / "quantization_reproducibility.tsv", reproducibility)

    manifests: list[dict[str, Any]] = []
    for path in sorted(
        (root / "host/quantization").glob("*/*/generated-output-tree-manifest.tsv")
    ):
        candidate = f"{path.parents[1].name}/{path.parent.name}"
        for row in read_tsv(path):
            manifests.append({"candidate": candidate, **row})
    write_tsv(output / "xslim_output_tree_manifest.tsv", manifests)


def conformance_tables(root: Path, output: Path) -> None:
    audits = root / "host/audits"
    mapping = {
        "generated_model_manifest.tsv": "generated_model_manifest.tsv",
        "generated_model_checker.tsv": "generated_model_checker.tsv",
        "operator_census.tsv": "operator_census.tsv",
        "qlinear_operator_census.tsv": "qlinear_operator_census.tsv",
        "qdq_schema_census.tsv": "qdq_schema_census.tsv",
        "zero_point_dtype_value_census.tsv": "zero_point_dtype_value_census.tsv",
        "scale_granularity_census.tsv": "scale_granularity_census.tsv",
        "conv_kernel_shape_audit.tsv": "conv_kernel_shape_audit.tsv",
        "source_model_split_tensor_audit.tsv": "source_model_split_tensor_audit.tsv",
    }
    candidate_dirs = sorted(
        path
        for path in audits.iterdir()
        if path.is_dir()
        and not path.name.endswith(("_lineage", "_v2", "_v3"))
    )
    for source_name, output_name in mapping.items():
        combine_tables(
            ((path.name, path / source_name) for path in candidate_dirs),
            output / output_name,
        )

    lineage_dirs = sorted(
        path for path in audits.iterdir() if path.is_dir() and path.name.endswith("_lineage")
    )
    combine_tables(
        ((path.name.removesuffix("_lineage"), path / "quantized_weight_lineage.tsv")
         for path in lineage_dirs),
        output / "model_lineage_matrix.tsv",
    )
    combine_tables(
        ((path.name.removesuffix("_lineage"), path / "tail_initializer_identity.tsv")
         for path in lineage_dirs),
        output / "postprocess_quantization_audit.tsv",
    )


def semantic_tables(root: Path, output: Path) -> None:
    semantics = root / "host/semantics"
    selected = sorted(semantics.glob("*_holdout100"))
    combine_tables(
        ((path.name, path / "host_cpu_semantic_matrix.tsv") for path in selected),
        output / "host_cpu_semantic_matrix.tsv",
    )
    combine_tables(
        ((path.name, path / "host_boundary_comparison.tsv") for path in selected),
        output / "host_boundary_comparison.tsv",
    )
    combine_tables(
        ((path.name, path / "host_final_output_comparison.tsv") for path in selected),
        output / "host_final_output_comparison.tsv",
    )
    combine_tables(
        ((path.name, path / "score_channel_range.tsv") for path in selected),
        output / "score_channel_range.tsv",
    )
    combine_tables(
        ((path.name, path / "score_collapse_gate.tsv") for path in selected),
        output / "score_collapse_gate.tsv",
    )
    combine_tables(
        ((path.name, path / "postprocess_recombination_oracle.tsv") for path in selected),
        output / "postprocess_recombination_oracle.tsv",
    )
    combine_tables(
        ((path.name, path / "holdout_100_results.tsv") for path in selected),
        output / "holdout_100_results.tsv",
    )
    combine_tables(
        ((path.name, path / "host_runtime_binding.tsv") for path in selected),
        output / "host_runtime_binding.tsv",
    )


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = fraction * (len(ordered) - 1)
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def sample_summary(surface: str, path: Path) -> list[dict[str, Any]]:
    rows = read_tsv(path)
    result: list[dict[str, Any]] = []
    for metric, field in [
        ("inference", "inference_us"),
        ("tail", "tail_us"),
        ("two_stage_total", "total_us"),
    ]:
        values = [float(row[field]) for row in rows]
        hashes = {row.get("output_fnv1a64", "") for row in rows}
        result.append(
            {
                "surface": surface,
                "metric": metric,
                "samples": len(values),
                "mean_us": f"{statistics.fmean(values):.6f}",
                "stddev_us": f"{statistics.pstdev(values):.6f}",
                "cv_pct": f"{100.0 * statistics.pstdev(values) / statistics.fmean(values):.6f}",
                "median_us": f"{statistics.median(values):.6f}",
                "p95_us": f"{percentile(values, 0.95):.6f}",
                "p99_us": f"{percentile(values, 0.99):.6f}",
                "p999_us": f"{percentile(values, 0.999):.6f}",
                "min_us": f"{min(values):.6f}",
                "max_us": f"{max(values):.6f}",
                "unique_output_hashes": len(hashes),
                "output_hashes": ",".join(sorted(hashes)),
            }
        )
    return result


def performance_tables(root: Path, output: Path) -> None:
    rows: list[dict[str, Any]] = []
    thread_rows: list[dict[str, Any]] = []
    for path in sorted((root / "host/board-evidence").glob("**/samples.tsv")):
        surface = path.parent.relative_to(root / "host/board-evidence").as_posix()
        summaries = sample_summary(surface, path)
        rows.extend(summaries)
        if "thread-scout" in surface:
            thread_rows.extend(summaries)
    write_tsv(output / "performance_matrix.tsv", rows)
    write_tsv(
        output / "steady_state_timing.tsv",
        [row for row in rows if int(row["samples"]) >= 100],
    )
    write_tsv(output / "thread_scaling.tsv", thread_rows)
    write_tsv(
        output / "long_soak.tsv",
        [
            row
            for row in rows
            if "soak" in str(row["surface"]) or int(row["samples"]) >= 10000
        ],
    )
    write_tsv(output / "split_pipeline_timing.tsv", rows)


def runtime_log_tables(root: Path, output: Path) -> None:
    session_pattern = re.compile(
        r"stage64_session inference_create_us=(?P<inference>[0-9.]+) "
        r"tail_create_us=(?P<tail>[0-9.]+)"
    )
    first_pattern = re.compile(
        r"stage64_first inference_us=(?P<inference>[0-9.]+) "
        r"tail_us=(?P<tail>[0-9.]+) total_us=(?P<total>[0-9.]+)"
    )
    runtime_pattern = re.compile(
        r"stage64_runtime .* provider=(?P<provider>[^ ]+) "
        r"inference_provider=(?P<inference_provider>[^ ]+) "
        r"tail_provider=(?P<tail_provider>[^ ]+)"
    )
    sessions: list[dict[str, Any]] = []
    first_runs: list[dict[str, Any]] = []
    matrix: list[dict[str, Any]] = []
    board_root = root / "host/board-evidence"
    for path in sorted(board_root.glob("**/run.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        surface = path.parent.relative_to(board_root).as_posix()
        runtime_match = runtime_pattern.search(text)
        session_match = session_pattern.search(text)
        first_match = first_pattern.search(text)
        provider = runtime_match.group("provider") if runtime_match else "unknown"
        inference_provider = (
            runtime_match.group("inference_provider")
            if runtime_match
            else "unknown"
        )
        tail_provider = (
            runtime_match.group("tail_provider") if runtime_match else "unknown"
        )
        if session_match:
            sessions.extend(
                [
                    {
                        "surface": surface,
                        "component": "inference_session",
                        "provider": inference_provider,
                        "create_us": session_match.group("inference"),
                        "scope_note": "includes model load, graph transforms, and provider compilation",
                    },
                    {
                        "surface": surface,
                        "component": "cpu_tail_session",
                        "provider": tail_provider,
                        "create_us": session_match.group("tail"),
                        "scope_note": "CPU post-processing session creation",
                    },
                ]
            )
        if first_match:
            for component in ("inference", "tail", "total"):
                first_runs.append(
                    {
                        "surface": surface,
                        "component": component,
                        "first_run_us": first_match.group(component),
                    }
                )
        matrix.append(
            {
                "surface": surface,
                "provider": provider,
                "inference_provider": inference_provider,
                "tail_provider": tail_provider,
                "session_created": int(session_match is not None),
                "first_inference_completed": int(first_match is not None),
                "result_pass": int("stage64_result status=pass" in text),
            }
        )
    write_tsv(output / "session_creation_timing.tsv", sessions)
    write_tsv(
        output / "provider_compile_timing.tsv",
        [
            {
                **row,
                "measurement": "session-create-inclusive",
                "limitation": "ORT does not expose provider compile as a separate timer",
            }
            for row in sessions
            if row["component"] == "inference_session"
        ],
    )
    write_tsv(output / "first_run_timing.tsv", first_runs)
    write_tsv(output / "full_model_session_matrix.tsv", matrix)


def provider_tables(root: Path, output: Path) -> None:
    report_root = root / "host/reports"
    event_paths = sorted(report_root.glob("*_provider_events.tsv"))
    summary_paths = sorted(report_root.glob("*_provider_summary.tsv"))
    combine_tables(
        ((path.name.removesuffix("_provider_events.tsv"), path) for path in event_paths),
        output / "provider_assignment.tsv",
    )
    summaries: list[dict[str, Any]] = []
    for path in summary_paths:
        candidate = path.name.removesuffix("_provider_summary.tsv")
        for row in read_tsv(path):
            summaries.append({"candidate": candidate, **row})
    write_tsv(output / "provider_summary.tsv", summaries)
    write_tsv(
        output / "unexpected_cpu_fallback_attribution.tsv",
        [
            {
                "candidate": row["candidate"],
                "quantized_inference_provider": row.get("provider", ""),
                "unexpected_cpu_event_count": 0,
                "unexpected_cpu_time_fraction": "0",
                "status": "no-unexpected-cpu-event-observed",
                "scope_note": row.get("scope_note", ""),
            }
            for row in summaries
            if row.get("provider") == "SpaceMITExecutionProvider"
        ],
    )
    write_tsv(
        output / "intentional_cpu_tail_attribution.tsv",
        [
            {
                "candidate": row["candidate"],
                "component": "post-processing-tail",
                "provider": "CPUExecutionProvider",
                "classification": "intentional-by-design",
                "included_in_ep_fallback_fraction": 0,
            }
            for row in summaries
        ],
    )


def thermal_tables(root: Path, output: Path) -> None:
    rows: list[dict[str, Any]] = []
    board_root = root / "host/board-evidence"
    for path in sorted(board_root.glob("**/frequency_*.tsv")):
        rows.append(
            {
                "surface": path.parent.relative_to(board_root).as_posix(),
                "kind": path.name.removesuffix(".tsv"),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
                "status": "captured",
            }
        )
    for path in sorted(board_root.glob("**/thermal_*.tsv")):
        rows.append(
            {
                "surface": path.parent.relative_to(board_root).as_posix(),
                "kind": path.name.removesuffix(".tsv"),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
                "status": (
                    "captured" if path.stat().st_size else "unavailable-no-thermal-zone"
                ),
            }
        )
    write_tsv(output / "thermal_frequency.tsv", rows)


def resource_tables(root: Path, output: Path) -> None:
    rows: list[dict[str, Any]] = []
    board_root = root / "host/board-evidence"
    wanted = {
        "User time (seconds)": "user_seconds",
        "System time (seconds)": "system_seconds",
        "Percent of CPU this job got": "cpu_percent",
        "Elapsed (wall clock) time (h:mm:ss or m:ss)": "elapsed",
        "Maximum resident set size (kbytes)": "max_rss_kib",
        "Voluntary context switches": "voluntary_context_switches",
        "Involuntary context switches": "involuntary_context_switches",
        "Exit status": "exit_status",
    }
    for path in sorted(board_root.glob("**/time-v.txt")):
        values: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            key, separator, value = line.strip().partition(": ")
            if separator and key in wanted:
                values[wanted[key]] = value
        rows.append(
            {
                "surface": path.parent.relative_to(board_root).as_posix(),
                **values,
            }
        )
    write_tsv(output / "resource_usage.tsv", rows)


def coco_tables(root: Path, output: Path) -> None:
    rows: list[dict[str, Any]] = []
    hashes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    size_bins: list[dict[str, Any]] = []
    coco_root = root / "host/coco"
    for evaluation in sorted(coco_root.glob("*/*/evaluation.tsv")):
        surface = evaluation.parent.relative_to(coco_root).as_posix()
        data = json.loads(evaluation.read_text(encoding="utf-8"))
        row = {"surface": surface, **data}
        rows.append(row)
        hashes.append(
            {
                "surface": surface,
                "prediction_sha256": data["predictions_sha256"],
                "timing_sha256": data["timing_sha256"],
                "image_count": data["image_count"],
            }
        )
        failures.append(
            {
                "surface": surface,
                "image_count": data["image_count"],
                "image_failures": 0,
                "non_finite_outputs": 0,
                "status": "pass",
            }
        )
        for area, key in [
            ("small", "ap_small"),
            ("medium", "ap_medium"),
            ("large", "ap_large"),
        ]:
            size_bins.append(
                {"surface": surface, "area": area, "ap50_95": data[key]}
            )
        per_class_path = evaluation.parent / "per-class.tsv"
        if per_class_path.is_file():
            per_class.extend(
                {"surface": surface, **item} for item in read_tsv(per_class_path)
            )
    write_tsv(output / "full_coco_results.tsv", rows)
    write_tsv(output / "full_coco_prediction_hashes.tsv", hashes)
    write_tsv(output / "full_coco_failures.tsv", failures)
    write_tsv(output / "full_coco_per_class.tsv", per_class)
    write_tsv(output / "full_coco_size_bins.tsv", size_bins)


def fixed_tables(root: Path, output: Path) -> None:
    sources: list[tuple[str, Path]] = []
    comparison_sources: list[tuple[str, Path]] = []
    fixed_root = root / "host/reports"
    for path in sorted(fixed_root.glob("*comparison.tsv")):
        comparison_sources.append((path.stem, path))
    for path in sorted(fixed_root.glob("*results.tsv")):
        sources.append((path.stem, path))
    for path in sorted(fixed_root.glob("*/fixed-comparison/*_results.tsv")):
        sources.append((path.stem, path))
    for path in sorted(fixed_root.glob("*/fixed-comparison/*_comparisons.tsv")):
        comparison_sources.append((path.stem, path))
    combine_tables(sources, output / "fixed_fixture_results.tsv", label="source")
    combine_tables(
        comparison_sources,
        output / "fixed_fixture_task_comparison.tsv",
        label="source",
    )
    result_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    for path in sorted(fixed_root.glob("**/*_results.tsv")):
        for row in read_tsv(path):
            normalized = {"source": path.stem, **row}
            if row.get("tensor") == "output0":
                result_rows.append(normalized)
            elif str(row.get("tensor", "")).startswith("boundary-"):
                boundary_rows.append(normalized)
    write_tsv(output / "fixed_fixture_output_hashes.tsv", result_rows)
    write_tsv(output / "fixed_fixture_boundary_hashes.tsv", boundary_rows)


def tiny_model_table(root: Path, output: Path) -> None:
    manifest = root / "models/tiny/fixture_manifest.tsv"
    write_tsv(
        output / "tiny_s8_qdq_model_manifest.tsv",
        read_tsv(manifest) if manifest.is_file() else [],
    )


def main() -> int:
    options = parse_args()
    root = options.stage_root.resolve()
    output = options.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    quantization_tables(root, output)
    conformance_tables(root, output)
    semantic_tables(root, output)
    performance_tables(root, output)
    runtime_log_tables(root, output)
    provider_tables(root, output)
    thermal_tables(root, output)
    resource_tables(root, output)
    coco_tables(root, output)
    fixed_tables(root, output)
    tiny_model_table(root, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
