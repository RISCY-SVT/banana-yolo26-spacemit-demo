#!/usr/bin/env python3
"""Assemble Stage60 measured evidence and reject incomplete resolution surfaces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import statistics
from pathlib import Path
from typing import Any, Iterable


RESOLUTIONS = (640, 512, 448, 416, 384, 352, 320, 256)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows:
        raise ValueError(f"empty evidence table: {path}")
    return rows


def write_tsv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"refusing to write an empty evidence table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(materialized[0]), delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def assemble_graph_identity(stage_root: Path, output: Path) -> None:
    package_rows = {
        int(row["resolution"]): row
        for row in read_tsv(stage_root / "reports/generated/resolution_package_hashes.tsv")
    }
    rows: list[dict[str, Any]] = []
    for resolution in RESOLUTIONS:
        source = read_tsv(stage_root / f"reports/graph/r{resolution}.identity.tsv")[0]
        package = package_rows[resolution]
        rows.append({
            "resolution": resolution,
            "profile_id": package["profile_id"],
            "source_model_sha256": source["source_model_sha256"],
            "static_model_sha256": package["model_sha256"],
            "node_count": source["node_count"],
            "initializer_count": source["initializer_count"],
            "initializer_payload_identity": source["initializer_identity"],
            "topology_identity": source["topology_identity"],
            "input_shape": source["input_shape"],
            "output_shape": source["output_shape"],
            "operator_classes": source["operator_classes"],
            "control_role": "frozen-source-control" if resolution == 640 else "static-shape-derivative",
        })
    write_tsv(output / "resolution_graph_identity.tsv", rows)


def assemble_performance(stage_root: Path, output: Path) -> dict[int, dict[str, str]]:
    raw_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, Any]] = []
    selected: dict[int, dict[str, str]] = {}
    for resolution in RESOLUTIONS:
        for surface in ("preprocessed", "rgb"):
            path = stage_root / f"benchmarks/r{resolution}_{surface}_raw.tsv"
            rows = read_tsv(path)
            if len(rows) != 500:
                raise ValueError(f"expected 500 samples, found {len(rows)} in {path}")
            hashes = {row["output_hash"] for row in rows}
            manifests = {row["manifest_sha256"] for row in rows}
            if len(hashes) != 1 or len(manifests) != 1:
                raise ValueError(f"mixed identity in {path}")
            for row in rows:
                raw_rows.append({"source_file": path.name, **row})
            walls = [float(row["total_us"]) for row in rows]
            process_cpu = [float(row["process_cpu_us"]) for row in rows]
            category_columns = (
                "input_us", "resident_core_us", "dense_us", "attention_us",
                "depthwise_us", "lut_us", "concat_us", "transform_us", "head_us",
            )
            mean = statistics.fmean(walls)
            stddev = statistics.stdev(walls)
            summary = {
                "resolution": resolution,
                "surface": surface,
                "wake_policy": rows[0]["wake_policy"],
                "samples": len(rows),
                "mean_us": f"{mean:.6f}",
                "stddev_us": f"{stddev:.6f}",
                "cv_pct": f"{100.0 * stddev / mean:.6f}",
                "median_us": f"{percentile(walls, 0.50):.6f}",
                "p90_us": f"{percentile(walls, 0.90):.6f}",
                "p95_us": f"{percentile(walls, 0.95):.6f}",
                "p99_us": f"{percentile(walls, 0.99):.6f}",
                "p999_us": "",
                "max_us": f"{max(walls):.6f}",
                "fps": f"{1_000_000.0 / mean:.9f}",
                "process_cpu_mean_us": f"{statistics.fmean(process_cpu):.6f}",
                **{
                    f"{column}_mean": f"{statistics.fmean(float(row[column]) for row in rows):.6f}"
                    for column in category_columns
                },
                "voluntary_context_switches": sum(int(row["voluntary_cs"]) for row in rows),
                "involuntary_context_switches": sum(int(row["involuntary_cs"]) for row in rows),
                "affinity_failures": sum(int(row["affinity_ok"]) != 1 for row in rows),
                "cpu4_7_ime_count": sum(int(row["cpu4_7_ime_count"]) for row in rows),
                "output_hash": next(iter(hashes)),
                "manifest_sha256": next(iter(manifests)),
                "surface_definition": (
                    "fixed-preprocessed-f32-pure-model" if surface == "preprocessed"
                    else "already-letterboxed-rgb8-executor-surface"
                ),
            }
            summary_rows.append(summary)
            if surface == "preprocessed":
                selected[resolution] = {key: str(value) for key, value in summary.items()}
    write_tsv(output / "resolution_performance_raw.tsv", raw_rows)
    write_tsv(output / "resolution_performance_summary.tsv", summary_rows)
    return selected


def assemble_coco(stage_root: Path, output: Path) -> dict[int, dict[str, str]]:
    results: list[dict[str, str]] = []
    per_class: list[dict[str, str]] = []
    size_bins: list[dict[str, str]] = []
    hashes: list[dict[str, Any]] = []
    by_resolution: dict[int, dict[str, str]] = {}
    for resolution in RESOLUTIONS:
        directory = stage_root / f"reports/coco/r{resolution}"
        result = read_tsv(directory / "results.tsv")
        if len(result) != 1 or int(result[0]["images"]) != 5000 or int(result[0]["failures"]) != 0:
            raise ValueError(f"incomplete COCO result for R{resolution}")
        prediction = stage_root / f"predictions/r{resolution}_predictions.json"
        if result[0]["prediction_sha256"] != sha256(prediction):
            raise ValueError(f"prediction hash mismatch for R{resolution}")
        results.extend(result)
        per_class.extend(read_tsv(directory / "per_class.tsv"))
        size_bins.extend(read_tsv(directory / "size_bins.tsv"))
        hashes.append({
            "resolution": resolution,
            "quantization_arm": "Q0",
            "prediction_file": prediction.name,
            "prediction_bytes": prediction.stat().st_size,
            "prediction_sha256": result[0]["prediction_sha256"],
        })
        by_resolution[resolution] = result[0]
    write_tsv(output / "resolution_coco_results.tsv", results)
    write_tsv(output / "resolution_coco_per_class.tsv", per_class)
    write_tsv(output / "resolution_size_bins.tsv", size_bins)
    write_tsv(output / "resolution_prediction_hashes.tsv", hashes)
    return by_resolution


def assemble_pareto(stage_root: Path, output: Path,
                    performance: dict[int, dict[str, str]],
                    coco: dict[int, dict[str, str]]) -> None:
    cache = {
        int(row["resolution"]): row
        for row in read_tsv(stage_root / "reports/generated/resolution_cache_model.tsv")
    }
    control_map = float(coco[640]["map50_95"])
    candidates: list[dict[str, Any]] = []
    for resolution in RESOLUTIONS:
        timing = performance[resolution]
        accuracy = coco[resolution]
        model_mean = float(timing["mean_us"])
        loss_ap = 100.0 * (control_map - float(accuracy["map50_95"]))
        strong = model_mean <= 70_000.0 and loss_ap <= 1.0
        latency = (model_mean <= 60_000.0 and loss_ap <= 1.5 and
                   float(accuracy["ap_small"]) > 0.0)
        candidates.append({
            "resolution": resolution,
            "quantization_arm": "Q0",
            "mac_count": cache[resolution]["mac_count"],
            "arena_bytes": cache[resolution]["arena_bytes"],
            "peak_live_bytes": cache[resolution]["peak_live_bytes"],
            "max_activation_bytes": cache[resolution]["max_tensor_bytes"],
            "conv_operations_with_M12_tail": cache[resolution]["conv_operations_with_M12_tail"],
            "pure_model_mean_us": timing["mean_us"],
            "pure_model_p95_us": timing["p95_us"],
            "pure_model_p99_us": timing["p99_us"],
            "pure_model_fps": timing["fps"],
            "map50_95": accuracy["map50_95"],
            "map50": accuracy["map50"],
            "ap_small": accuracy["ap_small"],
            "ap_medium": accuracy["ap_medium"],
            "ap_large": accuracy["ap_large"],
            "accuracy_loss_vs_r640_ap": f"{loss_ap:.9f}",
            "strong_no_training_gate": "pass" if strong else "fail",
            "latency_accuracy_gate": "pass" if latency else "fail",
            "near_20fps_diagnostic_only": int(model_mean <= 50_000.0 and loss_ap > 2.0),
            "camera_candidate_status": "pending-finalist-selection",
        })

    nondominated: set[int] = set()
    for candidate in candidates:
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            no_slower = float(other["pure_model_mean_us"]) <= float(candidate["pure_model_mean_us"])
            no_less_accurate = float(other["map50_95"]) >= float(candidate["map50_95"])
            strictly_better = (float(other["pure_model_mean_us"]) < float(candidate["pure_model_mean_us"]) or
                               float(other["map50_95"]) > float(candidate["map50_95"]))
            if no_slower and no_less_accurate and strictly_better:
                dominated = True
                break
        if not dominated:
            nondominated.add(int(candidate["resolution"]))
    for candidate in candidates:
        candidate["pareto_nondominated"] = int(int(candidate["resolution"]) in nondominated)
    write_tsv(output / "resolution_pareto.tsv", candidates)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assemble_graph_identity(args.stage_root, args.output)
    performance = assemble_performance(args.stage_root, args.output)
    coco = assemble_coco(args.stage_root, args.output)
    assemble_pareto(args.stage_root, args.output, performance, coco)
    print(f"resolutions={len(RESOLUTIONS)}")
    print("performance_surfaces=16")
    print("coco_surfaces=8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
