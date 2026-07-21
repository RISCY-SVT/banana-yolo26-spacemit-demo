#!/usr/bin/env python3
"""Assemble Stage61's nine-profile evidence with strict identity gates."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

from stage60_finalize_evidence import percentile, read_tsv, sha256, write_tsv
from stage60_parse_pipeline import parse as parse_pipeline


RESOLUTIONS = (640, 512, 448, 416, 384, 352, 320, 256, 768)
EXPECTED_PREDICTIONS = {
    640: "cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda",
    512: "9b0cc4aa1295d58c314a48ae3fd38d4ef8cb7e1386a0e0952dd3a17d88e97d13",
    448: "e6ade48fa813d85f7036b33b1ea67b1cb4ea108073debd914e30074ae0675284",
    416: "8154d17bd8384a18094961977cd84497e9d6d778c969c629aa01e2372df95a73",
    384: "3681ca3a0c2782f47f02aa560fea59c95b4ec34b1f52e082d733e9f0aa2922ef",
    352: "25dc80f94d4bd11681e3aa3c51ad9b34c7212074a17ed6adf3c4dada6ac2e12a",
    320: "86653c74e83f98d1e95f0f308beee23d103ad3096316938591690dd2679190d5",
    256: "4b94ceb69be8ca9950fa16f2c64c6b974982285273c8fc4b58de0f5a3855146d",
}
CATEGORY_COLUMNS = (
    "input_us",
    "resident_core_us",
    "dense_us",
    "attention_us",
    "depthwise_us",
    "lut_us",
    "concat_us",
    "transform_us",
    "head_us",
)
FIXTURES = ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7",
            "bus", "canonical", "zidane")


def numeric_environment(path: Path, resolution: int, surface: str) -> tuple[str, str, str, str, str]:
    rows = [
        row for row in read_tsv(path)
        if int(row["resolution"]) == resolution and row["surface"] == surface
    ]
    temperatures = [float(row["mean_thermal_c"]) for row in rows if row["mean_thermal_c"]]
    frequencies = [float(row["mean_cpu0_4_khz"]) for row in rows if row["mean_cpu0_4_khz"]]
    boot_ids = {row["boot_id"] for row in rows}
    if len(rows) != 2 or len(boot_ids) != 1:
        raise ValueError(f"incomplete environment snapshots for R{resolution} {surface}")
    return (
        f"{statistics.fmean(temperatures):.6f}" if temperatures else "",
        f"{max(temperatures):.6f}" if temperatures else "",
        f"{min(frequencies):.0f}" if frequencies else "",
        f"{max(frequencies):.0f}" if frequencies else "",
        next(iter(boot_ids)),
    )


def summarize_samples(rows: list[dict[str, str]], surface: str,
                      environment: tuple[str, str, str, str, str],
                      include_p999: bool) -> dict[str, Any]:
    walls = [float(row["total_us"]) for row in rows]
    mean = statistics.fmean(walls)
    stddev = statistics.stdev(walls)
    hashes = {row["output_hash"] for row in rows}
    manifests = {row["manifest_sha256"] for row in rows}
    resolutions = {int(row["resolution"]) for row in rows}
    if len(hashes) != 1 or len(manifests) != 1 or len(resolutions) != 1:
        raise ValueError("mixed identity in performance samples")
    if any(int(row["affinity_ok"]) != 1 for row in rows):
        raise ValueError("affinity failure in performance samples")
    if any(int(row["cpu4_7_ime_count"]) != 0 for row in rows):
        raise ValueError("CPU4-7 IME use in performance samples")
    temperature_mean, temperature_max, frequency_min, frequency_max, boot_id = environment
    result: dict[str, Any] = {
        "resolution": next(iter(resolutions)),
        "surface": surface,
        "wake_policy": rows[0]["wake_policy"],
        "route": "exact-attention-ntail-selected",
        "samples": len(rows),
        "mean_us": f"{mean:.6f}",
        "stddev_us": f"{stddev:.6f}",
        "cv_pct": f"{100.0 * stddev / mean:.6f}",
        "median_us": f"{percentile(walls, 0.50):.6f}",
        "p90_us": f"{percentile(walls, 0.90):.6f}",
        "p95_us": f"{percentile(walls, 0.95):.6f}",
        "p99_us": f"{percentile(walls, 0.99):.6f}",
        "p999_us": f"{percentile(walls, 0.999):.6f}" if include_p999 else "",
        "max_us": f"{max(walls):.6f}",
        "fps": f"{1_000_000.0 / mean:.9f}",
        "process_cpu_mean_us": f"{statistics.fmean(float(row['process_cpu_us']) for row in rows):.6f}",
        **{
            f"{column}_mean": f"{statistics.fmean(float(row[column]) for row in rows):.6f}"
            for column in CATEGORY_COLUMNS
        },
        "voluntary_context_switches": sum(int(row["voluntary_cs"]) for row in rows),
        "involuntary_context_switches": sum(int(row["involuntary_cs"]) for row in rows),
        "affinity_failures": 0,
        "cpu4_7_ime_count": 0,
        "mean_thermal_c": temperature_mean,
        "max_thermal_c": temperature_max,
        "min_cpu0_4_khz": frequency_min,
        "max_cpu0_4_khz": frequency_max,
        "boot_id": boot_id,
        "output_hash": next(iter(hashes)),
        "manifest_sha256": next(iter(manifests)),
        "surface_definition": (
            "fixed-preprocessed-f32-pure-model"
            if surface == "preprocessed"
            else "already-letterboxed-rgb8-executor-surface"
        ),
        "status": "pass",
    }
    return result


def assemble_performance(raw_root: Path, output: Path) -> None:
    source = raw_root / "benchmarks/final"
    environment_path = source / "environment.tsv"
    raw: list[dict[str, str]] = []
    summaries: list[dict[str, Any]] = []
    for resolution in RESOLUTIONS:
        for surface in ("preprocessed", "rgb"):
            path = source / f"r{resolution}_{surface}_raw.tsv"
            rows = read_tsv(path)
            if len(rows) != 500:
                raise ValueError(f"expected 500 rows in {path}, found {len(rows)}")
            raw.extend({"source_file": path.name, **row} for row in rows)
            summaries.append(summarize_samples(
                rows, surface, numeric_environment(environment_path, resolution, surface), False
            ))
    write_tsv(output / "resolution_performance_raw_v2.tsv", raw)
    write_tsv(output / "resolution_performance_summary_v2.tsv", summaries)


def soak_environment(path: Path, resolution: int) -> tuple[str, str, str, str, str]:
    rows = read_tsv(path)
    temperatures = [float(row["mean_thermal_c"]) for row in rows if row["mean_thermal_c"]]
    frequencies = [float(row["mean_cpu0_4_khz"]) for row in rows if row["mean_cpu0_4_khz"]]
    return (
        f"{statistics.fmean(temperatures):.6f}" if temperatures else "",
        f"{max(temperatures):.6f}" if temperatures else "",
        f"{min(frequencies):.0f}" if frequencies else "",
        f"{max(frequencies):.0f}" if frequencies else "",
        "recorded-in-system-state",
    )


def assemble_soaks(raw_root: Path, output: Path) -> None:
    source = raw_root / "long-soak"
    rows: list[dict[str, Any]] = []
    for resolution in (640, 768, 448, 416, 352, 320):
        samples = read_tsv(source / f"r{resolution}.raw.tsv")
        if len(samples) != 10_000:
            raise ValueError(f"expected 10000 soak rows for R{resolution}")
        summary = summarize_samples(
            samples, "preprocessed", soak_environment(source / f"r{resolution}.system.tsv", resolution), True
        )
        summary["source_file"] = f"r{resolution}.raw.tsv"
        rows.append(summary)
    write_tsv(output / "resolution_long_soak_v2.tsv", rows)


def assemble_pipelines(raw_root: Path, output: Path) -> None:
    source = raw_root / "pipeline"
    summaries: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for resolution in RESOLUTIONS:
        for kind in ("serial", "double_buffer"):
            path = source / f"r{resolution}_{kind}.tsv"
            rows, metadata = parse_pipeline(path, resolution, kind)
            summaries.extend(rows)
            identities.append({
                "resolution": resolution,
                "pipeline_kind": kind,
                "output_hash": metadata["output_hash"],
                "package_manifest_sha256": metadata["package_manifest_sha256"],
                "detections": metadata["detections"],
                "cpu4_7_ime_count": metadata["cpu4_7_ime_count"],
                "preprocessor_cpus": metadata.get("preprocessor_cpus", "controller-thread"),
                "executor_cpus": metadata.get("executor_cpus", "0-4"),
                "steady_state_fps": metadata.get("steady_state_fps", ""),
                "source_file": path.name,
            })
    write_tsv(output / "resolution_pipeline_summary_v2.tsv", summaries)
    write_tsv(output / "resolution_pipeline_identity_v2.tsv", identities)


def assemble_exactness(raw_root: Path, output: Path) -> None:
    source = raw_root / "exactness/full-fixture-parity/summary.tsv"
    rows = read_tsv(source)
    expected = {(resolution, fixture) for resolution in RESOLUTIONS for fixture in FIXTURES}
    observed = {(int(row["resolution"]), row["fixture"]) for row in rows}
    if observed != expected or len(rows) != len(expected):
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"incomplete exactness matrix: missing={missing} extra={extra}")
    for row in rows:
        if int(row["integer_boundary_count"]) != 215:
            raise ValueError("exactness row does not contain all 215 boundaries")
        for column in (
            "host_vs_board_scalar_boundaries",
            "host_vs_board_optimized_boundaries",
            "board_scalar_vs_optimized_all_files",
            "board_final_output_exact",
        ):
            if row[column] != "exact":
                raise ValueError(f"non-exact row in {source}: {row}")
    write_tsv(output / "attention_ntail_exactness.tsv", rows)
    write_tsv(
        output / "r768_exactness_matrix.tsv",
        [row for row in rows if int(row["resolution"]) == 768],
    )


def assemble_coco(raw_root: Path, output: Path) -> None:
    results: list[dict[str, str]] = []
    per_class: list[dict[str, str]] = []
    size_bins: list[dict[str, str]] = []
    hashes: list[dict[str, Any]] = []
    for resolution in RESOLUTIONS:
        directory = raw_root / f"coco/r{resolution}"
        result_rows = read_tsv(directory / "results.tsv")
        if len(result_rows) != 1:
            raise ValueError(f"multiple COCO summary rows for R{resolution}")
        result = result_rows[0]
        prediction = raw_root / f"predictions/r{resolution}_predictions.json"
        actual_hash = sha256(prediction)
        if int(result["images"]) != 5000 or int(result["failures"]) != 0:
            raise ValueError(f"incomplete COCO run for R{resolution}")
        if result["prediction_sha256"] != actual_hash:
            raise ValueError(f"prediction hash mismatch for R{resolution}")
        if resolution in EXPECTED_PREDICTIONS and actual_hash != EXPECTED_PREDICTIONS[resolution]:
            raise ValueError(f"Stage60 prediction identity changed for R{resolution}")
        results.append(result)
        per_class.extend(read_tsv(directory / "per_class.tsv"))
        size_bins.extend(read_tsv(directory / "size_bins.tsv"))
        hashes.append({
            "resolution": resolution,
            "quantization_arm": "Q0",
            "prediction_file": prediction.name,
            "prediction_bytes": prediction.stat().st_size,
            "prediction_sha256": actual_hash,
            "stage60_identity_required": int(resolution in EXPECTED_PREDICTIONS),
            "status": "exact" if resolution in EXPECTED_PREDICTIONS else "new-frozen-identity",
        })
    write_tsv(output / "resolution_coco_results_v2.tsv", results)
    write_tsv(output / "resolution_coco_per_class_v2.tsv", per_class)
    write_tsv(output / "resolution_size_bins_v2.tsv", size_bins)
    write_tsv(output / "resolution_prediction_hashes_v2.tsv", hashes)


def cache_rows(stage_root: Path, stage60_root: Path) -> dict[int, dict[str, str]]:
    rows = read_tsv(stage60_root / "reports/generated/resolution_cache_model.tsv")
    rows.extend(read_tsv(
        stage_root / "profiles/generated/r768/census/resolution_cache_model.tsv"
    ))
    return {int(row["resolution"]): row for row in rows}


def assemble_pareto(stage_root: Path, stage60_root: Path, output: Path) -> None:
    timing_rows = read_tsv(output / "resolution_performance_summary_v2.tsv")
    timings = {
        int(row["resolution"]): row for row in timing_rows if row["surface"] == "preprocessed"
    }
    coco_rows = read_tsv(output / "resolution_coco_results_v2.tsv")
    coco = {int(row["resolution"]): row for row in coco_rows}
    cache = cache_rows(stage_root, stage60_root)
    control_map = float(coco[640]["map50_95"])
    candidates: list[dict[str, Any]] = []
    for resolution in RESOLUTIONS:
        timing = timings[resolution]
        accuracy = coco[resolution]
        model_mean = float(timing["mean_us"])
        loss_ap = 100.0 * (control_map - float(accuracy["map50_95"]))
        strong = model_mean <= 70_000.0 and loss_ap <= 1.0
        latency = model_mean <= 60_000.0 and loss_ap <= 1.5 and float(accuracy["ap_small"]) > 0.0
        candidates.append({
            "resolution": resolution,
            "quantization_arm": "Q0",
            "mac_count": cache[resolution]["mac_count"],
            "flop_count_2_per_mac": cache[resolution]["flop_count_2_per_mac"],
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
            "camera_candidate_status": "matched-r640-r768" if resolution in (640, 768) else "not-selected",
        })
    nondominated: set[int] = set()
    for candidate in candidates:
        dominated = any(
            other is not candidate
            and float(other["pure_model_mean_us"]) <= float(candidate["pure_model_mean_us"])
            and float(other["map50_95"]) >= float(candidate["map50_95"])
            and (
                float(other["pure_model_mean_us"]) < float(candidate["pure_model_mean_us"])
                or float(other["map50_95"]) > float(candidate["map50_95"])
            )
            for other in candidates
        )
        if not dominated:
            nondominated.add(int(candidate["resolution"]))
    for candidate in candidates:
        candidate["pareto_nondominated"] = int(int(candidate["resolution"]) in nondominated)
    write_tsv(output / "resolution_pareto_q0_v2.tsv", candidates)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--stage60-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sections", default="exactness,performance,soak,pipeline,coco,pareto",
        help="comma-separated subset of exactness,performance,soak,pipeline,coco,pareto",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sections = set(args.sections.split(","))
    unknown = sections - {"exactness", "performance", "soak", "pipeline", "coco", "pareto"}
    if unknown:
        raise ValueError(f"unknown sections: {sorted(unknown)}")
    if "performance" in sections:
        assemble_performance(args.raw_root, args.output)
    if "exactness" in sections:
        assemble_exactness(args.raw_root, args.output)
    if "soak" in sections:
        assemble_soaks(args.raw_root, args.output)
    if "pipeline" in sections:
        assemble_pipelines(args.raw_root, args.output)
    if "coco" in sections:
        assemble_coco(args.raw_root, args.output)
    if "pareto" in sections:
        assemble_pareto(args.stage_root, args.stage60_root, args.output)
    print(f"sections={','.join(sorted(sections))}")
    print(f"resolutions={len(RESOLUTIONS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
