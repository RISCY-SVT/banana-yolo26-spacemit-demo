#!/usr/bin/env python3
"""Evaluate DEV-001A candidates on H500 and apply the frozen selection gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
from pathlib import Path
from typing import Any

LANES = ("A1", "A2", "A3", "A4", "A5", "A6")
B2_H500_MAP = 0.4446654879525213
FQ8_L3_H500_MAP = 0.47505674025452777
STRONG_DELTA = (FQ8_L3_H500_MAP - B2_H500_MAP) * 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--h500-list", required=True, type=Path)
    parser.add_argument("--b2-predictions", required=True, type=Path)
    parser.add_argument("--b2-expected-sha256", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty report: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in fields} for row in rows
        )


def invoke(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.run(
            command, stdout=output, stderr=subprocess.STDOUT, check=False
        )
    if process.returncode:
        raise RuntimeError(
            f"command failed ({process.returncode}): {' '.join(command)}"
        )


def metrics(options: argparse.Namespace, surface: str, predictions: Path) -> Path:
    output_dir = options.raw_root / "h500-metrics" / surface
    result = output_dir / "results.tsv"
    if result.exists():
        return result
    invoke(
        [
            str(options.python),
            str(options.repo / "vendor_ort_validation/stage65b_r1_coco_metrics.py"),
            "--annotations",
            str(options.annotations),
            "--predictions",
            str(predictions),
            "--image-list",
            str(options.h500_list),
            "--surface",
            surface,
            "--output-dir",
            str(output_dir),
        ],
        options.raw_root / "logs" / f"h500-metrics-{surface}.log",
    )
    return result


def candidate_predictions(options: argparse.Namespace, lane: str) -> Path:
    output_dir = options.raw_root / "h500" / lane
    predictions = output_dir / "predictions.json"
    if predictions.exists():
        return predictions
    candidate_root = options.raw_root / "postprocess" / lane / "models"
    inference = candidate_root / f"stage65b_r1_{lane.lower()}.inference.onnx"
    tail = candidate_root / f"stage65b_r1_{lane.lower()}.postprocess.onnx"
    invoke(
        [
            str(options.python),
            str(options.repo / "vendor_ort_validation/stage65b_r1_evaluate.py"),
            "run",
            "--candidate-inference",
            str(inference),
            "--tail",
            str(tail),
            "--image-list",
            str(options.h500_list),
            "--output-dir",
            str(output_dir),
            "--name",
            lane,
            "--threads",
            str(options.threads),
            "--log-every",
            "50",
        ],
        options.raw_root / "logs" / f"h500-predict-{lane}.log",
    )
    return predictions


def bootstrap(
    options: argparse.Namespace,
    lanes: list[str],
    predictions: dict[str, Path],
    replicates: int,
    seed: int,
    name: str,
) -> Path:
    output_dir = options.raw_root / "bootstrap" / name
    report = output_dir / "paired_bootstrap_results.tsv"
    if report.exists():
        return report
    command = [
        str(options.python),
        str(options.repo / "vendor_ort_validation/stage65b_r2_bootstrap.py"),
        "--annotations",
        str(options.annotations),
        "--image-list",
        str(options.h500_list),
    ]
    for lane in lanes:
        command.extend(
            [
                "--pair",
                f"{lane}-A0={predictions[lane]},{predictions['A0']}",
            ]
        )
    command.extend(
        [
            "--replicates",
            str(replicates),
            "--seed",
            str(seed),
            "--workers",
            str(options.workers),
            "--output-dir",
            str(output_dir),
        ]
    )
    invoke(command, options.raw_root / "logs" / f"bootstrap-{name}.log")
    return report


def main() -> int:
    options = parse_args()
    options.report_dir.mkdir(parents=True, exist_ok=True)
    if sha256(options.b2_predictions) != options.b2_expected_sha256:
        raise ValueError("frozen B2 H500 prediction identity mismatch")
    conformance = {
        row["lane"]: row
        for row in read_tsv(options.report_dir / "candidate_conformance.tsv")
    }
    passing = [
        lane for lane in LANES if conformance.get(lane, {}).get("status") == "pass"
    ]
    if not passing:
        raise RuntimeError("no candidate passed structural/semantic qualification")

    predictions: dict[str, Path] = {"A0": options.b2_predictions}
    metric_paths: dict[str, Path] = {
        "A0": metrics(options, "A0", options.b2_predictions)
    }
    for lane in passing:
        predictions[lane] = candidate_predictions(options, lane)
        metric_paths[lane] = metrics(options, lane, predictions[lane])

    metric_rows = [read_tsv(metric_paths[lane])[0] for lane in ("A0", *passing)]
    size_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for lane in ("A0", *passing):
        metric_root = metric_paths[lane].parent
        size_rows.extend(read_tsv(metric_root / "size_bins.tsv"))
        class_rows.extend(read_tsv(metric_root / "per_class.tsv"))
        prediction_rows.append(
            {
                "lane": lane,
                "predictions": str(predictions[lane]),
                "prediction_sha256": sha256(predictions[lane]),
                "bytes": predictions[lane].stat().st_size,
                "prediction_count": read_tsv(metric_paths[lane])[0]["prediction_count"],
            }
        )
    write_tsv(options.report_dir / "h500_metrics.tsv", metric_rows)
    write_tsv(options.report_dir / "h500_size_bins.tsv", size_rows)
    write_tsv(options.report_dir / "h500_per_class.tsv", class_rows)
    write_tsv(options.report_dir / "h500_prediction_hashes.tsv", prediction_rows)

    screen_path = bootstrap(options, passing, predictions, 1000, 65004, "h500-screen")
    screen_rows = read_tsv(screen_path)
    write_tsv(options.report_dir / "h500_bootstrap_screen.tsv", screen_rows)
    baseline = metric_rows[0]
    metrics_by_lane = {row["surface"]: row for row in metric_rows}
    bootstrap_by_lane = {
        row["pair"].split("-A0", 1)[0]: row
        for row in screen_rows
        if row["metric"] == "map50_95"
    }
    decisions: list[dict[str, Any]] = []
    for lane in passing:
        row = metrics_by_lane[lane]
        bootstrap_row = bootstrap_by_lane[lane]
        delta = float(row["map50_95"]) - float(baseline["map50_95"])
        size_deltas = {
            name: float(row[name]) - float(baseline[name])
            for name in ("ap_small", "ap_medium", "ap_large")
        }
        qualified = all(
            (
                delta >= 0.005,
                float(bootstrap_row["percentile_2_5"]) > 0.0,
                min(size_deltas.values()) >= -0.005,
            )
        )
        decisions.append(
            {
                "lane": lane,
                "map50_95": row["map50_95"],
                "delta_vs_a0": delta,
                "ci_lower": bootstrap_row["percentile_2_5"],
                "ci_upper": bootstrap_row["percentile_97_5"],
                "probability_delta_gt_zero": bootstrap_row["probability_delta_gt_zero"],
                "ap_small_delta": size_deltas["ap_small"],
                "ap_medium_delta": size_deltas["ap_medium"],
                "ap_large_delta": size_deltas["ap_large"],
                "strong_success_threshold": STRONG_DELTA,
                "strong_success": int(
                    delta >= STRONG_DELTA
                    and float(bootstrap_row["percentile_2_5"]) > 0.0
                ),
                "qualified": int(qualified),
            }
        )
    selected = sorted(
        (row for row in decisions if row["qualified"]),
        key=lambda row: (-float(row["map50_95"]), row["lane"]),
    )[:2]
    selected_lanes = [str(row["lane"]) for row in selected]
    if selected_lanes:
        final_path = bootstrap(
            options,
            selected_lanes,
            predictions,
            10_000,
            65005,
            "h500-final",
        )
        final_rows = read_tsv(final_path)
    else:
        final_rows = [
            {
                "status": "not-run-no-qualified-candidate",
                "replicates": 0,
                "seed": 65005,
            }
        ]
    write_tsv(options.report_dir / "h500_bootstrap_final.tsv", final_rows)
    for row in decisions:
        row["selected_for_full_val"] = int(row["lane"] in selected_lanes)
    write_tsv(options.report_dir / "h500_selection.tsv", decisions)
    (options.report_dir / "h500_selection_decision.md").write_text(
        "# H500 selection decision\n\n"
        f"Selected lanes: `{', '.join(selected_lanes) if selected_lanes else 'none'}`.\n\n"
        "Selection applied the predeclared +0.005 mAP, positive paired-CI, "
        "and no-size-bin-regression gates. The strong-success threshold was "
        f"computed from the accepted B2-to-FQ8-L3 H500 gap: `{STRONG_DELTA:.12f}`.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
