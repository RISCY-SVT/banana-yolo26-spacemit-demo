#!/usr/bin/env python3
"""Run bounded full-val evaluation for the H500-selected DEV-001A lanes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
from pathlib import Path
from typing import Any

B2_FULL_MAP = 0.3658592288412378
FQ8_L3_FULL_MAP = 0.3977524934214979
H8_FULL_MAP = 0.4018217950262668


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--val-list", required=True, type=Path)
    parser.add_argument("--b2-predictions", required=True, type=Path)
    parser.add_argument("--b2-sha256", required=True)
    parser.add_argument("--fq8-l3-predictions", required=True, type=Path)
    parser.add_argument("--fq8-l3-sha256", required=True)
    parser.add_argument("--h8-predictions", required=True, type=Path)
    parser.add_argument("--h8-sha256", required=True)
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
    output_dir = options.raw_root / "full-val-metrics" / surface
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
            str(options.val_list),
            "--surface",
            surface,
            "--output-dir",
            str(output_dir),
        ],
        options.raw_root / "logs" / f"full-val-metrics-{surface}.log",
    )
    return result


def predictions(options: argparse.Namespace, lane: str) -> Path:
    output_dir = options.raw_root / "full-val" / lane
    path = output_dir / "predictions.json"
    if path.exists():
        return path
    model_root = options.raw_root / "postprocess" / lane / "models"
    invoke(
        [
            str(options.python),
            str(options.repo / "vendor_ort_validation/stage65b_r1_evaluate.py"),
            "run",
            "--candidate-inference",
            str(model_root / f"stage65b_r1_{lane.lower()}.inference.onnx"),
            "--tail",
            str(model_root / f"stage65b_r1_{lane.lower()}.postprocess.onnx"),
            "--image-list",
            str(options.val_list),
            "--output-dir",
            str(output_dir),
            "--name",
            lane,
            "--threads",
            str(options.threads),
            "--log-every",
            "100",
        ],
        options.raw_root / "logs" / f"full-val-predict-{lane}.log",
    )
    return path


def main() -> int:
    options = parse_args()
    options.report_dir.mkdir(parents=True, exist_ok=True)
    expected = {
        options.b2_predictions: options.b2_sha256,
        options.fq8_l3_predictions: options.fq8_l3_sha256,
        options.h8_predictions: options.h8_sha256,
    }
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise ValueError(f"frozen full-val prediction identity mismatch: {path}")
    selection = read_tsv(options.report_dir / "h500_selection.tsv")
    lanes = [row["lane"] for row in selection if row["selected_for_full_val"] == "1"]
    if len(lanes) > 2:
        raise ValueError("more than two H500-selected lanes")
    if not lanes:
        marker = {"status": "not-run-no-h500-qualified-candidate"}
        for name in (
            "full_val_metrics.tsv",
            "full_val_size_bins.tsv",
            "full_val_per_class.tsv",
            "full_val_bootstrap.tsv",
        ):
            write_tsv(options.report_dir / name, [marker])
        (options.report_dir / "full_val_candidate_decision.md").write_text(
            "# Full-val candidate decision\n\n"
            "No candidate passed the H500 qualification gate; full val2017 was not opened.\n",
            encoding="utf-8",
        )
        return 0

    prediction_paths = {"A0": options.b2_predictions}
    metric_paths = {"A0": metrics(options, "A0", options.b2_predictions)}
    for lane in lanes:
        prediction_paths[lane] = predictions(options, lane)
        metric_paths[lane] = metrics(options, lane, prediction_paths[lane])
    metric_rows = [read_tsv(metric_paths[lane])[0] for lane in ("A0", *lanes)]
    size_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for lane in ("A0", *lanes):
        size_rows.extend(read_tsv(metric_paths[lane].parent / "size_bins.tsv"))
        class_rows.extend(read_tsv(metric_paths[lane].parent / "per_class.tsv"))

    bootstrap_root = options.raw_root / "bootstrap" / "full-val-final"
    bootstrap_report = bootstrap_root / "paired_bootstrap_results.tsv"
    if not bootstrap_report.exists():
        command = [
            str(options.python),
            str(options.repo / "vendor_ort_validation/stage65b_r2_bootstrap.py"),
            "--annotations",
            str(options.annotations),
            "--image-list",
            str(options.val_list),
        ]
        for lane in lanes:
            command.extend(
                [
                    "--pair",
                    f"{lane}-A0={prediction_paths[lane]},{prediction_paths['A0']}",
                ]
            )
        command.extend(
            [
                "--replicates",
                "10000",
                "--seed",
                "65006",
                "--workers",
                str(options.workers),
                "--output-dir",
                str(bootstrap_root),
            ]
        )
        invoke(command, options.raw_root / "logs" / "bootstrap-full-val-final.log")
    bootstrap_rows = read_tsv(bootstrap_report)
    bootstrap_map = {
        row["pair"].split("-A0", 1)[0]: row
        for row in bootstrap_rows
        if row["metric"] == "map50_95"
    }
    baseline = metric_rows[0]
    decision_rows: list[dict[str, Any]] = []
    for row in metric_rows[1:]:
        lane = row["surface"]
        delta = float(row["map50_95"]) - float(baseline["map50_95"])
        size_deltas = {
            name: float(row[name]) - float(baseline[name])
            for name in ("ap_small", "ap_medium", "ap_large")
        }
        boot = bootstrap_map[lane]
        success = all(
            (
                delta >= 0.005,
                float(boot["percentile_2_5"]) > 0.0,
                min(size_deltas.values()) >= -0.005,
            )
        )
        row.update(
            {
                "delta_vs_b2": delta,
                "bootstrap_ci_lower": boot["percentile_2_5"],
                "bootstrap_ci_upper": boot["percentile_97_5"],
                "bootstrap_probability_delta_gt_zero": boot[
                    "probability_delta_gt_zero"
                ],
                "ap_small_delta": size_deltas["ap_small"],
                "ap_medium_delta": size_deltas["ap_medium"],
                "ap_large_delta": size_deltas["ap_large"],
                "b2_to_fq8_l3_gap_recovered": delta / (FQ8_L3_FULL_MAP - B2_FULL_MAP),
                "b2_to_h8_gap_recovered": delta / (H8_FULL_MAP - B2_FULL_MAP),
                "full_val_success": int(success),
                "prediction_path": str(prediction_paths[lane]),
                "prediction_sha256_current": sha256(prediction_paths[lane]),
            }
        )
        decision_rows.append(row)
    write_tsv(options.report_dir / "full_val_metrics.tsv", metric_rows)
    write_tsv(options.report_dir / "full_val_size_bins.tsv", size_rows)
    write_tsv(options.report_dir / "full_val_per_class.tsv", class_rows)
    write_tsv(options.report_dir / "full_val_bootstrap.tsv", bootstrap_rows)
    write_tsv(options.report_dir / "full_val_candidate_decision.tsv", decision_rows)
    winners = [row["surface"] for row in decision_rows if row["full_val_success"]]
    partial = [
        row["surface"]
        for row in decision_rows
        if float(row["delta_vs_b2"]) > 0 and not row["full_val_success"]
    ]
    (options.report_dir / "full_val_candidate_decision.md").write_text(
        "# Full-val candidate decision\n\n"
        f"Full gate winners: `{', '.join(winners) if winners else 'none'}`.\n\n"
        f"Positive but below-gate candidates: `{', '.join(partial) if partial else 'none'}`.\n\n"
        "Every opened lane was selected on H500 first; val2017 was reporting and final gate confirmation.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
