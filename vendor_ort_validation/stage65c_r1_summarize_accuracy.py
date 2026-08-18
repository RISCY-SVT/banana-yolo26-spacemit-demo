#!/usr/bin/env python3
"""Aggregate four-surface metrics and apply Stage65C-R1 statistical rules."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SURFACE_PATHS = {
    "B2_CPU": "B2-cpu",
    "B2_EP": "B2-spacemit",
    "A1_CPU": "A1-cpu",
    "A1_EP": "A1-spacemit",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise ValueError(f"refusing empty TSV: {path}")
    fields = list(values[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def interval_class(row: dict[str, str], tolerance: float) -> str:
    lower = float(row["percentile_2_5"])
    upper = float(row["percentile_97_5"])
    if lower >= -tolerance and upper <= tolerance:
        return "equivalent"
    if upper < -tolerance:
        return "negative-interaction"
    return "inconclusive"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("h500", "val"), required=True)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--bootstrap-root", required=True, type=Path)
    parser.add_argument("--metrics-root", required=True, type=Path)
    parser.add_argument("--tracked-root", required=True, type=Path)
    options = parser.parse_args()
    dataset_root = Path("/data/datasets/coco2017-independent-stage65b-r1")
    repo = Path(__file__).resolve().parent
    if options.dataset == "h500":
        annotations = dataset_root / "annotations/instances_train2017.json"
        image_list = dataset_root / "lists/selection_H500_holdout.txt"
        metrics_name = "h500_complete_metrics.tsv"
        bootstrap_name = "h500_complete_bootstrap.tsv"
        interaction_name = "h500_interaction_bootstrap.tsv"
        decision_name = "h500_statistical_interpretation.md"
    else:
        annotations = dataset_root / "annotations/instances_val2017.json"
        image_list = dataset_root / "lists/val2017_all.txt"
        metrics_name = "full_val_diagnostic_metrics.tsv"
        bootstrap_name = "full_val_complete_bootstrap.tsv"
        interaction_name = "full_val_interaction_bootstrap.tsv"
        decision_name = "full_val_statistical_decision.md"
    options.metrics_root.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    size_rows = []
    class_rows = []
    for surface, directory in SURFACE_PATHS.items():
        prediction = options.prediction_root / directory / "predictions.json"
        output = options.metrics_root / surface
        if not (output / "results.tsv").is_file():
            subprocess.run(
                [
                    sys.executable, str(repo / "stage65b_r1_coco_metrics.py"),
                    "--annotations", str(annotations), "--predictions", str(prediction),
                    "--image-list", str(image_list), "--surface", surface,
                    "--output-dir", str(output),
                ],
                check=True,
            )
        metric_rows.extend(read_tsv(output / "results.tsv"))
        size_rows.extend(read_tsv(output / "size_bins.tsv"))
        class_rows.extend(read_tsv(output / "per_class.tsv"))
    write_tsv(options.tracked_root / metrics_name, metric_rows)
    if options.dataset == "val":
        write_tsv(options.tracked_root / "full_val_diagnostic_size_bins.tsv", size_rows)
        write_tsv(options.tracked_root / "full_val_diagnostic_per_class.tsv", class_rows)
    shutil.copy2(options.bootstrap_root / "complete_bootstrap.tsv", options.tracked_root / bootstrap_name)
    shutil.copy2(options.bootstrap_root / "interaction_bootstrap.tsv", options.tracked_root / interaction_name)

    pairs = {(row["pair"], row["metric"]): row for row in read_tsv(options.bootstrap_root / "complete_bootstrap.tsv")}
    interactions = {row["metric"]: row for row in read_tsv(options.bootstrap_root / "interaction_bootstrap.tsv")}
    interaction_classes = {
        metric: interval_class(row, 0.001 if metric == "map50_95" else 0.003)
        for metric, row in interactions.items()
        if metric in {"map50_95", "ap_small", "ap_medium", "ap_large", "ar_small", "ar_medium", "ar_large"}
    }
    lines = [
        f"# {'H500' if options.dataset == 'h500' else 'Full val2017'} statistical decision",
        "",
        "Point thresholds and interval classifications are reported separately.",
        "",
        "| Metric | Point interaction | 95% CI | Classification |",
        "|---|---:|---:|---|",
    ]
    for metric, classification in interaction_classes.items():
        row = interactions[metric]
        lines.append(
            f"| {metric} | {float(row['point_interaction']):.12f} | "
            f"[{float(row['percentile_2_5']):.12f}, {float(row['percentile_97_5']):.12f}] | {classification} |"
        )

    if options.dataset == "h500":
        lines.extend(
            [
                "",
                "This surface revalidates the historical H500 gate. Full val2017 remains the authority for the R1 classification.",
            ]
        )
    else:
        intrinsic = pairs[("A1_CPU-vs-B2_CPU", "ar_large")]
        intrinsic_point = float(intrinsic["point_delta"])
        intrinsic_upper = float(intrinsic["percentile_97_5"])
        intrinsic_confirmed = intrinsic_point < -0.005 and intrinsic_upper < 0.0
        intrinsic_strong = intrinsic_upper < -0.005
        provider_negative = interaction_classes["ar_large"] == "negative-interaction"
        a1_ep_map = pairs[("A1_EP-vs-B2_EP", "map50_95")]
        full_contract = float(a1_ep_map["point_delta"]) >= 0.005 and float(a1_ep_map["percentile_2_5"]) > 0.0
        for metric in ("ap_small", "ap_medium", "ap_large", "ar_small", "ar_medium", "ar_large"):
            full_contract = full_contract and float(pairs[("A1_EP-vs-B2_EP", metric)]["point_delta"]) >= -0.005
        no_material_negative = interaction_classes["map50_95"] != "negative-interaction" and interaction_classes["ar_large"] != "negative-interaction"
        sampling_artifact = full_contract and no_material_negative
        if intrinsic_confirmed and provider_negative:
            classification = "stage65c-r1-a1-model-intrinsic-and-spacemit-ep-specific-large-recall-loss-confirmed"
        elif intrinsic_confirmed:
            classification = "stage65c-r1-a1-model-intrinsic-precision-gain-large-recall-loss-confirmed"
        elif provider_negative:
            classification = "stage65c-r1-a1-spacemit-ep-specific-terminal-or-tail-large-recall-interaction-confirmed"
        elif sampling_artifact:
            classification = "stage65c-r1-h500-recall-failure-not-confirmed-full-val-a1-board-accuracy-route-ready-for-separate-performance-review"
        else:
            classification = "stage65c-r1-recall-causality-inconclusive-frozen-a1-remains-blocked"
        lines.extend(
            [
                "",
                f"- Model-intrinsic AR-large trade-off: `{'confirmed' if intrinsic_confirmed else 'not-confirmed'}`.",
                f"- Strong intrinsic threshold: `{'pass' if intrinsic_strong else 'not-pass'}`.",
                f"- Provider-specific AR-large interaction: `{'confirmed' if provider_negative else interaction_classes['ar_large']}`.",
                f"- H500 sampling-artifact rule: `{'satisfied' if sampling_artifact else 'not-satisfied'}`.",
                f"- Primary classification: `{classification}`.",
            ]
        )
        (options.tracked_root / "primary_classification.txt").write_text(classification + "\n", encoding="utf-8")
    (options.tracked_root / decision_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
