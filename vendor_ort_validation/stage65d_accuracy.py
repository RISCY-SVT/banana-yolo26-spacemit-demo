#!/usr/bin/env python3
"""Normalize Stage65D board accuracy and enforce frozen H500/full-val gates."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

SURFACES = ("B2_CPU", "B2_EP", "C2_CPU", "C2_EP")
DIRECTORIES = {
    "B2_CPU": "B2-cpu",
    "B2_EP": "B2-spacemit",
    "C2_CPU": "C2-cpu",
    "C2_EP": "C2-spacemit",
}
METRIC_COLUMNS = {
    "map50_95": "map50_95",
    "map50": "map50",
    "map75": "map75",
    "ap_small": "ap_small",
    "ap_medium": "ap_medium",
    "ap_large": "ap_large",
    "ar1": "ar_1",
    "ar10": "ar_10",
    "ar100": "ar_100",
    "ar_small": "ar_small",
    "ar_medium": "ar_medium",
    "ar_large": "ar_large",
    "prediction_count": "prediction_count",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, values: list[dict[str, object]]) -> None:
    if not values:
        raise ValueError(f"refusing empty output: {path}")
    fields = list(values[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def value(table: dict[str, dict[str, str]], surface: str, metric: str) -> float:
    return float(table[surface][METRIC_COLUMNS[metric]])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("h500", "val"), required=True)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--tracked-root", required=True, type=Path)
    options = parser.parse_args()

    repo = Path(__file__).resolve().parent
    dataset = Path("/data/datasets/coco2017-independent-stage65b-r1")
    if options.dataset == "h500":
        annotations = dataset / "annotations/instances_train2017.json"
        image_list = dataset / "lists/selection_H500_holdout.txt"
        prefix = "h500_board"
        seed = 65010
    else:
        annotations = dataset / "annotations/instances_val2017.json"
        image_list = dataset / "lists/val2017_all.txt"
        prefix = "full_val_board"
        seed = 65011

    metrics_root = options.raw_root / "metrics" / options.dataset
    bootstrap_root = options.raw_root / "bootstrap" / options.dataset
    metrics_root.mkdir(parents=True, exist_ok=True)
    prediction_paths: dict[str, Path] = {}
    for surface in SURFACES:
        prediction = options.prediction_root / DIRECTORIES[surface] / "predictions.json"
        if not prediction.is_file():
            raise FileNotFoundError(prediction)
        prediction_paths[surface] = prediction
        output = metrics_root / surface
        if not (output / "results.tsv").is_file():
            run([
                sys.executable,
                str(repo / "stage65b_r1_coco_metrics.py"),
                "--annotations", str(annotations),
                "--predictions", str(prediction),
                "--image-list", str(image_list),
                "--surface", surface,
                "--output-dir", str(output),
            ])

    if not (bootstrap_root / "complete_bootstrap.tsv").is_file():
        run([
            sys.executable,
            str(repo / "stage65d_bootstrap.py"),
            "--annotations", str(annotations),
            "--image-list", str(image_list),
            "--replicates", "10000",
            "--seed", str(seed),
            "--workers", "8",
            "--output-dir", str(bootstrap_root),
            *sum((["--surface", f"{name}={prediction_paths[name]}"] for name in SURFACES), []),
        ])

    aggregate: list[dict[str, str]] = []
    sizes: list[dict[str, str]] = []
    classes: list[dict[str, str]] = []
    for surface in SURFACES:
        aggregate.extend(read_tsv(metrics_root / surface / "results.tsv"))
        sizes.extend(read_tsv(metrics_root / surface / "size_bins.tsv"))
        classes.extend(read_tsv(metrics_root / surface / "per_class.tsv"))
    table = {row["surface"]: row for row in aggregate}
    write_tsv(options.tracked_root / f"{prefix}_metrics.tsv", aggregate)
    write_tsv(options.tracked_root / f"{prefix}_size_bins.tsv", sizes)
    write_tsv(options.tracked_root / f"{prefix}_per_class.tsv", classes)

    archive = np.load(bootstrap_root / "four_surface_replicates.npz")
    samples = archive["surface_metrics"]
    surfaces = [str(item) for item in archive["surfaces"]]
    metrics = [str(item) for item in archive["metrics"]]
    sidx = {name: surfaces.index(name) for name in surfaces}
    midx = {name: metrics.index(name) for name in metrics}

    bootstrap_rows = read_tsv(bootstrap_root / "complete_bootstrap.tsv")
    for row in bootstrap_rows:
        left, right = {
            "C2_EP-vs-B2_EP": ("C2_EP", "B2_EP"),
            "C2_CPU-vs-B2_CPU": ("C2_CPU", "B2_CPU"),
            "C2_EP-vs-C2_CPU": ("C2_EP", "C2_CPU"),
            "B2_EP-vs-B2_CPU": ("B2_EP", "B2_CPU"),
        }[row["pair"]]
        delta = samples[:, sidx[left], midx[row["metric"]]] - samples[:, sidx[right], midx[row["metric"]]]
        row["probability_ge_minus_0_005"] = str(float(np.mean(delta >= -0.005)))
    write_tsv(options.tracked_root / f"{prefix}_complete_bootstrap.tsv", bootstrap_rows)

    interaction_rows = read_tsv(bootstrap_root / "interaction_bootstrap.tsv")
    for row in interaction_rows:
        lower = float(row["percentile_2_5"])
        upper = float(row["percentile_97_5"])
        if row["metric"] == "map50_95":
            margin = 0.002
        elif row["metric"] == "prediction_count":
            margin = 0.0
        else:
            margin = 0.005
        if lower >= -margin and upper <= margin:
            classification = "provider-neutral"
        elif upper < -margin or lower > margin:
            classification = "provider-interaction-material"
        else:
            classification = "provider-interaction-inconclusive"
        row["practical_margin"] = str(margin)
        row["classification"] = classification
    write_tsv(options.tracked_root / f"{prefix}_provider_interactions.tsv", interaction_rows)

    checks: list[tuple[str, bool, str]] = []
    c2_b2_map = value(table, "C2_EP", "map50_95") - value(table, "B2_EP", "map50_95")
    map_bootstrap = next(row for row in bootstrap_rows if row["pair"] == "C2_EP-vs-B2_EP" and row["metric"] == "map50_95")
    if options.dataset == "h500":
        checks.extend([
            ("c2_ep_b2_ep_map_delta", c2_b2_map >= 0.004, f"{c2_b2_map:.12f} >= 0.004"),
            ("c2_ep_b2_ep_map_probability", float(map_bootstrap["probability_gt_zero"]) >= 0.95, f"{map_bootstrap['probability_gt_zero']} >= 0.95"),
        ])
        for name in ("ap_small", "ap_medium", "ap_large", "ar_small", "ar_medium", "ar_large"):
            delta = value(table, "C2_EP", name) - value(table, "B2_EP", name)
            row = next(item for item in bootstrap_rows if item["pair"] == "C2_EP-vs-B2_EP" and item["metric"] == name)
            checks.append((f"{name}_point", delta >= -0.005, f"{delta:.12f} >= -0.005"))
            checks.append((f"{name}_probability", float(row["probability_ge_minus_0_005"]) >= 0.90, f"{row['probability_ge_minus_0_005']} >= 0.90"))
    else:
        checks.extend([
            ("c2_ep_b2_ep_map_delta", c2_b2_map >= 0.005, f"{c2_b2_map:.12f} >= 0.005"),
            ("c2_ep_b2_ep_map_ci", float(map_bootstrap["percentile_2_5"]) > 0.0, f"{map_bootstrap['percentile_2_5']} > 0"),
        ])
        for name in ("ap_small", "ap_medium", "ap_large", "ar_small", "ar_medium", "ar_large"):
            delta = value(table, "C2_EP", name) - value(table, "B2_EP", name)
            row = next(item for item in bootstrap_rows if item["pair"] == "C2_EP-vs-B2_EP" and item["metric"] == name)
            checks.append((f"{name}_point", delta >= -0.003, f"{delta:.12f} >= -0.003"))
            checks.append((f"{name}_ci", float(row["percentile_2_5"]) >= -0.005, f"{row['percentile_2_5']} >= -0.005"))

    map_agreement = abs(value(table, "C2_EP", "map50_95") - value(table, "C2_CPU", "map50_95"))
    checks.append(("c2_cpu_ep_map_agreement", map_agreement <= 0.002, f"{map_agreement:.12f} <= 0.002"))
    for name in ("ap_small", "ap_medium", "ap_large", "ar_small", "ar_medium", "ar_large"):
        difference = abs(value(table, "C2_EP", name) - value(table, "C2_CPU", name))
        checks.append((f"c2_cpu_ep_{name}", difference <= 0.005, f"{difference:.12f} <= 0.005"))
    for surface in SURFACES:
        checks.append((f"{surface}_failures", int(table[surface]["failures"]) == 0, table[surface]["failures"]))
        checks.append((f"{surface}_non_finite", int(table[surface]["non_finite_predictions"]) == 0, table[surface]["non_finite_predictions"]))

    passed = all(check[1] for check in checks)
    lines = [
        f"# {'H500' if options.dataset == 'h500' else 'Full val2017'} board decision",
        "",
        f"Decision: `{'pass' if passed else 'fail'}`.",
        "",
        f"C2 EP - B2 EP mAP50-95: `{c2_b2_map:.12f}`.",
        f"C2 CPU/EP absolute mAP difference: `{map_agreement:.12f}`.",
        "",
        "| Check | Status | Evidence |",
        "|---|---|---|",
    ]
    lines.extend(f"| {name} | {'pass' if status else 'fail'} | {evidence} |" for name, status, evidence in checks)
    (options.tracked_root / f"{prefix}_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy2(bootstrap_root / "replicates.sha256", options.raw_root / f"{options.dataset}_bootstrap_replicates.sha256")

    if options.dataset == "val":
        host_rows = read_tsv(Path("/data/worktrees/banana-yolo26-xslim211-s8-qdq-validation/stages/BANANA-YOLO26-XSLIM-DEV-001C-C2-FROZEN-INDEPENDENT-HOLDOUT-ADJUDICATION-AND-VENDOR-PTQ-LANE-CLOSURE-001/full_val_metrics.tsv"))
        host = {row["surface"]: row for row in host_rows}
        transfer = []
        for model in ("B2", "C2"):
            for provider in ("CPU", "EP"):
                board_surface = f"{model}_{provider}"
                transfer.append({
                    "model": model,
                    "board_provider": provider,
                    "host_map50_95": host[model]["map50_95"],
                    "board_map50_95": table[board_surface]["map50_95"],
                    "board_minus_host": value(table, board_surface, "map50_95") - float(host[model]["map50_95"]),
                    "interpretation": "task-level transfer; prediction-byte equality not required",
                })
        write_tsv(options.tracked_root / "full_val_host_board_transfer.tsv", transfer)

    print(f"{options.dataset} status={'pass' if passed else 'fail'} c2_b2_map={c2_b2_map:.12f}")
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
