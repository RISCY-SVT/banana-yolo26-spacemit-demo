#!/usr/bin/env python3
"""Full-val four-surface adjudication for frozen Stage65D-R1 artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

SURFACES = ("B2_CPU", "B2_EP", "C2_CPU", "C2_EP")
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty output: {path}")
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_surface(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or name not in SURFACES or not raw_path:
        raise ValueError(f"invalid --surface: {value}")
    return name, Path(raw_path).resolve()


def metric(table: dict[str, dict[str, str]], surface: str, name: str) -> float:
    return float(table[surface][METRIC_COLUMNS[name]])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", action="append", required=True)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--tracked-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=65012)
    parser.add_argument("--workers", type=int, default=8)
    options = parser.parse_args()
    predictions = dict(parse_surface(value) for value in options.surface)
    if tuple(sorted(predictions)) != tuple(sorted(SURFACES)) or len(options.surface) != 4:
        raise ValueError(f"one path is required for every surface: {SURFACES}")
    for path in predictions.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    repo = Path(__file__).resolve().parent
    dataset = Path("/data/datasets/coco2017-independent-stage65b-r1")
    annotations = dataset / "annotations/instances_val2017.json"
    image_list = dataset / "lists/val2017_all.txt"
    metrics_root = options.raw_root / "metrics/full-val"
    bootstrap_root = options.raw_root / "bootstrap/full-val"
    metrics_root.mkdir(parents=True, exist_ok=True)

    for surface in SURFACES:
        output = metrics_root / surface
        if output.exists():
            raise RuntimeError(f"refusing existing metric root: {output}")
        subprocess.run(
            [
                sys.executable,
                str(repo / "stage65b_r1_coco_metrics.py"),
                "--annotations", str(annotations),
                "--predictions", str(predictions[surface]),
                "--image-list", str(image_list),
                "--surface", surface,
                "--output-dir", str(output),
            ],
            check=True,
        )

    if bootstrap_root.exists():
        raise RuntimeError(f"refusing existing bootstrap root: {bootstrap_root}")
    command = [
        sys.executable,
        str(repo / "stage65d_bootstrap.py"),
        "--annotations", str(annotations),
        "--image-list", str(image_list),
        "--replicates", "10000",
        "--seed", str(options.seed),
        "--workers", str(options.workers),
        "--output-dir", str(bootstrap_root),
    ]
    for surface in SURFACES:
        command.extend(["--surface", f"{surface}={predictions[surface]}"])
    subprocess.run(command, check=True)

    aggregate: list[dict[str, str]] = []
    sizes: list[dict[str, str]] = []
    classes: list[dict[str, str]] = []
    hashes: list[dict[str, object]] = []
    collapse_status: dict[str, bool] = {}
    for surface in SURFACES:
        aggregate.extend(read_tsv(metrics_root / surface / "results.tsv"))
        sizes.extend(read_tsv(metrics_root / surface / "size_bins.tsv"))
        classes.extend(read_tsv(metrics_root / surface / "per_class.tsv"))
        payload = json.loads(predictions[surface].read_text(encoding="utf-8"))
        scores = [float(row["score"]) for row in payload]
        classes_seen = {int(row["category_id"]) for row in payload}
        collapsed = (
            not scores
            or any(not math.isfinite(score) for score in scores)
            or min(scores) == max(scores)
            or len({score.hex() for score in scores}) < 2
            or len(classes_seen) < 2
        )
        collapse_status[surface] = collapsed
        hashes.append({
            "surface": surface,
            "path": predictions[surface],
            "bytes": predictions[surface].stat().st_size,
            "sha256": sha256(predictions[surface]),
            "score_collapse": "yes" if collapsed else "no",
            "status": "fail" if collapsed else "pass",
        })
    table = {row["surface"]: row for row in aggregate}
    write_tsv(options.tracked_root / "full_val_board_metrics.tsv", aggregate)
    write_tsv(options.tracked_root / "full_val_board_size_bins.tsv", sizes)
    write_tsv(options.tracked_root / "full_val_board_per_class.tsv", classes)
    write_tsv(options.tracked_root / "full_val_board_prediction_hashes.tsv", hashes)

    archive = np.load(bootstrap_root / "four_surface_replicates.npz")
    samples = archive["surface_metrics"]
    surfaces = [str(value) for value in archive["surfaces"]]
    metrics = [str(value) for value in archive["metrics"]]
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
    write_tsv(options.tracked_root / "full_val_board_complete_bootstrap.tsv", bootstrap_rows)

    interaction_rows = read_tsv(bootstrap_root / "interaction_bootstrap.tsv")
    count_rows: list[dict[str, object]] = []
    for row in interaction_rows:
        lower = float(row["percentile_2_5"])
        upper = float(row["percentile_97_5"])
        metric_name = row["metric"]
        if metric_name == "prediction_count":
            row["practical_margin"] = "descriptive-only"
            row["classification"] = "descriptive-only"
            count_rows.append(dict(row))
            continue
        margin = 0.002 if metric_name == "map50_95" else 0.005
        if lower >= -margin and upper <= margin:
            classification = "provider-neutral"
        elif upper < -margin:
            classification = "provider-interaction-material-negative"
        elif lower > margin:
            classification = "provider-interaction-material-positive"
        else:
            classification = "provider-interaction-inconclusive"
        row["practical_margin"] = str(margin)
        row["classification"] = classification
    write_tsv(options.tracked_root / "full_val_board_provider_interactions.tsv", interaction_rows)
    write_tsv(options.tracked_root / "full_val_prediction_count_interactions.tsv", count_rows)

    task_checks: list[tuple[str, bool, str]] = []
    c2_b2_map = metric(table, "C2_EP", "map50_95") - metric(table, "B2_EP", "map50_95")
    map_row = next(row for row in bootstrap_rows if row["pair"] == "C2_EP-vs-B2_EP" and row["metric"] == "map50_95")
    task_checks.extend([
        ("c2_ep_b2_ep_map_delta", c2_b2_map >= 0.005, f"{c2_b2_map:.12f} >= 0.005"),
        ("c2_ep_b2_ep_map_ci", float(map_row["percentile_2_5"]) > 0.0, f"{map_row['percentile_2_5']} > 0"),
    ])
    for name in ("ap_small", "ap_medium", "ap_large", "ar_small", "ar_medium", "ar_large"):
        delta = metric(table, "C2_EP", name) - metric(table, "B2_EP", name)
        row = next(item for item in bootstrap_rows if item["pair"] == "C2_EP-vs-B2_EP" and item["metric"] == name)
        task_checks.append((f"{name}_point", delta >= -0.003, f"{delta:.12f} >= -0.003"))
        task_checks.append((f"{name}_ci", float(row["percentile_2_5"]) >= -0.005, f"{row['percentile_2_5']} >= -0.005"))
    for surface in SURFACES:
        task_checks.append((f"{surface}_images", int(table[surface]["images"]) == 5000, table[surface]["images"]))
        task_checks.append((f"{surface}_failures", int(table[surface]["failures"]) == 0, table[surface]["failures"]))
        task_checks.append((f"{surface}_non_finite", int(table[surface]["non_finite_predictions"]) == 0, table[surface]["non_finite_predictions"]))
        task_checks.append((f"{surface}_score_collapse", not collapse_status[surface], str(collapse_status[surface])))

    warning_rows = []
    for name in ("map50_95", "ap_small", "ap_medium", "ap_large", "ar_small", "ar_medium", "ar_large"):
        absolute = abs(metric(table, "C2_EP", name) - metric(table, "C2_CPU", name))
        warning_rows.append({
            "metric": name,
            "c2_ep_minus_cpu": metric(table, "C2_EP", name) - metric(table, "C2_CPU", name),
            "absolute_difference": absolute,
            "historical_point_margin": 0.002 if name == "map50_95" else 0.005,
            "role": "secondary-warning-not-stage-stop",
        })
    write_tsv(options.tracked_root / "full_val_cpu_ep_absolute_warnings.tsv", warning_rows)

    host_metrics_path = Path(
        "/data/worktrees/banana-yolo26-xslim211-s8-qdq-validation/stages/"
        "BANANA-YOLO26-XSLIM-DEV-001C-C2-FROZEN-INDEPENDENT-HOLDOUT-"
        "ADJUDICATION-AND-VENDOR-PTQ-LANE-CLOSURE-001/full_val_metrics.tsv"
    )
    host = {row["surface"]: row for row in read_tsv(host_metrics_path)}
    transfer = []
    for model in ("B2", "C2"):
        for provider in ("CPU", "EP"):
            surface = f"{model}_{provider}"
            transfer.append({
                "model": model,
                "board_provider": provider,
                "host_map50_95": host[model]["map50_95"],
                "board_map50_95": table[surface]["map50_95"],
                "board_minus_host": metric(table, surface, "map50_95") - float(host[model]["map50_95"]),
                "interpretation": "task-level transfer; byte equality not required",
            })
    write_tsv(options.tracked_root / "full_val_host_board_transfer.tsv", transfer)

    task_pass = all(status for _, status, _ in task_checks)
    material = [row for row in interaction_rows if str(row["classification"]).startswith("provider-interaction-material")]
    lines = [
        "# Stage65D-R1 full val2017 decision",
        "",
        f"Primary C2 EP versus B2 EP task gate: `{'pass' if task_pass else 'fail'}`.",
        f"C2 EP - B2 EP mAP50-95: `{c2_b2_map:.12f}`; 95% CI `[{map_row['percentile_2_5']}, {map_row['percentile_97_5']}]`.",
        f"Provider interaction material metrics: `{len(material)}`. Absolute CPU/EP point gaps are warnings only in this Stage.",
        "",
        "| Check | Status | Evidence |",
        "|---|---|---|",
    ]
    lines.extend(f"| {name} | {'pass' if status else 'fail'} | {evidence} |" for name, status, evidence in task_checks)
    (options.tracked_root / "full_val_board_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy2(bootstrap_root / "replicates.sha256", options.raw_root / "full_val_bootstrap_replicates.sha256")
    print(f"full_val task_status={'pass' if task_pass else 'fail'} c2_ep_b2_ep_map={c2_b2_map:.12f} material_interactions={len(material)}")
    return 0 if task_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
