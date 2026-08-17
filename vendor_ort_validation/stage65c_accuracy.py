#!/usr/bin/env python3
"""Evaluate Stage65C board predictions, bootstrap pairs, and emit gate reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

SURFACES = ("B2-cpu", "B2-spacemit", "A1-cpu", "A1-spacemit")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write(path: Path, columns: list[str], values: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for value in values:
            writer.writerow({column: value.get(column, "") for column in columns})


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def metric(table: dict[str, dict[str, str]], surface: str, name: str) -> float:
    return float(table[surface][name])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("h500", "val"), required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--bootstrap-root", type=Path, required=True)
    parser.add_argument("--tracked-root", type=Path, required=True)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent
    dataset_root = Path("/data/datasets/coco2017-independent-stage65b-r1")
    if args.dataset == "h500":
        annotations = dataset_root / "annotations/instances_train2017.json"
        image_list = dataset_root / "lists/selection_H500_holdout.txt"
        prefix = "h500_board"
        pairs = (
            "A1_EP-vs-B2_EP=A1-spacemit,B2-spacemit",
            "A1_CPU-vs-B2_CPU=A1-cpu,B2-cpu",
            "A1_EP-vs-A1_CPU=A1-spacemit,A1-cpu",
        )
        seed = 65007
    else:
        annotations = dataset_root / "annotations/instances_val2017.json"
        image_list = dataset_root / "lists/val2017_all.txt"
        prefix = "full_coco_board"
        pairs = (
            "A1_EP-vs-B2_EP=A1-spacemit,B2-spacemit",
            "A1_CPU-vs-B2_CPU=A1-cpu,B2-cpu",
            "A1_EP-vs-A1_CPU=A1-spacemit,A1-cpu",
            "B2_EP-vs-B2_CPU=B2-spacemit,B2-cpu",
        )
        seed = 65008

    args.metrics_root.mkdir(parents=True, exist_ok=True)
    prediction_paths: dict[str, Path] = {}
    for surface in SURFACES:
        prediction = args.prediction_root / surface / "predictions.json"
        if not prediction.is_file():
            raise FileNotFoundError(prediction)
        prediction_paths[surface] = prediction
        output = args.metrics_root / surface
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

    args.bootstrap_root.mkdir(parents=True, exist_ok=True)
    pair_result_rows: list[dict[str, str]] = []
    pair_hash_lines: list[str] = []
    for spec in pairs:
        name, pair = spec.split("=", 1)
        left, right = pair.split(",", 1)
        pair_root = args.bootstrap_root / name
        if not (pair_root / "paired_bootstrap_results.tsv").is_file():
            run([
                sys.executable,
                str(repo / "stage65c_bootstrap.py"),
                "--annotations", str(annotations),
                "--image-list", str(image_list),
                "--replicates", "10000",
                "--seed", str(seed),
                "--workers", str(min(8, os.cpu_count() or 1)),
                "--validate-remap",
                "--output-dir", str(pair_root),
                "--pair",
                f"{name}={prediction_paths[left]},{prediction_paths[right]}",
            ])
        pair_result_rows.extend(rows(pair_root / "paired_bootstrap_results.tsv"))
        for line in (pair_root / "paired_bootstrap_replicates.sha256").read_text(
            encoding="utf-8"
        ).splitlines():
            digest, separator, filename = line.partition("  ")
            if separator:
                pair_hash_lines.append(f"{digest}  {name}/{filename}")
    write(
        args.bootstrap_root / "paired_bootstrap_results.tsv",
        list(pair_result_rows[0]),
        pair_result_rows,
    )
    (args.bootstrap_root / "paired_bootstrap_replicates.sha256").write_text(
        "\n".join(pair_hash_lines) + "\n", encoding="utf-8"
    )

    aggregate_metrics: list[dict[str, str]] = []
    aggregate_sizes: list[dict[str, str]] = []
    aggregate_classes: list[dict[str, str]] = []
    for surface in SURFACES:
        aggregate_metrics.extend(rows(args.metrics_root / surface / "results.tsv"))
        aggregate_sizes.extend(rows(args.metrics_root / surface / "size_bins.tsv"))
        aggregate_classes.extend(rows(args.metrics_root / surface / "per_class.tsv"))
    metric_table = {row["surface"]: row for row in aggregate_metrics}
    write(args.tracked_root / f"{prefix}_metrics.tsv", list(aggregate_metrics[0]), aggregate_metrics)
    write(args.tracked_root / f"{prefix}_size_bins.tsv", list(aggregate_sizes[0]), aggregate_sizes)
    write(args.tracked_root / f"{prefix}_per_class.tsv", list(aggregate_classes[0]), aggregate_classes)

    bootstrap_rows = rows(args.bootstrap_root / "paired_bootstrap_results.tsv")
    write(args.tracked_root / f"{prefix}_bootstrap.tsv", list(bootstrap_rows[0]), bootstrap_rows)
    shutil.copy2(
        args.bootstrap_root / "paired_bootstrap_replicates.sha256",
        args.tracked_root / f"{prefix}_bootstrap_replicates.sha256",
    )

    bootstrap_table = {(row["pair"], row["metric"]): row for row in bootstrap_rows}
    a1_ep_delta = metric(metric_table, "A1-spacemit", "map50_95") - metric(metric_table, "B2-spacemit", "map50_95")
    map_ci = bootstrap_table[("A1_EP-vs-B2_EP", "map50_95")]
    checks: list[tuple[str, bool, str]] = [
        ("map_delta", a1_ep_delta >= 0.005, f"{a1_ep_delta:.12f} >= 0.005"),
        ("map_ci_lower", float(map_ci["percentile_2_5"]) > 0.0, f"{map_ci['percentile_2_5']} > 0"),
    ]
    for metric_name in ("ap_small", "ap_medium", "ap_large", "ar_small", "ar_medium", "ar_large"):
        delta = metric(metric_table, "A1-spacemit", metric_name) - metric(metric_table, "B2-spacemit", metric_name)
        checks.append((f"{metric_name}_delta", delta >= -0.005, f"{delta:.12f} >= -0.005"))
    cpu_ep_map = abs(metric(metric_table, "A1-spacemit", "map50_95") - metric(metric_table, "A1-cpu", "map50_95"))
    checks.append(("a1_cpu_ep_map", cpu_ep_map <= 0.001, f"{cpu_ep_map:.12f} <= 0.001"))
    for metric_name in ("ap_small", "ap_medium", "ap_large", "ar_small", "ar_medium", "ar_large"):
        difference = abs(metric(metric_table, "A1-spacemit", metric_name) - metric(metric_table, "A1-cpu", metric_name))
        checks.append((f"a1_cpu_ep_{metric_name}", difference <= 0.003, f"{difference:.12f} <= 0.003"))
    for surface in SURFACES:
        checks.append((f"{surface}_failures", int(metric_table[surface]["failures"]) == 0, metric_table[surface]["failures"]))
        checks.append((f"{surface}_non_finite", int(metric_table[surface]["non_finite_predictions"]) == 0, metric_table[surface]["non_finite_predictions"]))

    b2_count = int(metric_table["B2-spacemit"]["prediction_count"])
    a1_count = int(metric_table["A1-spacemit"]["prediction_count"])
    ratio = a1_count / b2_count
    if args.dataset == "val":
        checks.append(("prediction_count_ratio", abs(ratio - 0.960105089) <= 0.03, f"{ratio:.12f}, host=0.960105089 +/- 0.03"))

    decision = all(value for _, value, _ in checks)
    lines = [
        f"# {'H500' if args.dataset == 'h500' else 'Full COCO'} board decision",
        "",
        f"Decision: `{'pass' if decision else 'fail'}`.",
        "",
        f"A1 EP - B2 EP mAP50-95: `{a1_ep_delta:.12f}`.",
        f"A1 EP/B2 EP prediction-count ratio: `{ratio:.12f}`.",
        "",
        "| Check | Status | Evidence |",
        "|---|---|---|",
    ]
    lines.extend(f"| {name} | {'pass' if value else 'fail'} | {evidence} |" for name, value, evidence in checks)
    (args.tracked_root / f"{prefix}_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{args.dataset} decision={'pass' if decision else 'fail'} delta={a1_ep_delta:.12f}")
    return 0 if decision else 3


if __name__ == "__main__":
    raise SystemExit(main())
