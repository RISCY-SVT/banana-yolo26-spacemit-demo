#!/usr/bin/env python3
"""Apply the frozen DEV-001C H5000 and conditional full-val gates."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


SURFACES = ("B2", "A1", "C2")
SIZE_METRICS = (
    "ap_small",
    "ap_medium",
    "ap_large",
    "ar_small",
    "ar_medium",
    "ar_large",
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows:
        raise ValueError(f"refusing empty TSV: {path}")
    return rows


def write_tsv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise ValueError(f"refusing empty TSV: {path}")
    fields: list[str] = []
    for row in values:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(values)


def metric_rows(metrics_root: Path) -> list[dict[str, str]]:
    rows = []
    for surface in SURFACES:
        values = read_tsv(metrics_root / surface / "results.tsv")
        if len(values) != 1 or values[0]["surface"] != surface:
            raise ValueError(f"invalid aggregate metrics for {surface}")
        rows.extend(values)
    return rows


def combined_rows(metrics_root: Path, filename: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for surface in SURFACES:
        values = read_tsv(metrics_root / surface / filename)
        if any(row["surface"] != surface for row in values):
            raise ValueError(f"surface mismatch in {surface}/{filename}")
        rows.extend(values)
    return rows


def indexed(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate {key} rows")
    return result


def bootstrap_index(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    result = {(row["pair"], row["metric"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate pair/metric bootstrap rows")
    return result


def interval_classification(lower: float, upper: float) -> str:
    if lower >= -0.005:
        return "interval-pass"
    if upper < -0.005:
        return "interval-fail"
    return "interval-inconclusive"


def check_metric_identity(
    point_rows: list[dict[str, str]], bootstrap_metrics: list[dict[str, str]]
) -> None:
    points = indexed(point_rows, "surface")
    bootstrap = indexed(bootstrap_metrics, "surface")
    mapping = {"ar1": "ar_1", "ar10": "ar_10", "ar100": "ar_100"}
    for surface in SURFACES:
        if points[surface]["prediction_sha256"] != bootstrap[surface]["prediction_sha256"]:
            raise RuntimeError(f"prediction hash mismatch for {surface}")
        for metric in (
            "map50_95",
            "map50",
            "map75",
            *SIZE_METRICS,
            "ar1",
            "ar10",
            "ar100",
            "prediction_count",
        ):
            point_name = mapping.get(metric, metric)
            difference = abs(
                float(points[surface][point_name])
                - float(bootstrap[surface][metric])
            )
            if difference > 1.0e-12:
                raise RuntimeError(
                    f"metric mismatch for {surface}/{metric}: {difference}"
                )


def h5000_decision(
    metrics: list[dict[str, str]], bootstrap: list[dict[str, str]]
) -> tuple[bool, list[dict[str, Any]]]:
    points = indexed(metrics, "surface")
    paired = bootstrap_index(bootstrap)
    gates: list[dict[str, Any]] = []

    def gate(name: str, observed: Any, required: str, passed: bool) -> None:
        gates.append(
            {
                "gate": name,
                "observed": observed,
                "required": required,
                "status": "pass" if passed else "fail",
            }
        )

    map_row = paired[("C2-vs-B2", "map50_95")]
    map_delta = float(map_row["point_delta"])
    map_probability = float(map_row["probability_gt_zero"])
    gate("C2-B2 mAP point", map_delta, ">=0.004", map_delta >= 0.004)
    gate(
        "C2-B2 P(mAP delta>0)",
        map_probability,
        ">=0.95",
        map_probability >= 0.95,
    )
    for metric in SIZE_METRICS:
        row = paired[("C2-vs-B2", metric)]
        point = float(row["point_delta"])
        probability = float(row["probability_ge_minus_0_005"])
        lower = float(row["percentile_2_5"])
        upper = float(row["percentile_97_5"])
        gate(f"C2-B2 {metric} point", point, ">=-0.005", point >= -0.005)
        gate(
            f"C2-B2 P({metric} delta>=-0.005)",
            probability,
            ">=0.90",
            probability >= 0.90,
        )
        gate(
            f"C2-B2 {metric} interval",
            f"[{lower},{upper}] {interval_classification(lower, upper)}",
            "descriptive-only",
            True,
        )

    c2_map = float(points["C2"]["map50_95"])
    a1_map = float(points["A1"]["map50_95"])
    c2_ar_large = float(points["C2"]["ar_large"])
    a1_ar_large = float(points["A1"]["ar_large"])
    gate(
        "C2 mAP Pareto vs A1",
        c2_map - a1_map,
        ">=-0.001",
        c2_map >= a1_map - 0.001,
    )
    gate(
        "C2 AR-large repair vs A1",
        c2_ar_large - a1_ar_large,
        ">=0.002",
        c2_ar_large >= a1_ar_large + 0.002,
    )
    return all(row["status"] == "pass" for row in gates), gates


def full_val_decision(
    metrics: list[dict[str, str]], bootstrap: list[dict[str, str]]
) -> tuple[bool, list[dict[str, Any]]]:
    points = indexed(metrics, "surface")
    paired = bootstrap_index(bootstrap)
    gates: list[dict[str, Any]] = []

    def gate(name: str, observed: Any, required: str, passed: bool) -> None:
        gates.append(
            {
                "gate": name,
                "observed": observed,
                "required": required,
                "status": "pass" if passed else "fail",
            }
        )

    map_row = paired[("C2-vs-B2", "map50_95")]
    point = float(map_row["point_delta"])
    lower = float(map_row["percentile_2_5"])
    gate("C2-B2 mAP point", point, ">=0.005", point >= 0.005)
    gate("C2-B2 mAP CI lower", lower, ">0", lower > 0)
    for metric in SIZE_METRICS:
        row = paired[("C2-vs-B2", metric)]
        point = float(row["point_delta"])
        lower = float(row["percentile_2_5"])
        gate(f"C2-B2 {metric} point", point, ">=-0.003", point >= -0.003)
        gate(f"C2-B2 {metric} CI lower", lower, ">=-0.005", lower >= -0.005)

    c2_map = float(points["C2"]["map50_95"])
    a1_map = float(points["A1"]["map50_95"])
    ar_row = paired[("C2-vs-A1", "ar_large")]
    ar_delta = float(ar_row["point_delta"])
    ar_probability = float(ar_row["probability_gt_zero"])
    gate(
        "C2 mAP Pareto vs A1",
        c2_map - a1_map,
        ">=-0.001",
        c2_map >= a1_map - 0.001,
    )
    gate("C2 AR-large repair vs A1", ar_delta, ">=0.002", ar_delta >= 0.002)
    gate(
        "P(C2-A1 AR-large>0)",
        ar_probability,
        ">=0.95",
        ar_probability >= 0.95,
    )
    return all(row["status"] == "pass" for row in gates), gates


def decision_markdown(kind: str, opened: bool, gates: list[dict[str, Any]]) -> str:
    result = "pass" if opened else "fail"
    next_step = (
        "conditional full val2017 opened"
        if kind == "h5000" and opened
        else "conditional full val2017 not opened; vendor PTQ lane closed"
        if kind == "h5000"
        else "frozen C2 host Pareto pass"
        if opened
        else "full-val Pareto fail; vendor PTQ lane closed"
    )
    lines = [
        f"# DEV-001C {kind} decision",
        "",
        f"Decision: `{result}`.",
        "",
        f"Disposition: {next_step}.",
        "",
        "| Gate | Observed | Required | Status |",
        "|---|---:|---:|---|",
    ]
    lines.extend(
        f"| {row['gate']} | {row['observed']} | {row['required']} | {row['status']} |"
        for row in gates
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("h5000", "full-val"))
    parser.add_argument("--metrics-root", required=True, type=Path)
    parser.add_argument("--bootstrap-root", required=True, type=Path)
    parser.add_argument("--tracked-root", required=True, type=Path)
    options = parser.parse_args()

    metrics = metric_rows(options.metrics_root)
    per_class = combined_rows(options.metrics_root, "per_class.tsv")
    size_bins = combined_rows(options.metrics_root, "size_bins.tsv")
    bootstrap_metrics = read_tsv(options.bootstrap_root / "complete_metrics.tsv")
    bootstrap = read_tsv(options.bootstrap_root / "complete_bootstrap.tsv")
    check_metric_identity(metrics, bootstrap_metrics)

    if options.kind == "h5000":
        opened, gates = h5000_decision(metrics, bootstrap)
        prefix = "h5000"
    else:
        opened, gates = full_val_decision(metrics, bootstrap)
        prefix = "full_val"

    write_tsv(options.tracked_root / f"{prefix}_metrics.tsv", metrics)
    write_tsv(options.tracked_root / f"{prefix}_per_class.tsv", per_class)
    write_tsv(options.tracked_root / f"{prefix}_size_bins.tsv", size_bins)
    annotated_bootstrap = []
    for row in bootstrap:
        value: dict[str, Any] = dict(row)
        if row["pair"] == "C2-vs-B2" and row["metric"] in SIZE_METRICS:
            value["interval_classification_at_minus_0_005"] = interval_classification(
                float(row["percentile_2_5"]), float(row["percentile_97_5"])
            )
        annotated_bootstrap.append(value)
    write_tsv(
        options.tracked_root / f"{prefix}_complete_bootstrap.tsv",
        annotated_bootstrap,
    )
    (options.tracked_root / f"{prefix}_decision.md").write_text(
        decision_markdown(options.kind, opened, gates), encoding="utf-8"
    )
    decision = {
        "kind": options.kind,
        "pass": opened,
        "conditional_full_val_opened": opened if options.kind == "h5000" else True,
        "gates": gates,
    }
    print(json.dumps(decision, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
