#!/usr/bin/env python3
"""Summarize Stage53/54 operation-profile logs for controlled route comparisons."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="append", required=True, help="ARM=PATH")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    observations: list[dict[str, object]] = []
    for specification in args.log:
        arm, path_text = specification.split("=", 1)
        path = Path(path_text)
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) != 8 or fields[0] != "stage53_op":
                continue
            observations.append({
                "arm": arm,
                "run": int(fields[1]),
                "operation_index": int(fields[2]),
                "resident_operation_index": int(fields[3]),
                "kind": fields[4],
                "scope": fields[5],
                "wall_us": float(fields[6]),
                "name": fields[7],
            })

    raw_fields = [
        "arm", "run", "operation_index", "resident_operation_index", "kind",
        "scope", "wall_us", "name",
    ]
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    with args.raw.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=raw_fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(observations)

    grouped: dict[tuple[str, int, str, str, str], list[float]] = defaultdict(list)
    for row in observations:
        grouped[(
            str(row["arm"]), int(row["operation_index"]), str(row["kind"]),
            str(row["scope"]), str(row["name"]),
        )].append(float(row["wall_us"]))
    summary: list[dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        summary.append({
            "arm": key[0],
            "operation_index": key[1],
            "kind": key[2],
            "scope": key[3],
            "name": key[4],
            "samples": len(values),
            "mean_us": f"{statistics.fmean(values):.6f}",
            "median_us": f"{statistics.median(values):.6f}",
            "p95_us": f"{percentile(values, 0.95):.6f}",
            "min_us": f"{min(values):.6f}",
            "max_us": f"{max(values):.6f}",
        })
    summary_fields = [
        "arm", "operation_index", "kind", "scope", "name", "samples",
        "mean_us", "median_us", "p95_us", "min_us", "max_us",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
