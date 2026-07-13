#!/usr/bin/env python3
"""Normalize Stage 52 CLI benchmark output into reproducible TSV summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def parse_value(value: str) -> object:
    if value.startswith("0x"):
        return value
    try:
        return int(value)
    except ValueError:
        return float(value)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if not line.startswith("raw\t"):
            continue
        row: dict[str, object] = {"surface": args.label}
        for item in line.split("\t")[1:]:
            key, value = item.split("=", 1)
            row[key] = parse_value(value)
        rows.append(row)
    if not rows:
        raise ValueError(f"no raw benchmark rows in {args.input}")
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    with args.raw.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    wall = [float(row["wall_us"]) for row in rows]
    process_cpu = [float(row.get("process_cpu_us", 0.0)) for row in rows]
    repeat_means: dict[int, list[float]] = {}
    for row in rows:
        repeat_means.setdefault(int(row["repeat"]), []).append(float(row["wall_us"]))
    repeat_values = [statistics.fmean(values) for values in repeat_means.values()]
    summary = {
        "surface": args.label,
        "samples": len(wall),
        "repeats": len(repeat_values),
        "mean_us": statistics.fmean(wall),
        "stddev_us": statistics.stdev(wall) if len(wall) > 1 else 0.0,
        "cv_pct": (statistics.stdev(wall) / statistics.fmean(wall) * 100.0) if len(wall) > 1 else 0.0,
        "min_us": min(wall),
        "median_us": percentile(wall, 0.5),
        "p90_us": percentile(wall, 0.9),
        "p95_us": percentile(wall, 0.95),
        "p99_us": percentile(wall, 0.99),
        "p999_us": percentile(wall, 0.999),
        "max_us": max(wall),
        "process_cpu_mean_us": statistics.fmean(process_cpu),
        "repeat_mean_cv_pct": (statistics.stdev(repeat_values) / statistics.fmean(repeat_values) * 100.0)
                              if len(repeat_values) > 1 else 0.0,
        "voluntary_context_switches_mean": statistics.fmean(
            float(row.get("voluntary_cs", 0)) for row in rows),
        "involuntary_context_switches_mean": statistics.fmean(
            float(row.get("involuntary_cs", 0)) for row in rows),
        "affinity_all_pass": all(int(row.get("affinity_ok", 0)) == 1 for row in rows),
        "cpu4_7_ime_count": max(int(row.get("cpu4_7_ime_count", 0)) for row in rows),
        "output_hashes": sorted({str(row["hash"]) for row in rows}),
    }
    with args.summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary), delimiter="\t")
        writer.writeheader()
        writer.writerow(summary)
    args.summary.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
