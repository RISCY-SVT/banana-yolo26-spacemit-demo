#!/usr/bin/env python3
"""Summarize paired Stage57 full-model ABBA evidence and confidence bounds."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    cycle = position = block = -1
    arm = ""
    for line in args.input.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("block\t"):
            _, cycle_text, position_text, arm, block_text = line.split("\t")
            cycle, position, block = int(cycle_text), int(position_text), int(block_text)
            continue
        if not line.startswith("raw\t"):
            continue
        row: dict[str, object] = {
            "block": block, "cycle": cycle, "position": position, "arm": arm,
        }
        for field in line.split("\t")[1:]:
            key, value = field.split("=", 1)
            row[key] = value
        rows.append(row)
    if not rows:
        raise ValueError("no raw samples")
    write_tsv(args.raw_output, rows)

    summaries: list[dict[str, object]] = []
    arm_values: dict[str, list[float]] = {}
    for selected_arm in ("A", "B"):
        selected = [row for row in rows if row["arm"] == selected_arm]
        values = [float(row["wall_us"]) for row in selected]
        arm_values[selected_arm] = values
        mean = statistics.fmean(values)
        summaries.append({
            "row_kind": "arm",
            "arm": selected_arm,
            "samples": len(values),
            "blocks": len({int(row["block"]) for row in selected}),
            "mean_us": f"{mean:.6f}",
            "stddev_us": f"{statistics.pstdev(values):.6f}",
            "cv_pct": f"{statistics.pstdev(values) / mean * 100.0:.9f}",
            "median_us": f"{statistics.median(values):.6f}",
            "p90_us": f"{percentile(values, 0.90):.6f}",
            "p95_us": f"{percentile(values, 0.95):.6f}",
            "p99_us": f"{percentile(values, 0.99):.6f}",
            "max_us": f"{max(values):.6f}",
            "paired_delta_us": "",
            "paired_delta_pct": "",
            "ci95_low_us": "",
            "ci95_high_us": "",
            "output_hash_count": len({str(row["hash"]) for row in selected}),
            "output_hash": sorted({str(row["hash"]) for row in selected})[0],
        })

    by_cycle: dict[int, dict[str, list[float]]] = {}
    for row in rows:
        by_cycle.setdefault(int(row["cycle"]), {}).setdefault(str(row["arm"]), []).append(
            float(row["wall_us"])
        )
    paired = [statistics.fmean(values["B"]) - statistics.fmean(values["A"])
              for _, values in sorted(by_cycle.items()) if set(values) == {"A", "B"}]
    if len(paired) < 2:
        raise ValueError("at least two complete ABBA pairs are required")
    paired_mean = statistics.fmean(paired)
    standard_error = statistics.stdev(paired) / math.sqrt(len(paired))
    critical = T95.get(len(paired) - 1, 1.96)
    a_mean = statistics.fmean(arm_values["A"])
    summaries.append({
        "row_kind": "paired_B_minus_A",
        "arm": "B-A",
        "samples": len(paired),
        "blocks": len(paired) * 2,
        "mean_us": "",
        "stddev_us": f"{statistics.stdev(paired):.6f}",
        "cv_pct": "",
        "median_us": "",
        "p90_us": "",
        "p95_us": "",
        "p99_us": "",
        "max_us": "",
        "paired_delta_us": f"{paired_mean:.6f}",
        "paired_delta_pct": f"{paired_mean / a_mean * 100.0:.9f}",
        "ci95_low_us": f"{paired_mean - critical * standard_error:.6f}",
        "ci95_high_us": f"{paired_mean + critical * standard_error:.6f}",
        "output_hash_count": "",
        "output_hash": "",
    })
    write_tsv(args.summary_output, summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
