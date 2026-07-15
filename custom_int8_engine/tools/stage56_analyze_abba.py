#!/usr/bin/env python3
"""Summarize Stage56 interleaved full-model A/B evidence."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


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
    cycle = -1
    position = -1
    arm = ""
    block = -1
    for line in args.input.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("block\t"):
            _, cycle_text, position_text, arm, block_text = line.split("\t")
            cycle = int(cycle_text)
            position = int(position_text)
            block = int(block_text)
            continue
        if not line.startswith("raw\t"):
            continue
        row: dict[str, object] = {
            "block": block,
            "cycle": cycle,
            "position": position,
            "arm": arm,
        }
        for field in line.split("\t")[1:]:
            key, value = field.split("=", 1)
            row[key] = value
        rows.append(row)
    if not rows:
        raise ValueError("no raw samples")

    write_tsv(args.raw_output, rows)
    summaries: list[dict[str, object]] = []
    for selected_arm in sorted({str(row["arm"]) for row in rows}):
        selected = [row for row in rows if row["arm"] == selected_arm]
        values = [float(row["wall_us"]) for row in selected]
        mean = statistics.fmean(values)
        stddev = statistics.pstdev(values)
        summaries.append(
            {
                "arm": selected_arm,
                "samples": len(values),
                "blocks": len({int(row["block"]) for row in selected}),
                "mean_us": f"{mean:.6f}",
                "stddev_us": f"{stddev:.6f}",
                "cv_pct": f"{stddev / mean * 100.0:.9f}",
                "median_us": f"{statistics.median(values):.6f}",
                "p90_us": f"{percentile(values, 0.90):.6f}",
                "p95_us": f"{percentile(values, 0.95):.6f}",
                "p99_us": f"{percentile(values, 0.99):.6f}",
                "max_us": f"{max(values):.6f}",
                "process_cpu_mean_us": f"{statistics.fmean(float(row['process_cpu_us']) for row in selected):.6f}",
                "voluntary_cs_mean": f"{statistics.fmean(float(row['voluntary_cs']) for row in selected):.6f}",
                "involuntary_cs_mean": f"{statistics.fmean(float(row['involuntary_cs']) for row in selected):.6f}",
                "input_mean_us": f"{statistics.fmean(float(row['input_us']) for row in selected):.6f}",
                "core_mean_us": f"{statistics.fmean(float(row['core_us']) for row in selected):.6f}",
                "dense_mean_us": f"{statistics.fmean(float(row['dense_us']) for row in selected):.6f}",
                "depthwise_mean_us": f"{statistics.fmean(float(row['depthwise_us']) for row in selected):.6f}",
                "attention_mean_us": f"{statistics.fmean(float(row['attention_us']) for row in selected):.6f}",
                "lut_mean_us": f"{statistics.fmean(float(row['lut_us']) for row in selected):.6f}",
                "concat_mean_us": f"{statistics.fmean(float(row['concat_us']) for row in selected):.6f}",
                "transform_mean_us": f"{statistics.fmean(float(row['transform_us']) for row in selected):.6f}",
                "head_mean_us": f"{statistics.fmean(float(row['head_us']) for row in selected):.6f}",
                "output_hash_count": len({str(row["hash"]) for row in selected}),
                "output_hash": sorted({str(row["hash"]) for row in selected})[0],
                "cpu4_7_ime_count": max(int(row["cpu4_7_ime_count"]) for row in selected),
            }
        )
    write_tsv(args.summary_output, summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
