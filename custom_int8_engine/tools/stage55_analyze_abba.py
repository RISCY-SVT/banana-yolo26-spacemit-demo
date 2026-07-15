#!/usr/bin/env python3
"""Parse interleaved full-executor logs into raw and summary TSV evidence."""

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


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    cycle = ""
    arm = ""
    block = -1
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if line.startswith("block\t"):
            _, cycle, arm = line.split("\t")
            block += 1
            continue
        if not line.startswith("raw\t"):
            continue
        values: dict[str, object] = {"block": block, "cycle": cycle, "arm": arm}
        for field in line.split("\t")[1:]:
            key, value = field.split("=", 1)
            values[key] = value
        rows.append(values)
    if not rows:
        raise ValueError("no raw samples")

    raw_fields = ["block", "cycle", "arm"] + [
        key for key in rows[0] if key not in {"block", "cycle", "arm"}
    ]
    write_tsv(args.raw_output, rows, raw_fields)

    summaries: list[dict[str, object]] = []
    for selected_arm in sorted({str(row["arm"]) for row in rows}):
        selected = [row for row in rows if row["arm"] == selected_arm]
        values = [float(row["wall_us"]) for row in selected]
        block_means = [
            statistics.fmean(float(row["wall_us"]) for row in selected if int(row["block"]) == block_id)
            for block_id in sorted({int(row["block"]) for row in selected})
        ]
        block_mean_cv = (statistics.stdev(block_means) / statistics.fmean(block_means) * 100.0
                         if len(block_means) > 1 else 0.0)
        summaries.append({
            "arm": selected_arm,
            "samples": len(values),
            "mean_us": f"{statistics.fmean(values):.6f}",
            "stddev_us": f"{statistics.stdev(values):.6f}",
            "cv_pct": f"{statistics.stdev(values) / statistics.fmean(values) * 100.0:.9f}",
            "median_us": f"{statistics.median(values):.6f}",
            "p90_us": f"{percentile(values, 0.90):.6f}",
            "p95_us": f"{percentile(values, 0.95):.6f}",
            "p99_us": f"{percentile(values, 0.99):.6f}",
            "max_us": f"{max(values):.6f}",
            "block_mean_cv_pct": f"{block_mean_cv:.9f}",
            "process_cpu_mean_us": f"{statistics.fmean(float(row['process_cpu_us']) for row in selected):.6f}",
            "input_mean_us": f"{statistics.fmean(float(row['input_us']) for row in selected):.6f}",
            "core_mean_us": f"{statistics.fmean(float(row['core_us']) for row in selected):.6f}",
            "dense_mean_us": f"{statistics.fmean(float(row['dense_us']) for row in selected):.6f}",
            "depthwise_mean_us": f"{statistics.fmean(float(row['depthwise_us']) for row in selected):.6f}",
            "attention_mean_us": f"{statistics.fmean(float(row['attention_us']) for row in selected):.6f}",
            "lut_mean_us": f"{statistics.fmean(float(row['lut_us']) for row in selected):.6f}",
            "concat_mean_us": f"{statistics.fmean(float(row['concat_us']) for row in selected):.6f}",
            "transform_mean_us": f"{statistics.fmean(float(row['transform_us']) for row in selected):.6f}",
            "head_mean_us": f"{statistics.fmean(float(row['head_us']) for row in selected):.6f}",
            "voluntary_cs_mean": f"{statistics.fmean(float(row['voluntary_cs']) for row in selected):.6f}",
            "involuntary_cs_mean": f"{statistics.fmean(float(row['involuntary_cs']) for row in selected):.6f}",
            "output_hash_count": len({str(row["hash"]) for row in selected}),
            "output_hash": sorted({str(row["hash"]) for row in selected})[0],
        })
    write_tsv(args.summary_output, summaries, list(summaries[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
