#!/usr/bin/env python3
"""Normalize Stage65C 10k alternating-segment stability evidence."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stats(values: list[float]) -> dict[str, object]:
    mean = statistics.fmean(values)
    stddev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "samples": len(values),
        "mean_us": mean,
        "stddev_us": stddev,
        "cv_pct": 0.0 if mean == 0 else 100.0 * stddev / mean,
        "p50_us": statistics.median(values),
        "p95_us": percentile(values, 0.95),
        "p99_us": percentile(values, 0.99),
        "p999_us": percentile(values, 0.999),
        "max_us": max(values),
    }


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def numeric_column(rows: list[dict[str, str]], key: str) -> list[int]:
    return [int(row[key]) for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--tracked-root", required=True, type=Path)
    args = parser.parse_args()
    soak = args.raw_root / "board" / "stability" / "soak10k"

    aggregate: dict[tuple[str, str], list[float]] = defaultdict(list)
    output_hashes: dict[str, set[str]] = defaultdict(set)
    fnv_hashes: dict[str, set[str]] = defaultdict(set)
    windows: list[dict[str, object]] = []
    resources: list[dict[str, object]] = []
    status_by_dir: dict[str, dict[str, str]] = {}
    with (soak / "status.raw.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            directory = f"segment-{row['segment']}-{row['position']}-{row['model']}"
            status_by_dir[directory] = row
            output_hashes[row["model"]].add(row["output_sha256"])

    for segment_dir in sorted(soak.glob("segment-*-*-*")):
        _, segment_text, position_text, model = segment_dir.name.split("-", maxsplit=3)
        segment, position = int(segment_text), int(position_text)
        values = {"inference": [], "tail": [], "two_stage": []}
        with (segment_dir / "samples.tsv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                values["inference"].append(float(row["inference_us"]))
                values["tail"].append(float(row["tail_us"]))
                values["two_stage"].append(float(row["total_us"]))
                fnv_hashes[model].add(row["output_fnv1a64"])
        for metric, metric_values in values.items():
            aggregate[(model, metric)].extend(metric_values)
            windows.append(
                {
                    "segment": segment,
                    "position": position,
                    "model": model,
                    "metric": metric,
                    **stats(metric_values),
                }
            )

        resource_rows = []
        with (segment_dir / "resource.tsv").open(encoding="utf-8", newline="") as stream:
            resource_rows.extend(csv.DictReader(stream, delimiter="\t"))
        numeric = {
            key: numeric_column(resource_rows, key)
            for key in (
                "rss_kib",
                "peak_rss_kib",
                "fds",
                "threads",
                "voluntary_ctxt",
                "nonvoluntary_ctxt",
            )
        }
        resources.append(
            {
                "segment": segment,
                "position": position,
                "model": model,
                "samples": len(resource_rows),
                "rss_first_kib": numeric["rss_kib"][0],
                "rss_last_kib": numeric["rss_kib"][-1],
                "rss_max_kib": max(numeric["rss_kib"]),
                "peak_rss_max_kib": max(numeric["peak_rss_kib"]),
                "fds_min": min(numeric["fds"]),
                "fds_max": max(numeric["fds"]),
                "threads_min": min(numeric["threads"]),
                "threads_max": max(numeric["threads"]),
                "voluntary_ctxt_last": numeric["voluntary_ctxt"][-1],
                "nonvoluntary_ctxt_last": numeric["nonvoluntary_ctxt"][-1],
                "exit_code": status_by_dir[segment_dir.name]["exit_code"],
            }
        )

    long_rows = []
    for model in ("B2", "A1"):
        for metric in ("inference", "tail", "two_stage"):
            long_rows.append({"model": model, "metric": metric, **stats(aggregate[(model, metric)])})
    metric_fields = [
        "model",
        "metric",
        "samples",
        "mean_us",
        "stddev_us",
        "cv_pct",
        "p50_us",
        "p95_us",
        "p99_us",
        "p999_us",
        "max_us",
    ]
    write_tsv(args.tracked_root / "long_soak.tsv", long_rows, metric_fields)
    write_tsv(
        args.tracked_root / "long_soak_windows.tsv",
        windows,
        ["segment", "position", *metric_fields],
    )
    write_tsv(
        args.tracked_root / "resource_stability.tsv",
        resources,
        [
            "segment",
            "position",
            "model",
            "samples",
            "rss_first_kib",
            "rss_last_kib",
            "rss_max_kib",
            "peak_rss_max_kib",
            "fds_min",
            "fds_max",
            "threads_min",
            "threads_max",
            "voluntary_ctxt_last",
            "nonvoluntary_ctxt_last",
            "exit_code",
        ],
    )

    hash_rows = []
    passed = True
    for model in ("B2", "A1"):
        model_windows = [row for row in windows if row["model"] == model and row["metric"] == "two_stage"]
        first_half = statistics.median(float(row["p50_us"]) for row in model_windows[:5])
        second_half = statistics.median(float(row["p50_us"]) for row in model_windows[5:])
        drift_ratio = second_half / first_half
        model_pass = (
            len(output_hashes[model]) == 1
            and len(fnv_hashes[model]) == 1
            and len(aggregate[(model, "two_stage")]) == 10000
            and 0.95 <= drift_ratio <= 1.05
        )
        passed = passed and model_pass
        hash_rows.append(
            {
                "model": model,
                "measured_runs": len(aggregate[(model, "two_stage")]),
                "output_sha256_count": len(output_hashes[model]),
                "output_sha256": ",".join(sorted(output_hashes[model])),
                "output_fnv1a64_count": len(fnv_hashes[model]),
                "output_fnv1a64": ",".join(sorted(fnv_hashes[model])),
                "second_half_over_first_half_median": drift_ratio,
                "status": "pass" if model_pass else "fail",
            }
        )
    for row in resources:
        resource_pass = (
            int(row["exit_code"]) == 0
            and int(row["fds_max"]) - int(row["fds_min"]) <= 2
            and int(row["threads_max"]) - int(row["threads_min"]) <= 2
            and int(row["rss_last_kib"]) <= int(row["rss_first_kib"]) + 16384
        )
        passed = passed and resource_pass

    write_tsv(
        args.tracked_root / "output_hash_stability.tsv",
        hash_rows,
        [
            "model",
            "measured_runs",
            "output_sha256_count",
            "output_sha256",
            "output_fnv1a64_count",
            "output_fnv1a64",
            "second_half_over_first_half_median",
            "status",
        ],
    )
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
