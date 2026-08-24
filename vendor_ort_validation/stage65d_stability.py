#!/usr/bin/env python3
"""Normalize Stage65D short and C2 10k stability evidence."""

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


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    if not rows:
        raise ValueError(f"refusing empty report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def collect(root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, list[float]], dict[str, set[str]], dict[str, set[str]]]:
    status = {
        f"segment-{row['segment']}-{row['position']}-{row['model']}": row
        for row in read_tsv(root / "status.raw.tsv")
    }
    windows: list[dict[str, object]] = []
    resources: list[dict[str, object]] = []
    aggregate: dict[str, list[float]] = defaultdict(list)
    output_hashes: dict[str, set[str]] = defaultdict(set)
    fnv_hashes: dict[str, set[str]] = defaultdict(set)
    for directory, status_row in sorted(status.items()):
        segment_dir = root / directory
        model = status_row["model"]
        output_hashes[model].add(status_row["output_sha256"])
        values: dict[str, list[float]] = {"inference": [], "tail": [], "two_stage": []}
        for row in read_tsv(segment_dir / "samples.tsv"):
            values["inference"].append(float(row["inference_us"]))
            values["tail"].append(float(row["tail_us"]))
            values["two_stage"].append(float(row["total_us"]))
            fnv_hashes[model].add(row["output_fnv1a64"])
        for metric, metric_values in values.items():
            aggregate[f"{model}:{metric}"].extend(metric_values)
            windows.append({
                "segment": status_row["segment"],
                "position": status_row["position"],
                "model": model,
                "metric": metric,
                **stats(metric_values),
            })

        resource = read_tsv(segment_dir / "resource.tsv")
        numeric = {
            name: [int(row[name]) for row in resource]
            for name in ("rss_kib", "peak_rss_kib", "fds", "threads")
        }
        resources.append({
            "root": root.name,
            "segment": status_row["segment"],
            "position": status_row["position"],
            "model": model,
            "samples": len(resource),
            "rss_first_kib": numeric["rss_kib"][0],
            "rss_last_kib": numeric["rss_kib"][-1],
            "rss_max_kib": max(numeric["rss_kib"]),
            "peak_rss_max_kib": max(numeric["peak_rss_kib"]),
            "fds_min": min(numeric["fds"]),
            "fds_max": max(numeric["fds"]),
            "threads_min": min(numeric["threads"]),
            "threads_max": max(numeric["threads"]),
            "exit_code": status_row["exit_code"],
        })
    return windows, resources, aggregate, output_hashes, fnv_hashes


def thermal_rows(*roots: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for root in roots:
        for path in sorted(root.glob("segment-*/state-*.tsv")):
            for line in path.read_text(encoding="utf-8").splitlines():
                fields = line.split("\t")
                if len(fields) == 3:
                    rows.append({
                        "root": root.name,
                        "segment": path.parent.name,
                        "moment": path.stem.removeprefix("state-"),
                        "kind": fields[0],
                        "source": fields[1],
                        "value": fields[2],
                    })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--tracked-root", required=True, type=Path)
    args = parser.parse_args()
    short_root = args.raw_root / "board/stability/short-1k"
    long_root = args.raw_root / "board/stability/c2-10k"
    short = collect(short_root)
    long = collect(long_root)

    metric_fields = [
        "model", "metric", "samples", "mean_us", "stddev_us", "cv_pct",
        "p50_us", "p95_us", "p99_us", "p999_us", "max_us",
    ]
    short_rows = []
    for model in ("B2", "C2"):
        for metric in ("inference", "tail", "two_stage"):
            short_rows.append({"model": model, "metric": metric, **stats(short[2][f"{model}:{metric}"])})
    long_rows = [
        {"model": "C2", "metric": metric, **stats(long[2][f"C2:{metric}"])}
        for metric in ("inference", "tail", "two_stage")
    ]
    write_tsv(args.tracked_root / "short_soak.tsv", short_rows, metric_fields)
    write_tsv(args.tracked_root / "c2_10k_soak.tsv", long_rows, metric_fields)

    resources = short[1] + long[1]
    resource_fields = [
        "root", "segment", "position", "model", "samples", "rss_first_kib",
        "rss_last_kib", "rss_max_kib", "peak_rss_max_kib", "fds_min", "fds_max",
        "threads_min", "threads_max", "exit_code", "status",
    ]
    passed = True
    for row in resources:
        row_pass = (
            int(row["exit_code"]) == 0
            and int(row["fds_max"]) - int(row["fds_min"]) <= 2
            and int(row["threads_max"]) - int(row["threads_min"]) <= 2
            and int(row["rss_last_kib"]) <= int(row["rss_first_kib"]) + 16384
        )
        row["status"] = "pass" if row_pass else "fail"
        passed = passed and row_pass
    write_tsv(args.tracked_root / "resource_drift.tsv", resources, resource_fields)

    thermal = thermal_rows(short_root, long_root)
    write_tsv(
        args.tracked_root / "thermal_frequency_log.tsv",
        thermal,
        ["root", "segment", "moment", "kind", "source", "value"],
    )

    hash_rows = []
    for model in ("B2", "C2"):
        model_pass = (
            len(short[3][model]) == 1
            and len(short[4][model]) == 1
            and len(short[2][f"{model}:two_stage"]) == 1000
        )
        passed = passed and model_pass
        hash_rows.append({
            "surface": f"short-1k-{model}",
            "runs": len(short[2][f"{model}:two_stage"]),
            "output_sha256_count": len(short[3][model]),
            "output_fnv1a64_count": len(short[4][model]),
            "status": "pass" if model_pass else "fail",
        })

    c2_windows = [row for row in long[0] if row["model"] == "C2" and row["metric"] == "two_stage"]
    first = statistics.median(float(row["p50_us"]) for row in c2_windows[:5])
    second = statistics.median(float(row["p50_us"]) for row in c2_windows[5:])
    drift = second / first
    long_pass = (
        len(long[3]["C2"]) == 1
        and len(long[4]["C2"]) == 1
        and len(long[2]["C2:two_stage"]) == 10000
        and 0.95 <= drift <= 1.05
    )
    passed = passed and long_pass
    hash_rows.append({
        "surface": "c2-10k",
        "runs": len(long[2]["C2:two_stage"]),
        "output_sha256_count": len(long[3]["C2"]),
        "output_fnv1a64_count": len(long[4]["C2"]),
        "status": "pass" if long_pass else "fail",
    })
    write_tsv(
        args.tracked_root / "output_hash_stability.tsv",
        hash_rows,
        ["surface", "runs", "output_sha256_count", "output_fnv1a64_count", "status"],
    )
    (args.tracked_root / "stability_decision.md").write_text(
        "# Stability decision\n\n"
        f"Decision: `{'pass' if passed else 'fail'}`.\n\n"
        f"C2 10k second-half/first-half median ratio: `{drift:.9f}` "
        "(predeclared range 0.95..1.05). Resource gates require FD/thread drift "
        "no greater than 2 and RSS growth no greater than 16 MiB per isolated segment.\n",
        encoding="utf-8",
    )
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
