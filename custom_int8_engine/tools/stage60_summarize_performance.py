#!/usr/bin/env python3
"""Summarize buffered Stage60 executor benchmark samples without mixing surfaces."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    thermal_by_key: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    frequency_by_key: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    sources_by_key: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for path in args.inputs:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        if not rows:
            raise ValueError(f"empty benchmark input: {path}")
        for row in rows:
            key = (row["resolution"], row["surface"], row["wake_policy"],
                   row.get("route", "selected-m12n16"))
            grouped[key].append(row)
            sources_by_key[key].add(path.name)
        keys = {
            (row["resolution"], row["surface"], row["wake_policy"],
             row.get("route", "selected-m12n16"))
            for row in rows
        }
        system_path = path.with_name(path.name.replace(".raw.tsv", ".system.tsv"))
        if system_path.is_file():
            with system_path.open(newline="", encoding="utf-8") as stream:
                system_rows = list(csv.DictReader(stream, delimiter="\t"))
            temperatures = [float(row["mean_thermal_c"])
                            for row in system_rows if row.get("mean_thermal_c")]
            frequencies = [float(row["mean_cpu0_4_khz"])
                           for row in system_rows if row.get("mean_cpu0_4_khz")]
            for key in keys:
                thermal_by_key[key].extend(temperatures)
                frequency_by_key[key].extend(frequencies)

    output: list[dict[str, Any]] = []
    for (resolution, surface, wake, route), rows in sorted(grouped.items(), reverse=True):
        walls = [float(row["total_us"]) for row in rows]
        process_cpu = [float(row["process_cpu_us"]) for row in rows]
        mean = statistics.fmean(walls)
        stddev = statistics.stdev(walls) if len(walls) > 1 else 0.0
        hashes = sorted({row["output_hash"] for row in rows})
        manifests = sorted({row["manifest_sha256"] for row in rows})
        if len(hashes) != 1 or len(manifests) != 1:
            raise ValueError(f"mixed identity in {resolution}/{surface}/{wake}/{route}")
        temperatures = thermal_by_key[(resolution, surface, wake, route)]
        frequencies = frequency_by_key[(resolution, surface, wake, route)]
        output.append({
            "resolution": resolution,
            "surface": surface,
            "wake_policy": wake,
            "route": route,
            "samples": len(rows),
            "mean_us": f"{mean:.6f}",
            "stddev_us": f"{stddev:.6f}",
            "cv_pct": f"{100.0 * stddev / mean:.6f}",
            "median_us": f"{percentile(walls, 0.50):.6f}",
            "p90_us": f"{percentile(walls, 0.90):.6f}",
            "p95_us": f"{percentile(walls, 0.95):.6f}",
            "p99_us": f"{percentile(walls, 0.99):.6f}",
            "p999_us": f"{percentile(walls, 0.999):.6f}" if len(walls) >= 1000 else "",
            "max_us": f"{max(walls):.6f}",
            "pure_model_fps": f"{1_000_000.0 / mean:.9f}",
            "process_cpu_mean_us": f"{statistics.fmean(process_cpu):.6f}",
            "voluntary_context_switches": sum(int(row["voluntary_cs"]) for row in rows),
            "involuntary_context_switches": sum(int(row["involuntary_cs"]) for row in rows),
            "affinity_failures": sum(int(row["affinity_ok"]) != 1 for row in rows),
            "cpu4_7_ime_count": sum(int(row["cpu4_7_ime_count"]) for row in rows),
            "mean_thermal_c": (f"{statistics.fmean(temperatures):.6f}"
                               if temperatures else ""),
            "max_thermal_c": f"{max(temperatures):.6f}" if temperatures else "",
            "min_cpu0_4_khz": f"{min(frequencies):.0f}" if frequencies else "",
            "max_cpu0_4_khz": f"{max(frequencies):.0f}" if frequencies else "",
            "output_hash": hashes[0],
            "manifest_sha256": manifests[0],
            "source_files": ",".join(sorted(sources_by_key[
                (resolution, surface, wake, route)])),
            "status": "pass" if math.isfinite(mean) else "invalid",
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]), delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    print(f"surfaces={len(output)}")
    print(f"samples={sum(int(row['samples']) for row in output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
