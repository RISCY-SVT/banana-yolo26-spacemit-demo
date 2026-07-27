#!/usr/bin/env python3
"""Summarize one-inference Stage46 runner repeat records."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from pathlib import Path


REPEAT_RE = re.compile(
    r"stage46_repeat index=(?P<index>\d+) runs=(?P<runs>\d+) "
    r"wall_mean_us=(?P<wall>[0-9.]+) process_cpu_mean_us=(?P<cpu>[0-9.]+)"
)
RESULT_RE = re.compile(r"stage46_tensor .*output_fnv1a64=(?P<hash>\d+)")
SESSION_RE = re.compile(r"stage46_session status=created create_us=(?P<value>[0-9.]+)")
FIRST_RUN_RE = re.compile(r"stage46_first_run first_run_us=(?P<value>[0-9.]+)")
TIME_FIELDS = {
    "Maximum resident set size (kbytes)": "max_rss_kib",
    "Voluntary context switches": "voluntary_context_switches",
    "Involuntary context switches": "involuntary_context_switches",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot calculate percentile of an empty set")
    ordered = sorted(values)
    rank = fraction * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def parse_time_file(path: Path) -> dict[str, str]:
    values = {output: "" for output in TIME_FIELDS.values()}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        for source, output in TIME_FIELDS.items():
            prefix = source + ":"
            if stripped.startswith(prefix):
                values[output] = stripped[len(prefix) :].strip()
    return values


def parse_state_file(path: Path) -> tuple[list[int], list[int]]:
    frequencies: list[int] = []
    temperatures: list[int] = []
    if not path.exists():
        return frequencies, temperatures
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.rsplit("=", 1)
        try:
            numeric = int(value)
        except ValueError:
            continue
        if key.endswith("/scaling_cur_freq"):
            frequencies.append(numeric)
        elif "/thermal_zone" in key and key.endswith("/temp"):
            temperatures.append(numeric)
    return frequencies, temperatures


def main() -> int:
    options = parse_args()
    options.samples.parent.mkdir(parents=True, exist_ok=True)
    options.summary.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for log in options.logs:
        text = log.read_text(encoding="utf-8", errors="replace")
        matches = list(REPEAT_RE.finditer(text))
        if not matches:
            raise ValueError(f"{log}: no stage46_repeat records")
        run_counts = {int(match["runs"]) for match in matches}
        if len(run_counts) != 1:
            raise ValueError(f"{log}: inconsistent runs-per-repeat values")
        runs_per_repeat = next(iter(run_counts))
        output_hashes = set(RESULT_RE.findall(text))
        session_match = SESSION_RE.search(text)
        first_run_match = FIRST_RUN_RE.search(text)
        wall = [float(match["wall"]) for match in matches]
        cpu = [float(match["cpu"]) for match in matches]
        resource = parse_time_file(log.with_suffix(".time.txt"))
        state_root = log.parent.parent / "state"
        frequencies: list[int] = []
        temperatures: list[int] = []
        for suffix in ("before", "after"):
            state_frequencies, state_temperatures = parse_state_file(
                state_root / f"{log.stem}.{suffix}.txt"
            )
            frequencies.extend(state_frequencies)
            temperatures.extend(state_temperatures)
        for match in matches:
            rows.append(
                {
                    "arm": log.stem,
                    "sample": int(match["index"]),
                    "wall_us": match["wall"],
                    "process_cpu_us": match["cpu"],
                }
            )
        summaries.append(
            {
                "arm": log.stem,
                "samples": len(wall),
                "runs_per_repeat": runs_per_repeat,
                "total_inferences": len(wall) * runs_per_repeat,
                "session_create_us": session_match["value"] if session_match else "",
                "first_run_us": first_run_match["value"] if first_run_match else "",
                "mean_us": f"{statistics.fmean(wall):.6f}",
                "stddev_us": f"{statistics.pstdev(wall):.6f}",
                "cv_pct": f"{statistics.pstdev(wall) / statistics.fmean(wall) * 100:.6f}",
                "median_us": f"{statistics.median(wall):.6f}",
                "p90_us": f"{percentile(wall, 0.90):.6f}",
                "p95_us": f"{percentile(wall, 0.95):.6f}",
                "p99_us": f"{percentile(wall, 0.99):.6f}",
                "p999_us": f"{percentile(wall, 0.999):.6f}",
                "min_us": f"{min(wall):.6f}",
                "max_us": f"{max(wall):.6f}",
                "process_cpu_mean_us": f"{statistics.fmean(cpu):.6f}",
                **resource,
                "frequency_min_khz": min(frequencies) if frequencies else "",
                "frequency_max_khz": max(frequencies) if frequencies else "",
                "temperature_min_millic": min(temperatures) if temperatures else "",
                "temperature_max_millic": max(temperatures) if temperatures else "",
                "output_hash": (
                    next(iter(output_hashes)) if len(output_hashes) == 1 else "unstable-or-missing"
                ),
            }
        )

    with options.samples.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            delimiter="\t",
            fieldnames=["arm", "sample", "wall_us", "process_cpu_us"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    with options.summary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            delimiter="\t",
            fieldnames=list(summaries[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
