#!/usr/bin/env python3
"""Correlate Stage56 inference markers with low-perturbation trace events."""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import re
import statistics
from pathlib import Path
from typing import TextIO


TRACE_LINE = re.compile(
    r".*\[(?P<cpu>\d+)\].*?\s(?P<timestamp>\d+\.\d+):\s"
    r"(?P<event>[A-Za-z0-9_]+):\s(?P<details>.*)$"
)
MARKER = re.compile(r"y26_(?P<phase>begin|end) repeat=(?P<repeat>\d+) run=(?P<run>\d+)")


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def parse_raw_log(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    rows: dict[tuple[int, int], dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("raw\t"):
            continue
        fields = line.split("\t")[1:]
        if any("=" not in field for field in fields):
            continue
        row = dict(field.split("=", 1) for field in fields)
        if "repeat" not in row or "run" not in row:
            continue
        rows[(int(row["repeat"]), int(row["run"]))] = row
    if not rows:
        raise ValueError(f"no raw samples in {path}")
    return rows


def open_trace(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return math.nan
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left) *
        sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator else math.nan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--benchmark-log", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--slow-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    benchmark = parse_raw_log(args.benchmark_log)
    completed: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    irq_stack: dict[int, list[tuple[str, float]]] = {}
    softirq_stack: dict[int, list[tuple[str, float]]] = {}
    workqueue_stack: dict[int, list[float]] = {}
    lost_event_lines = 0
    malformed_marker_count = 0

    with open_trace(args.trace) as stream:
        for line in stream:
            if "LOST EVENTS" in line:
                lost_event_lines += 1
            match = TRACE_LINE.match(line)
            if not match:
                continue
            cpu = int(match.group("cpu"))
            timestamp = float(match.group("timestamp"))
            event = match.group("event")
            details = match.group("details")
            marker = MARKER.search(details) if event == "tracing_mark_write" else None
            if marker is not None:
                phase = marker.group("phase")
                key = (int(marker.group("repeat")), int(marker.group("run")))
                if phase == "begin":
                    if current is not None:
                        malformed_marker_count += 1
                    current = {
                        "repeat": key[0], "run": key[1], "trace_begin_s": timestamp,
                        "irq_count": 0, "irq_duration_us": 0.0,
                        "softirq_count": 0, "softirq_duration_us": 0.0,
                        "workqueue_count": 0, "workqueue_duration_us": 0.0,
                        "sched_switch_count": 0, "block_issue_count": 0,
                        "block_complete_count": 0, "frequency_event_count": 0,
                    }
                    irq_stack.clear()
                    softirq_stack.clear()
                    workqueue_stack.clear()
                elif current is None or (current["repeat"], current["run"]) != key:
                    malformed_marker_count += 1
                else:
                    current["trace_end_s"] = timestamp
                    current["trace_duration_us"] = (
                        timestamp - float(current["trace_begin_s"])
                    ) * 1_000_000.0
                    sample = benchmark.get(key)
                    if sample is not None:
                        current.update(sample)
                    else:
                        current["benchmark_sample_missing"] = 1
                    completed.append(current)
                    current = None
                continue
            if current is None:
                continue

            if event == "irq_handler_entry":
                identity = re.search(r"irq=(\d+)", details)
                irq_stack.setdefault(cpu, []).append(
                    (identity.group(1) if identity else "unknown", timestamp)
                )
                current["irq_count"] = int(current["irq_count"]) + 1
            elif event == "irq_handler_exit":
                stack = irq_stack.get(cpu, [])
                if stack:
                    identity, begin = stack.pop()
                    duration = max(0.0, (timestamp - begin) * 1_000_000.0)
                    current["irq_duration_us"] = float(current["irq_duration_us"]) + duration
                    key = f"irq_{identity}_count"
                    current[key] = int(current.get(key, 0)) + 1
            elif event == "softirq_entry":
                identity = re.search(r"vec=(\d+)", details)
                softirq_stack.setdefault(cpu, []).append(
                    (identity.group(1) if identity else "unknown", timestamp)
                )
                current["softirq_count"] = int(current["softirq_count"]) + 1
            elif event == "softirq_exit":
                stack = softirq_stack.get(cpu, [])
                if stack:
                    _, begin = stack.pop()
                    current["softirq_duration_us"] = float(current["softirq_duration_us"]) + max(
                        0.0, (timestamp - begin) * 1_000_000.0
                    )
            elif event == "workqueue_execute_start":
                workqueue_stack.setdefault(cpu, []).append(timestamp)
                current["workqueue_count"] = int(current["workqueue_count"]) + 1
            elif event == "workqueue_execute_end":
                stack = workqueue_stack.get(cpu, [])
                if stack:
                    begin = stack.pop()
                    current["workqueue_duration_us"] = float(current["workqueue_duration_us"]) + max(
                        0.0, (timestamp - begin) * 1_000_000.0
                    )
            elif event == "sched_switch":
                current["sched_switch_count"] = int(current["sched_switch_count"]) + 1
            elif event == "block_rq_issue":
                current["block_issue_count"] = int(current["block_issue_count"]) + 1
            elif event == "block_rq_complete":
                current["block_complete_count"] = int(current["block_complete_count"]) + 1
            elif event == "cpu_frequency":
                current["frequency_event_count"] = int(current["frequency_event_count"]) + 1

    if not completed:
        raise ValueError("no complete marked intervals in trace")
    write_tsv(args.raw_output, completed)

    matched = [row for row in completed if "wall_us" in row]
    wall = [float(row["wall_us"]) for row in matched]
    threshold = percentile(wall, 0.999)
    slow = [row for row in matched if float(row["wall_us"]) >= threshold]
    write_tsv(args.slow_output, slow)

    metrics = [
        "trace_duration_us", "irq_count", "irq_duration_us", "softirq_count",
        "softirq_duration_us", "workqueue_count", "workqueue_duration_us",
        "sched_switch_count", "block_issue_count", "block_complete_count",
        "frequency_event_count", "involuntary_cs",
    ]
    summary: list[dict[str, object]] = []
    for metric in metrics:
        values = [float(row.get(metric, 0.0)) for row in matched]
        summary.append(
            {
                "metric": metric,
                "samples": len(values),
                "mean": f"{statistics.fmean(values):.9f}",
                "p95": f"{percentile(values, 0.95):.9f}",
                "p999": f"{percentile(values, 0.999):.9f}",
                "max": f"{max(values):.9f}",
                "correlation_with_wall": f"{correlation(wall, values):.9f}",
                "lost_event_lines": lost_event_lines,
                "malformed_marker_count": malformed_marker_count,
                "complete_intervals": len(completed),
                "benchmark_samples": len(benchmark),
            }
        )
    write_tsv(args.summary_output, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
