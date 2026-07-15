#!/usr/bin/env python3
"""Normalize Stage56 per-task HPM logs without cross-process subtraction."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


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


def scaled(count: int, enabled: int, running: int) -> float:
    if running <= 0:
        return 0.0
    return count * enabled / running


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--shape-output", type=Path, required=True)
    parser.add_argument("--full-output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    occurrence: defaultdict[tuple[str, str, int, int], int] = defaultdict(int)
    for path in sorted(args.input_dir.glob("shapes_*.log")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("stage56_shape_counter\t"):
                continue
            fields = line.split("\t")
            if len(fields) != 15:
                raise ValueError(f"malformed shape counter in {path}: {line}")
            (_, op, m, n, k, worker, cpu, event, status, error, event_id,
             iterations, count, enabled, running) = fields
            key = (path.name, event, int(op), int(worker))
            sample = occurrence[key]
            occurrence[key] += 1
            rows.append({
                "scope": "shape", "source_file": path.name, "sample": sample,
                "operation_index": int(op), "M": int(m), "N": int(n), "K": int(k),
                "worker": int(worker), "cpu": int(cpu), "event": event, "status": status,
                "errno": int(error), "event_id": int(event_id), "iterations": int(iterations),
                "raw_u64": int(count), "time_enabled": int(enabled), "time_running": int(running),
                "scaled_count": f"{scaled(int(count), int(enabled), int(running)):.6f}",
            })

    for path in sorted(args.input_dir.glob("single_*.log")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("stage55_worker_counter\t"):
                continue
            fields = line.split("\t")
            if len(fields) != 12:
                raise ValueError(f"malformed full counter in {path}: {line}")
            (_, worker, tid, cpu, event, status, error, event_id, iterations,
             count, enabled, running) = fields
            key = (path.name, event, -1, int(worker))
            sample = occurrence[key]
            occurrence[key] += 1
            rows.append({
                "scope": "full_model", "source_file": path.name, "sample": sample,
                "operation_index": -1, "M": "", "N": "", "K": "",
                "worker": int(worker), "tid": int(tid), "cpu": int(cpu), "event": event,
                "status": status, "errno": int(error), "event_id": int(event_id),
                "iterations": int(iterations), "raw_u64": int(count),
                "time_enabled": int(enabled), "time_running": int(running),
                "scaled_count": f"{scaled(int(count), int(enabled), int(running)):.6f}",
            })
    if not rows:
        raise ValueError("no HPM rows")
    write_tsv(args.raw_output, rows)

    shape_rows: list[dict[str, object]] = []
    shapes = sorted({
        (int(row["operation_index"]), int(row["M"]), int(row["N"]), int(row["K"]), str(row["source_file"]))
        for row in rows if row["scope"] == "shape"
    })
    for operation, m, n, k, source in shapes:
        selected = [
            row for row in rows
            if row["scope"] == "shape" and row["operation_index"] == operation
            and row["source_file"] == source
        ]
        cycles = [float(row["scaled_count"]) for row in selected if row["event"] == "cycles"]
        events = sorted({str(row["event"]) for row in selected if row["event"] != "cycles"})
        for event in events:
            values = [float(row["scaled_count"]) for row in selected if row["event"] == event]
            cycle_mean = statistics.fmean(cycles)
            event_mean = statistics.fmean(values)
            shape_rows.append({
                "operation_index": operation, "M": m, "N": n, "K": k,
                "source_file": source, "event": event, "worker_samples": len(values),
                "cycles_mean_per_worker": f"{cycle_mean:.6f}",
                "event_mean_per_worker": f"{event_mean:.6f}",
                "event_per_cycle": f"{event_mean / cycle_mean:.9f}",
                "all_available": int(all(row["status"] == "available" for row in selected)),
                "all_time_running": int(all(int(row["time_running"]) > 0 for row in selected)),
            })
    write_tsv(args.shape_output, shape_rows)

    full_rows: list[dict[str, object]] = []
    for source in sorted({str(row["source_file"]) for row in rows if row["scope"] == "full_model"}):
        selected = [row for row in rows if row["scope"] == "full_model" and row["source_file"] == source]
        cycles = [float(row["scaled_count"]) for row in selected if row["event"] == "cycles"]
        events = sorted({str(row["event"]) for row in selected if row["event"] != "cycles"})
        for event in events:
            values = [float(row["scaled_count"]) for row in selected if row["event"] == event]
            cycle_mean = statistics.fmean(cycles)
            event_mean = statistics.fmean(values)
            full_rows.append({
                "source_file": source, "event": event, "worker_samples": len(values),
                "cycles_mean_per_worker": f"{cycle_mean:.6f}",
                "event_mean_per_worker": f"{event_mean:.6f}",
                "event_per_cycle": f"{event_mean / cycle_mean:.9f}",
                "ipc": f"{event_mean / cycle_mean:.9f}" if event == "instructions" else "not-applicable",
                "all_available": int(all(row["status"] == "available" for row in selected)),
                "all_time_running": int(all(int(row["time_running"]) > 0 for row in selected)),
                "negative_value_count": sum(int(row["raw_u64"]) < 0 for row in selected),
            })
    write_tsv(args.full_output, full_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
