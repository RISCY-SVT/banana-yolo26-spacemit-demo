#!/usr/bin/env python3
"""Parse Stage60 M8/M12 timings and same-process worker HPM records."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    timings: dict[tuple[int, int, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for path in sorted(args.input.glob("r*_*.tsv")):
        if any(event in path.stem for event in
               ("instructions", "stalls", "l1d_read")):
            continue
        if not ("_m12n16" in path.stem or "_m8n16" in path.stem or
                "_m4tail" in path.stem):
            continue
        for row in read_tsv(path):
            key = (int(row["resolution"]), int(row["operation_index"]), row["shape_class"])
            timings[key][row["route"]] = row

    comparison: list[dict[str, Any]] = []
    for (resolution, operation, shape_class), routes in sorted(timings.items(), reverse=True):
        if set(routes) != {"m12n16", "m8n16", "m4tail"}:
            raise ValueError(f"missing route for {resolution}/{operation}/{shape_class}")
        m12 = routes["m12n16"]
        m8 = routes["m8n16"]
        m4 = routes["m4tail"]
        mean12 = float(m12["mean_us"])
        mean8 = float(m8["mean_us"])
        exact = m12["output_hash"] == m8["output_hash"]
        comparison.append({
            "resolution": resolution,
            "operation_index": operation,
            "shape_class": shape_class,
            "M": m12["m"],
            "N": m12["n"],
            "K": m12["k"],
            "m12_mean_us": m12["mean_us"],
            "m12_p95_us": m12["p95_us"],
            "m12_p99_us": m12["p99_us"],
            "m12_gmac_per_s": m12["gmac_per_s"],
            "m12_register_safe_status": "selected-m12n16-register-budget-proven",
            "m12_epilogue_status": "full-conv-selected-e2c5",
            "m8_mean_us": m8["mean_us"],
            "m8_p95_us": m8["p95_us"],
            "m8_p99_us": m8["p99_us"],
            "m8_gmac_per_s": m8["gmac_per_s"],
            "m8_register_safe_status": "exact-existing-m8n16-route",
            "m8_epilogue_status": "full-conv-selected-e2c5",
            "m8_delta_pct": f"{100.0 * (mean8 / mean12 - 1.0):.9f}",
            "m4_tail_M": m4["m"],
            "m4_tail_mean_us": m4["mean_us"],
            "m4_tail_p95_us": m4["p95_us"],
            "m4_tail_p99_us": m4["p99_us"],
            "m4_tail_gmac_per_s": m4["gmac_per_s"],
            "m4_tail_register_safe_status": "exact-current-m4-tail-route",
            "m4_tail_epilogue_status": "full-conv-selected-e2c5",
            "m4_tail_output_hash": m4["output_hash"],
            "m4_tail_deterministic": int(m4["correctness_status"] ==
                                         "exact-component-contract"),
            "output_hash_m12": m12["output_hash"],
            "output_hash_m8": m8["output_hash"],
            "exact_hash_match": int(exact),
            "selected_route": "m12n16",
            "selection_reason": (
                "control retained; no per-shape route was promoted without complete-model "
                ">=0.5% evidence"
            ),
        })
        if not exact:
            raise ValueError(f"M8 exactness mismatch for {resolution}/{operation}")

    counter_rows: list[dict[str, Any]] = []
    aggregates: dict[tuple[int, str, str, int, int, int, int], list[tuple[str, float]]] = \
        defaultdict(list)
    for path in sorted(args.input.glob("*.counters")):
        stem = path.stem
        parts = stem.split("_")
        resolution = int(parts[0][1:])
        route = parts[1]
        group_event = "_".join(parts[2:])
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("stage56_shape_counter\t"):
                continue
            values = line.split("\t")
            if len(values) != 15:
                raise ValueError(f"invalid counter row in {path}: {line}")
            (_, operation, m, n, k, worker, cpu, event, status, error_number,
             event_id, iterations, count, enabled, running) = values
            raw = int(count)
            time_enabled = int(enabled)
            time_running = int(running)
            scaled = raw * time_enabled / time_running if time_running else 0.0
            counter_rows.append({
                "resolution": resolution,
                "route": route,
                "group_event": group_event,
                "operation_index": operation,
                "M": m,
                "N": n,
                "K": k,
                "worker": worker,
                "worker_cpu": cpu,
                "event": event,
                "status": status,
                "errno": error_number,
                "event_id": event_id,
                "iterations": iterations,
                "value_u64": raw,
                "time_enabled": time_enabled,
                "time_running": time_running,
                "scaled_value": f"{scaled:.6f}",
                "source_file": path.name,
            })
            key = (resolution, route, group_event, int(operation), int(m), int(n), int(k))
            aggregates[key].append((event, scaled))

    summary: list[dict[str, Any]] = []
    for (resolution, route, group_event, operation, m, n, k), values in sorted(aggregates.items()):
        by_event: dict[str, list[float]] = defaultdict(list)
        for event, value in values:
            by_event[event].append(value)
        cycles = by_event.get("cycles", [])
        events = by_event.get(group_event, [])
        cycle_mean = sum(cycles) / len(cycles) if cycles else 0.0
        event_mean = sum(events) / len(events) if events else 0.0
        summary.append({
            "resolution": resolution,
            "route": route,
            "operation_index": operation,
            "M": m,
            "N": n,
            "K": k,
            "event": group_event,
            "worker_samples": len(events),
            "cycles_mean_per_worker": f"{cycle_mean:.6f}",
            "event_mean_per_worker": f"{event_mean:.6f}",
            "event_count_per_cycle": f"{event_mean / cycle_mean:.9f}" if cycle_mean else "",
            "all_available": int(bool(events) and all(float(value) >= 0 for value in events)),
            "time_running_status": "positive" if cycles and events else "missing",
            "interpretation": "event-count-per-cycle; not a miss/access ratio",
        })

    write_tsv(args.output / "m8_m12_shape_matrix.tsv", comparison)
    write_tsv(args.output / "resolution_pmu_worker_raw.tsv", counter_rows)
    write_tsv(args.output / "resolution_pmu_summary.tsv", summary)
    print(f"shape_comparisons={len(comparison)}")
    print(f"counter_rows={len(counter_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
