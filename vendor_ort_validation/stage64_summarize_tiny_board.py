#!/usr/bin/env python3
"""Normalize Stage64 tiny-control board results and ORT profiles."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def profile_suffix(
    row: dict[str, str], occurrence: int
) -> str:
    test_id = row["test_id"]
    provider = row["provider"]
    cpus = row["cpus"]
    if provider == "cpu":
        return "primary"
    if test_id.startswith(("c4_", "m3_")):
        return "negative-once"
    if test_id not in {"c1_s8_conv_pc_explicit", "m1_s8_matmul"}:
        return "primary"
    if cpus == "0-3" and occurrence == 1:
        return "primary"
    return f"affinity-{cpus.replace('-', '_')}"


def profile_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "profile_json": "",
            "profile_event_count": 0,
            "spacemit_kernel_events": 0,
            "cpu_kernel_events": 0,
            "spacemit_unique_nodes": 0,
            "cpu_unique_nodes": 0,
        }
    files = sorted(path.glob("*.json"))
    if not files:
        return {
            "profile_json": "",
            "profile_event_count": 0,
            "spacemit_kernel_events": 0,
            "cpu_kernel_events": 0,
            "spacemit_unique_nodes": 0,
            "cpu_unique_nodes": 0,
        }
    try:
        events = json.loads(files[-1].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {
            "profile_json": str(files[-1]),
            "profile_event_count": 0,
            "spacemit_kernel_events": 0,
            "cpu_kernel_events": 0,
            "spacemit_unique_nodes": 0,
            "cpu_unique_nodes": 0,
        }
    kernel_events: list[tuple[str, str]] = []
    for event in events:
        args = event.get("args", {})
        provider = str(args.get("provider", ""))
        if provider:
            kernel_events.append((provider, str(args.get("op_name", event.get("name", "")))))
    spacemit = [item for item in kernel_events if item[0] == "SpaceMITExecutionProvider"]
    cpu = [item for item in kernel_events if item[0] == "CPUExecutionProvider"]
    return {
        "profile_json": str(files[-1]),
        "profile_event_count": len(kernel_events),
        "spacemit_kernel_events": len(spacemit),
        "cpu_kernel_events": len(cpu),
        "spacemit_unique_nodes": len(set(spacemit)),
        "cpu_unique_nodes": len(set(cpu)),
    }


def classify(row: dict[str, Any]) -> str:
    if row["exit_code"] == "0" and row["exact"] == "1":
        if row["provider"] == "spacemit":
            if int(row["spacemit_kernel_events"]) > 0:
                return "exact-spacemit-assigned"
            return "placement-not-proven"
        return "exact-cpu"
    if row["exit_code"] == "134" and row["signal"] == "6":
        return "unsupported-abort"
    if row["timed_out"] == "1":
        return "timeout"
    return "failed"


def main() -> int:
    options = parse_args()
    raw = read_tsv(options.evidence_root / "tiny_vendor_contract_matrix.raw.tsv")
    occurrence_counter: Counter[tuple[str, str, str]] = Counter()
    rows: list[dict[str, Any]] = []
    for source in raw:
        key = (source["test_id"], source["provider"], source["cpus"])
        occurrence_counter[key] += 1
        occurrence = occurrence_counter[key]
        suffix = profile_suffix(source, occurrence)
        case_id = f"{source['test_id']}__{source['provider']}__{suffix}"
        profile = profile_summary(options.evidence_root / "profiles" / case_id)
        log = options.evidence_root / "logs" / f"{case_id}.log"
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        error = ""
        for line in reversed(text.splitlines()):
            if "what():" in line or "error message:" in line:
                error = line.strip()
                break
        row: dict[str, Any] = {
            **source,
            "case_id": case_id,
            "suffix": suffix,
            "log": str(log) if log.exists() else "",
            "error_summary": error,
            **profile,
        }
        row["classification"] = classify(row)
        rows.append(row)

    common_fields = [
        "case_id",
        "test_id",
        "provider",
        "cpus",
        "suffix",
        "exit_code",
        "signal",
        "timed_out",
        "session_created",
        "exact",
        "output_sha256",
        "oracle_sha256",
        "spacemit_kernel_events",
        "cpu_kernel_events",
        "spacemit_unique_nodes",
        "cpu_unique_nodes",
        "classification",
        "error_summary",
        "log",
        "profile_json",
    ]
    write_tsv(
        options.output_dir / "tiny_vendor_contract_matrix.tsv",
        rows,
        common_fields,
    )
    write_tsv(
        options.output_dir / "tiny_control_correctness.tsv",
        rows,
        [
            "case_id",
            "test_id",
            "provider",
            "cpus",
            "output_sha256",
            "oracle_sha256",
            "exact",
            "classification",
        ],
    )
    write_tsv(
        options.output_dir / "tiny_control_exit_signals.tsv",
        rows,
        [
            "case_id",
            "test_id",
            "provider",
            "cpus",
            "exit_code",
            "signal",
            "timed_out",
            "session_created",
            "classification",
            "error_summary",
        ],
    )
    write_tsv(
        options.output_dir / "tiny_provider_assignment.tsv",
        [row for row in rows if row["provider"] == "spacemit"],
        [
            "case_id",
            "test_id",
            "cpus",
            "spacemit_kernel_events",
            "cpu_kernel_events",
            "spacemit_unique_nodes",
            "cpu_unique_nodes",
            "classification",
            "profile_json",
        ],
    )
    write_tsv(
        options.output_dir / "tiny_affinity_cluster_matrix.tsv",
        [row for row in rows if str(row["suffix"]).startswith("affinity-")],
        [
            "case_id",
            "test_id",
            "cpus",
            "exit_code",
            "signal",
            "exact",
            "spacemit_kernel_events",
            "cpu_kernel_events",
            "classification",
        ],
    )
    write_tsv(
        options.output_dir / "fault_inventory.tsv",
        [row for row in rows if row["exit_code"] != "0"],
        [
            "case_id",
            "test_id",
            "provider",
            "cpus",
            "exit_code",
            "signal",
            "timed_out",
            "classification",
            "error_summary",
            "log",
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
