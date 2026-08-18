#!/usr/bin/env python3
"""Summarize Stage65C-R1 one-session and clean-session determinism."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def write_tsv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise ValueError(f"refusing empty TSV: {path}")
    fields = list(values[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def values(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def boundary_manifest_sha256(directory: Path) -> str:
    lines = []
    for path in sorted(directory.glob("boundary-*.bin"), key=lambda item: item.name):
        lines.append(f"{sha256_file(path)}  {path.name}\n")
    if len(lines) != 6:
        raise ValueError(f"expected six boundary files under {directory}, got {len(lines)}")
    return hashlib.sha256("".join(sorted(lines)).encode()).hexdigest()


def read_numbers(path: Path) -> list[int]:
    result = []
    for line in path.read_text().splitlines():
        _name, _separator, value = line.rpartition("\t")
        if value:
            result.append(int(value))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--hash-repeat-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    options = parser.parse_args()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    rows = values(options.root / "status.tsv")
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    session_rows: list[dict[str, Any]] = []
    for row in rows:
        key = (row["case_group"], row["image_id"], row["model"], row["provider"])
        grouped[key].append(row)
        directory = (
            options.root / f"{row['case_group']}-{row['image_id']}" /
            f"{row['model']}-{row['provider']}" / f"{row['mode']}-{row['recreation']}"
        )
        sample_rows = values(directory / "samples.tsv")
        temperatures = read_numbers(directory / "thermal_after.tsv")
        frequencies = read_numbers(directory / "frequency_after.tsv")
        expected_runs = 100 if row["mode"] == "one-session" else 1
        frozen_status = (
            row["exit_code"] == "0"
            and len(sample_rows) == expected_runs
            and sha256_file(directory / "samples.tsv") == row["samples_sha256"]
            and sha256_file(directory / "output.bin") == row["output_sha256"]
            and boundary_manifest_sha256(directory / "boundaries")
            == row["boundary_manifest_sha256"]
            and "output_fnv1a64" not in (sample_rows[0] if sample_rows else {})
        )
        session_rows.append(
            {
                "evidence_surface": "frozen-session-matrix",
                **row,
                "sample_runs": len(sample_rows),
                "sample_hash_schema": "not-exposed-by-frozen-runner",
                "unique_sample_output_hashes": "not-available",
                "sample_output_hash": "not-available",
                "cpu_set": "0-3",
                "minimum_frequency_khz": min(frequencies) if frequencies else "",
                "maximum_frequency_khz": max(frequencies) if frequencies else "",
                "minimum_temperature_millic": min(temperatures) if temperatures else "",
                "maximum_temperature_millic": max(temperatures) if temperatures else "",
                "signal_or_error": "none" if row["exit_code"] == "0" else "nonzero-exit",
                "status": "pass" if frozen_status else "fail",
            }
        )

    diagnostic_rows = values(options.hash_repeat_root / "status.tsv")
    expected_keys = set(grouped)
    diagnostic_keys: set[tuple[str, str, str, str]] = set()
    for row in diagnostic_rows:
        key = (row["case_group"], row["image_id"], row["model"], row["provider"])
        if key in diagnostic_keys:
            raise ValueError(f"duplicate diagnostic key: {key}")
        diagnostic_keys.add(key)
        directory = (
            options.hash_repeat_root / f"{row['case_group']}-{row['image_id']}" /
            f"{row['model']}-{row['provider']}"
        )
        sample_rows = values(directory / "samples.tsv")
        sample_hashes = {item["output_fnv1a64"] for item in sample_rows}
        temperatures = read_numbers(directory / "thermal_after.tsv")
        frequencies = read_numbers(directory / "frequency_after.tsv")
        frozen_one_session = [
            item for item in grouped.get(key, []) if item["mode"] == "one-session"
        ]
        diagnostic_status = (
            row["status"] == "pass"
            and row["runs"] == "100"
            and row["unique_sample_hashes"] == "1"
            and len(sample_rows) == 100
            and len(sample_hashes) == 1
            and next(iter(sample_hashes), "") == row["sample_output_fnv1a64"]
            and sha256_file(directory / "samples.tsv") == row["samples_sha256"]
            and sha256_file(directory / "output.bin") == row["output_sha256"]
            and boundary_manifest_sha256(directory / "boundaries")
            == row["boundary_manifest_sha256"]
            and len(frozen_one_session) == 1
            and row["output_sha256"] == frozen_one_session[0]["output_sha256"]
            and row["boundary_manifest_sha256"]
            == frozen_one_session[0]["boundary_manifest_sha256"]
        )
        session_rows.append(
            {
                "evidence_surface": "diagnostic-per-run-hash",
                "case_group": row["case_group"],
                "image_id": row["image_id"],
                "model": row["model"],
                "provider": row["provider"],
                "mode": "one-session-per-run-hash",
                "recreation": "0",
                "exit_code": "0" if row["status"] == "pass" else "1",
                "output_sha256": row["output_sha256"],
                "boundary_manifest_sha256": row["boundary_manifest_sha256"],
                "samples_sha256": row["samples_sha256"],
                "sample_runs": len(sample_rows),
                "sample_hash_schema": "output_fnv1a64",
                "unique_sample_output_hashes": len(sample_hashes),
                "sample_output_hash": next(iter(sample_hashes)) if len(sample_hashes) == 1 else "multiple",
                "cpu_set": "0-3",
                "minimum_frequency_khz": min(frequencies) if frequencies else "",
                "maximum_frequency_khz": max(frequencies) if frequencies else "",
                "minimum_temperature_millic": min(temperatures) if temperatures else "",
                "maximum_temperature_millic": max(temperatures) if temperatures else "",
                "signal_or_error": "none" if row["status"] == "pass" else "nonzero-exit",
                "status": "pass" if diagnostic_status else "fail",
            }
        )
    if diagnostic_keys != expected_keys:
        missing = sorted(expected_keys - diagnostic_keys)
        extra = sorted(diagnostic_keys - expected_keys)
        raise ValueError(f"diagnostic key mismatch: missing={missing}, extra={extra}")
    write_tsv(options.output_dir / "session_recreation_determinism.tsv", session_rows)

    summary = []
    for (case_group, image_id, model, provider), group in sorted(grouped.items()):
        one_session = [row for row in group if row["mode"] == "one-session"]
        recreations = [row for row in group if row["mode"] == "recreate"]
        output_hashes = {row["output_sha256"] for row in group}
        boundary_hashes = {row["boundary_manifest_sha256"] for row in group}
        diagnostic = [
            row for row in diagnostic_rows
            if (row["case_group"], row["image_id"], row["model"], row["provider"])
            == (case_group, image_id, model, provider)
        ]
        status = (
            len(one_session) == 1
            and len(recreations) == 10
            and len(output_hashes) == 1
            and len(boundary_hashes) == 1
            and all(row["exit_code"] == "0" for row in group)
            and len(diagnostic) == 1
            and diagnostic[0]["runs"] == "100"
            and diagnostic[0]["unique_sample_hashes"] == "1"
            and diagnostic[0]["output_sha256"] == next(iter(output_hashes))
            and diagnostic[0]["boundary_manifest_sha256"] == next(iter(boundary_hashes))
            and diagnostic[0]["status"] == "pass"
        )
        summary.append(
            {
                "case_group": case_group, "image_id": image_id, "model": model, "provider": provider,
                "one_session_runs": 100, "clean_session_recreations": len(recreations),
                "diagnostic_per_run_hash_runs": int(diagnostic[0]["runs"]) if diagnostic else 0,
                "diagnostic_unique_per_run_hashes": (
                    int(diagnostic[0]["unique_sample_hashes"]) if diagnostic else 0
                ),
                "diagnostic_per_run_output_hash": (
                    diagnostic[0]["sample_output_fnv1a64"] if diagnostic else "missing"
                ),
                "unique_all_session_output_hashes": len(output_hashes),
                "all_session_output_sha256": next(iter(output_hashes)) if len(output_hashes) == 1 else "multiple",
                "unique_all_session_boundary_manifests": len(boundary_hashes),
                "all_session_boundary_manifest_sha256": next(iter(boundary_hashes)) if len(boundary_hashes) == 1 else "multiple",
                "status": "pass" if status else "fail",
            }
        )
    write_tsv(options.output_dir / "output_determinism.tsv", summary)
    if any(row["status"] != "pass" for row in summary + session_rows):
        raise RuntimeError("determinism gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
