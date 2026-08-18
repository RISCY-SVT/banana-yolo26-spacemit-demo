#!/usr/bin/env python3
"""Summarize Stage65C-R1 one-session and clean-session determinism."""

from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--output-dir", required=True, type=Path)
    options = parser.parse_args()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    rows = values(options.root / "status.tsv")
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    session_rows = []
    for row in rows:
        key = (row["case_group"], row["image_id"], row["model"], row["provider"])
        grouped[key].append(row)
        directory = (
            options.root / f"{row['case_group']}-{row['image_id']}" /
            f"{row['model']}-{row['provider']}" / f"{row['mode']}-{row['recreation']}"
        )
        sample_rows = values(directory / "samples.tsv")
        hashes = {item["output_fnv1a64"] for item in sample_rows}
        temperatures = read_numbers(directory / "thermal_after.tsv")
        frequencies = read_numbers(directory / "frequency_after.tsv")
        session_rows.append(
            {
                **row,
                "sample_runs": len(sample_rows),
                "unique_sample_output_hashes": len(hashes),
                "sample_output_hash": next(iter(hashes)) if len(hashes) == 1 else "multiple",
                "cpu_set": "0-3",
                "minimum_frequency_khz": min(frequencies) if frequencies else "",
                "maximum_frequency_khz": max(frequencies) if frequencies else "",
                "minimum_temperature_millic": min(temperatures) if temperatures else "",
                "maximum_temperature_millic": max(temperatures) if temperatures else "",
                "signal_or_error": "none" if row["exit_code"] == "0" else "nonzero-exit",
                "status": "pass" if row["exit_code"] == "0" and len(hashes) == 1 else "fail",
            }
        )
    write_tsv(options.output_dir / "session_recreation_determinism.tsv", session_rows)

    summary = []
    for (case_group, image_id, model, provider), group in sorted(grouped.items()):
        one_session = [row for row in group if row["mode"] == "one-session"]
        recreations = [row for row in group if row["mode"] == "recreate"]
        output_hashes = {row["output_sha256"] for row in recreations}
        boundary_hashes = {row["boundary_manifest_sha256"] for row in recreations}
        status = (
            len(one_session) == 1
            and len(recreations) == 10
            and len(output_hashes) == 1
            and len(boundary_hashes) == 1
            and all(row["exit_code"] == "0" for row in group)
        )
        summary.append(
            {
                "case_group": case_group, "image_id": image_id, "model": model, "provider": provider,
                "one_session_runs": 100, "clean_session_recreations": len(recreations),
                "unique_recreated_output_hashes": len(output_hashes),
                "recreated_output_sha256": next(iter(output_hashes)) if len(output_hashes) == 1 else "multiple",
                "unique_recreated_boundary_manifests": len(boundary_hashes),
                "recreated_boundary_manifest_sha256": next(iter(boundary_hashes)) if len(boundary_hashes) == 1 else "multiple",
                "status": "pass" if status else "fail",
            }
        )
    write_tsv(options.output_dir / "output_determinism.tsv", summary)
    if any(row["status"] != "pass" for row in summary + session_rows):
        raise RuntimeError("determinism gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
