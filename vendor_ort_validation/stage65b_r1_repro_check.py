#!/usr/bin/env python3
"""Verify a Stage65B-R1 XSlim generation or generation pair."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path
from typing import Any


TIMESTAMP = re.compile(
    r"20\d\d-\d\d-\d\d[ T]\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:?\d\d)?"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def summary(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 1:
        raise ValueError(f"expected one summary row: {path}")
    row = rows[0]
    model = Path(row["output_model"])
    expected = row["output_sha256"]
    passed = (
        row["returncode"] == "0"
        and row["output_exists"] == "1"
        and row["checker"] == "pass"
        and model.is_file()
        and sha256(model) == expected
    )
    if not passed:
        raise RuntimeError(f"generation summary did not pass identity checks: {path}")
    return row


def report(row: dict[str, str]) -> tuple[Path, Path]:
    model = Path(row["output_model"])
    run_root = model.parent.parent
    matches = sorted(model.parent.glob("*_report.md"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one Graphwise report under {model.parent}, found {len(matches)}"
        )
    return run_root, matches[0]


def normalized_report(path: Path, run_root: Path) -> bytes:
    text = path.read_text(encoding="utf-8")
    text = text.replace(str(run_root), "<RUN_ROOT>")
    text = TIMESTAMP.sub("<TIMESTAMP>", text)
    return text.encode("utf-8")


def write_tsv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(row),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--run1-summary", required=True, type=Path)
    parser.add_argument("--run2-summary", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()

    first = summary(options.run1_summary)
    first_root, first_report = report(first)
    row: dict[str, Any] = {
        "lane": options.lane.upper(),
        "run1_summary": str(options.run1_summary),
        "run1_model_sha256": first["output_sha256"],
        "run1_report_sha256": sha256(first_report),
        "run2_summary": "",
        "run2_model_sha256": "",
        "run2_report_sha256": "",
        "deployable_onnx_byte_equal": "not-applicable",
        "normalized_analysis_report_equal": "not-applicable",
        "status": "single-generation-pass",
    }
    if options.run2_summary is not None:
        second = summary(options.run2_summary)
        second_root, second_report = report(second)
        model_equal = first["output_sha256"] == second["output_sha256"]
        analysis_equal = normalized_report(
            first_report, first_root
        ) == normalized_report(second_report, second_root)
        row.update(
            {
                "run2_summary": str(options.run2_summary),
                "run2_model_sha256": second["output_sha256"],
                "run2_report_sha256": sha256(second_report),
                "deployable_onnx_byte_equal": int(model_equal),
                "normalized_analysis_report_equal": int(analysis_equal),
                "status": "pass" if model_equal and analysis_equal else "fail",
            }
        )
        if not model_equal or not analysis_equal:
            write_tsv(options.output, row)
            raise RuntimeError(
                f"reproducibility mismatch for {options.lane}: "
                f"model_equal={model_equal}, analysis_equal={analysis_equal}"
            )
    write_tsv(options.output, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
