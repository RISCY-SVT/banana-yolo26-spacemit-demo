#!/usr/bin/env python3
"""Aggregate Stage63 COCO evaluator outputs without recomputing metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


RESULT_COLUMNS = [
    "surface",
    "runtime",
    "provider",
    "model_surface",
    "status",
    "image_count",
    "prediction_count",
    "map50_95",
    "map50",
    "map75",
    "ap_small",
    "ap_medium",
    "ap_large",
    "ar_maxdet100",
    "prediction_sha256",
    "timing_sha256",
    "source",
]


def write_tsv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def identity_values(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["field"]: row["value"] for row in csv.DictReader(stream, delimiter="\t")}


def infer_surface_parts(surface: str, identity: dict[str, str]) -> tuple[str, str, str]:
    provider = identity.get("provider", "unknown")
    runtime = "rt206" if surface.startswith("rt206_") else "historical"
    if "_fp16" in surface:
        model_surface = "fp16"
    elif "_fp32" in surface:
        model_surface = "fp32"
    elif "_int8" in surface:
        model_surface = "int8"
    else:
        model_surface = "unknown"
    return runtime, provider, model_surface


def load_accepted(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    for row in rows:
        missing = set(RESULT_COLUMNS) - row.keys()
        if missing:
            raise ValueError(f"{path}: missing accepted-control columns: {sorted(missing)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--accepted-controls", type=Path)
    args = parser.parse_args()

    rows = load_accepted(args.accepted_controls)
    failures: list[dict[str, Any]] = []
    for summary_path in sorted(args.root.rglob("*.eval.json")):
        surface = summary_path.name.removesuffix(".eval.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        identity = identity_values(summary_path.with_name(f"{surface}.identity.tsv"))
        runtime, provider, model_surface = infer_surface_parts(surface, identity)
        rows.append(
            {
                "surface": surface,
                "runtime": runtime,
                "provider": provider,
                "model_surface": model_surface,
                "status": "pass" if int(summary["image_count"]) == 5000 else "partial",
                "image_count": summary["image_count"],
                "prediction_count": summary["prediction_count"],
                "map50_95": summary["map50_95"],
                "map50": summary["map50"],
                "map75": summary["map75"],
                "ap_small": summary["ap_small"],
                "ap_medium": summary["ap_medium"],
                "ap_large": summary["ap_large"],
                "ar_maxdet100": summary["ar_maxdet100"],
                "prediction_sha256": summary["predictions_sha256"],
                "timing_sha256": summary["timing_sha256"],
                "source": "Stage63 measured",
            }
        )

    for identity_path in sorted(args.root.rglob("*.identity.tsv")):
        surface = identity_path.name.removesuffix(".identity.tsv")
        if any(row["surface"] == surface for row in rows):
            continue
        values = identity_values(identity_path)
        failures.append(
            {
                "surface": surface,
                "exit_code": values.get("exit_code", "missing"),
                "predictions_present": int("predictions_sha256" in values),
                "timing_present": int("timing_sha256" in values),
                "reason": "no evaluator summary",
            }
        )

    rows.sort(key=lambda row: str(row["surface"]))
    output_dir = args.output_dir
    write_tsv(output_dir / "full_coco_results.tsv", rows, RESULT_COLUMNS)
    write_tsv(
        output_dir / "full_coco_prediction_hashes.tsv",
        [
            {
                "surface": row["surface"],
                "image_count": row["image_count"],
                "prediction_sha256": row["prediction_sha256"],
                "source": row["source"],
            }
            for row in rows
        ],
        ["surface", "image_count", "prediction_sha256", "source"],
    )
    write_tsv(
        output_dir / "full_coco_failures.tsv",
        failures,
        ["surface", "exit_code", "predictions_present", "timing_present", "reason"],
    )

    measured = [row for row in rows if row["source"] == "Stage63 measured"]
    lines = [
        "# Full COCO comparison",
        "",
        "All Stage63 metric rows were evaluated with the accepted COCO val2017",
        "annotations and the exact image IDs in each timing TSV. Historical rows",
        "are retained only when their archive, model, evaluator, and fixed-subset",
        "identity gates passed.",
        "",
        "| Surface | Images | mAP50-95 | mAP50 | AP small | AP medium | AP large |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in measured:
        lines.append(
            f"| {row['surface']} | {row['image_count']} | {float(row['map50_95']):.12f} "
            f"| {float(row['map50']):.12f} | {float(row['ap_small']):.12f} "
            f"| {float(row['ap_medium']):.12f} | {float(row['ap_large']):.12f} |"
        )
    lines.extend(
        [
            "",
            "The SpacemiT INT8 surface is absent because session creation aborts;",
            "it is not represented as a zero-accuracy run. Cross-runtime final",
            "tensor byte identity is not used as the correctness oracle.",
        ]
    )
    (output_dir / "full_coco_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
