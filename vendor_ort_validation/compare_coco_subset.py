#!/usr/bin/env python3
"""Compare per-image output and detection hashes for a bounded COCO control."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    required = {"image_id", "output_sha256", "detections_sha256"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path}: expected columns {sorted(required)}")
    return {int(row["image_id"]): row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    accepted = load(args.accepted)
    candidate = load(args.candidate)
    rows: list[dict[str, str | int]] = []
    for image_id in sorted(candidate):
        if image_id not in accepted:
            raise ValueError(f"candidate image {image_id} absent from accepted manifest")
        lhs = accepted[image_id]
        rhs = candidate[image_id]
        output_equal = lhs["output_sha256"] == rhs["output_sha256"]
        detections_equal = lhs["detections_sha256"] == rhs["detections_sha256"]
        rows.append(
            {
                "image_id": image_id,
                "accepted_output_sha256": lhs["output_sha256"],
                "candidate_output_sha256": rhs["output_sha256"],
                "output_equal": int(output_equal),
                "accepted_detections_sha256": lhs["detections_sha256"],
                "candidate_detections_sha256": rhs["detections_sha256"],
                "detections_equal": int(detections_equal),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    if not all(row["output_equal"] and row["detections_equal"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
