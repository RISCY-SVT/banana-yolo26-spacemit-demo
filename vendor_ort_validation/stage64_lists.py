#!/usr/bin/env python3
"""Build deterministic, hash-bound Stage64 calibration and holdout lists."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-root", required=True, type=Path)
    parser.add_argument("--holdout-root", required=True, type=Path)
    parser.add_argument("--calibration-count", type=int, default=50)
    parser.add_argument("--holdout-count", type=int, default=100)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def images(root: Path) -> list[Path]:
    return sorted(
        (
            path.resolve()
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.name,
    )


def write_list(path: Path, paths: list[Path]) -> None:
    path.write_text("".join(f"{item}\n" for item in paths), encoding="utf-8")


def write_manifest(path: Path, set_name: str, paths: list[Path]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        fields = ["set", "index", "filename", "path", "bytes", "sha256"]
        writer = csv.DictWriter(
            output, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for index, item in enumerate(paths):
            writer.writerow(
                {
                    "set": set_name,
                    "index": index,
                    "filename": item.name,
                    "path": str(item),
                    "bytes": item.stat().st_size,
                    "sha256": sha256(item),
                }
            )


def main() -> int:
    options = parse_args()
    calibration_all = images(options.calibration_root)
    holdout_all = images(options.holdout_root)
    if len(calibration_all) < options.calibration_count:
        raise RuntimeError("insufficient calibration images")

    calibration = calibration_all[: options.calibration_count]
    calibration_names = {path.name for path in calibration}
    holdout = [
        path for path in holdout_all if path.name not in calibration_names
    ][: options.holdout_count]
    if len(holdout) < options.holdout_count:
        raise RuntimeError("insufficient disjoint holdout images")

    options.output_dir.mkdir(parents=True, exist_ok=True)
    write_list(options.output_dir / "calibration_list.txt", calibration)
    write_list(options.output_dir / "holdout_list.txt", holdout)
    write_manifest(
        options.output_dir / "calibration_manifest.tsv", "calibration", calibration
    )
    write_manifest(options.output_dir / "holdout_manifest.tsv", "holdout", holdout)

    corpus_overlap = {
        path.name for path in calibration_all
    } & {path.name for path in holdout_all}
    selected_overlap = calibration_names & {path.name for path in holdout}
    with (options.output_dir / "calibration_holdout_overlap.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        fields = ["check", "count", "status", "note"]
        writer = csv.DictWriter(
            output, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerow(
            {
                "check": "selected_calibration_vs_selected_holdout",
                "count": len(selected_overlap),
                "status": "pass" if not selected_overlap else "fail",
                "note": "selected lists must be disjoint by filename",
            }
        )
        writer.writerow(
            {
                "check": "calibration_corpus_vs_coco_val_corpus",
                "count": len(corpus_overlap),
                "status": "documented-evaluation-leakage",
                "note": (
                    "the accepted coco_calib2K corpus is a 2015-image subset "
                    "of COCO val2017; 50 selected calibration images are "
                    "excluded from the 100-image holdout but remain in full COCO"
                ),
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
