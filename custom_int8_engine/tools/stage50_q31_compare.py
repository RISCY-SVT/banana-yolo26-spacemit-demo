#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from stage49_slice_package import write_tsv


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-package", type=Path, required=True)
    parser.add_argument("--q31-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    v1_manifest = read_tsv(args.v1_package / "fixture_manifest.tsv")
    q31_manifest = read_tsv(args.q31_package / "fixture_manifest.tsv")
    q31_by_key = {(row["fixture_id"], row["tensor_id"]): row for row in q31_manifest}
    rows: list[dict[str, object]] = []
    total_histogram: Counter[int] = Counter()
    for v1_row in v1_manifest:
        key = (v1_row["fixture_id"], v1_row["tensor_id"])
        q31_row = q31_by_key[key]
        v1 = np.fromfile(args.v1_package / v1_row["nchw_file"], dtype=np.uint8)
        q31 = np.fromfile(args.q31_package / q31_row["nchw_file"], dtype=np.uint8)
        difference = q31.astype(np.int16) - v1.astype(np.int16)
        histogram = Counter(int(value) for value in difference)
        total_histogram.update(histogram)
        rows.append({
            "fixture_id": key[0],
            "tensor_id": key[1],
            "tensor_key": v1_row["tensor_key"],
            "elements": v1.size,
            "mismatches": int(np.count_nonzero(difference)),
            "max_abs_difference": int(np.max(np.abs(difference))),
            "mean_signed_difference": float(np.mean(difference)),
            "difference_histogram": ",".join(f"{code}:{count}" for code, count in sorted(histogram.items())),
        })
    write_tsv(args.output, rows)
    total_elements = sum(int(row["elements"]) for row in rows)
    total_mismatches = sum(int(row["mismatches"]) for row in rows)
    print(f"rows={len(rows)}")
    print(f"elements={total_elements}")
    print(f"mismatches={total_mismatches}")
    print(f"mismatch_pct={100.0 * total_mismatches / total_elements:.9f}")
    print("histogram=" + ",".join(f"{code}:{count}" for code, count in sorted(total_histogram.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
