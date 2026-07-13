#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from stage49_slice_package import generate_fixtures, round_fraction_even, sha256_file, write_tsv


CONTRACT_ID = "K1X_INT8_V2_Q31_CANDIDATE"
PROFILE_ID = "K1X_INT8_V2_Q31_CANDIDATE_GENERAL"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def encode_q31(ratio: Fraction) -> tuple[int, int]:
    if ratio < 0:
        raise ValueError("negative Q31 ratio")
    if ratio == 0:
        return 0, 0
    right_shift = 31
    multiplier = round_fraction_even(ratio * (1 << right_shift))
    while multiplier > (1 << 31) - 1 and right_shift > 0:
        right_shift -= 1
        multiplier = round_fraction_even(ratio * (1 << right_shift))
    if multiplier < 0 or multiplier > (1 << 31) - 1:
        raise ValueError("ratio cannot be represented by the Q31 candidate")
    return multiplier, right_shift


def generate(args: argparse.Namespace) -> None:
    source = args.v1_package.resolve()
    output = args.out_dir.resolve()
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(source, output)
    shutil.rmtree(output / "oracles")
    (output / "fixture_manifest.tsv").unlink()
    (output / "asset_hashes.tsv").unlink()

    operations = read_tsv(output / "operations.tsv")
    tensors = read_tsv(output / "tensors.tsv")
    scale_rows = read_tsv(output / "scale_encoding.tsv")
    rows_by_operation: dict[int, list[dict[str, str]]] = {}
    for row in scale_rows:
        rows_by_operation.setdefault(int(row["operation_index"]), []).append(row)

    q31_rows: list[dict[str, object]] = []
    for operation in operations:
        if operation["kind"] != "conv":
            continue
        operation_index = int(operation["index"])
        channel_rows = sorted(rows_by_operation[operation_index], key=lambda row: int(row["channel"]))
        multipliers = np.empty(len(channel_rows), dtype="<i8")
        shifts = np.empty(len(channel_rows), dtype="<i4")
        for channel, row in enumerate(channel_rows):
            ratio = Fraction(int(row["ratio_numerator"]), int(row["ratio_denominator"]))
            multiplier, right_shift = encode_q31(ratio)
            multipliers[channel] = multiplier
            shifts[channel] = right_shift
            q31_rows.append({
                "operation_index": operation_index,
                "operation_name": operation["name"],
                "channel": channel,
                "ratio_numerator": ratio.numerator,
                "ratio_denominator": ratio.denominator,
                "v1_multiplier": row["multiplier"],
                "v1_right_shift": row["right_shift"],
                "q31_multiplier": multiplier,
                "q31_right_shift": right_shift,
                "multiplier_storage": "little-endian-int64",
                "multiplier_effective_bits": 31,
            })
        multipliers.tofile(output / operation["requant_multiplier_file"])
        shifts.tofile(output / operation["requant_shift_file"])

    write_tsv(output / "q31_candidate_assets.tsv", q31_rows)
    fixture_args = SimpleNamespace(stage43_oracle_root=args.stage43_oracle_root.resolve())
    fixture_rows = generate_fixtures(fixture_args, output, tensors, operations)
    for row in fixture_rows:
        row["oracle"] = "K1X_INT8_V2_Q31_CANDIDATE_python_arbitrary_precision"
    write_tsv(output / "fixture_manifest.tsv", fixture_rows)

    package_path = output / "package.json"
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["contract_id"] = CONTRACT_ID
    package["profile_id"] = PROFILE_ID
    package["source_lineage_id"] += ":q31-candidate-sidecar"
    package["q31_multiplier_effective_width"] = 31
    package["q31_multiplier_storage"] = "little-endian-int64"
    package["q31_right_shift_range"] = "0..31"
    package["q31_rounding"] = "round-to-nearest-ties-to-even"
    package["q31_promotion_authorized"] = False
    package_path.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    files = [path for path in sorted(output.rglob("*")) if path.is_file() and path.name != "asset_hashes.tsv"]
    hashes = [{"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
               "sha256": sha256_file(path)} for path in files]
    write_tsv(output / "asset_hashes.tsv", hashes, ["path", "bytes", "sha256"])
    print(json.dumps({
        "contract_id": CONTRACT_ID,
        "fixtures": len(fixture_rows),
        "manifest_sha256": sha256_file(output / "asset_hashes.tsv"),
        "operations": len(operations),
        "package": str(output),
        "q31_assets": len(q31_rows),
        "tensors": len(tensors),
    }, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-package", type=Path, required=True)
    parser.add_argument("--stage43-oracle-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    generate(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
