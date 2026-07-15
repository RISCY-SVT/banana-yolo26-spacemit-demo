#!/usr/bin/env python3
"""Census and exhaustively classify the full-executor LUT2 contracts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import onnx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage49_slice_package import fraction_from_f32_bits, f32_bits, round_fraction_even
from stage52_full_package import Graph, PackageBuilder


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expression_text(expression: Any) -> str:
    kind = expression[0]
    if kind == "x":
        return f"x{expression[1]}"
    if kind == "sigmoid":
        return f"sigmoid({expression_text(expression[1])})"
    return f"{kind}({expression_text(expression[1])},{expression_text(expression[2])})"


def expression_kinds(expression: Any) -> set[str]:
    kind = expression[0]
    if kind == "x":
        return {kind}
    if kind == "sigmoid":
        return {kind} | expression_kinds(expression[1])
    return {kind} | expression_kinds(expression[1]) | expression_kinds(expression[2])


def round_shift_even(value: int, shift: int) -> int:
    negative = value < 0
    absolute = -value if negative else value
    quotient = absolute >> shift
    remainder = absolute & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and quotient & 1):
        quotient += 1
    return -quotient if negative else quotient


def factorization_result(
    table: np.ndarray,
    root_specs: list[Any],
    output_spec: Any,
) -> dict[str, Any]:
    semantic = table.astype(np.int16) + 128
    ratios = [
        fraction_from_f32_bits(f32_bits(spec.scale)) /
        fraction_from_f32_bits(f32_bits(output_spec.scale))
        for spec in root_specs
    ]
    rational_mismatches = 0
    first_rational_mismatch = ""
    for left in range(256):
        left_value = (left - root_specs[0].zero_point) * ratios[0]
        for right in range(256):
            value = left_value + (right - root_specs[1].zero_point) * ratios[1]
            expected = min(255, max(0, round_fraction_even(value) + output_spec.zero_point))
            actual = int(semantic[left, right])
            if expected != actual:
                rational_mismatches += 1
                if not first_rational_mismatch:
                    first_rational_mismatch = f"{left},{right}:{expected}!={actual}"

    selected_shift = -1
    selected_multipliers = [0, 0]
    q_mismatches = 65536
    first_q_mismatch = ""
    for shift in range(8, 57):
        multipliers = [round_fraction_even(ratio * (1 << shift)) for ratio in ratios]
        mismatches = 0
        first = ""
        for left in range(256):
            left_term = (left - root_specs[0].zero_point) * multipliers[0]
            for right in range(256):
                accumulator = left_term + (right - root_specs[1].zero_point) * multipliers[1]
                expected = min(
                    255,
                    max(0, round_shift_even(accumulator, shift) + output_spec.zero_point),
                )
                actual = int(semantic[left, right])
                if expected != actual:
                    mismatches += 1
                    if not first:
                        first = f"{left},{right}:{expected}!={actual}"
        if mismatches < q_mismatches:
            q_mismatches = mismatches
            first_q_mismatch = first
        if mismatches == 0:
            selected_shift = shift
            selected_multipliers = multipliers
            q_mismatches = 0
            first_q_mismatch = ""
            break

    maximum = max(
        abs((code - spec.zero_point) * multiplier)
        for spec, multiplier in zip(root_specs, selected_multipliers)
        for code in range(256)
    ) if selected_shift >= 0 else 0
    return {
        "rational_mismatches": rational_mismatches,
        "rational_first_mismatch": first_rational_mismatch,
        "factor_shift": selected_shift,
        "left_multiplier": selected_multipliers[0],
        "right_multiplier": selected_multipliers[1],
        "factor_mismatches": q_mismatches,
        "factor_first_mismatch": first_q_mismatch,
        "maximum_term_bits": maximum.bit_length(),
        "factorable": int(selected_shift >= 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    graph = Graph(onnx.load(args.model, load_external_data=False))
    builder = PackageBuilder(graph, args.output / "scratch")
    operations = read_tsv(args.package / "operations.tsv")
    tensors = {row["id"]: row for row in read_tsv(args.package / "tensors.tsv")}
    profile = {row["operation_index"]: row for row in read_tsv(args.profile)}

    census: list[dict[str, Any]] = []
    exhaustive: list[dict[str, Any]] = []
    unique: dict[str, dict[str, Any]] = {}
    for operation in operations:
        if operation["kind"] != "lut2":
            continue
        index = operation["index"]
        name = operation["name"]
        output_tensor = tensors[operation["output"]]
        if "@" in name:
            source_name, target_name = name.split("@", 1)
        else:
            target_name = output_tensor["name"]
            quantizer = graph.producer[target_name]
            source_name = quantizer.input[0]
        roots, expression = builder.scalar_expression_roots(source_name)
        output_spec = graph.qspec(target_name)
        root_specs = [graph.qspec(root) for root in roots]
        table_path = args.package / operation["lut_file"]
        table = np.fromfile(table_path, dtype=np.int8).reshape(256, 256)
        table_sha = sha256(table_path)
        kinds = expression_kinds(expression)
        classification = (
            "pure_add" if kinds <= {"x", "add"} and expression[0] == "add"
            else "mul" if "mul" in kinds and "sigmoid" not in kinds
            else "add_mul_sigmoid_composite" if "sigmoid" in kinds
            else "other_composite"
        )
        timing = profile.get(index, {})
        active = int(index in profile)
        census_row = {
            "operation_index": index,
            "name": name,
            "source_expression": expression_text(expression),
            "expression_class": classification,
            "root_tensor_ids": operation["inputs"],
            "root_names": "|".join(roots),
            "root_scales": "|".join(f"{float(spec.scale):.17g}" for spec in root_specs),
            "root_zero_points": "|".join(str(spec.zero_point) for spec in root_specs),
            "output_scale": f"{float(output_spec.scale):.17g}",
            "output_zero_point": output_spec.zero_point,
            "table_sha256": table_sha,
            "tensor_bytes": output_tensor["storage_bytes"],
            "active_full_executor": active,
            "current_mean_us": timing.get("mean_us", ""),
            "current_p95_us": timing.get("p95_us", ""),
        }
        census.append(census_row)
        unique.setdefault(table_sha, {
            "table_sha256": table_sha,
            "expression_class": classification,
            "first_operation_index": index,
            "operation_indices": [],
            "active_operation_count": 0,
            "table_bytes": table_path.stat().st_size,
        })
        unique[table_sha]["operation_indices"].append(index)
        unique[table_sha]["active_operation_count"] += active

        if classification == "pure_add":
            result = factorization_result(table, root_specs, output_spec)
        else:
            result = {
                "rational_mismatches": "not-applicable",
                "rational_first_mismatch": "",
                "factor_shift": -1,
                "left_multiplier": 0,
                "right_multiplier": 0,
                "factor_mismatches": "not-attempted-without-proof",
                "factor_first_mismatch": "",
                "maximum_term_bits": 0,
                "factorable": 0,
            }
        exhaustive.append({
            "operation_index": index,
            "table_sha256": table_sha,
            "expression_class": classification,
            "pairs_tested": 65536 if classification == "pure_add" else 0,
            **result,
        })

    unique_rows = []
    for row in unique.values():
        row = dict(row)
        row["operation_count"] = len(row["operation_indices"])
        row["operation_indices"] = ",".join(row["operation_indices"])
        unique_rows.append(row)

    write_tsv(args.output / "lut2_expression_census.tsv", census, list(census[0]))
    write_tsv(args.output / "lut2_unique_table_manifest.tsv", unique_rows, list(unique_rows[0]))
    write_tsv(args.output / "lut2_factorization_exhaustive.tsv", exhaustive, list(exhaustive[0]))
    (args.output / "lut2_analysis_summary.json").write_text(json.dumps({
        "operation_count": len(census),
        "active_operation_count": sum(int(row["active_full_executor"]) for row in census),
        "unique_table_count": len(unique_rows),
        "pure_add_count": sum(row["expression_class"] == "pure_add" for row in census),
        "factorable_count": sum(int(row["factorable"]) for row in exhaustive),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
