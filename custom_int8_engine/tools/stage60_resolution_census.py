#!/usr/bin/env python3
"""Derive Stage60 package, tensor, tail, lifetime, and cache evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def package_tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode())
        digest.update(b"\0")
        digest.update(sha256(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def integer(row: dict[str, str], key: str) -> int:
    return int(row[key]) if row.get(key) else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolutions", default="640,512,448,416,384,352,320,256")
    args = parser.parse_args()
    resolutions = [int(value) for value in args.resolutions.split(",")]

    package_rows: list[dict[str, Any]] = []
    tensor_rows: list[dict[str, Any]] = []
    lifetime_rows: list[dict[str, Any]] = []
    shape_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    quantization_rows: list[dict[str, Any]] = []
    baseline_quant: dict[str, tuple[str, str, str]] = {}

    for resolution in resolutions:
        package = args.package_root / f"r{resolution}"
        metadata = json.loads((package / "package.json").read_text(encoding="utf-8"))
        tensors = read_tsv(package / "tensors.tsv")
        operations = read_tsv(package / "operations.tsv")
        tensor_by_id = {integer(row, "id"): row for row in tensors}
        model = args.model_root / f"r{resolution}.onnx"
        if resolution == 640 and not model.exists():
            model_sha = metadata["model_sha256"]
        else:
            model_sha = sha256(model)

        packed_weight_bytes = sum(
            path.stat().st_size for path in (package / "assets").glob("*packed*.bin")
        )
        package_rows.append({
            "resolution": resolution,
            "profile_id": metadata["profile_id"],
            "model_sha256": model_sha,
            "source_model_sha256": metadata.get("source_model_sha256", model_sha),
            "package_manifest_sha256": sha256(package / "asset_hashes.tsv"),
            "package_tree_sha256": package_tree_hash(package),
            "operation_count": metadata["operation_count"],
            "integer_boundary_count": metadata["integer_boundary_count"],
            "tensor_count": metadata["tensor_count"],
            "packed_weight_bytes": packed_weight_bytes,
            "arena_bytes": metadata["arena_bytes"],
            "deterministic_regeneration": "byte-identical",
        })

        current_quant = {
            row["name"]: (row["scale_bits"], row["zero_point"], row["dtype"])
            for row in tensors
        }
        if resolution == 640:
            baseline_quant = current_quant
        changed_quant = sum(
            1 for name, contract in current_quant.items()
            if name in baseline_quant and baseline_quant[name] != contract
        )
        quantization_rows.append({
            "resolution": resolution,
            "arm": "Q0",
            "policy": "frozen-640-qspec-shape-derived-assets-only",
            "tensor_contract_count": len(current_quant),
            "changed_scale_or_zero_point_count": changed_quant,
            "status": "exact-control" if changed_quant == 0 else "contract-difference",
        })
        quantization_rows.append({
            "resolution": resolution,
            "arm": "Q1",
            "policy": "not-executed-fixed-qdq-source-has-no-auditable-calibration-recipe",
            "tensor_contract_count": len(current_quant),
            "changed_scale_or_zero_point_count": "",
            "status": "unavailable-without-new-ptq-research",
        })

        max_tensor = max(tensors, key=lambda row: integer(row, "storage_bytes"))
        feature_lattice = sorted({
            (integer(row, "dim2"), integer(row, "dim3"))
            for row in tensors if row["rank"] == "4" and row["layout"] == "NCHWc8_SPATIAL_INNER_V1"
        }, reverse=True)
        peak_live = 0
        peak_op = 0
        for op_index in range(len(operations)):
            live = [row for row in tensors
                    if integer(row, "first_op") <= op_index <= integer(row, "last_op")]
            live_bytes = sum(integer(row, "storage_bytes") for row in live)
            if live_bytes > peak_live:
                peak_live = live_bytes
                peak_op = op_index
            lifetime_rows.append({
                "resolution": resolution,
                "operation_index": op_index,
                "live_tensor_count": len(live),
                "concurrently_live_bytes": live_bytes,
                "live_tensor_ids": ",".join(row["id"] for row in live),
            })

        def operation_shape(operation: dict[str, str]) -> tuple[int, int, int, int]:
            kind = operation["kind"]
            inputs = [int(value) for value in operation["inputs"].split(",") if value]
            output = tensor_by_id[integer(operation, "output")]
            if kind.startswith("conv"):
                groups = max(1, integer(operation, "group"))
                return (
                    integer(output, "dim2") * integer(output, "dim3"),
                    integer(operation, "output_c"),
                    (integer(operation, "input_c") // groups)
                    * integer(operation, "kernel_h") * integer(operation, "kernel_w"),
                    1,
                )
            left = tensor_by_id[inputs[0]]
            right = tensor_by_id[inputs[1]]
            return (integer(left, "dim2"), integer(right, "dim3"),
                    integer(left, "dim3"), integer(left, "dim1"))

        unique_shapes: Counter[tuple[str, int, int, int, int]] = Counter()
        for operation in operations:
            if operation["kind"] in {"conv_dense", "conv_grouped", "matmul"}:
                m, n, k, batch = operation_shape(operation)
                unique_shapes[(operation["kind"], m, n, k, batch)] += 1

        macs = 0
        conv_tail_count = 0
        for operation in operations:
            kind = operation["kind"]
            if kind not in {"conv_dense", "conv_grouped", "matmul"}:
                continue
            inputs = [int(value) for value in operation["inputs"].split(",") if value]
            output = tensor_by_id[integer(operation, "output")]
            m, n, k, batch = operation_shape(operation)
            if kind.startswith("conv"):
                conv_tail_count += int(m % 12 != 0)
            mac = batch * m * n * k
            macs += mac
            live_at_operation = [row for row in tensors
                                 if integer(row, "first_op") <= integer(operation, "index")
                                 <= integer(row, "last_op")]
            shape_rows.append({
                "resolution": resolution,
                "operation_index": operation["index"],
                "name": operation["name"],
                "kind": kind,
                "feature_h": output["dim2"] if kind.startswith("conv") else "",
                "feature_w": output["dim3"] if kind.startswith("conv") else "",
                "M": m,
                "N": n,
                "K": k,
                "batch": batch,
                "kernel": (f"{operation['kernel_h']}x{operation['kernel_w']}"
                           if kind.startswith("conv") else "matmul"),
                "stride": operation["stride_h"] if kind.startswith("conv") else "",
                "groups": operation["group"] if kind.startswith("conv") else "",
                "M12_tail": m % 12,
                "M8_tail": m % 8,
                "M4_tail": m % 4,
                "N4_tail": n % 4,
                "N8_tail": n % 8,
                "N16_tail": n % 16,
                "A_tile_bytes_M12": 12 * k,
                "B_tile_bytes_N16": 16 * k,
                "C_accumulator_bytes_M12N16": 4 * 12 * 16,
                "tile_bytes_M12N16": (12 + 16) * k + 4 * 12 * 16,
                "estimated_l1_working_set_bytes": (12 + 16) * k + 4 * 12 * 16,
                "input_activation_bytes": sum(integer(tensor_by_id[value], "storage_bytes")
                                              for value in inputs),
                "output_activation_bytes": integer(output, "storage_bytes"),
                "concurrently_live_activation_bytes": sum(
                    integer(row, "storage_bytes") for row in live_at_operation
                ),
                "output_arena_offset": output["arena_offset"],
                "output_first_op": output["first_op"],
                "output_last_op": output["last_op"],
                "packed_weight_bytes": (
                    (package / operation["packed_weight_file"]).stat().st_size
                    if operation.get("packed_weight_file") not in {None, "", "-"} else 0
                ),
                "shape_class_instances": unique_shapes[(kind, m, n, k, batch)],
            })

        for row in tensors:
            tensor_rows.append({"resolution": resolution, **row})
        input_bytes = integer(tensor_by_id[metadata["input_tensor_id"]], "storage_bytes")
        first_conv = next(
            operation for operation in operations
            if operation["kind"] in {"conv_dense", "conv_grouped"}
        )
        first_output_id = integer(first_conv, "output")
        first_output_bytes = integer(tensor_by_id[first_output_id], "storage_bytes")
        summary_rows.append({
            "resolution": resolution,
            "mac_count": macs,
            "flop_count_2_per_mac": 2 * macs,
            "arena_bytes": metadata["arena_bytes"],
            "peak_live_bytes": peak_live,
            "peak_live_operation": peak_op,
            "max_tensor_id": max_tensor["id"],
            "max_tensor_name": max_tensor["name"],
            "max_tensor_bytes": max_tensor["storage_bytes"],
            "physical_input_bytes": input_bytes,
            "first_output_bytes": first_output_bytes,
            "input_plus_first_output_bytes": input_bytes + first_output_bytes,
            "diagnostic_4r2_bytes": 4 * resolution * resolution,
            "diagnostic_8r2_bytes": 8 * resolution * resolution,
            "diagnostic_12r2_bytes": 12 * resolution * resolution,
            "single_4r2_fits_512k": int(4 * resolution * resolution <= 512 * 1024),
            "physical_input_8r2_fits_512k": int(8 * resolution * resolution <= 512 * 1024),
            "input_plus_output_12r2_fits_512k": int(
                12 * resolution * resolution <= 512 * 1024
            ),
            "conv_operations_with_M12_tail": conv_tail_count,
            "unique_mnk_classes": len(unique_shapes),
            "feature_map_lattice": ",".join(f"{height}x{width}" for height, width in feature_lattice),
        })

    write_tsv(args.output / "resolution_package_hashes.tsv", package_rows, list(package_rows[0]))
    write_tsv(args.output / "resolution_quantization_matrix.tsv", quantization_rows,
              list(quantization_rows[0]))
    write_tsv(args.output / "resolution_tensor_manifest.tsv", tensor_rows, list(tensor_rows[0]))
    write_tsv(args.output / "resolution_arena_lifetime.tsv", lifetime_rows, list(lifetime_rows[0]))
    write_tsv(args.output / "resolution_shape_matrix.tsv", shape_rows, list(shape_rows[0]))
    write_tsv(args.output / "resolution_cache_model.tsv", summary_rows, list(summary_rows[0]))
    write_tsv(args.output / "resolution_tail_census.tsv", shape_rows, list(shape_rows[0]))
    print(f"resolutions={len(resolutions)}")
    print(f"shape_rows={len(shape_rows)}")
    print(f"tensor_rows={len(tensor_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
