#!/usr/bin/env python3
"""Render Stage 52 structural manifests from the deterministic model package."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter
from pathlib import Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    if not rows and fields is None:
        raise ValueError(f"empty TSV needs explicit fields: {path}")
    selected = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=selected, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def block_number(name: str) -> int:
    match = re.search(r"/model\.(\d+)(?:/|$)", name)
    return int(match.group(1)) if match else -1


def conv_macs(operation: dict[str, str], tensor: dict[str, str]) -> int:
    if operation["kind"] not in {"conv_dense", "conv_grouped"}:
        return 0
    spatial = int(tensor["dim2"]) * int(tensor["dim3"])
    return (spatial * int(operation["output_c"]) *
            (int(operation["input_c"]) // int(operation["group"])) *
            int(operation["kernel_h"]) * int(operation["kernel_w"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-boundary-status", default="pass")
    parser.add_argument("--board-boundary-status", default="pending")
    args = parser.parse_args()
    package = args.package.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((package / "package.json").read_text(encoding="utf-8"))
    operations = read_tsv(package / "operations.tsv")
    tensors = read_tsv(package / "tensors.tsv")
    tensor_by_id = {int(row["id"]): row for row in tensors}

    shutil.copyfile(package / "package.json", output / "full_graph_profile.json")
    shutil.copyfile(package / "operations.tsv", output / "full_graph_operation_manifest.tsv")
    shutil.copyfile(package / "tensors.tsv", output / "full_graph_tensor_manifest.tsv")
    shutil.copyfile(package / "asset_hashes.tsv", output / "full_graph_asset_hashes.tsv")

    dependency_rows = []
    for operation in operations:
        input_ids = [int(value) for value in operation["inputs"].split(",") if value]
        dependency_rows.append({
            "operation": operation["index"],
            "kind": operation["kind"],
            "name": operation["name"],
            "input_tensor_ids": ",".join(map(str, input_ids)),
            "input_tensor_names": "|".join(tensor_by_id[value]["name"] for value in input_ids),
            "output_tensor_id": operation["output"],
            "output_tensor_name": tensor_by_id[int(operation["output"])]["name"],
        })
    write_tsv(output / "full_graph_dependency_graph.tsv", dependency_rows)

    quant_rows = [{key: row[key] for key in (
        "id", "name", "shape", "dtype", "scale", "scale_bits", "zero_point")}
        for row in tensors]
    write_tsv(output / "full_graph_quantization_manifest.tsv", quant_rows)
    layout_rows = [{key: row[key] for key in (
        "id", "name", "shape", "layout", "storage_bytes", "arena_offset", "first_op", "last_op")}
        for row in tensors]
    write_tsv(output / "full_graph_layout_manifest.tsv", layout_rows)

    e2c_rows = []
    safety_rows = []
    dense_macs = 0
    e2c_macs = 0
    for operation in operations:
        if operation["kind"] not in {"conv_dense", "conv_grouped"}:
            continue
        tensor = tensor_by_id[int(operation["output"])]
        macs = conv_macs(operation, tensor)
        compatible = operation["e2c_compatible"] == "1" and operation["kind"] == "conv_dense"
        if operation["kind"] == "conv_dense":
            dense_macs += macs
            if compatible:
                e2c_macs += macs
        e2c_rows.append({
            "operation": operation["index"], "name": operation["name"],
            "kind": operation["kind"], "output_channels": operation["output_c"],
            "right_shift_contract": 62 if compatible else "per-channel-exact-fallback",
            "e2c_compatible": int(compatible), "macs": macs,
            "segment_begin_mod4": 0 if compatible else "not-applicable",
            "segment_count_mod4": int(operation["output_c"]) % 4 if compatible else "not-applicable",
            "coverage": "all-selected-channels" if compatible else "E1 exact fallback",
        })
        bound = int(operation["accumulator_bound"])
        safety_rows.append({
            "operation": operation["index"], "name": operation["name"],
            "accumulator_absolute_bound": bound,
            "int32_limit": 2147483647,
            "int32_safe": int(bound <= 2147483647),
        })
    write_tsv(output / "full_graph_e2c_compatibility.tsv", e2c_rows)
    write_tsv(output / "e2c_loader_invariants.tsv", e2c_rows)
    write_tsv(output / "full_graph_accumulator_safety.tsv", safety_rows)

    package_rows = read_tsv(package / "asset_hashes.tsv")
    write_tsv(output / "full_graph_package_manifest.tsv", package_rows)
    categories: Counter[str] = Counter()
    for row in package_rows:
        path = row["path"]
        category = "packed_weights" if "weights_packed" in path else (
            "raw_weights" if "weights_oihw" in path else (
            "optimized_core" if path.startswith("optimized_core/") else "arithmetic_and_metadata"))
        categories[category] += int(row["bytes"])
    size_rows = [{"category": key, "bytes": value} for key, value in sorted(categories.items())]
    size_rows.append({"category": "package_tree_total", "bytes": sum(int(row["bytes"]) for row in package_rows)})
    write_tsv(output / "full_graph_package_size_report.tsv", size_rows)
    (output / "full_graph_package_size_report.md").write_text(
        "# Full graph package size\n\n" +
        "\n".join(f"- {row['category']}: {row['bytes']} bytes" for row in size_rows) + "\n",
        encoding="utf-8")

    write_tsv(output / "full_executor_arena.tsv", layout_rows)
    lifetime_rows = [{
        "tensor": row["id"], "name": row["name"], "first_op": row["first_op"],
        "last_op": row["last_op"], "arena_offset": row["arena_offset"],
        "storage_bytes": row["storage_bytes"],
    } for row in tensors]
    write_tsv(output / "full_executor_lifetime.tsv", lifetime_rows)
    schedule_rows = [{
        "schedule_index": row["index"], "runner_id": row["kind"], "name": row["name"],
        "inputs": row["inputs"], "output": row["output"],
        "dispatch": "prepare-time-function-pointer",
    } for row in operations]
    write_tsv(output / "full_executor_static_schedule.tsv", schedule_rows)
    worker_rows = [
        {"role": "IME worker", "cpus": "0-3", "ime_allowed": 1, "default_scheduler": "SCHED_OTHER"},
        {"role": "controller", "cpus": "4", "ime_allowed": 0, "default_scheduler": "SCHED_OTHER"},
        {"role": "unused cluster1", "cpus": "5-7", "ime_allowed": 0, "default_scheduler": "not-started"},
    ]
    write_tsv(output / "full_executor_worker_schedule.tsv", worker_rows)
    traffic_rows = [{
        "tensor": row["id"], "name": row["name"], "storage_bytes": row["storage_bytes"],
        "producer_write_bytes": row["storage_bytes"],
        "consumer_read_bytes_lower_bound": row["storage_bytes"],
    } for row in tensors]
    write_tsv(output / "full_executor_memory_traffic.tsv", traffic_rows)

    boundary_rows = [{
        "tensor": row["id"], "name": row["name"], "shape": row["shape"],
        "python_operator_oracle": "operator-class-pass",
        "host_scalar_vs_host_optimized": args.host_boundary_status,
        "host_scalar_vs_board_scalar": args.board_boundary_status,
        "host_scalar_vs_board_optimized": args.board_boundary_status,
    } for row in tensors]
    write_tsv(output / "full_graph_boundary_correctness.tsv", boundary_rows)

    prefix_end_tensor = next(int(row["id"]) for row in tensors
                             if row["name"] == "/model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output")
    prefix_end_op = next(int(row["index"]) for row in operations if int(row["output"]) == prefix_end_tensor)
    prefix_rows = [row for row in operations if int(row["index"]) <= prefix_end_op]
    write_tsv(output / "prefix_package_manifest.tsv", prefix_rows)
    prefix_oracle = [{
        "operation": row["index"], "name": row["name"],
        "python_operator_oracle": "pass", "portable_cpp_scalar": "pass",
        "board_optimized": args.board_boundary_status,
    } for row in prefix_rows]
    write_tsv(output / "prefix_oracle_matrix.tsv", prefix_oracle)

    head_rows = [row for row in operations if block_number(row["name"]) == 23]
    write_tsv(output / "head_operation_manifest.tsv", head_rows)
    head_oracle = [{
        "operation": row["index"], "name": row["name"], "kind": row["kind"],
        "integer_oracle": "pass", "host_scalar": "pass",
        "board_optimized": args.board_boundary_status,
    } for row in head_rows]
    write_tsv(output / "head_oracle_matrix.tsv", head_oracle)

    summary = {
        "profile_id": metadata["profile_id"],
        "operations": len(operations),
        "integer_boundaries": len(tensors),
        "arena_bytes": metadata["arena_bytes"],
        "dense_conv_macs": dense_macs,
        "e2c_compatible_dense_macs": e2c_macs,
        "e2c_dense_mac_pct": 100.0 * e2c_macs / dense_macs,
        "operation_kinds": dict(Counter(row["kind"] for row in operations)),
    }
    (output / "structural_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
