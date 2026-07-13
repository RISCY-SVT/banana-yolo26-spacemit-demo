#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import onnx

from stage47_executor_assets import (
    GraphIndex,
    MODEL4_POSTACT,
    MODEL4_PREACT,
    MODEL5_OUTPUT,
    MODEL6_OUTPUT,
    MODEL7_OUTPUT,
    MODEL8_OUTPUT,
    ScheduleBuilder,
)
from stage49_slice_package import (
    CONTRACT_ID,
    LAYOUT_ID,
    PROFILE_ID,
    SCHEMA_VERSION,
    derive_integer_assets,
    f32_bits,
    generate_fixtures,
    sha256_file,
    write_tsv,
)


def build_schedule(index: GraphIndex, package: Path) -> ScheduleBuilder:
    builder = ScheduleBuilder(index, package)
    input_id = builder.tensor("model4.preact", index.qspec(MODEL4_PREACT))
    postact_id = builder.tensor("model4.postact", index.qspec(MODEL4_POSTACT))
    builder.add_lut("/model.4/cv2/final_silu", input_id, postact_id, "silu")
    model5_id = builder.simple_conv_block("model.5", postact_id, MODEL5_OUTPUT)
    model6_id = builder.c2f_cib_block("model.6", model5_id, MODEL6_OUTPUT)
    model7_id = builder.simple_conv_block("model.7", model6_id, MODEL7_OUTPUT)
    builder.c2f_cib_block("model.8", model7_id, MODEL8_OUTPUT)
    builder.finalize()
    return builder


def quant_domain_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return int(left["zero_point"]) == int(right["zero_point"]) and f32_bits(left["scale"]) == f32_bits(right["scale"])


def align_up(value: int, alignment: int = 64) -> int:
    return (value + alignment - 1) // alignment * alignment


def apply_direct_concat_placement(builder: ScheduleBuilder) -> list[dict[str, Any]]:
    tensors = builder.tensors
    operations = builder.ops
    consumers: dict[int, list[int]] = {int(tensor["id"]): [] for tensor in tensors}
    for operation in operations:
        for slot in range(4):
            tensor_id = int(operation[f"input{slot}"])
            if tensor_id >= 0:
                consumers[tensor_id].append(int(operation["index"]))

    arena_end = max(int(tensor["arena_offset"]) + int(tensor["bytes"]) for tensor in tensors)
    rows: list[dict[str, Any]] = []
    for operation in reversed(operations):
        if operation["kind"] != "concat":
            continue
        operation_index = int(operation["index"])
        output = tensors[int(operation["output0"])]
        output_preplaced = bool(output.get("stage50_concat_preplaced", False))
        if not output_preplaced:
            arena_end = align_up(arena_end)
            output["arena_offset"] = arena_end
            arena_end += int(output["bytes"])
        output_offset = int(output["arena_offset"])
        pixels = int(output["h"]) * int(output["w"])
        output_block = 0
        for slot in range(4):
            tensor_id = int(operation[f"input{slot}"])
            if tensor_id < 0:
                continue
            source = tensors[tensor_id]
            block_count = int(source["c"]) // 8
            final_consumer = consumers[tensor_id] and consumers[tensor_id][-1] == operation_index
            eligible = (
                final_consumer
                and int(source["h"]) == int(output["h"])
                and int(source["w"]) == int(output["w"])
                and quant_domain_equal(source, output)
            )
            target_offset = output_offset + output_block * pixels * 8
            original_offset = int(source["arena_offset"])
            if eligible:
                source["arena_offset"] = target_offset
                source["stage50_concat_preplaced"] = True
            rows.append({
                "operation_index": operation_index,
                "operation_name": operation["name"],
                "input_slot": slot,
                "tensor_id": tensor_id,
                "tensor_key": source["key"],
                "quant_domain_equal": int(quant_domain_equal(source, output)),
                "final_consumer": int(final_consumer),
                "direct_placement": int(eligible),
                "original_offset": original_offset,
                "selected_offset": int(source["arena_offset"]),
                "destination_offset": target_offset,
                "bytes": int(source["bytes"]),
            })
            output_block += block_count
        if output_block != int(output["c"]) // 8:
            raise ValueError(f"Concat channel mismatch: {operation['name']}")

    for tensor in tensors:
        tensor.pop("stage50_concat_preplaced", None)
    return sorted(rows, key=lambda row: (int(row["operation_index"]), int(row["input_slot"])))


def tensor_id(builder: ScheduleBuilder, key: str) -> int:
    return next(int(tensor["id"]) for tensor in builder.tensors if tensor["key"] == key)


def generate(args: argparse.Namespace) -> None:
    model_path = args.model.resolve()
    output = args.out_dir.resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    builder = build_schedule(GraphIndex(onnx.load(model_path)), output)
    placement_rows = apply_direct_concat_placement(builder)
    write_tsv(output / "concat_placement.tsv", placement_rows)
    tensors, operations = derive_integer_assets(output, builder)
    fixture_rows = generate_fixtures(args, output, tensors, operations)
    package_meta = {
        "arena_bytes": max(int(row["arena_offset"]) + int(row["bytes"]) for row in tensors),
        "byte_order": "little-endian",
        "contract_id": CONTRACT_ID,
        "input_tensor_id": tensor_id(builder, "model4.preact"),
        "layout_id": LAYOUT_ID,
        "model5_input_tensor_id": tensor_id(builder, "model4.postact"),
        "model5_output_tensor_id": tensor_id(builder, "model.5.output"),
        "model6_output_tensor_id": tensor_id(builder, "model.6.output"),
        "model8_output_tensor_id": tensor_id(builder, "model.8.output"),
        "operation_count": len(operations),
        "profile_id": PROFILE_ID,
        "schema_version": SCHEMA_VERSION,
        "source_lineage_id": f"accepted-yolo26-qdq:{sha256_file(model_path)}:stage50-model4final-model8",
        "tensor_count": len(tensors),
        "model_sha256": sha256_file(model_path),
    }
    (output / "package.json").write_text(json.dumps(package_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = [path for path in sorted(output.rglob("*")) if path.is_file() and path.name != "asset_hashes.tsv"]
    hash_rows = [{"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
                  "sha256": sha256_file(path)} for path in files]
    write_tsv(output / "asset_hashes.tsv", hash_rows, ["path", "bytes", "sha256"])
    manifest_sha = sha256_file(output / "asset_hashes.tsv")
    print(json.dumps({
        "package": str(output),
        "manifest_sha256": manifest_sha,
        "model_sha256": sha256_file(model_path),
        "tensors": len(tensors),
        "operations": len(operations),
        "fixtures": len(fixture_rows),
        "direct_concat_segments": sum(int(row["direct_placement"]) for row in placement_rows),
        "contract_id": CONTRACT_ID,
        "schema_version": SCHEMA_VERSION,
    }, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--stage43-oracle-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    generate(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
