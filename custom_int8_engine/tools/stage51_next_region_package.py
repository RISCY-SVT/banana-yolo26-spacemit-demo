#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

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
    generate_fixtures,
    sha256_file,
    write_tsv,
)
from stage50_slice_package import apply_direct_concat_placement


MODEL9_CONCAT = "/model.9/Concat_output_0_QuantizeLinear_Output"
MODEL9_OUTPUT = "/model.9/Add_output_0_QuantizeLinear_Output"
NEXT_REGION_FIRST_OPERATION = 29
NEXT_REGION_LAST_OPERATION = 35


def build_schedule(index: GraphIndex, package: Path) -> ScheduleBuilder:
    builder = ScheduleBuilder(index, package)
    input_id = builder.tensor("model4.preact", index.qspec(MODEL4_PREACT))
    postact_id = builder.tensor("model4.postact", index.qspec(MODEL4_POSTACT))
    builder.add_lut("/model.4/cv2/final_silu", input_id, postact_id, "silu")
    model5_id = builder.simple_conv_block("model.5", postact_id, MODEL5_OUTPUT)
    model6_id = builder.c2f_cib_block("model.6", model5_id, MODEL6_OUTPUT)
    model7_id = builder.simple_conv_block("model.7", model6_id, MODEL7_OUTPUT)
    model8_id = builder.c2f_cib_block("model.8", model7_id, MODEL8_OUTPUT)

    concat_spec = index.qspec(MODEL9_CONCAT)
    cv1_node = index.nodes_by_name["/model.9/cv1/conv/Conv"]
    cv1_channels = int(index.conv_assets(cv1_node)[0].shape[0])
    cv1_id = builder.tensor(
        "model.9.cv1_act", concat_spec, cv1_channels,
        "/model.9/cv1/act@model9_concat_scale",
    )
    # The accepted ONNX graph feeds model9 cv1 directly into SPPF MaxPool.
    # Unlike the preceding Conv blocks, this cv1 has no SiLU node.
    builder.add_conv(cv1_node.name, model8_id, cv1_id, "none")

    pool0_id = builder.tensor(
        "model.9.pool0", concat_spec, cv1_channels,
        "/model.9/m/MaxPool_output_0@model9_concat_scale",
    )
    pool1_id = builder.tensor(
        "model.9.pool1", concat_spec, cv1_channels,
        "/model.9/m_1/MaxPool_output_0@model9_concat_scale",
    )
    pool2_id = builder.tensor(
        "model.9.pool2", concat_spec, cv1_channels,
        "/model.9/m_2/MaxPool_output_0@model9_concat_scale",
    )
    builder.add_maxpool("/model.9/m/MaxPool", cv1_id, pool0_id)
    builder.add_maxpool("/model.9/m_1/MaxPool", pool0_id, pool1_id)
    builder.add_maxpool("/model.9/m_2/MaxPool", pool1_id, pool2_id)

    concat_id = builder.tensor("model.9.concat", concat_spec)
    builder.add_concat("/model.9/Concat", [cv1_id, pool0_id, pool1_id, pool2_id], concat_id)

    cv2_node = index.nodes_by_name["/model.9/cv2/conv/Conv"]
    cv2_preact_spec = index.qspec(index.conv_output_quant_tensor(cv2_node))
    cv2_preact_id = builder.tensor("model.9.cv2_preact", cv2_preact_spec)
    builder.add_conv(cv2_node.name, concat_id, cv2_preact_id, "none")
    model9_id = builder.tensor("model.9.output", index.qspec(MODEL9_OUTPUT))
    builder.add_add_silu("/model.9/Add", model8_id, cv2_preact_id, model9_id)
    builder.finalize()
    if len(builder.ops) - 1 != NEXT_REGION_LAST_OPERATION:
        raise ValueError("unexpected model9 operation boundary")
    return builder


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
        "model9_output_tensor_id": tensor_id(builder, "model.9.output"),
        "next_region_first_operation": NEXT_REGION_FIRST_OPERATION,
        "next_region_input_tensor_id": tensor_id(builder, "model.8.output"),
        "next_region_last_operation": NEXT_REGION_LAST_OPERATION,
        "next_region_output_tensor_id": tensor_id(builder, "model.9.output"),
        "operation_count": len(operations),
        "profile_id": PROFILE_ID,
        "schema_version": SCHEMA_VERSION,
        "source_lineage_id": (
            f"accepted-yolo26-qdq:{sha256_file(model_path)}:stage51-model4final-model9"
        ),
        "tensor_count": len(tensors),
        "model_sha256": sha256_file(model_path),
    }
    (output / "package.json").write_text(
        json.dumps(package_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files = [
        path for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "asset_hashes.tsv"
    ]
    hash_rows = [
        {"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
         "sha256": sha256_file(path)}
        for path in files
    ]
    write_tsv(output / "asset_hashes.tsv", hash_rows, ["path", "bytes", "sha256"])
    print(json.dumps({
        "contract_id": CONTRACT_ID,
        "direct_concat_segments": sum(int(row["direct_placement"]) for row in placement_rows),
        "fixtures": len(fixture_rows),
        "manifest_sha256": sha256_file(output / "asset_hashes.tsv"),
        "model_sha256": sha256_file(model_path),
        "operations": len(operations),
        "package": str(output),
        "tensors": len(tensors),
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
