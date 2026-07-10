#!/usr/bin/env python3
"""Build Stage41 suffix inventory and cumulative cut manifest."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import onnx
import onnx.shape_inference
import onnx.utils

from stage40_skeleton_common import MODEL4_CUT_OUTPUT, sha256_file


def top_model_index(name: str) -> int | None:
    match = re.match(r"/model\.([0-9]+)(?:/|$)", name)
    return int(match.group(1)) if match else None


def tensor_shape_map(model: onnx.ModelProto) -> dict[str, str]:
    values = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    shapes: dict[str, str] = {}
    for value in values:
        tensor = value.type.tensor_type
        if not tensor.HasField("shape"):
            continue
        dims = []
        for dim in tensor.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(str(dim.dim_value))
            elif dim.HasField("dim_param"):
                dims.append(dim.dim_param)
            else:
                dims.append("?")
        shapes[value.name] = "x".join(dims)
    return shapes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--inventory-tsv", required=True)
    parser.add_argument("--profile-cuts-tsv", required=True)
    parser.add_argument("--contracts-tsv", required=True)
    parser.add_argument("--operator-mix-md", required=True)
    parser.add_argument("--report-md", required=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    out_dir = Path(args.out_dir)
    cut_dir = out_dir / "suffix_cuts"
    cut_dir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(model_path)
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:
        inferred = model
    shapes = tensor_shape_map(inferred)
    final_output = model.graph.output[0].name

    blocks: dict[int, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        idx = top_model_index(node.name)
        if idx is not None and idx >= 5:
            blocks.setdefault(idx, []).append(node)

    inventory_rows: list[dict[str, str]] = []
    cut_rows: list[dict[str, str]] = []
    contract_rows: list[dict[str, str]] = []
    operator_lines = ["# Stage41 Suffix Operator Mix", ""]

    for idx in sorted(blocks):
        nodes = blocks[idx]
        block_id = f"model.{idx}"
        ops: dict[str, int] = {}
        for node in nodes:
            ops[node.op_type] = ops.get(node.op_type, 0) + 1
        first_node = nodes[0]
        last_node = nodes[-1]
        output_name = last_node.output[-1]
        if idx == max(blocks):
            output_name = final_output
        input_names = [name for node in nodes for name in node.input if name]
        output_names = [name for node in nodes for name in node.output if name]
        conv_count = ops.get("Conv", 0)
        q_count = ops.get("QuantizeLinear", 0)
        dq_count = ops.get("DequantizeLinear", 0)
        cut_path = cut_dir / f"model4_to_model_{idx}.onnx"
        cut_status = "not_attempted"
        cut_sha = ""
        cut_nodes = ""
        try:
            onnx.utils.extract_model(str(model_path), str(cut_path), [MODEL4_CUT_OUTPUT], [output_name])
            cut_model = onnx.load(cut_path)
            cut_status = "pass"
            cut_sha = sha256_file(cut_path)
            cut_nodes = str(len(cut_model.graph.node))
            cut_rows.append(
                {
                    "block_id": block_id,
                    "cut_path": str(cut_path),
                    "input_name": MODEL4_CUT_OUTPUT,
                    "output_name": output_name,
                    "cut_nodes": cut_nodes,
                    "cut_sha256": cut_sha,
                }
            )
        except Exception as exc:  # noqa: BLE001 - report exact extraction blocker.
            cut_status = f"blocked:{type(exc).__name__}:{str(exc)[:160]}"
            cut_path.unlink(missing_ok=True)

        inventory_rows.append(
            {
                "block_id": block_id,
                "first_node": first_node.name,
                "last_node": last_node.name,
                "node_count": str(len(nodes)),
                "operator_mix": ",".join(f"{k}:{ops[k]}" for k in sorted(ops)),
                "conv_count": str(conv_count),
                "matmul_count": str(ops.get("MatMul", 0)),
                "quantize_count": str(q_count),
                "dequantize_count": str(dq_count),
                "start_tensor_candidates": ";".join(dict.fromkeys(input_names[:8])),
                "end_tensor": output_name,
                "end_shape": shapes.get(output_name, ""),
                "cumulative_cut_status": cut_status,
                "cumulative_cut_nodes": cut_nodes,
                "cumulative_cut_sha256": cut_sha,
                "custom_kernel_reuse": "model4_c2f_reuse_candidate" if conv_count >= 2 else "conv_or_postprocess_specific",
                "risk_score": "high" if idx >= 22 else ("medium" if conv_count >= 4 else "low"),
            }
        )
        contract_rows.append(
            {
                "block_id": block_id,
                "input_boundary": MODEL4_CUT_OUTPUT,
                "input_dtype": "uint8",
                "input_shape": shapes.get(MODEL4_CUT_OUTPUT, "1x128x80x80"),
                "output_boundary": output_name,
                "output_shape": shapes.get(output_name, ""),
                "output_is_quantized": "1" if output_name.endswith("_QuantizeLinear_Output") else "0",
                "cut_status": cut_status,
            }
        )
        operator_lines.extend(
            [
                f"## {block_id}",
                "",
                f"- nodes: `{len(nodes)}`",
                f"- operators: `{', '.join(f'{k}:{ops[k]}' for k in sorted(ops))}`",
                f"- cumulative_cut_status: `{cut_status}`",
                "",
            ]
        )

    for path, rows in [
        (Path(args.inventory_tsv), inventory_rows),
        (Path(args.profile_cuts_tsv), cut_rows),
        (Path(args.contracts_tsv), contract_rows),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(rows[0].keys()),
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
    Path(args.operator_mix_md).write_text(
        "\n".join(operator_lines).rstrip() + "\n",
        encoding="utf-8",
    )
    Path(args.report_md).write_text(
        "# Stage41 Suffix Inventory Report\n\n"
        f"- model: `{model_path}`\n"
        f"- model_sha256: `{sha256_file(model_path)}`\n"
        f"- suffix_blocks: `{len(inventory_rows)}`\n"
        f"- cumulative_cuts_extracted: `{len(cut_rows)}`\n"
        f"- input_boundary: `{MODEL4_CUT_OUTPUT}`\n"
        "- cumulative cut timings must be interpreted as suffix-prefix timings from the same input boundary, not isolated block runtimes.\n",
        encoding="utf-8",
    )
    print(f"suffix_blocks={len(inventory_rows)} cumulative_cuts={len(cut_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
