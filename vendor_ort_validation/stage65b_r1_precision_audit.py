#!/usr/bin/env python3
"""Inventory Q/DQ coverage without inferring execution-provider placement."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import onnx
from onnx import numpy_helper


def rows(lane: str, path: Path) -> list[dict[str, Any]]:
    model = onnx.load(path, load_external_data=True)
    producers = {
        output: node for node in model.graph.node for output in node.output
    }
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for value in node.input:
            consumers[value].append(node)
    classified: dict[str, Counter[str]] = defaultdict(Counter)
    qdq_count = 0
    qlinear_count = 0
    for node in model.graph.node:
        if node.op_type in {"QuantizeLinear", "DequantizeLinear"}:
            qdq_count += 1
            continue
        if node.op_type.startswith("QLinear"):
            qlinear_count += 1
        input_dq = sum(
            producers.get(value) is not None
            and producers[value].op_type == "DequantizeLinear"
            for value in node.input
        )
        output_q = sum(
            consumer.op_type == "QuantizeLinear"
            for value in node.output
            for consumer in consumers.get(value, [])
        )
        if input_dq and output_q:
            classification = "qdq-input-and-output"
        elif input_dq:
            classification = "dequantized-input-only"
        elif output_q:
            classification = "quantized-output-only"
        else:
            classification = "no-adjacent-qdq"
        classified[node.op_type][classification] += 1

    initializer_dtypes = Counter(
        str(numpy_helper.to_array(item).dtype)
        for item in model.graph.initializer
    )
    summary = {
        "lane": lane,
        "model": path.name,
        "operator_type": "__summary__",
        "total": sum(sum(counts.values()) for counts in classified.values()),
        "qdq_input_and_output": sum(
            counts["qdq-input-and-output"] for counts in classified.values()
        ),
        "dequantized_input_only": sum(
            counts["dequantized-input-only"] for counts in classified.values()
        ),
        "quantized_output_only": sum(
            counts["quantized-output-only"] for counts in classified.values()
        ),
        "no_adjacent_qdq": sum(
            counts["no-adjacent-qdq"] for counts in classified.values()
        ),
        "qdq_node_count": qdq_count,
        "qlinear_count": qlinear_count,
        "float32_initializers": initializer_dtypes["float32"],
        "float16_initializers": initializer_dtypes["float16"],
        "int8_initializers": initializer_dtypes["int8"],
        "placement_claim": "none-host-graph-inventory-only",
    }
    result = [summary]
    for op_type in sorted(classified):
        counts = classified[op_type]
        result.append(
            {
                **summary,
                "operator_type": op_type,
                "total": sum(counts.values()),
                "qdq_input_and_output": counts["qdq-input-and-output"],
                "dequantized_input_only": counts["dequantized-input-only"],
                "quantized_output_only": counts["quantized-output-only"],
                "no_adjacent_qdq": counts["no-adjacent-qdq"],
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", action="append", required=True, help="LANE=/path/model.onnx"
    )
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    result: list[dict[str, Any]] = []
    for value in options.model:
        lane, separator, raw_path = value.partition("=")
        if not separator or not lane or not raw_path:
            raise ValueError(f"invalid --model value: {value}")
        result.extend(rows(lane, Path(raw_path)))
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(result[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
