#!/usr/bin/env python3
"""Add only missing Conv kernel_shape attributes from static weight shapes."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import onnx
from onnx import helper, numpy_helper


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    options = parser.parse_args()

    model = onnx.load(options.input, load_external_data=True)
    initializers = {
        item.name: numpy_helper.to_array(item) for item in model.graph.initializer
    }
    producers = {
        name: node for node in model.graph.node for name in node.output
    }
    rows: list[list[object]] = []
    for index, node in enumerate(model.graph.node):
        if node.op_type != "Conv":
            continue
        if any(item.name == "kernel_shape" for item in node.attribute):
            rows.append([index, node.name, "preserved", ""])
            continue
        weight_name = node.input[1]
        producer = producers.get(weight_name)
        if producer and producer.op_type == "DequantizeLinear":
            weight_name = producer.input[0]
        weight = initializers.get(weight_name)
        if weight is None or weight.ndim < 4:
            raise RuntimeError(
                f"cannot derive kernel_shape for {node.name or index}: {weight_name}"
            )
        kernel = list(map(int, weight.shape[-2:]))
        node.attribute.append(helper.make_attribute("kernel_shape", kernel))
        rows.append([index, node.name, "added", ",".join(map(str, kernel))])

    options.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, options.output)
    options.report.parent.mkdir(parents=True, exist_ok=True)
    with options.report.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(["node_index", "node_name", "action", "kernel_shape"])
        writer.writerows(rows)
        writer.writerow(
            [
                "identity",
                "",
                "input_sha256",
                sha256(options.input),
            ]
        )
        writer.writerow(
            [
                "identity",
                "",
                "output_sha256",
                sha256(options.output),
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
