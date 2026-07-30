#!/usr/bin/env python3
"""Extract Stage64 inference and CPU-tail graphs at the six vendor boundaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import onnx


SPLIT_NAMES = [
    "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0",
    "/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0",
    "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0",
    "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0",
    "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0",
    "/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    options = parser.parse_args()

    model = onnx.load(options.model, load_external_data=True)
    known_names = {
        name
        for node in model.graph.node
        for name in [*node.input, *node.output]
    }
    missing = [name for name in SPLIT_NAMES if name not in known_names]
    if missing:
        raise RuntimeError(f"missing split tensors: {missing}")
    input_names = [item.name for item in model.graph.input]
    output_names = [item.name for item in model.graph.output]
    if input_names != ["images"] or output_names != ["output0"]:
        raise RuntimeError(
            f"unexpected model contract inputs={input_names} outputs={output_names}"
        )

    options.output_dir.mkdir(parents=True, exist_ok=True)
    inference = options.output_dir / f"{options.prefix}.inference.onnx"
    tail = options.output_dir / f"{options.prefix}.postprocess.onnx"
    onnx.utils.extract_model(
        str(options.model), str(inference), input_names, SPLIT_NAMES, check_model=True
    )
    onnx.utils.extract_model(
        str(options.model), str(tail), SPLIT_NAMES, output_names, check_model=True
    )

    rows = []
    for role, path in (
        ("source-merged", options.model),
        ("quantized-inference", inference),
        ("float-postprocess-tail", tail),
    ):
        item = onnx.load(path, load_external_data=False)
        rows.append(
            {
                "role": role,
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "node_count": len(item.graph.node),
                "input_names": ",".join(value.name for value in item.graph.input),
                "output_names": ",".join(value.name for value in item.graph.output),
                "qdq_count": sum(
                    node.op_type in {"QuantizeLinear", "DequantizeLinear"}
                    for node in item.graph.node
                ),
                "qlinear_count": sum(
                    node.op_type.startswith("QLinear") for node in item.graph.node
                ),
            }
        )
    options.manifest.parent.mkdir(parents=True, exist_ok=True)
    with options.manifest.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            delimiter="\t",
            lineterminator="\n",
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
