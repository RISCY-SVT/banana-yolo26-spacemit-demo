#!/usr/bin/env python3
"""Generate deterministic inputs and CPU-oracle outputs for tiny Stage64 models."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shape(value: onnx.ValueInfoProto) -> list[int]:
    return [
        int(item.dim_value) if item.HasField("dim_value") and item.dim_value > 0 else 1
        for item in value.type.tensor_type.shape.dim
    ]


def dtype(value: onnx.ValueInfoProto) -> np.dtype[Any]:
    return np.dtype(
        onnx.helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    options = parser.parse_args()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for model_path in sorted(options.model_dir.glob("*.onnx")):
        model = onnx.load(model_path)
        if len(model.graph.input) != 1 or len(model.graph.output) != 1:
            continue
        input_info = model.graph.input[0]
        input_shape = shape(input_info)
        input_dtype = dtype(input_info)
        count = int(np.prod(input_shape))
        if np.issubdtype(input_dtype, np.floating):
            values = np.linspace(-1.0, 1.0, count, dtype=np.float32).astype(
                input_dtype
            )
        else:
            values = np.arange(count, dtype=np.int64).astype(input_dtype)
        input_array = np.ascontiguousarray(values.reshape(input_shape))
        input_path = options.output_dir / f"{model_path.stem}.input.bin"
        input_array.tofile(input_path)

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        outputs = session.run(None, {input_info.name: input_array})
        if len(outputs) != 1:
            raise RuntimeError(f"unexpected output count for {model_path}")
        output_array = np.ascontiguousarray(outputs[0])
        oracle_path = options.output_dir / f"{model_path.stem}.oracle.bin"
        output_array.tofile(oracle_path)
        rows.append(
            {
                "test_id": model_path.stem,
                "model": str(model_path.resolve()),
                "model_sha256": sha256(model_path),
                "input_name": input_info.name,
                "input_shape": "x".join(map(str, input_array.shape)),
                "input_dtype": str(input_array.dtype),
                "input_path": str(input_path.resolve()),
                "input_sha256": sha256(input_path),
                "output_name": model.graph.output[0].name,
                "output_shape": "x".join(map(str, output_array.shape)),
                "output_dtype": str(output_array.dtype),
                "oracle_path": str(oracle_path.resolve()),
                "oracle_sha256": sha256(oracle_path),
                "oracle_min": float(output_array.min()),
                "oracle_max": float(output_array.max()),
            }
        )
    with (options.output_dir / "tiny_oracle_manifest.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
