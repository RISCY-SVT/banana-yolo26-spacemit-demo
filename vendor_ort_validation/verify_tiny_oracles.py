#!/usr/bin/env python3
"""Compute independent reference outputs for the Stage63 tiny controls."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def initializers(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}


def quantize_u8(values: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return np.clip(np.rint(values.astype(np.float64) / scale) + zero_point, 0, 255).astype(np.uint8)


def qdq_conv(model_path: Path, input_path: Path) -> np.ndarray:
    model = onnx.load(model_path)
    values = initializers(model)
    input_shape = tuple(d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim)
    x = np.fromfile(input_path, dtype=np.float32).reshape(input_shape)
    x_scale = float(values["input_scale"].reshape(-1)[0])
    x_zero = int(values["input_zero_point"].reshape(-1)[0])
    w_scale = float(values["weight_scale"].reshape(-1)[0])
    w_zero = int(values["weight_zero_point"].reshape(-1)[0])
    y_scale = float(values["output_scale"].reshape(-1)[0])
    y_zero = int(values["output_zero_point"].reshape(-1)[0])
    weights = values["conv_w_quantized"].astype(np.int64) - w_zero
    bias = values["conv_b"].astype(np.float64)
    xq = quantize_u8(x, x_scale, x_zero).astype(np.int64) - x_zero

    conv_node = next(node for node in model.graph.node if node.op_type == "Conv")
    attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in conv_node.attribute}
    pads = attrs.get("pads", [0, 0, 0, 0])
    strides = attrs.get("strides", [1, 1])
    output_shape = tuple(d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim)
    output = np.empty(output_shape, dtype=np.float32)

    for n in range(output_shape[0]):
        for oc in range(output_shape[1]):
            for oy in range(output_shape[2]):
                for ox in range(output_shape[3]):
                    accumulator = 0
                    for ic in range(weights.shape[1]):
                        for ky in range(weights.shape[2]):
                            iy = oy * strides[0] + ky - pads[0]
                            if iy < 0 or iy >= xq.shape[2]:
                                continue
                            for kx in range(weights.shape[3]):
                                ix = ox * strides[1] + kx - pads[1]
                                if 0 <= ix < xq.shape[3]:
                                    accumulator += int(xq[n, ic, iy, ix]) * int(weights[oc, ic, ky, kx])
                    real = float(accumulator) * x_scale * w_scale + float(bias[oc])
                    quantized = int(np.clip(np.rint(real / y_scale) + y_zero, 0, 255))
                    output[n, oc, oy, ox] = np.float32(
                        (np.float32(quantized) - np.float32(y_zero)) * np.float32(y_scale)
                    )
    return output


def qlinear_conv(model_path: Path, input_path: Path) -> np.ndarray:
    model = onnx.load(model_path)
    values = initializers(model)
    input_shape = tuple(d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim)
    x = np.fromfile(input_path, dtype=np.float32).reshape(input_shape)
    x_scale = float(values["input_scale"].reshape(-1)[0])
    x_zero = int(values["input_zero_point"].reshape(-1)[0])
    w_scale = float(values["weight_scale"].reshape(-1)[0])
    w_zero = int(values["weight_zero_point"].reshape(-1)[0])
    y_scale = float(values["output_scale"].reshape(-1)[0])
    y_zero = int(values["output_zero_point"].reshape(-1)[0])
    weights = values["conv_w_quantized"].astype(np.int64) - w_zero
    bias = values["conv_b_qlinear"].astype(np.int64)
    xq = quantize_u8(x, x_scale, x_zero).astype(np.int64) - x_zero
    output_shape = tuple(d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim)
    output = np.empty(output_shape, dtype=np.float32)
    multiplier = x_scale * w_scale / y_scale

    for n in range(output_shape[0]):
        for oc in range(output_shape[1]):
            for oy in range(output_shape[2]):
                for ox in range(output_shape[3]):
                    accumulator = int(bias[oc])
                    for ic in range(weights.shape[1]):
                        for ky in range(weights.shape[2]):
                            iy = oy + ky - 1
                            if iy < 0 or iy >= xq.shape[2]:
                                continue
                            for kx in range(weights.shape[3]):
                                ix = ox + kx - 1
                                if 0 <= ix < xq.shape[3]:
                                    accumulator += int(xq[n, ic, iy, ix]) * int(weights[oc, ic, ky, kx])
                    quantized = int(np.clip(np.rint(float(accumulator) * multiplier) + y_zero, 0, 255))
                    output[n, oc, oy, ox] = np.float32(
                        (np.float32(quantized) - np.float32(y_zero)) * np.float32(y_scale)
                    )
    return output


def qlinear_matmul(model_path: Path, input_path: Path) -> np.ndarray:
    model = onnx.load(model_path)
    values = initializers(model)
    input_shape = tuple(d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim)
    x = np.fromfile(input_path, dtype=np.float32).reshape(input_shape)
    a_scale = float(values["a_scale"].reshape(-1)[0])
    a_zero = int(values["a_zero_point"].reshape(-1)[0])
    b_scale = float(values["b_scale"].reshape(-1)[0])
    b_zero = int(values["b_zero_point"].reshape(-1)[0])
    c_scale = float(values["c_scale"].reshape(-1)[0])
    c_zero = int(values["c_zero_point"].reshape(-1)[0])
    aq = quantize_u8(x, a_scale, a_zero).astype(np.int64) - a_zero
    bq = values["matmul_b_quantized"].astype(np.int64) - b_zero
    accumulators = aq @ bq
    quantized = np.clip(np.rint(accumulators.astype(np.float64) * a_scale * b_scale / c_scale) + c_zero, 0, 255)
    return ((quantized.astype(np.float32) - np.float32(c_zero)) * np.float32(c_scale)).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repro_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    controls = [
        ("A0", "03_conv_qdq.onnx", "input_1x3x8x8_f32.bin", qdq_conv),
        ("A1", "15_conv_qdq_attr_kernel_shape.onnx", "input_1x3x8x8_f32.bin", qdq_conv),
        ("B", "08_qlinearconv.onnx", "input_1x3x8x8_f32.bin", qlinear_conv),
        ("C", "10_qlinearmatmul_smoke.onnx", "input_1x4_f32.bin", qlinear_matmul),
    ]
    rows = ["test_id\tmodel_sha256\tinput_sha256\toracle_sha256\tbytes"]
    for test_id, model_name, input_name, oracle in controls:
        model_path = args.repro_dir / model_name
        input_path = args.repro_dir / input_name
        output = oracle(model_path, input_path)
        output_path = args.output_dir / f"{test_id}_independent_oracle.bin"
        output.tofile(output_path)
        rows.append(
            "\t".join(
                [
                    test_id,
                    sha256(model_path.read_bytes()),
                    sha256(input_path.read_bytes()),
                    sha256(output.tobytes()),
                    str(output.nbytes),
                ]
            )
        )
    args.summary.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
