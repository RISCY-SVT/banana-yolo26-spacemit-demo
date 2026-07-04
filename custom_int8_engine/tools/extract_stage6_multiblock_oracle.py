#!/usr/bin/env python3
"""Extract Stage 6 bounded multi-block oracle fixtures.

This is an offline oracle/converter helper. The runtime library must not depend
on Python, ONNX, protobuf, ONNX Runtime, or dynamic graph execution.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, shape_inference


SUBSET_ID = "candidate_C_block0_silu_model1_conv"
CONV0_NODE = "/model.0/conv/Conv"
CONV1_NODE = "/model.1/conv/Conv"


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def c_array(name: str, values: np.ndarray, ctype: str, cols: int = 16) -> str:
    flat = values.reshape(-1).tolist()
    lines = [f"inline constexpr {ctype} {name}[] = {{"]
    for offset in range(0, len(flat), cols):
        row_values = flat[offset : offset + cols]
        if ctype == "float":
            row = ", ".join(f"{float(v):.12g}f" for v in row_values)
        else:
            row = ", ".join(str(int(v)) for v in row_values)
        suffix = "," if offset + cols < len(flat) else ""
        lines.append(f"    {row}{suffix}")
    lines.append("};")
    return "\n".join(lines)


def value_info_map(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    values: dict[str, onnx.ValueInfoProto] = {}
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        values[value.name] = value
    return values


def producer_map(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    producers: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for output in node.output:
            producers[output] = node
    return producers


def consumer_map(model: onnx.ModelProto) -> dict[str, list[onnx.NodeProto]]:
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)
    return consumers


def attrs(node: onnx.NodeProto) -> dict[str, Any]:
    return {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}


def tensor_shape(model: onnx.ModelProto, name: str) -> list[int | str]:
    values = value_info_map(model)
    if name not in values:
        return []
    tt = values[name].type.tensor_type
    if not tt.HasField("shape"):
        return []
    result: list[int | str] = []
    for dim in tt.shape.dim:
        if dim.HasField("dim_value"):
            result.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            result.append(dim.dim_param)
        else:
            result.append("?")
    return result


def selected_outputs_model(model: onnx.ModelProto, outputs: list[str]) -> onnx.ModelProto:
    inferred = shape_inference.infer_shapes(model)
    values = value_info_map(inferred)
    clone = copy.deepcopy(inferred)
    existing = {output.name for output in clone.graph.output}
    for name in outputs:
        if name not in values:
            raise KeyError(f"missing value_info for output {name}")
        if name not in existing:
            clone.graph.output.append(copy.deepcopy(values[name]))
            existing.add(name)
    return clone


def synthetic_seeded_input() -> np.ndarray:
    rng = np.random.default_rng(20260704)
    return rng.random((1, 3, 640, 640), dtype=np.float32)


def synthetic_gradient_input() -> np.ndarray:
    x = np.linspace(0.0, 1.0, 640, dtype=np.float32)
    y = np.linspace(0.0, 1.0, 640, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    img = np.stack([xx, yy, 0.25 * xx + 0.75 * yy], axis=0)
    return img.reshape(1, 3, 640, 640).astype(np.float32)


def weight_ohwi(weight_q: np.ndarray) -> np.ndarray:
    return np.transpose(weight_q, (0, 2, 3, 1)).astype(np.int8)


def nchw_q_to_nhwc_s8(q: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    cropped = q[0, :, 0:crop_h, 0:crop_w]
    nhwc_i16 = np.transpose(cropped, (1, 2, 0)).astype(np.int16) - 128
    return nhwc_i16.astype(np.int8)


def nhwc_s8_to_q_u8_nhwc(values: np.ndarray) -> np.ndarray:
    return (values.astype(np.int16) + 128).astype(np.uint8)


def quantize_u8_nearest_even(values: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    q = np.rint(values.astype(np.float64) / float(scale)) + int(zero_point)
    q = np.clip(q, 0, 255)
    return q.astype(np.uint8)


def dequantize_u8(values: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (values.astype(np.int32) - int(zero_point)).astype(np.float32) * float(scale)


def silu(values: np.ndarray) -> np.ndarray:
    values64 = values.astype(np.float64)
    return (values64 / (1.0 + np.exp(-values64))).astype(np.float32)


def conv_i32_nhwc(
    input_q_u8_nhwc: np.ndarray,
    weight_q_oihw: np.ndarray,
    bias_i32: np.ndarray,
    activation_zero_point: int,
    weight_zero_point: np.ndarray,
    node_attrs: dict[str, Any],
) -> np.ndarray:
    kernel_h, kernel_w = node_attrs["kernel_shape"]
    stride_h, stride_w = node_attrs["strides"]
    pad_top, pad_left = node_attrs["pads"][0], node_attrs["pads"][1]
    input_h, input_w, input_c = input_q_u8_nhwc.shape
    output_c = weight_q_oihw.shape[0]
    output_h = (input_h + 2 * pad_top - kernel_h) // stride_h + 1
    output_w = (input_w + 2 * pad_left - kernel_w) // stride_w + 1
    out = np.zeros((output_h, output_w, output_c), dtype=np.int32)
    for oh in range(output_h):
        for ow in range(output_w):
            for oc in range(output_c):
                acc = int(bias_i32[oc])
                wzp = int(weight_zero_point[oc] if weight_zero_point.ndim else weight_zero_point.item())
                for kh in range(kernel_h):
                    ih = oh * stride_h + kh - pad_top
                    for kw in range(kernel_w):
                        iw = ow * stride_w + kw - pad_left
                        inside = 0 <= ih < input_h and 0 <= iw < input_w
                        for ic in range(input_c):
                            aq = int(input_q_u8_nhwc[ih, iw, ic]) if inside else int(activation_zero_point)
                            acc += (aq - int(activation_zero_point)) * (int(weight_q_oihw[oc, ic, kh, kw]) - wzp)
                out[oh, ow, oc] = acc
    return out


def write_binary(path: Path, arr: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(arr)
    path.write_bytes(contiguous.tobytes())
    return {
        "path": str(path.resolve()),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": sha256_array(arr),
        "bytes": path.stat().st_size,
        "min": float(np.min(arr)) if arr.size else 0.0,
        "max": float(np.max(arr)) if arr.size else 0.0,
    }


def init_summary(name: str, initializers: dict[str, np.ndarray]) -> dict[str, Any]:
    arr = initializers[name]
    flat = arr.reshape(-1) if arr.shape else np.array([arr.item()])
    return {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "first": flat[:8].tolist(),
        "sha256": sha256_array(arr),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dump-dir", required=True, type=Path)
    parser.add_argument("--fixture-header-out", required=True, type=Path)
    parser.add_argument("--metadata-out", required=True, type=Path)
    args = parser.parse_args()

    args.dump_dir.mkdir(parents=True, exist_ok=True)
    args.fixture_header_out.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(args.model), load_external_data=True)
    inferred = shape_inference.infer_shapes(model)
    producers = producer_map(inferred)
    consumers = consumer_map(inferred)
    initializers = {tensor.name: numpy_helper.to_array(tensor) for tensor in inferred.graph.initializer}
    nodes = {node.name: node for node in inferred.graph.node}

    conv0 = nodes[CONV0_NODE]
    conv1 = nodes[CONV1_NODE]
    conv0_attrs = attrs(conv0)
    conv1_attrs = attrs(conv1)
    input_q = nodes["images_QuantizeLinear"]
    input_dq = consumers[input_q.output[0]][0]
    conv0_q = consumers[conv0.output[0]][0]
    conv0_dq = consumers[conv0_q.output[0]][0]
    sigmoid = consumers[conv0_dq.output[0]][0]
    mul = consumers[sigmoid.output[0]][0]
    act_q = consumers[mul.output[0]][0]
    act_dq = consumers[act_q.output[0]][0]
    conv1_q = consumers[conv1.output[0]][0]
    conv1_dq = consumers[conv1_q.output[0]][0]
    w0_dq = producers[conv0.input[1]]
    b0_dq = producers[conv0.input[2]]
    w1_dq = producers[conv1.input[1]]
    b1_dq = producers[conv1.input[2]]

    outputs = [
        input_q.output[0],
        input_dq.output[0],
        conv0.output[0],
        conv0_q.output[0],
        conv0_dq.output[0],
        sigmoid.output[0],
        mul.output[0],
        act_q.output[0],
        act_dq.output[0],
        conv1.output[0],
        conv1_q.output[0],
        conv1_dq.output[0],
    ]
    oracle_model = selected_outputs_model(model, outputs)
    oracle_model_path = args.dump_dir / "stage6_multiblock_outputs.onnx"
    onnx.save(oracle_model, str(oracle_model_path))

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(oracle_model_path), sess_options=session_options, providers=["CPUExecutionProvider"])

    input_cases = [
        ("synthetic_seeded", synthetic_seeded_input(), "seed=20260704 random float32 [0,1)"),
        ("synthetic_gradient", synthetic_gradient_input(), "deterministic H/W gradient float32 [0,1]"),
    ]

    weight0_q = initializers[w0_dq.input[0]]
    weight1_q = initializers[w1_dq.input[0]]
    weight0_scale = initializers[w0_dq.input[1]].astype(np.float32)
    weight1_scale = initializers[w1_dq.input[1]].astype(np.float32)
    weight0_zero_point = initializers[w0_dq.input[2]]
    weight1_zero_point = initializers[w1_dq.input[2]]
    bias0_q = initializers[b0_dq.input[0]].astype(np.int32)
    bias1_q = initializers[b1_dq.input[0]].astype(np.int32)
    images_scale = float(initializers[input_q.input[1]].item())
    images_zero_point = int(initializers[input_q.input[2]].item())
    conv0_output_scale = float(initializers[conv0_q.input[1]].item())
    conv0_output_zero_point = int(initializers[conv0_q.input[2]].item())
    act0_output_scale = float(initializers[act_q.input[1]].item())
    act0_output_zero_point = int(initializers[act_q.input[2]].item())
    conv1_output_scale = float(initializers[conv1_q.input[1]].item())
    conv1_output_zero_point = int(initializers[conv1_q.input[2]].item())
    weights0 = weight_ohwi(weight0_q)
    weights1 = weight_ohwi(weight1_q)

    input_crop_h = 8
    input_crop_w = 8

    header_parts = [
        "#pragma once",
        "",
        "#include \"y26_k1x_conv_kernels.h\"",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace y26_stage6_multiblock_fixture {",
        "",
        "struct MultiblockFixture {",
        "    const char* label;",
        "    const char* subset_id;",
        "    const char* conv0_node_name;",
        "    const char* conv1_node_name;",
        "    Y26Conv2DParams conv0_params;",
        "    Y26Conv2DParams conv1_params;",
        "    int conv0_kernel_h;",
        "    int conv0_kernel_w;",
        "    int conv1_kernel_h;",
        "    int conv1_kernel_w;",
        "    int conv0_activation_zero_point_u8;",
        "    int conv0_input_storage_zero_point_s8;",
        "    int conv0_output_zero_point_u8;",
        "    int act0_output_zero_point_u8;",
        "    int conv1_input_storage_zero_point_s8;",
        "    int conv1_output_zero_point_u8;",
        "    float images_scale;",
        "    float conv0_output_scale;",
        "    float act0_output_scale;",
        "    float conv1_output_scale;",
        "    const float* conv0_weight_scales;",
        "    std::size_t conv0_weight_scale_count;",
        "    const float* conv1_weight_scales;",
        "    std::size_t conv1_weight_scale_count;",
        "    const std::int8_t* input_nhwc_s8;",
        "    std::size_t input_count;",
        "    const std::int8_t* conv0_weights_ohwi_s8;",
        "    std::size_t conv0_weight_count;",
        "    const std::int32_t* conv0_bias_i32;",
        "    std::size_t conv0_bias_count;",
        "    const std::int8_t* conv1_weights_ohwi_s8;",
        "    std::size_t conv1_weight_count;",
        "    const std::int32_t* conv1_bias_i32;",
        "    std::size_t conv1_bias_count;",
        "    const std::int32_t* expected_conv0_i32_nhwc;",
        "    std::size_t expected_conv0_count;",
        "    const std::int8_t* expected_act0_s8_nhwc;",
        "    std::size_t expected_act0_count;",
        "    const std::int32_t* expected_conv1_i32_nhwc;",
        "    std::size_t expected_conv1_count;",
        "};",
        "",
        c_array("kConv0WeightsOhwiS8", weights0, "std::int8_t"),
        "",
        c_array("kConv1WeightsOhwiS8", weights1, "std::int8_t"),
        "",
        c_array("kConv0BiasI32", bias0_q, "std::int32_t", cols=8),
        "",
        c_array("kConv1BiasI32", bias1_q, "std::int32_t", cols=8),
        "",
        c_array("kConv0WeightScales", weight0_scale, "float", cols=8),
        "",
        c_array("kConv1WeightScales", weight1_scale, "float", cols=8),
        "",
    ]

    report: dict[str, Any] = {
        "subset_id": SUBSET_ID,
        "model": str(args.model.resolve()),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "oracle_model": str(oracle_model_path.resolve()),
        "python": {
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
        },
        "xslim_used": False,
        "nodes": [
            {"name": node.name, "op_type": node.op_type, "inputs": list(node.input), "outputs": list(node.output)}
            for node in [input_q, input_dq, conv0, conv0_q, conv0_dq, sigmoid, mul, act_q, act_dq, conv1, conv1_q, conv1_dq]
        ],
        "shapes": {
            name: tensor_shape(inferred, name)
            for name in outputs
        },
        "quantization": {
            "images_scale": images_scale,
            "images_zero_point_u8": images_zero_point,
            "images_storage_zero_point_s8": images_zero_point - 128,
            "conv0_output_scale": conv0_output_scale,
            "conv0_output_zero_point_u8": conv0_output_zero_point,
            "act0_output_scale": act0_output_scale,
            "act0_output_zero_point_u8": act0_output_zero_point,
            "conv1_input_storage_zero_point_s8": act0_output_zero_point - 128,
            "conv1_output_scale": conv1_output_scale,
            "conv1_output_zero_point_u8": conv1_output_zero_point,
            "conv0_weight": init_summary(w0_dq.input[0], initializers),
            "conv1_weight": init_summary(w1_dq.input[0], initializers),
            "conv0_weight_scale": init_summary(w0_dq.input[1], initializers),
            "conv1_weight_scale": init_summary(w1_dq.input[1], initializers),
            "conv0_weight_zero_point": init_summary(w0_dq.input[2], initializers),
            "conv1_weight_zero_point": init_summary(w1_dq.input[2], initializers),
        },
        "cases": [],
        "dumps": {},
    }

    fixture_names: list[str] = []
    for case_name, input_tensor, source in input_cases:
        outputs_values = session.run(None, {"images": input_tensor})
        output_map = {output.name: value for output, value in zip(session.get_outputs(), outputs_values)}
        input_q_full = output_map[input_q.output[0]]
        conv0_float_full = output_map[conv0.output[0]]
        conv1_float_full = output_map[conv1.output[0]]
        input_s8 = nchw_q_to_nhwc_s8(input_q_full, input_crop_h, input_crop_w)
        input_q_nhwc = nhwc_s8_to_q_u8_nhwc(input_s8)

        conv0_i32 = conv_i32_nhwc(
            input_q_nhwc,
            weight0_q,
            bias0_q,
            images_zero_point,
            weight0_zero_point,
            conv0_attrs,
        )
        conv0_float = np.empty(conv0_i32.shape, dtype=np.float32)
        for oc in range(weight0_q.shape[0]):
            conv0_float[:, :, oc] = conv0_i32[:, :, oc].astype(np.float32) * (images_scale * float(weight0_scale[oc]))
        conv0_q_local = quantize_u8_nearest_even(conv0_float, conv0_output_scale, conv0_output_zero_point)
        conv0_dq_local = dequantize_u8(conv0_q_local, conv0_output_scale, conv0_output_zero_point)
        silu_local = silu(conv0_dq_local)
        act0_q_local = quantize_u8_nearest_even(silu_local, act0_output_scale, act0_output_zero_point)
        act0_s8 = (act0_q_local.astype(np.int16) - 128).astype(np.int8)
        conv1_i32 = conv_i32_nhwc(
            act0_q_local,
            weight1_q,
            bias1_q,
            act0_output_zero_point,
            weight1_zero_point,
            conv1_attrs,
        )
        conv1_dequant = np.empty(conv1_i32.shape, dtype=np.float32)
        for oc in range(weight1_q.shape[0]):
            conv1_dequant[:, :, oc] = conv1_i32[:, :, oc].astype(np.float32) * (
                act0_output_scale * float(weight1_scale[oc])
            )
        conv0_ort_roi = np.transpose(conv0_float_full[0, :, 0 : conv0_i32.shape[0], 0 : conv0_i32.shape[1]], (1, 2, 0))
        conv1_ort_roi = np.transpose(conv1_float_full[0, :, 0 : conv1_i32.shape[0], 0 : conv1_i32.shape[1]], (1, 2, 0))
        conv0_max_abs_diff = float(np.max(np.abs(conv0_float - conv0_ort_roi)))
        conv1_max_abs_diff = float(np.max(np.abs(conv1_dequant - conv1_ort_roi)))

        safe = case_name.replace("-", "_")
        camel = "".join(part.capitalize() for part in safe.split("_"))
        input_name = f"k{camel}InputS8"
        conv0_expected_name = f"k{camel}ExpectedConv0I32Nhwc"
        act0_name = f"k{camel}ExpectedAct0S8Nhwc"
        conv1_expected_name = f"k{camel}ExpectedConv1I32Nhwc"
        fixture_name = f"k{camel}Fixture"
        fixture_names.append(fixture_name)

        header_parts.append(c_array(input_name, input_s8, "std::int8_t"))
        header_parts.append("")
        header_parts.append(c_array(conv0_expected_name, conv0_i32, "std::int32_t", cols=8))
        header_parts.append("")
        header_parts.append(c_array(act0_name, act0_s8, "std::int8_t"))
        header_parts.append("")
        header_parts.append(c_array(conv1_expected_name, conv1_i32, "std::int32_t", cols=8))
        header_parts.append("")
        header_parts.append(f"inline constexpr MultiblockFixture {fixture_name} = {{")
        header_parts.append(f'    "{case_name}",')
        header_parts.append(f'    "{SUBSET_ID}",')
        header_parts.append(f'    "{CONV0_NODE}",')
        header_parts.append(f'    "{CONV1_NODE}",')
        header_parts.append(f"    Y26Conv2DParams{{{input_crop_h}, {input_crop_w}, 3, 16, 2, 2, 1, 1}},")
        header_parts.append(f"    Y26Conv2DParams{{{conv0_i32.shape[0]}, {conv0_i32.shape[1]}, 16, 32, 2, 2, 1, 1}},")
        header_parts.append("    3,")
        header_parts.append("    3,")
        header_parts.append("    3,")
        header_parts.append("    3,")
        header_parts.append(f"    {images_zero_point},")
        header_parts.append(f"    {images_zero_point - 128},")
        header_parts.append(f"    {conv0_output_zero_point},")
        header_parts.append(f"    {act0_output_zero_point},")
        header_parts.append(f"    {act0_output_zero_point - 128},")
        header_parts.append(f"    {conv1_output_zero_point},")
        header_parts.append(f"    {images_scale:.12g}f,")
        header_parts.append(f"    {conv0_output_scale:.12g}f,")
        header_parts.append(f"    {act0_output_scale:.12g}f,")
        header_parts.append(f"    {conv1_output_scale:.12g}f,")
        header_parts.append("    kConv0WeightScales,")
        header_parts.append("    sizeof(kConv0WeightScales) / sizeof(kConv0WeightScales[0]),")
        header_parts.append("    kConv1WeightScales,")
        header_parts.append("    sizeof(kConv1WeightScales) / sizeof(kConv1WeightScales[0]),")
        header_parts.append(f"    {input_name},")
        header_parts.append(f"    sizeof({input_name}) / sizeof({input_name}[0]),")
        header_parts.append("    kConv0WeightsOhwiS8,")
        header_parts.append("    sizeof(kConv0WeightsOhwiS8) / sizeof(kConv0WeightsOhwiS8[0]),")
        header_parts.append("    kConv0BiasI32,")
        header_parts.append("    sizeof(kConv0BiasI32) / sizeof(kConv0BiasI32[0]),")
        header_parts.append("    kConv1WeightsOhwiS8,")
        header_parts.append("    sizeof(kConv1WeightsOhwiS8) / sizeof(kConv1WeightsOhwiS8[0]),")
        header_parts.append("    kConv1BiasI32,")
        header_parts.append("    sizeof(kConv1BiasI32) / sizeof(kConv1BiasI32[0]),")
        header_parts.append(f"    {conv0_expected_name},")
        header_parts.append(f"    sizeof({conv0_expected_name}) / sizeof({conv0_expected_name}[0]),")
        header_parts.append(f"    {act0_name},")
        header_parts.append(f"    sizeof({act0_name}) / sizeof({act0_name}[0]),")
        header_parts.append(f"    {conv1_expected_name},")
        header_parts.append(f"    sizeof({conv1_expected_name}) / sizeof({conv1_expected_name}[0]),")
        header_parts.append("};")
        header_parts.append("")

        case_dir = args.dump_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        report["cases"].append(
            {
                "label": case_name,
                "source": source,
                "input_crop_h": input_crop_h,
                "input_crop_w": input_crop_w,
                "conv0_output_shape": list(conv0_i32.shape),
                "conv1_output_shape": list(conv1_i32.shape),
                "conv0_dequant_max_abs_diff_vs_ort_roi": conv0_max_abs_diff,
                "conv1_dequant_max_abs_diff_vs_ort_roi": conv1_max_abs_diff,
                "input_s8_sha256": sha256_array(input_s8),
                "expected_act0_s8_sha256": sha256_array(act0_s8),
                "expected_conv1_i32_sha256": sha256_array(conv1_i32),
            }
        )
        report["dumps"][case_name] = {
            "model_input_float": write_binary(case_dir / "model_input_float32.bin", input_tensor),
            "model_input_q_u8": write_binary(case_dir / "model_input_q_u8.bin", input_q_full),
            "fixture_input_s8_nhwc": write_binary(case_dir / "fixture_input_s8_nhwc.bin", input_s8),
            "conv0_float_full": write_binary(case_dir / "conv0_float32.bin", conv0_float_full),
            "conv0_expected_i32_nhwc": write_binary(case_dir / "conv0_expected_i32_nhwc.bin", conv0_i32),
            "act0_expected_s8_nhwc": write_binary(case_dir / "act0_expected_s8_nhwc.bin", act0_s8),
            "conv1_float_full": write_binary(case_dir / "conv1_float32.bin", conv1_float_full),
            "conv1_expected_i32_nhwc": write_binary(case_dir / "conv1_expected_i32_nhwc.bin", conv1_i32),
        }

    header_parts.append("inline constexpr const MultiblockFixture* kFixtures[] = {")
    for fixture_name in fixture_names:
        header_parts.append(f"    &{fixture_name},")
    header_parts.append("};")
    header_parts.append("")
    header_parts.append("}  // namespace y26_stage6_multiblock_fixture")
    header_parts.append("")

    args.fixture_header_out.write_text("\n".join(header_parts))
    args.metadata_out.write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

