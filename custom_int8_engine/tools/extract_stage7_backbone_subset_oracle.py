#!/usr/bin/env python3
"""Extract Stage 7 bounded backbone-subset oracle fixtures.

This is an offline oracle/converter helper. The runtime library must not depend
on Python, ONNX, protobuf, ONNX Runtime, or dynamic graph execution.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper, shape_inference

from extract_stage6_multiblock_oracle import (
    attrs,
    c_array,
    consumer_map,
    conv_i32_nhwc,
    dequantize_u8,
    init_summary,
    nchw_q_to_nhwc_s8,
    nhwc_s8_to_q_u8_nhwc,
    producer_map,
    quantize_u8_nearest_even,
    selected_outputs_model,
    sha256_array,
    silu,
    synthetic_gradient_input,
    synthetic_seeded_input,
    tensor_shape,
    weight_ohwi,
    write_binary,
)


SUBSET_ID = "candidate_D_block0_silu_model1_silu_model2_cv1_conv"
CONV0_NODE = "/model.0/conv/Conv"
CONV1_NODE = "/model.1/conv/Conv"
CONV2_NODE = "/model.2/cv1/conv/Conv"


def first_consumer(consumers: dict[str, list[onnx.NodeProto]], tensor: str, op_type: str, name_contains: str = ""):
    for node in consumers.get(tensor, []):
        if node.op_type == op_type and name_contains in node.name:
            return node
    raise KeyError(f"missing {op_type} consumer for {tensor}")


def dequantize_conv_i32_nhwc(i32: np.ndarray, input_scale: float, weight_scale: np.ndarray) -> np.ndarray:
    result = np.empty(i32.shape, dtype=np.float32)
    for oc in range(i32.shape[2]):
        result[:, :, oc] = i32[:, :, oc].astype(np.float32) * (float(input_scale) * float(weight_scale[oc]))
    return result


def fixture_case_name(case_name: str) -> tuple[str, str]:
    safe = case_name.replace("-", "_")
    camel = "".join(part.capitalize() for part in safe.split("_"))
    return safe, camel


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
    conv2 = nodes[CONV2_NODE]
    conv0_attrs = attrs(conv0)
    conv1_attrs = attrs(conv1)
    conv2_attrs = attrs(conv2)

    input_q = nodes["images_QuantizeLinear"]
    input_dq = first_consumer(consumers, input_q.output[0], "DequantizeLinear")

    conv0_q = first_consumer(consumers, conv0.output[0], "QuantizeLinear")
    conv0_dq = first_consumer(consumers, conv0_q.output[0], "DequantizeLinear")
    sigmoid0 = first_consumer(consumers, conv0_dq.output[0], "Sigmoid")
    mul0 = first_consumer(consumers, conv0_dq.output[0], "Mul", "/model.0/act")
    act0_q = first_consumer(consumers, mul0.output[0], "QuantizeLinear")
    act0_dq = first_consumer(consumers, act0_q.output[0], "DequantizeLinear")

    conv1_q = first_consumer(consumers, conv1.output[0], "QuantizeLinear")
    conv1_dq = first_consumer(consumers, conv1_q.output[0], "DequantizeLinear")
    sigmoid1 = first_consumer(consumers, conv1_dq.output[0], "Sigmoid")
    mul1 = first_consumer(consumers, conv1_dq.output[0], "Mul", "/model.1/act")
    act1_q = first_consumer(consumers, mul1.output[0], "QuantizeLinear")
    act1_dq = first_consumer(consumers, act1_q.output[0], "DequantizeLinear")

    conv2_q = first_consumer(consumers, conv2.output[0], "QuantizeLinear")
    conv2_dq = first_consumer(consumers, conv2_q.output[0], "DequantizeLinear")

    w0_dq = producers[conv0.input[1]]
    b0_dq = producers[conv0.input[2]]
    w1_dq = producers[conv1.input[1]]
    b1_dq = producers[conv1.input[2]]
    w2_dq = producers[conv2.input[1]]
    b2_dq = producers[conv2.input[2]]

    outputs = [
        input_q.output[0],
        input_dq.output[0],
        conv0.output[0],
        conv0_q.output[0],
        conv0_dq.output[0],
        sigmoid0.output[0],
        mul0.output[0],
        act0_q.output[0],
        act0_dq.output[0],
        conv1.output[0],
        conv1_q.output[0],
        conv1_dq.output[0],
        sigmoid1.output[0],
        mul1.output[0],
        act1_q.output[0],
        act1_dq.output[0],
        conv2.output[0],
        conv2_q.output[0],
        conv2_dq.output[0],
    ]
    oracle_model = selected_outputs_model(model, outputs)
    oracle_model_path = args.dump_dir / "stage7_backbone_subset_outputs.onnx"
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
    weight2_q = initializers[w2_dq.input[0]]
    weight0_scale = initializers[w0_dq.input[1]].astype(np.float32)
    weight1_scale = initializers[w1_dq.input[1]].astype(np.float32)
    weight2_scale = initializers[w2_dq.input[1]].astype(np.float32)
    weight0_zero_point = initializers[w0_dq.input[2]]
    weight1_zero_point = initializers[w1_dq.input[2]]
    weight2_zero_point = initializers[w2_dq.input[2]]
    bias0_q = initializers[b0_dq.input[0]].astype(np.int32)
    bias1_q = initializers[b1_dq.input[0]].astype(np.int32)
    bias2_q = initializers[b2_dq.input[0]].astype(np.int32)

    images_scale = float(initializers[input_q.input[1]].item())
    images_zero_point = int(initializers[input_q.input[2]].item())
    conv0_output_scale = float(initializers[conv0_q.input[1]].item())
    conv0_output_zero_point = int(initializers[conv0_q.input[2]].item())
    act0_output_scale = float(initializers[act0_q.input[1]].item())
    act0_output_zero_point = int(initializers[act0_q.input[2]].item())
    conv1_output_scale = float(initializers[conv1_q.input[1]].item())
    conv1_output_zero_point = int(initializers[conv1_q.input[2]].item())
    act1_output_scale = float(initializers[act1_q.input[1]].item())
    act1_output_zero_point = int(initializers[act1_q.input[2]].item())
    conv2_output_scale = float(initializers[conv2_q.input[1]].item())
    conv2_output_zero_point = int(initializers[conv2_q.input[2]].item())

    weights0 = weight_ohwi(weight0_q)
    weights1 = weight_ohwi(weight1_q)
    weights2 = weight_ohwi(weight2_q)
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
        "namespace y26_stage7_backbone_subset_fixture {",
        "",
        "struct BackboneSubsetFixture {",
        "    const char* label;",
        "    const char* subset_id;",
        "    const char* conv0_node_name;",
        "    const char* conv1_node_name;",
        "    const char* conv2_node_name;",
        "    Y26Conv2DParams conv0_params;",
        "    Y26Conv2DParams conv1_params;",
        "    Y26Conv2DParams conv2_params;",
        "    int conv0_kernel_h;",
        "    int conv0_kernel_w;",
        "    int conv1_kernel_h;",
        "    int conv1_kernel_w;",
        "    int conv2_kernel_h;",
        "    int conv2_kernel_w;",
        "    int conv0_activation_zero_point_u8;",
        "    int conv0_input_storage_zero_point_s8;",
        "    int conv0_output_zero_point_u8;",
        "    int act0_output_zero_point_u8;",
        "    int conv1_input_storage_zero_point_s8;",
        "    int conv1_output_zero_point_u8;",
        "    int act1_output_zero_point_u8;",
        "    int conv2_input_storage_zero_point_s8;",
        "    int conv2_output_zero_point_u8;",
        "    float images_scale;",
        "    float conv0_output_scale;",
        "    float act0_output_scale;",
        "    float conv1_output_scale;",
        "    float act1_output_scale;",
        "    float conv2_output_scale;",
        "    const float* conv0_weight_scales;",
        "    std::size_t conv0_weight_scale_count;",
        "    const float* conv1_weight_scales;",
        "    std::size_t conv1_weight_scale_count;",
        "    const float* conv2_weight_scales;",
        "    std::size_t conv2_weight_scale_count;",
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
        "    const std::int8_t* conv2_weights_ohwi_s8;",
        "    std::size_t conv2_weight_count;",
        "    const std::int32_t* conv2_bias_i32;",
        "    std::size_t conv2_bias_count;",
        "    const std::int32_t* expected_conv0_i32_nhwc;",
        "    std::size_t expected_conv0_count;",
        "    const std::int8_t* expected_act0_s8_nhwc;",
        "    std::size_t expected_act0_count;",
        "    const std::int32_t* expected_conv1_i32_nhwc;",
        "    std::size_t expected_conv1_count;",
        "    const std::int8_t* expected_act1_s8_nhwc;",
        "    std::size_t expected_act1_count;",
        "    const std::int32_t* expected_conv2_i32_nhwc;",
        "    std::size_t expected_conv2_count;",
        "};",
        "",
        c_array("kConv0WeightsOhwiS8", weights0, "std::int8_t"),
        "",
        c_array("kConv1WeightsOhwiS8", weights1, "std::int8_t"),
        "",
        c_array("kConv2WeightsOhwiS8", weights2, "std::int8_t"),
        "",
        c_array("kConv0BiasI32", bias0_q, "std::int32_t", cols=8),
        "",
        c_array("kConv1BiasI32", bias1_q, "std::int32_t", cols=8),
        "",
        c_array("kConv2BiasI32", bias2_q, "std::int32_t", cols=8),
        "",
        c_array("kConv0WeightScales", weight0_scale, "float", cols=8),
        "",
        c_array("kConv1WeightScales", weight1_scale, "float", cols=8),
        "",
        c_array("kConv2WeightScales", weight2_scale, "float", cols=8),
        "",
    ]

    report: dict[str, Any] = {
        "subset_id": SUBSET_ID,
        "model": str(args.model.resolve()),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "oracle_model": str(oracle_model_path.resolve()),
        "python": {"onnx": onnx.__version__, "onnxruntime": ort.__version__, "numpy": np.__version__},
        "xslim_used": False,
        "nodes": [
            {"name": node.name, "op_type": node.op_type, "inputs": list(node.input), "outputs": list(node.output)}
            for node in [
                input_q,
                input_dq,
                conv0,
                conv0_q,
                conv0_dq,
                sigmoid0,
                mul0,
                act0_q,
                act0_dq,
                conv1,
                conv1_q,
                conv1_dq,
                sigmoid1,
                mul1,
                act1_q,
                act1_dq,
                conv2,
                conv2_q,
                conv2_dq,
            ]
        ],
        "shapes": {name: tensor_shape(inferred, name) for name in outputs},
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
            "act1_output_scale": act1_output_scale,
            "act1_output_zero_point_u8": act1_output_zero_point,
            "conv2_input_storage_zero_point_s8": act1_output_zero_point - 128,
            "conv2_output_scale": conv2_output_scale,
            "conv2_output_zero_point_u8": conv2_output_zero_point,
            "conv0_weight": init_summary(w0_dq.input[0], initializers),
            "conv1_weight": init_summary(w1_dq.input[0], initializers),
            "conv2_weight": init_summary(w2_dq.input[0], initializers),
            "conv0_weight_scale": init_summary(w0_dq.input[1], initializers),
            "conv1_weight_scale": init_summary(w1_dq.input[1], initializers),
            "conv2_weight_scale": init_summary(w2_dq.input[1], initializers),
            "conv0_weight_zero_point": init_summary(w0_dq.input[2], initializers),
            "conv1_weight_zero_point": init_summary(w1_dq.input[2], initializers),
            "conv2_weight_zero_point": init_summary(w2_dq.input[2], initializers),
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
        conv2_float_full = output_map[conv2.output[0]]
        input_s8 = nchw_q_to_nhwc_s8(input_q_full, input_crop_h, input_crop_w)
        input_q_nhwc = nhwc_s8_to_q_u8_nhwc(input_s8)

        conv0_i32 = conv_i32_nhwc(
            input_q_nhwc, weight0_q, bias0_q, images_zero_point, weight0_zero_point, conv0_attrs
        )
        conv0_float = dequantize_conv_i32_nhwc(conv0_i32, images_scale, weight0_scale)
        conv0_q_local = quantize_u8_nearest_even(conv0_float, conv0_output_scale, conv0_output_zero_point)
        conv0_dq_local = dequantize_u8(conv0_q_local, conv0_output_scale, conv0_output_zero_point)
        act0_q_local = quantize_u8_nearest_even(silu(conv0_dq_local), act0_output_scale, act0_output_zero_point)
        act0_s8 = (act0_q_local.astype(np.int16) - 128).astype(np.int8)

        conv1_i32 = conv_i32_nhwc(
            act0_q_local, weight1_q, bias1_q, act0_output_zero_point, weight1_zero_point, conv1_attrs
        )
        conv1_float = dequantize_conv_i32_nhwc(conv1_i32, act0_output_scale, weight1_scale)
        conv1_q_local = quantize_u8_nearest_even(conv1_float, conv1_output_scale, conv1_output_zero_point)
        conv1_dq_local = dequantize_u8(conv1_q_local, conv1_output_scale, conv1_output_zero_point)
        act1_q_local = quantize_u8_nearest_even(silu(conv1_dq_local), act1_output_scale, act1_output_zero_point)
        act1_s8 = (act1_q_local.astype(np.int16) - 128).astype(np.int8)

        conv2_i32 = conv_i32_nhwc(
            act1_q_local, weight2_q, bias2_q, act1_output_zero_point, weight2_zero_point, conv2_attrs
        )
        conv2_float = dequantize_conv_i32_nhwc(conv2_i32, act1_output_scale, weight2_scale)

        conv0_ort_roi = np.transpose(conv0_float_full[0, :, 0 : conv0_i32.shape[0], 0 : conv0_i32.shape[1]], (1, 2, 0))
        conv1_ort_roi = np.transpose(conv1_float_full[0, :, 0 : conv1_i32.shape[0], 0 : conv1_i32.shape[1]], (1, 2, 0))
        conv2_ort_roi = np.transpose(conv2_float_full[0, :, 0 : conv2_i32.shape[0], 0 : conv2_i32.shape[1]], (1, 2, 0))
        conv0_max_abs_diff = float(np.max(np.abs(conv0_float - conv0_ort_roi)))
        conv1_max_abs_diff = float(np.max(np.abs(conv1_float - conv1_ort_roi)))
        conv2_max_abs_diff = float(np.max(np.abs(conv2_float - conv2_ort_roi)))

        _, camel = fixture_case_name(case_name)
        input_name = f"k{camel}InputS8"
        conv0_expected_name = f"k{camel}ExpectedConv0I32Nhwc"
        act0_name = f"k{camel}ExpectedAct0S8Nhwc"
        conv1_expected_name = f"k{camel}ExpectedConv1I32Nhwc"
        act1_name = f"k{camel}ExpectedAct1S8Nhwc"
        conv2_expected_name = f"k{camel}ExpectedConv2I32Nhwc"
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
        header_parts.append(c_array(act1_name, act1_s8, "std::int8_t"))
        header_parts.append("")
        header_parts.append(c_array(conv2_expected_name, conv2_i32, "std::int32_t", cols=8))
        header_parts.append("")
        header_parts.append(f"inline constexpr BackboneSubsetFixture {fixture_name} = {{")
        header_parts.append(f'    "{case_name}",')
        header_parts.append(f'    "{SUBSET_ID}",')
        header_parts.append(f'    "{CONV0_NODE}",')
        header_parts.append(f'    "{CONV1_NODE}",')
        header_parts.append(f'    "{CONV2_NODE}",')
        header_parts.append(f"    Y26Conv2DParams{{{input_crop_h}, {input_crop_w}, 3, 16, 2, 2, 1, 1}},")
        header_parts.append(f"    Y26Conv2DParams{{{conv0_i32.shape[0]}, {conv0_i32.shape[1]}, 16, 32, 2, 2, 1, 1}},")
        header_parts.append(f"    Y26Conv2DParams{{{conv1_i32.shape[0]}, {conv1_i32.shape[1]}, 32, 32, 1, 1, 0, 0}},")
        header_parts.append("    3, 3, 3, 3, 1, 1,")
        header_parts.append(f"    {images_zero_point},")
        header_parts.append(f"    {images_zero_point - 128},")
        header_parts.append(f"    {conv0_output_zero_point},")
        header_parts.append(f"    {act0_output_zero_point},")
        header_parts.append(f"    {act0_output_zero_point - 128},")
        header_parts.append(f"    {conv1_output_zero_point},")
        header_parts.append(f"    {act1_output_zero_point},")
        header_parts.append(f"    {act1_output_zero_point - 128},")
        header_parts.append(f"    {conv2_output_zero_point},")
        header_parts.append(f"    {images_scale:.12g}f,")
        header_parts.append(f"    {conv0_output_scale:.12g}f,")
        header_parts.append(f"    {act0_output_scale:.12g}f,")
        header_parts.append(f"    {conv1_output_scale:.12g}f,")
        header_parts.append(f"    {act1_output_scale:.12g}f,")
        header_parts.append(f"    {conv2_output_scale:.12g}f,")
        header_parts.append("    kConv0WeightScales,")
        header_parts.append("    sizeof(kConv0WeightScales) / sizeof(kConv0WeightScales[0]),")
        header_parts.append("    kConv1WeightScales,")
        header_parts.append("    sizeof(kConv1WeightScales) / sizeof(kConv1WeightScales[0]),")
        header_parts.append("    kConv2WeightScales,")
        header_parts.append("    sizeof(kConv2WeightScales) / sizeof(kConv2WeightScales[0]),")
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
        header_parts.append("    kConv2WeightsOhwiS8,")
        header_parts.append("    sizeof(kConv2WeightsOhwiS8) / sizeof(kConv2WeightsOhwiS8[0]),")
        header_parts.append("    kConv2BiasI32,")
        header_parts.append("    sizeof(kConv2BiasI32) / sizeof(kConv2BiasI32[0]),")
        header_parts.append(f"    {conv0_expected_name},")
        header_parts.append(f"    sizeof({conv0_expected_name}) / sizeof({conv0_expected_name}[0]),")
        header_parts.append(f"    {act0_name},")
        header_parts.append(f"    sizeof({act0_name}) / sizeof({act0_name}[0]),")
        header_parts.append(f"    {conv1_expected_name},")
        header_parts.append(f"    sizeof({conv1_expected_name}) / sizeof({conv1_expected_name}[0]),")
        header_parts.append(f"    {act1_name},")
        header_parts.append(f"    sizeof({act1_name}) / sizeof({act1_name}[0]),")
        header_parts.append(f"    {conv2_expected_name},")
        header_parts.append(f"    sizeof({conv2_expected_name}) / sizeof({conv2_expected_name}[0]),")
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
                "conv2_output_shape": list(conv2_i32.shape),
                "conv0_dequant_max_abs_diff_vs_ort_roi": conv0_max_abs_diff,
                "conv1_dequant_max_abs_diff_vs_ort_roi": conv1_max_abs_diff,
                "conv2_dequant_max_abs_diff_vs_ort_roi": conv2_max_abs_diff,
                "input_s8_sha256": sha256_array(input_s8),
                "expected_act0_s8_sha256": sha256_array(act0_s8),
                "expected_conv1_i32_sha256": sha256_array(conv1_i32),
                "expected_act1_s8_sha256": sha256_array(act1_s8),
                "expected_conv2_i32_sha256": sha256_array(conv2_i32),
            }
        )
        report["dumps"][case_name] = {
            "model_input_float": write_binary(case_dir / "model_input_float32.bin", input_tensor),
            "model_input_q_u8": write_binary(case_dir / "model_input_q_u8.bin", input_q_full),
            "fixture_input_s8_nhwc": write_binary(case_dir / "fixture_input_s8_nhwc.bin", input_s8),
            "conv0_expected_i32_nhwc": write_binary(case_dir / "conv0_expected_i32_nhwc.bin", conv0_i32),
            "act0_expected_s8_nhwc": write_binary(case_dir / "act0_expected_s8_nhwc.bin", act0_s8),
            "conv1_expected_i32_nhwc": write_binary(case_dir / "conv1_expected_i32_nhwc.bin", conv1_i32),
            "act1_expected_s8_nhwc": write_binary(case_dir / "act1_expected_s8_nhwc.bin", act1_s8),
            "conv2_expected_i32_nhwc": write_binary(case_dir / "conv2_expected_i32_nhwc.bin", conv2_i32),
            "conv2_float32_full": write_binary(case_dir / "conv2_float32_full.bin", conv2_float_full),
        }

    header_parts.append("inline constexpr const BackboneSubsetFixture* kFixtures[] = {")
    for fixture_name in fixture_names:
        header_parts.append(f"    &{fixture_name},")
    header_parts.append("};")
    header_parts.append("")
    header_parts.append("}  // namespace y26_stage7_backbone_subset_fixture")
    header_parts.append("")

    args.fixture_header_out.write_text("\n".join(header_parts))
    args.metadata_out.write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
