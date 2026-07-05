#!/usr/bin/env python3
"""Generate compact Stage 14 fixtures from the accepted YOLO26 Q/DQ graph.

The generated fixture extends the Stage 12 compact C2f oracle:

  Stage12 /model.2/cv2/conv corrected int32
  -> /model.2/cv2 activation Q/DQ
  -> /model.3/conv corrected int32
  -> /model.3 activation Q/DQ
  -> /model.4/cv1/conv corrected int32

This is a host-side oracle/converter helper only. The runtime library does not
depend on ONNX, ONNX Runtime, Python, or protobuf.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_array(text: str, name: str, dtype) -> np.ndarray:
    pattern = re.compile(rf"inline constexpr [^=]+ {re.escape(name)}\[\] = \{{(.*?)\}};", re.S)
    match = pattern.search(text)
    if not match:
        raise KeyError(f"missing array {name}")
    values = []
    for raw in match.group(1).replace("\n", " ").split(","):
        token = raw.strip()
        if not token:
            continue
        if token.endswith("f"):
            token = token[:-1]
        values.append(dtype(token))
    return np.asarray(values)


def c_array(name: str, values: np.ndarray, c_type: str, per_line: int = 12) -> str:
    flat = values.reshape(-1)
    out = [f"inline constexpr {c_type} {name}[] = {{"]
    line = []
    for i, value in enumerate(flat):
        if np.issubdtype(flat.dtype, np.floating):
            item = f"{float(value):.12g}f"
        else:
            item = str(int(value))
        line.append(item)
        if len(line) == per_line or i == flat.size - 1:
            out.append("    " + ", ".join(line) + ("," if i != flat.size - 1 else ""))
            line = []
    out.append("};")
    return "\n".join(out)


def quantize_u8_nearest_even(value: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    q = np.rint(value.astype(np.float32) / np.float32(scale)).astype(np.int64) + int(zero_point)
    return np.clip(q, 0, 255).astype(np.uint8)


def silu(value: np.ndarray) -> np.ndarray:
    value = value.astype(np.float32)
    return value / (np.float32(1.0) + np.exp(-value, dtype=np.float32))


def activation_s8_from_i32(
    producer_i32: np.ndarray,
    input_scale: float,
    weight_scales: np.ndarray,
    conv_output_scale: float,
    conv_output_zp: int,
    act_output_scale: float,
    act_output_zp: int,
) -> np.ndarray:
    channels = int(weight_scales.size)
    reshaped = producer_i32.reshape(-1, channels).astype(np.float32)
    acc_scale = (np.float32(input_scale) * weight_scales.astype(np.float32)).reshape(1, channels)
    conv_float = reshaped * acc_scale
    conv_code = quantize_u8_nearest_even(conv_float, conv_output_scale, conv_output_zp).astype(np.int32)
    conv_dq = (conv_code - int(conv_output_zp)).astype(np.float32) * np.float32(conv_output_scale)
    act_float = silu(conv_dq)
    act_code = quantize_u8_nearest_even(act_float, act_output_scale, act_output_zp).astype(np.int32)
    return (act_code - 128).astype(np.int8).reshape(producer_i32.shape)


def conv_corrected_nhwc(
    input_s8: np.ndarray,
    input_h: int,
    input_w: int,
    weights_ohwi: np.ndarray,
    bias: np.ndarray,
    activation_zero_point_u8: int,
    stride: int,
    pad: int,
) -> np.ndarray:
    output_c, kernel_h, kernel_w, input_c = weights_ohwi.shape
    output_h = (input_h + 2 * pad - kernel_h) // stride + 1
    output_w = (input_w + 2 * pad - kernel_w) // stride + 1
    input_nhwc = input_s8.reshape(input_h, input_w, input_c)
    raw = np.zeros((output_h, output_w, output_c), dtype=np.int32)
    pad_value = np.int8(int(activation_zero_point_u8) - 128)
    for oh in range(output_h):
        for ow in range(output_w):
            for oc in range(output_c):
                acc = np.int32(0)
                for kh in range(kernel_h):
                    ih = oh * stride + kh - pad
                    for kw in range(kernel_w):
                        iw = ow * stride + kw - pad
                        if 0 <= ih < input_h and 0 <= iw < input_w:
                            a = input_nhwc[ih, iw, :].astype(np.int32)
                        else:
                            a = np.full((input_c,), pad_value, dtype=np.int8).astype(np.int32)
                        w = weights_ohwi[oc, kh, kw, :].astype(np.int32)
                        acc = np.int32(acc + np.sum(a * w, dtype=np.int32))
                raw[oh, ow, oc] = acc
    weight_sums = weights_ohwi.astype(np.int32).reshape(output_c, -1).sum(axis=1).astype(np.int32)
    correction_offset = np.int64(128 - int(activation_zero_point_u8))
    corrected = raw.astype(np.int64)
    corrected += correction_offset * weight_sums.reshape(1, 1, output_c).astype(np.int64)
    corrected += bias.reshape(1, 1, output_c).astype(np.int64)
    return corrected.astype(np.int32).reshape(-1)


def build_activation_micro_model(path: Path, conv_scale: float, conv_zp: int, act_scale: float, act_zp: int) -> None:
    x = helper.make_tensor_value_info("conv_code", TensorProto.UINT8, [256])
    y = helper.make_tensor_value_info("act_code", TensorProto.UINT8, [256])
    scale0 = numpy_helper.from_array(np.asarray(conv_scale, dtype=np.float32), "conv_scale")
    zp0 = numpy_helper.from_array(np.asarray(conv_zp, dtype=np.uint8), "conv_zp")
    scale1 = numpy_helper.from_array(np.asarray(act_scale, dtype=np.float32), "act_scale")
    zp1 = numpy_helper.from_array(np.asarray(act_zp, dtype=np.uint8), "act_zp")
    nodes = [
        helper.make_node("DequantizeLinear", ["conv_code", "conv_scale", "conv_zp"], ["conv_f32"]),
        helper.make_node("Sigmoid", ["conv_f32"], ["sigmoid"]),
        helper.make_node("Mul", ["conv_f32", "sigmoid"], ["silu"]),
        helper.make_node("QuantizeLinear", ["silu", "act_scale", "act_zp"], ["act_code"]),
    ]
    graph = helper.make_graph(nodes, "stage14_activation_micro_oracle", [x], [y], [scale0, zp0, scale1, zp1])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = 10
    onnx.save(model, path)


def run_activation_micro_model(path: Path) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    codes = np.arange(256, dtype=np.uint8)
    return session.run(None, {"conv_code": codes})[0].astype(np.uint8)


def activation_lut_internal(conv_scale: float, conv_zp: int, act_scale: float, act_zp: int) -> np.ndarray:
    codes = np.arange(256, dtype=np.uint8).astype(np.int32)
    conv_dq = (codes - int(conv_zp)).astype(np.float32) * np.float32(conv_scale)
    act_float = silu(conv_dq)
    return quantize_u8_nearest_even(act_float, act_scale, act_zp)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--stage12-fixture", default="custom_int8_engine/tests/stage12_c2f_block_fixture.h")
    parser.add_argument("--out-header", default="custom_int8_engine/tests/stage14_next_c2f_fixture.h")
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--out-lut-report", required=True)
    parser.add_argument("--out-micro-dir", required=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    model = onnx.load(model_path)
    init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    stage12_text = Path(args.stage12_fixture).read_text()

    model2_cv2_output_scale = float(init["/model.2/cv2/conv/Conv_output_0_scale"])
    model2_cv2_output_zp = int(init["/model.2/cv2/conv/Conv_output_0_zero_point"])
    model2_cv2_weight_scales = init["model.2.cv2.conv.weight_scale"].astype(np.float32)
    model2_cv2_act_scale = float(init["/model.2/cv2/act/Mul_output_0_scale"])
    model2_cv2_act_zp = int(init["/model.2/cv2/act/Mul_output_0_zero_point"])

    model3_output_scale = float(init["/model.3/conv/Conv_output_0_scale"])
    model3_output_zp = int(init["/model.3/conv/Conv_output_0_zero_point"])
    model3_weight_scales = init["model.3.conv.weight_scale"].astype(np.float32)
    model3_weight_zp = init["model.3.conv.weight_zero_point"].astype(np.int8)
    model3_bias = init["model.3.conv.bias_quantized"].astype(np.int32)
    model3_weights_oihw = init["model.3.conv.weight_quantized"].astype(np.int8)
    model3_weights_ohwi = np.transpose(model3_weights_oihw, (0, 2, 3, 1)).copy()
    model3_act_scale = float(init["/model.3/act/Mul_output_0_scale"])
    model3_act_zp = int(init["/model.3/act/Mul_output_0_zero_point"])

    model4_cv1_output_scale = float(init["/model.4/cv1/conv/Conv_output_0_scale"])
    model4_cv1_output_zp = int(init["/model.4/cv1/conv/Conv_output_0_zero_point"])
    model4_cv1_weight_scales = init["model.4.cv1.conv.weight_scale"].astype(np.float32)
    model4_cv1_weight_zp = init["model.4.cv1.conv.weight_zero_point"].astype(np.int8)
    model4_cv1_bias = init["model.4.cv1.conv.bias_quantized"].astype(np.int32)
    model4_cv1_weights_oihw = init["model.4.cv1.conv.weight_quantized"].astype(np.int8)
    model4_cv1_weights_ohwi = np.transpose(model4_cv1_weights_oihw, (0, 2, 3, 1)).copy()

    seeded_model2_cv2_i32 = parse_array(stage12_text, "kSeededExpectedModel2Cv2I32Nhwc", int).astype(np.int32)
    gradient_model2_cv2_i32 = parse_array(stage12_text, "kGradientExpectedModel2Cv2I32Nhwc", int).astype(np.int32)

    micro_dir = Path(args.out_micro_dir)
    micro_dir.mkdir(parents=True, exist_ok=True)
    act0_micro = micro_dir / "stage14_model2_cv2_activation.onnx"
    act1_micro = micro_dir / "stage14_model3_activation.onnx"
    build_activation_micro_model(act0_micro, model2_cv2_output_scale, model2_cv2_output_zp, model2_cv2_act_scale, model2_cv2_act_zp)
    build_activation_micro_model(act1_micro, model3_output_scale, model3_output_zp, model3_act_scale, model3_act_zp)
    act0_ort = run_activation_micro_model(act0_micro)
    act1_ort = run_activation_micro_model(act1_micro)
    act0_internal = activation_lut_internal(model2_cv2_output_scale, model2_cv2_output_zp, model2_cv2_act_scale, model2_cv2_act_zp)
    act1_internal = activation_lut_internal(model3_output_scale, model3_output_zp, model3_act_scale, model3_act_zp)
    act0_mismatches = int(np.count_nonzero(act0_ort != act0_internal))
    act1_mismatches = int(np.count_nonzero(act1_ort != act1_internal))
    act0_max_abs_diff = int(np.max(np.abs(act0_ort.astype(np.int16) - act0_internal.astype(np.int16))))
    act1_max_abs_diff = int(np.max(np.abs(act1_ort.astype(np.int16) - act1_internal.astype(np.int16))))

    def compute_fixture(model2_cv2_i32: np.ndarray):
        model3_input_s8 = activation_s8_from_i32(
            model2_cv2_i32,
            0.3288085460662842,
            model2_cv2_weight_scales,
            model2_cv2_output_scale,
            model2_cv2_output_zp,
            model2_cv2_act_scale,
            model2_cv2_act_zp,
        )
        model3_i32 = conv_corrected_nhwc(
            model3_input_s8,
            2,
            2,
            model3_weights_ohwi,
            model3_bias,
            model2_cv2_act_zp,
            2,
            1,
        )
        model4_cv1_input_s8 = activation_s8_from_i32(
            model3_i32,
            model2_cv2_act_scale,
            model3_weight_scales,
            model3_output_scale,
            model3_output_zp,
            model3_act_scale,
            model3_act_zp,
        )
        model4_cv1_i32 = conv_corrected_nhwc(
            model4_cv1_input_s8,
            1,
            1,
            model4_cv1_weights_ohwi,
            model4_cv1_bias,
            model3_act_zp,
            1,
            0,
        )
        return model3_input_s8, model3_i32, model4_cv1_input_s8, model4_cv1_i32

    seeded = compute_fixture(seeded_model2_cv2_i32)
    gradient = compute_fixture(gradient_model2_cv2_i32)

    header = """#pragma once

#include "stage12_c2f_block_fixture.h"

#include <cstddef>
#include <cstdint>

namespace y26_stage14_next_c2f_fixture {

struct NextC2fFixture {
    const char* label;
    const char* subset_id;
    const y26_stage12_c2f_block_fixture::C2fBlockFixture* stage12_fixture;
    const char* model3_node_name;
    Y26Conv2DParams model3_params;
    int model3_kernel_h;
    int model3_kernel_w;
    int model3_activation_zero_point_u8;
    int model3_input_storage_zero_point_s8;
    int model3_output_zero_point_u8;
    float model3_input_scale;
    float model3_output_scale;
    float model3_act_output_scale;
    int model3_act_output_zero_point_u8;
    const float* model3_weight_scales;
    std::size_t model3_weight_scale_count;
    const std::int8_t* model3_weights_ohwi_s8;
    std::size_t model3_weight_count;
    const std::int32_t* model3_bias_i32;
    std::size_t model3_bias_count;
    const char* model4_cv1_node_name;
    Y26Conv2DParams model4_cv1_params;
    int model4_cv1_kernel_h;
    int model4_cv1_kernel_w;
    int model4_cv1_activation_zero_point_u8;
    int model4_cv1_input_storage_zero_point_s8;
    int model4_cv1_output_zero_point_u8;
    float model4_cv1_input_scale;
    float model4_cv1_output_scale;
    const float* model4_cv1_weight_scales;
    std::size_t model4_cv1_weight_scale_count;
    const std::int8_t* model4_cv1_weights_ohwi_s8;
    std::size_t model4_cv1_weight_count;
    const std::int32_t* model4_cv1_bias_i32;
    std::size_t model4_cv1_bias_count;
    const std::int8_t* expected_model3_input_s8_nhwc;
    std::size_t expected_model3_input_count;
    const std::int32_t* expected_model3_i32_nhwc;
    std::size_t expected_model3_count;
    const std::int8_t* expected_model4_cv1_input_s8_nhwc;
    std::size_t expected_model4_cv1_input_count;
    const std::int32_t* expected_model4_cv1_i32_nhwc;
    std::size_t expected_model4_cv1_count;
};

"""
    header += c_array("kModel3WeightsOhwiS8", model3_weights_ohwi, "std::int8_t", 16)
    header += "\n" + c_array("kModel3BiasI32", model3_bias, "std::int32_t", 8)
    header += "\n" + c_array("kModel3WeightScales", model3_weight_scales, "float", 8)
    header += "\n" + c_array("kModel4Cv1WeightsOhwiS8", model4_cv1_weights_ohwi, "std::int8_t", 16)
    header += "\n" + c_array("kModel4Cv1BiasI32", model4_cv1_bias, "std::int32_t", 8)
    header += "\n" + c_array("kModel4Cv1WeightScales", model4_cv1_weight_scales, "float", 8)
    header += "\n" + c_array("kSeededExpectedModel3InputS8Nhwc", seeded[0], "std::int8_t", 16)
    header += "\n" + c_array("kSeededExpectedModel3I32Nhwc", seeded[1], "std::int32_t", 8)
    header += "\n" + c_array("kSeededExpectedModel4Cv1InputS8Nhwc", seeded[2], "std::int8_t", 16)
    header += "\n" + c_array("kSeededExpectedModel4Cv1I32Nhwc", seeded[3], "std::int32_t", 8)
    header += "\n" + c_array("kGradientExpectedModel3InputS8Nhwc", gradient[0], "std::int8_t", 16)
    header += "\n" + c_array("kGradientExpectedModel3I32Nhwc", gradient[1], "std::int32_t", 8)
    header += "\n" + c_array("kGradientExpectedModel4Cv1InputS8Nhwc", gradient[2], "std::int8_t", 16)
    header += "\n" + c_array("kGradientExpectedModel4Cv1I32Nhwc", gradient[3], "std::int32_t", 8)

    def fixture_text(label: str, stage12_name: str, arrays_prefix: str) -> str:
        return f"""
inline constexpr NextC2fFixture k{stage12_name}Fixture = {{
    "{label}",
    "candidate_H3_model2_act_model3_act_model4_cv1_conv",
    &y26_stage12_c2f_block_fixture::k{stage12_name}Fixture,
    "/model.3/conv/Conv",
    Y26Conv2DParams{{2, 2, 64, 64, 2, 2, 1, 1}},
    3,
    3,
    {model2_cv2_act_zp},
    {model2_cv2_act_zp - 128},
    {model3_output_zp},
    {model2_cv2_act_scale:.12g}f,
    {model3_output_scale:.12g}f,
    {model3_act_scale:.12g}f,
    {model3_act_zp},
    kModel3WeightScales,
    sizeof(kModel3WeightScales) / sizeof(kModel3WeightScales[0]),
    kModel3WeightsOhwiS8,
    sizeof(kModel3WeightsOhwiS8) / sizeof(kModel3WeightsOhwiS8[0]),
    kModel3BiasI32,
    sizeof(kModel3BiasI32) / sizeof(kModel3BiasI32[0]),
    "/model.4/cv1/conv/Conv",
    Y26Conv2DParams{{1, 1, 64, 64, 1, 1, 0, 0}},
    1,
    1,
    {model3_act_zp},
    {model3_act_zp - 128},
    {model4_cv1_output_zp},
    {model3_act_scale:.12g}f,
    {model4_cv1_output_scale:.12g}f,
    kModel4Cv1WeightScales,
    sizeof(kModel4Cv1WeightScales) / sizeof(kModel4Cv1WeightScales[0]),
    kModel4Cv1WeightsOhwiS8,
    sizeof(kModel4Cv1WeightsOhwiS8) / sizeof(kModel4Cv1WeightsOhwiS8[0]),
    kModel4Cv1BiasI32,
    sizeof(kModel4Cv1BiasI32) / sizeof(kModel4Cv1BiasI32[0]),
    k{arrays_prefix}ExpectedModel3InputS8Nhwc,
    sizeof(k{arrays_prefix}ExpectedModel3InputS8Nhwc) / sizeof(k{arrays_prefix}ExpectedModel3InputS8Nhwc[0]),
    k{arrays_prefix}ExpectedModel3I32Nhwc,
    sizeof(k{arrays_prefix}ExpectedModel3I32Nhwc) / sizeof(k{arrays_prefix}ExpectedModel3I32Nhwc[0]),
    k{arrays_prefix}ExpectedModel4Cv1InputS8Nhwc,
    sizeof(k{arrays_prefix}ExpectedModel4Cv1InputS8Nhwc) / sizeof(k{arrays_prefix}ExpectedModel4Cv1InputS8Nhwc[0]),
    k{arrays_prefix}ExpectedModel4Cv1I32Nhwc,
    sizeof(k{arrays_prefix}ExpectedModel4Cv1I32Nhwc) / sizeof(k{arrays_prefix}ExpectedModel4Cv1I32Nhwc[0]),
}};
"""

    header += fixture_text("synthetic_seeded", "SyntheticSeeded", "Seeded")
    header += fixture_text("synthetic_gradient", "SyntheticGradient", "Gradient")
    header += """
inline constexpr const NextC2fFixture* kFixtures[] = {
    &kSyntheticSeededFixture,
    &kSyntheticGradientFixture,
};

}  // namespace y26_stage14_next_c2f_fixture
"""
    Path(args.out_header).write_text(header)

    lut_report = f"""# Stage 14 Boundary LUT Oracle Report

model: `{args.model}`
model_sha256: `{sha256_file(model_path)}`

## Boundaries

| boundary | conv_scale | conv_zp | act_scale | act_zp | ort_mismatches | max_abs_diff_u8 | micro_model |
|---|---:|---:|---:|---:|---:|---:|---|
| `/model.2/cv2/act` | `{model2_cv2_output_scale}` | `{model2_cv2_output_zp}` | `{model2_cv2_act_scale}` | `{model2_cv2_act_zp}` | `{act0_mismatches}` | `{act0_max_abs_diff}` | `{act0_micro}` |
| `/model.3/act` | `{model3_output_scale}` | `{model3_output_zp}` | `{model3_act_scale}` | `{model3_act_zp}` | `{act1_mismatches}` | `{act1_max_abs_diff}` | `{act1_micro}` |

## Decision

Both Stage 14 activation boundaries use boundary-specific 256-code ONNX Runtime
micro-oracles. Accepted path requires `ort_mismatches=0`.
"""
    Path(args.out_lut_report).write_text(lut_report)

    report = f"""# Stage 14 Block Oracle Report

model: `{args.model}`
model_sha256: `{sha256_file(model_path)}`
selected_subset: `candidate_H3_model2_act_model3_act_model4_cv1_conv`
output_boundary: corrected int32 output of `/model.4/cv1/conv/Conv`

## Included Nodes

1. Stage 13 selected subset through corrected int32 output of `/model.2/cv2/conv/Conv`
2. `/model.2/cv2/act/Sigmoid`
3. `/model.2/cv2/act/Mul`
4. `/model.2/cv2/act/Mul_output_0` Q/DQ handoff
5. `/model.3/conv/Conv`
6. `/model.3/act/Sigmoid`
7. `/model.3/act/Mul`
8. `/model.3/act/Mul_output_0` Q/DQ handoff
9. `/model.4/cv1/conv/Conv`

The subset stops before `/model.4/Split`.

## Conv Metadata

| node | compact input NHWC | compact output NHWC | kernel | stride | pad | output_scale | output_zp | weight_shape_oihw | weight_zp_all_zero |
|---|---|---|---|---|---|---:|---:|---|---|
| `/model.3/conv/Conv` | `[2,2,64]` | `[1,1,64]` | `3x3` | `2` | `1` | `{model3_output_scale}` | `{model3_output_zp}` | `{list(model3_weights_oihw.shape)}` | `{bool(np.all(model3_weight_zp == 0))}` |
| `/model.4/cv1/conv/Conv` | `[1,1,64]` | `[1,1,64]` | `1x1` | `1` | `0` | `{model4_cv1_output_scale}` | `{model4_cv1_output_zp}` | `{list(model4_cv1_weights_oihw.shape)}` | `{bool(np.all(model4_cv1_weight_zp == 0))}` |

## Compact Fixture Checksums

| fixture | model3_input_s8_sum | model3_i32_sum | model4_cv1_input_s8_sum | model4_cv1_i32_sum |
|---|---:|---:|---:|---:|
| synthetic_seeded | `{int(seeded[0].astype(np.int64).sum())}` | `{int(seeded[1].astype(np.int64).sum())}` | `{int(seeded[2].astype(np.int64).sum())}` | `{int(seeded[3].astype(np.int64).sum())}` |
| synthetic_gradient | `{int(gradient[0].astype(np.int64).sum())}` | `{int(gradient[1].astype(np.int64).sum())}` | `{int(gradient[2].astype(np.int64).sum())}` | `{int(gradient[3].astype(np.int64).sum())}` |

## Boundary LUT Oracle

- `/model.2/cv2/act` mismatches: `{act0_mismatches}`, max_abs_diff_u8: `{act0_max_abs_diff}`
- `/model.3/act` mismatches: `{act1_mismatches}`, max_abs_diff_u8: `{act1_max_abs_diff}`

## Decision

Stage 14 selects Candidate H3 because it expands past `/model.2` to `/model.3`
and `/model.4/cv1` without crossing the next `/model.4/Split` branch point.
"""
    Path(args.out_report).write_text(report)
    print(f"wrote {args.out_header}")
    print(f"wrote {args.out_report}")
    print(f"wrote {args.out_lut_report}")
    print(f"act0_mismatches={act0_mismatches} act1_mismatches={act1_mismatches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
