#!/usr/bin/env python3
"""Generate compact Stage 16 model4 C2f completion fixtures.

This host-side oracle tool extends Stage 15 compact fixtures through:

  /model.4/m.0/cv2/conv/Conv
  /model.4/m.0/cv2/act/Sigmoid + Mul
  /model.4/m.0/Add
  /model.4/Concat
  post-Concat Q/DQ
  /model.4/cv2/conv/Conv

The runtime C++ library does not depend on ONNX, ONNX Runtime, Python, or protobuf.
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
    line: list[str] = []
    for i, value in enumerate(flat):
        if np.issubdtype(flat.dtype, np.floating):
            item = f"{float(value):.12g}f"
        else:
            item = str(int(value))
        line.append(item)
        if len(line) == per_line or i == flat.size - 1:
            suffix = "," if i != flat.size - 1 else ""
            out.append("    " + ", ".join(line) + suffix)
            line = []
    out.append("};")
    return "\n".join(out)


def quantize_u8_nearest_even(value: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    q = np.rint(value.astype(np.float32) / np.float32(scale)).astype(np.int64) + int(zero_point)
    return np.clip(q, 0, 255).astype(np.uint8)


def silu(value: np.ndarray) -> np.ndarray:
    value = value.astype(np.float32)
    return value / (np.float32(1.0) + np.exp(-value, dtype=np.float32))


def signed_storage_from_u8(code: np.ndarray) -> np.ndarray:
    return (code.astype(np.int16) - 128).astype(np.int8)


def dequant_from_signed_storage(value_s8: np.ndarray, scale: float, zero_point_u8: int) -> np.ndarray:
    code = value_s8.astype(np.int16) + 128
    return (code - int(zero_point_u8)).astype(np.float32) * np.float32(scale)


def activation_float_and_s8_from_i32(
    producer_i32: np.ndarray,
    input_scale: float,
    weight_scales: np.ndarray,
    conv_output_scale: float,
    conv_output_zp: int,
    act_output_scale: float | None,
    act_output_zp: int | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    channels = int(weight_scales.size)
    reshaped = producer_i32.reshape(-1, channels).astype(np.float32)
    acc_scale = (np.float32(input_scale) * weight_scales.astype(np.float32)).reshape(1, channels)
    conv_float = reshaped * acc_scale
    conv_code = quantize_u8_nearest_even(conv_float, conv_output_scale, conv_output_zp).astype(np.int32)
    conv_dq = (conv_code - int(conv_output_zp)).astype(np.float32) * np.float32(conv_output_scale)
    act_float = silu(conv_dq)
    if act_output_scale is None or act_output_zp is None:
        return act_float.reshape(producer_i32.shape), None
    act_code = quantize_u8_nearest_even(act_float, act_output_scale, act_output_zp)
    return act_float.reshape(producer_i32.shape), signed_storage_from_u8(act_code).reshape(producer_i32.shape)


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
    graph = helper.make_graph(nodes, "stage16_activation_micro_oracle", [x], [y], [scale0, zp0, scale1, zp1])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = 10
    onnx.save(model, path)


def run_micro_model(path: Path) -> np.ndarray:
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
    parser.add_argument("--stage14-fixture", default="custom_int8_engine/tests/stage14_next_c2f_fixture.h")
    parser.add_argument("--stage15-fixture", default="custom_int8_engine/tests/stage15_model4_branch_fixture.h")
    parser.add_argument("--out-header", default="custom_int8_engine/tests/stage16_model4_c2f_fixture.h")
    parser.add_argument("--out-oracle-report", required=True)
    parser.add_argument("--out-add-report", required=True)
    parser.add_argument("--out-concat-report", required=True)
    parser.add_argument("--out-lut-report", required=True)
    parser.add_argument("--out-scale-report", required=True)
    parser.add_argument("--out-micro-dir", required=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    model = onnx.load(model_path)
    init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    stage14_text = Path(args.stage14_fixture).read_text()
    stage15_text = Path(args.stage15_fixture).read_text()

    model4_cv1_input_scale = float(init["/model.3/act/Mul_output_0_scale"])
    model4_cv1_output_scale = float(init["/model.4/cv1/conv/Conv_output_0_scale"])
    model4_cv1_output_zp = int(init["/model.4/cv1/conv/Conv_output_0_zero_point"])
    model4_cv1_weight_scales = init["model.4.cv1.conv.weight_scale"].astype(np.float32)
    split1_scale = float(init["/model.4/Split_output_1_scale"])
    split1_zp = int(init["/model.4/Split_output_1_zero_point"])

    branch0_act_scale = float(init["/model.4/m.0/cv1/act/Mul_output_0_scale"])
    branch0_act_zp = int(init["/model.4/m.0/cv1/act/Mul_output_0_zero_point"])

    branch1_output_scale = float(init["/model.4/m.0/cv2/conv/Conv_output_0_scale"])
    branch1_output_zp = int(init["/model.4/m.0/cv2/conv/Conv_output_0_zero_point"])
    branch1_weight_scales = init["model.4.m.0.cv2.conv.weight_scale"].astype(np.float32)
    branch1_weight_zp = init["model.4.m.0.cv2.conv.weight_zero_point"].astype(np.int8)
    branch1_bias = init["model.4.m.0.cv2.conv.bias_quantized"].astype(np.int32)
    branch1_weights_oihw = init["model.4.m.0.cv2.conv.weight_quantized"].astype(np.int8)
    branch1_weights_ohwi = np.transpose(branch1_weights_oihw, (0, 2, 3, 1)).copy()

    concat_scale = float(init["/model.4/Concat_output_0_scale"])
    concat_zp = int(init["/model.4/Concat_output_0_zero_point"])

    model4_cv2_output_scale = float(init["/model.4/cv2/conv/Conv_output_0_scale"])
    model4_cv2_output_zp = int(init["/model.4/cv2/conv/Conv_output_0_zero_point"])
    model4_cv2_weight_scales = init["model.4.cv2.conv.weight_scale"].astype(np.float32)
    model4_cv2_weight_zp = init["model.4.cv2.conv.weight_zero_point"].astype(np.int8)
    model4_cv2_bias = init["model.4.cv2.conv.bias_quantized"].astype(np.int32)
    model4_cv2_weights_oihw = init["model.4.cv2.conv.weight_quantized"].astype(np.int8)
    model4_cv2_weights_ohwi = np.transpose(model4_cv2_weights_oihw, (0, 2, 3, 1)).copy()

    if np.any(branch1_weight_zp != 0) or np.any(model4_cv2_weight_zp != 0):
        raise RuntimeError("Stage16 requires symmetric signed weights with zero-point 0")

    seeded_model4_cv1_i32 = parse_array(stage14_text, "kSeededExpectedModel4Cv1I32Nhwc", int).astype(np.int32)
    gradient_model4_cv1_i32 = parse_array(stage14_text, "kGradientExpectedModel4Cv1I32Nhwc", int).astype(np.int32)
    seeded_branch0_act = parse_array(stage15_text, "kSeededExpectedBranch0ActS8Nhwc", int).astype(np.int8)
    gradient_branch0_act = parse_array(stage15_text, "kGradientExpectedBranch0ActS8Nhwc", int).astype(np.int8)

    def make_fixture(model4_cv1_i32: np.ndarray, branch0_act_s8: np.ndarray):
        model4_cv1_act_float, split_all_s8 = activation_float_and_s8_from_i32(
            model4_cv1_i32,
            model4_cv1_input_scale,
            model4_cv1_weight_scales,
            model4_cv1_output_scale,
            model4_cv1_output_zp,
            split1_scale,
            split1_zp,
        )
        model4_cv1_act_float = model4_cv1_act_float.reshape(-1, 64)
        split_all_s8 = split_all_s8.reshape(-1, 64)
        split0_float = model4_cv1_act_float[:, 0:32]
        split1_s8 = split_all_s8[:, 32:64].reshape(-1).astype(np.int8)
        split1_qdq_float = dequant_from_signed_storage(split1_s8, split1_scale, split1_zp).reshape(-1, 32)

        branch1_i32 = conv_corrected_nhwc(
            branch0_act_s8,
            1,
            1,
            branch1_weights_ohwi,
            branch1_bias,
            branch0_act_zp,
            stride=1,
            pad=1,
        )
        branch1_act_float, _ = activation_float_and_s8_from_i32(
            branch1_i32,
            branch0_act_scale,
            branch1_weight_scales,
            branch1_output_scale,
            branch1_output_zp,
            None,
            None,
        )
        branch1_act_float = branch1_act_float.reshape(-1, 32)
        add_float = split1_qdq_float + branch1_act_float
        concat_float = np.concatenate([split0_float, split1_qdq_float, add_float], axis=1)
        concat_code = quantize_u8_nearest_even(concat_float, concat_scale, concat_zp)
        concat_s8 = signed_storage_from_u8(concat_code).reshape(-1).astype(np.int8)
        model4_cv2_i32 = conv_corrected_nhwc(
            concat_s8,
            1,
            1,
            model4_cv2_weights_ohwi,
            model4_cv2_bias,
            concat_zp,
            stride=1,
            pad=0,
        )
        return branch1_i32, concat_s8, model4_cv2_i32

    seeded_branch1_i32, seeded_concat_s8, seeded_model4_cv2_i32 = make_fixture(
        seeded_model4_cv1_i32, seeded_branch0_act
    )
    gradient_branch1_i32, gradient_concat_s8, gradient_model4_cv2_i32 = make_fixture(
        gradient_model4_cv1_i32, gradient_branch0_act
    )

    micro_dir = Path(args.out_micro_dir)
    micro_dir.mkdir(parents=True, exist_ok=True)
    branch0_micro = micro_dir / "stage16_model4_m0_cv1_activation.onnx"
    build_activation_micro_model(
        branch0_micro,
        float(init["/model.4/m.0/cv1/conv/Conv_output_0_scale"]),
        int(init["/model.4/m.0/cv1/conv/Conv_output_0_zero_point"]),
        branch0_act_scale,
        branch0_act_zp,
    )
    branch0_ort = run_micro_model(branch0_micro)
    branch0_internal = activation_lut_internal(
        float(init["/model.4/m.0/cv1/conv/Conv_output_0_scale"]),
        int(init["/model.4/m.0/cv1/conv/Conv_output_0_zero_point"]),
        branch0_act_scale,
        branch0_act_zp,
    )

    header = f"""#pragma once

#include "stage15_model4_branch_fixture.h"

#include <cstddef>
#include <cstdint>

namespace y26_stage16_model4_c2f_fixture {{

struct Model4C2fFixture {{
    const char* label;
    const char* subset_id;
    const y26_stage15_model4_branch_fixture::Model4BranchFixture* stage15_fixture;
    Y26Conv2DParams branch1_params;
    int branch1_kernel_h;
    int branch1_kernel_w;
    int branch1_activation_zero_point_u8;
    int branch1_input_storage_zero_point_s8;
    float branch1_input_scale;
    float branch1_output_scale;
    int branch1_output_zero_point_u8;
    const float* branch1_weight_scales;
    std::size_t branch1_weight_scale_count;
    const std::int8_t* branch1_weights_ohwi_s8;
    std::size_t branch1_weight_count;
    const std::int32_t* branch1_bias_i32;
    std::size_t branch1_bias_count;
    Y26Conv2DParams model4_cv2_params;
    int model4_cv2_kernel_h;
    int model4_cv2_kernel_w;
    int model4_cv2_activation_zero_point_u8;
    int model4_cv2_input_storage_zero_point_s8;
    float concat_output_scale;
    int concat_output_zero_point_u8;
    float model4_cv2_output_scale;
    int model4_cv2_output_zero_point_u8;
    const float* model4_cv2_weight_scales;
    std::size_t model4_cv2_weight_scale_count;
    const std::int8_t* model4_cv2_weights_ohwi_s8;
    std::size_t model4_cv2_weight_count;
    const std::int32_t* model4_cv2_bias_i32;
    std::size_t model4_cv2_bias_count;
    const std::int32_t* expected_branch1_i32_nhwc;
    std::size_t expected_branch1_count;
    const std::int8_t* expected_concat_s8_nhwc;
    std::size_t expected_concat_count;
    const std::int32_t* expected_model4_cv2_i32_nhwc;
    std::size_t expected_model4_cv2_count;
}};

{c_array("kBranch1WeightScales", branch1_weight_scales, "float")}

{c_array("kBranch1WeightsOhwiS8", branch1_weights_ohwi.astype(np.int8), "std::int8_t")}

{c_array("kBranch1BiasI32", branch1_bias, "std::int32_t")}

{c_array("kModel4Cv2WeightScales", model4_cv2_weight_scales, "float")}

{c_array("kModel4Cv2WeightsOhwiS8", model4_cv2_weights_ohwi.astype(np.int8), "std::int8_t")}

{c_array("kModel4Cv2BiasI32", model4_cv2_bias, "std::int32_t")}

{c_array("kSeededExpectedBranch1I32Nhwc", seeded_branch1_i32, "std::int32_t")}

{c_array("kSeededExpectedConcatS8Nhwc", seeded_concat_s8, "std::int8_t")}

{c_array("kSeededExpectedModel4Cv2I32Nhwc", seeded_model4_cv2_i32, "std::int32_t")}

{c_array("kGradientExpectedBranch1I32Nhwc", gradient_branch1_i32, "std::int32_t")}

{c_array("kGradientExpectedConcatS8Nhwc", gradient_concat_s8, "std::int8_t")}

{c_array("kGradientExpectedModel4Cv2I32Nhwc", gradient_model4_cv2_i32, "std::int32_t")}

inline constexpr Model4C2fFixture kSyntheticSeededFixture = {{
    "synthetic_seeded",
    "candidate_J_model4_c2f_complete_compact",
    &y26_stage15_model4_branch_fixture::kSyntheticSeededFixture,
    Y26Conv2DParams{{1, 1, 16, 32, 1, 1, 1, 1}},
    3,
    3,
    {branch0_act_zp},
    {branch0_act_zp - 128},
    {branch0_act_scale:.12g}f,
    {branch1_output_scale:.12g}f,
    {branch1_output_zp},
    kBranch1WeightScales,
    {branch1_weight_scales.size},
    kBranch1WeightsOhwiS8,
    {branch1_weights_ohwi.size},
    kBranch1BiasI32,
    {branch1_bias.size},
    Y26Conv2DParams{{1, 1, 96, 128, 1, 1, 0, 0}},
    1,
    1,
    {concat_zp},
    {concat_zp - 128},
    {concat_scale:.12g}f,
    {concat_zp},
    {model4_cv2_output_scale:.12g}f,
    {model4_cv2_output_zp},
    kModel4Cv2WeightScales,
    {model4_cv2_weight_scales.size},
    kModel4Cv2WeightsOhwiS8,
    {model4_cv2_weights_ohwi.size},
    kModel4Cv2BiasI32,
    {model4_cv2_bias.size},
    kSeededExpectedBranch1I32Nhwc,
    {seeded_branch1_i32.size},
    kSeededExpectedConcatS8Nhwc,
    {seeded_concat_s8.size},
    kSeededExpectedModel4Cv2I32Nhwc,
    {seeded_model4_cv2_i32.size}
}};

inline constexpr Model4C2fFixture kSyntheticGradientFixture = {{
    "synthetic_gradient",
    "candidate_J_model4_c2f_complete_compact",
    &y26_stage15_model4_branch_fixture::kSyntheticGradientFixture,
    Y26Conv2DParams{{1, 1, 16, 32, 1, 1, 1, 1}},
    3,
    3,
    {branch0_act_zp},
    {branch0_act_zp - 128},
    {branch0_act_scale:.12g}f,
    {branch1_output_scale:.12g}f,
    {branch1_output_zp},
    kBranch1WeightScales,
    {branch1_weight_scales.size},
    kBranch1WeightsOhwiS8,
    {branch1_weights_ohwi.size},
    kBranch1BiasI32,
    {branch1_bias.size},
    Y26Conv2DParams{{1, 1, 96, 128, 1, 1, 0, 0}},
    1,
    1,
    {concat_zp},
    {concat_zp - 128},
    {concat_scale:.12g}f,
    {concat_zp},
    {model4_cv2_output_scale:.12g}f,
    {model4_cv2_output_zp},
    kModel4Cv2WeightScales,
    {model4_cv2_weight_scales.size},
    kModel4Cv2WeightsOhwiS8,
    {model4_cv2_weights_ohwi.size},
    kModel4Cv2BiasI32,
    {model4_cv2_bias.size},
    kGradientExpectedBranch1I32Nhwc,
    {gradient_branch1_i32.size},
    kGradientExpectedConcatS8Nhwc,
    {gradient_concat_s8.size},
    kGradientExpectedModel4Cv2I32Nhwc,
    {gradient_model4_cv2_i32.size}
}};

inline constexpr const Model4C2fFixture* kFixtures[] = {{
    &kSyntheticSeededFixture,
    &kSyntheticGradientFixture
}};

}}  // namespace y26_stage16_model4_c2f_fixture
"""
    out_header = Path(args.out_header)
    out_header.write_text(header)

    Path(args.out_oracle_report).write_text(
        "\n".join(
            [
                "# Stage 16 Model4 C2f Oracle Report",
                "",
                f"model: `{model_path}`",
                f"model_sha256: `{sha256_file(model_path)}`",
                "provider: `CPUExecutionProvider` for 256-code activation micro-oracles",
                "selected_subset: `candidate_J_model4_c2f_complete_compact`",
                "",
                "## Compact Fixture Checksums",
                "",
                f"seeded_branch1_i32_checksum: `{int(seeded_branch1_i32.astype(np.int64).sum())}`",
                f"seeded_concat_s8_checksum: `{int(seeded_concat_s8.astype(np.int64).sum())}`",
                f"seeded_model4_cv2_i32_checksum: `{int(seeded_model4_cv2_i32.astype(np.int64).sum())}`",
                f"gradient_branch1_i32_checksum: `{int(gradient_branch1_i32.astype(np.int64).sum())}`",
                f"gradient_concat_s8_checksum: `{int(gradient_concat_s8.astype(np.int64).sum())}`",
                f"gradient_model4_cv2_i32_checksum: `{int(gradient_model4_cv2_i32.astype(np.int64).sum())}`",
                "",
                "## Contract",
                "",
                "`/model.4/m.0/Add` and `/model.4/Concat` are float-domain operations in the accepted Q/DQ graph.",
                "The compact fixture preserves this by computing float Split0, Q/DQ float Split1, float branch cv2 SiLU,",
                "float Add, float Concat, post-Concat Q/DQ, then corrected int32 `/model.4/cv2/conv/Conv`.",
            ]
        )
        + "\n"
    )

    Path(args.out_add_report).write_text(
        "\n".join(
            [
                "# Model4 Add Contract Report",
                "",
                "node: `/model.4/m.0/Add`",
                "inputs:",
                "- `/model.4/Split_output_1_DequantizeLinear_Output`: float-domain Q/DQ output from Split output1",
                "- `/model.4/m.0/cv2/act/Mul_output_0`: float-domain SiLU output without intermediate Q/DQ",
                "output: `/model.4/m.0/Add_output_0`",
                "classification: `float-domain Add`",
                "accepted Stage 16 path: measured float fallback, not an integer-domain shortcut",
            ]
        )
        + "\n"
    )
    Path(args.out_concat_report).write_text(
        "\n".join(
            [
                "# Model4 Concat Contract Report",
                "",
                "node: `/model.4/Concat`",
                "axis: `1` in ONNX NCHW, represented as channel concatenation in NHWC runner fixtures",
                "inputs:",
                "- channels `[0,32)`: `/model.4/Split_output_0`, float-domain direct Split output",
                "- channels `[32,64)`: `/model.4/Split_output_1_DequantizeLinear_Output`, float-domain Q/DQ output",
                "- channels `[64,96)`: `/model.4/m.0/Add_output_0`, float-domain Add output",
                "post_concat_qdq_scale: `{:.12g}`".format(concat_scale),
                f"post_concat_qdq_zero_point_u8: `{concat_zp}`",
                "accepted Stage 16 compact path: materialize post-Concat signed int8 storage for `/model.4/cv2/conv/Conv`",
            ]
        )
        + "\n"
    )
    Path(args.out_lut_report).write_text(
        "\n".join(
            [
                "# Stage 16 Boundary LUT Oracle Report",
                "",
                f"micro_model: `{branch0_micro}`",
                f"micro_model_sha256: `{sha256_file(branch0_micro)}`",
                "boundary: `/model.4/m.0/cv1/conv/Conv_output_0` -> `/model.4/m.0/cv1/act/Mul_output_0`",
                f"mismatches: `{int(np.count_nonzero(branch0_ort != branch0_internal))}`",
                f"max_abs_diff_u8: `{int(np.max(np.abs(branch0_ort.astype(np.int16) - branch0_internal.astype(np.int16))))}`",
            ]
        )
        + "\n"
    )
    Path(args.out_scale_report).write_text(
        "\n".join(
            [
                "# Stage 16 Scale Zero Point Report",
                "",
                f"split1_scale: `{split1_scale:.12g}`",
                f"split1_zero_point_u8: `{split1_zp}`",
                f"branch0_act_scale: `{branch0_act_scale:.12g}`",
                f"branch0_act_zero_point_u8: `{branch0_act_zp}`",
                f"branch1_conv_output_scale: `{branch1_output_scale:.12g}`",
                f"branch1_conv_output_zero_point_u8: `{branch1_output_zp}`",
                f"concat_output_scale: `{concat_scale:.12g}`",
                f"concat_output_zero_point_u8: `{concat_zp}`",
                f"model4_cv2_conv_output_scale: `{model4_cv2_output_scale:.12g}`",
                f"model4_cv2_conv_output_zero_point_u8: `{model4_cv2_output_zp}`",
            ]
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
