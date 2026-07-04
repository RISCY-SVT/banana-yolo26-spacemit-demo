#!/usr/bin/env python3
"""Generate Stage 11 branch-block fixture metadata.

This host-side tool reads the accepted Q/DQ ONNX artifact and the compact
Stage 10 fixture header. It emits a small deterministic C++ fixture for:

Stage 10 subset -> /model.2/m.0/cv1/act/Sigmoid+Mul
-> /model.2/m.0/cv2/conv/Conv corrected int32 output.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def parse_array(text: str, name: str, dtype) -> np.ndarray:
    pattern = re.compile(rf"inline constexpr [^=]+ {re.escape(name)}\[\] = \{{(.*?)\}};", re.S)
    match = pattern.search(text)
    if not match:
        raise KeyError(f"array not found: {name}")
    values = [item.strip() for item in match.group(1).replace("\n", " ").split(",") if item.strip()]
    return np.asarray([dtype(item.rstrip("f")) for item in values])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def round_even(value: float) -> int:
    return int(np.rint(value))


def clamp_u8(value: int) -> int:
    return max(0, min(255, value))


def signed_storage_from_u8(value: int) -> np.int8:
    return np.int8(value - 128)


def silu(value: float) -> float:
    return value / (1.0 + math.exp(-value))


def quantize_u8(value: float, scale: float, zero_point: int) -> int:
    return clamp_u8(round_even(value / scale) + zero_point)


def build_lut(conv_scale: float, conv_zp: int, act_scale: float, act_zp: int) -> tuple[np.ndarray, np.ndarray]:
    out_u8 = np.zeros(256, dtype=np.uint8)
    out_s8 = np.zeros(256, dtype=np.int8)
    for q in range(256):
        x = float(q - conv_zp) * conv_scale
        qy = quantize_u8(silu(x), act_scale, act_zp)
        out_u8[q] = qy
        out_s8[q] = signed_storage_from_u8(qy)
    return out_u8, out_s8


def build_onnx_lut_model(path: Path, conv_scale: float, conv_zp: int, act_scale: float, act_zp: int) -> None:
    codes = helper.make_tensor_value_info("codes", TensorProto.UINT8, [256])
    out = helper.make_tensor_value_info("act_codes", TensorProto.UINT8, [256])
    initializers = [
        numpy_helper.from_array(np.asarray(conv_scale, dtype=np.float32), "conv_scale"),
        numpy_helper.from_array(np.asarray(conv_zp, dtype=np.uint8), "conv_zp"),
        numpy_helper.from_array(np.asarray(act_scale, dtype=np.float32), "act_scale"),
        numpy_helper.from_array(np.asarray(act_zp, dtype=np.uint8), "act_zp"),
    ]
    nodes = [
        helper.make_node("DequantizeLinear", ["codes", "conv_scale", "conv_zp"], ["x"]),
        helper.make_node("Sigmoid", ["x"], ["sigmoid"]),
        helper.make_node("Mul", ["x", "sigmoid"], ["silu"]),
        helper.make_node("QuantizeLinear", ["silu", "act_scale", "act_zp"], ["act_codes"]),
    ]
    graph = helper.make_graph(nodes, "stage11_silu_lut_oracle", [codes], [out], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])
    model.ir_version = 10
    onnx.save(model, path)


def run_onnx_lut_oracle(model_path: Path) -> np.ndarray:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    codes = np.arange(256, dtype=np.uint8)
    (out,) = session.run(None, {"codes": codes})
    return out.astype(np.uint8)


def activation_from_i32(
    values: np.ndarray,
    channels: int,
    input_scale: float,
    weight_scales: np.ndarray,
    conv_scale: float,
    conv_zp: int,
    act_scale: float,
    act_zp: int,
) -> np.ndarray:
    _, lut = build_lut(conv_scale, conv_zp, act_scale, act_zp)
    out = np.zeros(values.size, dtype=np.int8)
    for i, acc in enumerate(values.astype(np.int64)):
        c = i % channels
        conv_q = clamp_u8(round_even(float(acc) * input_scale * float(weight_scales[c]) / conv_scale) + conv_zp)
        out[i] = lut[conv_q]
    return out


def branch_conv_corrected(
    input_s8: np.ndarray,
    input_h: int,
    input_w: int,
    weights_ohwi: np.ndarray,
    bias_i32: np.ndarray,
    activation_zp_u8: int,
) -> np.ndarray:
    output_c, kernel_h, kernel_w, input_c = weights_ohwi.shape
    pad = np.int8(activation_zp_u8 - 128)
    weight_sums = weights_ohwi.astype(np.int32).sum(axis=(1, 2, 3))
    raw = np.zeros((input_h, input_w, output_c), dtype=np.int32)
    inp = input_s8.reshape(input_h, input_w, input_c)
    for oh in range(input_h):
        for ow in range(input_w):
            for oc in range(output_c):
                acc = 0
                for kh in range(kernel_h):
                    ih = oh + kh - 1
                    for kw in range(kernel_w):
                        iw = ow + kw - 1
                        for ic in range(input_c):
                            a = int(inp[ih, iw, ic]) if 0 <= ih < input_h and 0 <= iw < input_w else int(pad)
                            acc += a * int(weights_ohwi[oc, kh, kw, ic])
                raw[oh, ow, oc] = acc
    corrected = (
        raw.astype(np.int64)
        + (128 - activation_zp_u8) * weight_sums.reshape(1, 1, output_c)
        + bias_i32.reshape(1, 1, output_c)
    )
    return corrected.astype(np.int32).reshape(-1)


def c_array(name: str, values: np.ndarray, ctype: str, per_line: int = 12) -> str:
    flat = values.reshape(-1)
    items = []
    for v in flat:
        if np.issubdtype(flat.dtype, np.floating):
            items.append(f"{float(v):.12g}f")
        else:
            items.append(str(int(v)))
    lines = []
    for i in range(0, len(items), per_line):
        lines.append("    " + ", ".join(items[i : i + per_line]))
    return f"inline constexpr {ctype} {name}[] = {{\n" + ",\n".join(lines) + "\n};\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--stage10-fixture", required=True)
    parser.add_argument("--out-header", required=True)
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--out-lut-onnx", required=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    model = onnx.load(model_path)
    init = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}
    stage10_text = Path(args.stage10_fixture).read_text()

    branch0_weight_scales = parse_array(stage10_text, "kBranch0WeightScales", float).astype(np.float32)
    seeded_branch0_i32 = parse_array(stage10_text, "kSeededExpectedBranch0I32Nhwc", int).astype(np.int32)
    gradient_branch0_i32 = parse_array(stage10_text, "kGradientExpectedBranch0I32Nhwc", int).astype(np.int32)

    split1_scale = float(init["/model.2/Split_output_1_scale"])
    split1_zp = int(init["/model.2/Split_output_1_zero_point"])
    branch0_conv_scale = float(init["/model.2/m.0/cv1/conv/Conv_output_0_scale"])
    branch0_conv_zp = int(init["/model.2/m.0/cv1/conv/Conv_output_0_zero_point"])
    branch0_act_scale = float(init["/model.2/m.0/cv1/act/Mul_output_0_scale"])
    branch0_act_zp = int(init["/model.2/m.0/cv1/act/Mul_output_0_zero_point"])
    branch1_scale = float(init["/model.2/m.0/cv2/conv/Conv_output_0_scale"])
    branch1_zp = int(init["/model.2/m.0/cv2/conv/Conv_output_0_zero_point"])
    branch1_weight_scales = init["model.2.m.0.cv2.conv.weight_scale"].astype(np.float32)
    branch1_bias = init["model.2.m.0.cv2.conv.bias_quantized"].astype(np.int32)
    branch1_weights_oihw = init["model.2.m.0.cv2.conv.weight_quantized"].astype(np.int8)
    branch1_weights_ohwi = np.transpose(branch1_weights_oihw, (0, 2, 3, 1)).copy()

    out_lut_onnx = Path(args.out_lut_onnx)
    out_lut_onnx.parent.mkdir(parents=True, exist_ok=True)
    build_onnx_lut_model(out_lut_onnx, branch0_conv_scale, branch0_conv_zp, branch0_act_scale, branch0_act_zp)
    internal_lut_u8, internal_lut_s8 = build_lut(branch0_conv_scale, branch0_conv_zp, branch0_act_scale, branch0_act_zp)
    ort_lut_u8 = run_onnx_lut_oracle(out_lut_onnx)
    lut_mismatches = int(np.count_nonzero(internal_lut_u8 != ort_lut_u8))
    lut_max_abs_diff = int(np.max(np.abs(internal_lut_u8.astype(np.int16) - ort_lut_u8.astype(np.int16))))

    def compute_fixture(branch0_i32: np.ndarray):
        act = activation_from_i32(
            branch0_i32,
            8,
            split1_scale,
            branch0_weight_scales,
            branch0_conv_scale,
            branch0_conv_zp,
            branch0_act_scale,
            branch0_act_zp,
        )
        branch1 = branch_conv_corrected(act, 2, 2, branch1_weights_ohwi, branch1_bias, branch0_act_zp)
        return act, branch1

    seeded_act, seeded_branch1 = compute_fixture(seeded_branch0_i32)
    gradient_act, gradient_branch1 = compute_fixture(gradient_branch0_i32)

    header = """#pragma once

#include "stage10_backbone_expansion_fixture.h"

#include <cstddef>
#include <cstdint>

namespace y26_stage11_branch_block_fixture {

struct BranchBlockFixture {
    const char* label;
    const char* subset_id;
    const y26_stage10_backbone_expansion_fixture::BackboneExpansionFixture* stage10_fixture;
    const char* branch1_node_name;
    Y26Conv2DParams branch1_params;
    int branch1_kernel_h;
    int branch1_kernel_w;
    int branch1_activation_zero_point_u8;
    int branch1_input_storage_zero_point_s8;
    int branch1_output_zero_point_u8;
    float branch0_act_output_scale;
    float branch1_output_scale;
    const float* branch1_weight_scales;
    std::size_t branch1_weight_scale_count;
    const std::int8_t* branch1_weights_ohwi_s8;
    std::size_t branch1_weight_count;
    const std::int32_t* branch1_bias_i32;
    std::size_t branch1_bias_count;
    const std::int8_t* expected_branch0_act_s8_nhwc;
    std::size_t expected_branch0_act_count;
    const std::int32_t* expected_branch1_i32_nhwc;
    std::size_t expected_branch1_count;
};

"""
    header += c_array("kBranch1WeightsOhwiS8", branch1_weights_ohwi, "std::int8_t", 16)
    header += "\n" + c_array("kBranch1BiasI32", branch1_bias, "std::int32_t", 8)
    header += "\n" + c_array("kBranch1WeightScales", branch1_weight_scales, "float", 8)
    header += "\n" + c_array("kBranch0ActLutS8", internal_lut_s8, "std::int8_t", 16)
    header += "\n" + c_array("kSeededExpectedBranch0ActS8Nhwc", seeded_act, "std::int8_t", 16)
    header += "\n" + c_array("kSeededExpectedBranch1I32Nhwc", seeded_branch1, "std::int32_t", 8)
    header += "\n" + c_array("kGradientExpectedBranch0ActS8Nhwc", gradient_act, "std::int8_t", 16)
    header += "\n" + c_array("kGradientExpectedBranch1I32Nhwc", gradient_branch1, "std::int32_t", 8)
    header += f"""
inline constexpr BranchBlockFixture kSyntheticSeededFixture = {{
    "synthetic_seeded",
    "candidate_F_model2_m0_cv1_act_cv2_conv",
    &y26_stage10_backbone_expansion_fixture::kSyntheticSeededFixture,
    "/model.2/m.0/cv2/conv/Conv",
    Y26Conv2DParams{{2, 2, 8, 16, 1, 1, 1, 1}},
    3,
    3,
    {branch0_act_zp},
    {branch0_act_zp - 128},
    {branch1_zp},
    {branch0_act_scale:.12g}f,
    {branch1_scale:.12g}f,
    kBranch1WeightScales,
    sizeof(kBranch1WeightScales) / sizeof(kBranch1WeightScales[0]),
    kBranch1WeightsOhwiS8,
    sizeof(kBranch1WeightsOhwiS8) / sizeof(kBranch1WeightsOhwiS8[0]),
    kBranch1BiasI32,
    sizeof(kBranch1BiasI32) / sizeof(kBranch1BiasI32[0]),
    kSeededExpectedBranch0ActS8Nhwc,
    sizeof(kSeededExpectedBranch0ActS8Nhwc) / sizeof(kSeededExpectedBranch0ActS8Nhwc[0]),
    kSeededExpectedBranch1I32Nhwc,
    sizeof(kSeededExpectedBranch1I32Nhwc) / sizeof(kSeededExpectedBranch1I32Nhwc[0]),
}};

inline constexpr BranchBlockFixture kSyntheticGradientFixture = {{
    "synthetic_gradient",
    "candidate_F_model2_m0_cv1_act_cv2_conv",
    &y26_stage10_backbone_expansion_fixture::kSyntheticGradientFixture,
    "/model.2/m.0/cv2/conv/Conv",
    Y26Conv2DParams{{2, 2, 8, 16, 1, 1, 1, 1}},
    3,
    3,
    {branch0_act_zp},
    {branch0_act_zp - 128},
    {branch1_zp},
    {branch0_act_scale:.12g}f,
    {branch1_scale:.12g}f,
    kBranch1WeightScales,
    sizeof(kBranch1WeightScales) / sizeof(kBranch1WeightScales[0]),
    kBranch1WeightsOhwiS8,
    sizeof(kBranch1WeightsOhwiS8) / sizeof(kBranch1WeightsOhwiS8[0]),
    kBranch1BiasI32,
    sizeof(kBranch1BiasI32) / sizeof(kBranch1BiasI32[0]),
    kGradientExpectedBranch0ActS8Nhwc,
    sizeof(kGradientExpectedBranch0ActS8Nhwc) / sizeof(kGradientExpectedBranch0ActS8Nhwc[0]),
    kGradientExpectedBranch1I32Nhwc,
    sizeof(kGradientExpectedBranch1I32Nhwc) / sizeof(kGradientExpectedBranch1I32Nhwc[0]),
}};

inline constexpr const BranchBlockFixture* kFixtures[] = {{
    &kSyntheticSeededFixture,
    &kSyntheticGradientFixture,
}};

}}  // namespace y26_stage11_branch_block_fixture
"""
    Path(args.out_header).write_text(header)

    report = f"""# Stage 11 Branch Block Oracle Report

model: `{args.model}`
model_sha256: `{sha256_file(model_path)}`
selected_subset: `candidate_F_model2_m0_cv1_act_cv2_conv`
output_boundary: corrected int32 output of `/model.2/m.0/cv2/conv/Conv`

## New Boundary

- producer: `/model.2/m.0/cv1/conv/Conv`
- activation: `/model.2/m.0/cv1/act/Sigmoid` + `/model.2/m.0/cv1/act/Mul`
- consumer: `/model.2/m.0/cv2/conv/Conv`
- conv_output_scale: `{branch0_conv_scale}`
- conv_output_zero_point_u8: `{branch0_conv_zp}`
- act_output_scale: `{branch0_act_scale}`
- act_output_zero_point_u8: `{branch0_act_zp}`
- act_input_storage_zero_point_s8: `{branch0_act_zp - 128}`

## Boundary 256-Code LUT Oracle

- onnx_lut_model: `{out_lut_onnx}`
- onnx_lut_model_sha256: `{sha256_file(out_lut_onnx)}`
- mismatches: `{lut_mismatches}`
- max_abs_diff_u8: `{lut_max_abs_diff}`

## Branch Conv2

- node: `/model.2/m.0/cv2/conv/Conv`
- input_shape_nhwc: `[2, 2, 8]` for compact fixture
- output_shape_nhwc: `[2, 2, 16]` for compact fixture
- kernel: `3x3`
- stride: `1x1`
- padding: `1`
- output_scale: `{branch1_scale}`
- output_zero_point_u8: `{branch1_zp}`
- weight_scale_count: `{branch1_weight_scales.size}`
- weight_zero_points: all `0`

## Small Fixture Checksums

- seeded_branch0_act_sum: `{int(seeded_act.astype(np.int64).sum())}`
- seeded_branch1_i32_sum: `{int(seeded_branch1.astype(np.int64).sum())}`
- gradient_branch0_act_sum: `{int(gradient_act.astype(np.int64).sum())}`
- gradient_branch1_i32_sum: `{int(gradient_branch1.astype(np.int64).sum())}`

## Residual Add

Residual Add is not included in the generated Stage 11A fixture. ONNX represents
`/model.2/m.0/Add` as a float-domain Add of
`/model.2/Split_output_1_DequantizeLinear_Output` and
`/model.2/m.0/cv2/act/Mul_output_0`. There is no clean integer Add output
contract before the later Concat Q/DQ boundary in this stage.
"""
    Path(args.out_report).write_text(report)
    print(f"wrote {args.out_header}")
    print(f"wrote {args.out_report}")
    print(f"lut_mismatches={lut_mismatches} max_abs_diff_u8={lut_max_abs_diff}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
