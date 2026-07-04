#!/usr/bin/env python3
"""Generate Stage 10 narrow backbone fixture metadata.

This tool is host-side only. It reads the accepted Q/DQ ONNX artifact and the
small Stage 7 fixture header, then emits a compact Stage 10 C++ fixture for the
selected boundary:

Stage 9 subset -> /model.2/cv1/act/Sigmoid+Mul -> /model.2/Split output_1
-> /model.2/m.0/cv1/conv/Conv corrected int32 output.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def parse_array(text: str, name: str, dtype) -> np.ndarray:
    pattern = re.compile(rf"inline constexpr [^=]+ {re.escape(name)}\[\] = \{{(.*?)\}};", re.S)
    match = pattern.search(text)
    if not match:
        raise KeyError(f"array not found: {name}")
    values = [item.strip() for item in match.group(1).replace("\n", " ").split(",") if item.strip()]
    return np.asarray([dtype(v.rstrip("f")) for v in values])


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


def build_lut(conv_scale: float, conv_zp: int, act_scale: float, act_zp: int) -> np.ndarray:
    out = np.zeros(256, dtype=np.int8)
    for q in range(256):
        x = float(q - conv_zp) * conv_scale
        out[q] = signed_storage_from_u8(quantize_u8(silu(x), act_scale, act_zp))
    return out


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
    lut = build_lut(conv_scale, conv_zp, act_scale, act_zp)
    out = np.zeros(values.size, dtype=np.int8)
    for i, acc in enumerate(values.astype(np.int64)):
        c = i % channels
        conv_q = clamp_u8(round_even(float(acc) * input_scale * float(weight_scales[c]) / conv_scale) + conv_zp)
        out[i] = lut[conv_q]
    return out


def branch_conv_corrected(
    split_s8: np.ndarray,
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
    inp = split_s8.reshape(input_h, input_w, input_c)
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
    corrected = raw.astype(np.int64) + (128 - activation_zp_u8) * weight_sums.reshape(1, 1, output_c) + bias_i32.reshape(1, 1, output_c)
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
    parser.add_argument("--stage7-fixture", required=True)
    parser.add_argument("--out-header", required=True)
    parser.add_argument("--out-report", required=True)
    args = parser.parse_args()

    model = onnx.load(args.model)
    init = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}
    stage7_text = Path(args.stage7_fixture).read_text()

    conv2_weight_scales = parse_array(stage7_text, "kConv2WeightScales", float).astype(np.float32)
    seeded_conv2_i32 = parse_array(stage7_text, "kSyntheticSeededExpectedConv2I32Nhwc", int).astype(np.int32)
    gradient_conv2_i32 = parse_array(stage7_text, "kSyntheticGradientExpectedConv2I32Nhwc", int).astype(np.int32)

    conv2_scale = float(init["/model.2/cv1/conv/Conv_output_0_scale"])
    conv2_zp = int(init["/model.2/cv1/conv/Conv_output_0_zero_point"])
    split1_scale = float(init["/model.2/Split_output_1_scale"])
    split1_zp = int(init["/model.2/Split_output_1_zero_point"])
    branch_scale = float(init["/model.2/m.0/cv1/conv/Conv_output_0_scale"])
    branch_zp = int(init["/model.2/m.0/cv1/conv/Conv_output_0_zero_point"])
    branch_weight_scales = init["model.2.m.0.cv1.conv.weight_scale"].astype(np.float32)
    branch_bias = init["model.2.m.0.cv1.conv.bias_quantized"].astype(np.int32)
    weights_oihw = init["model.2.m.0.cv1.conv.weight_quantized"].astype(np.int8)
    weights_ohwi = np.transpose(weights_oihw, (0, 2, 3, 1)).copy()

    def compute_fixture(conv2_i32: np.ndarray):
        act = activation_from_i32(
            conv2_i32,
            32,
            0.582500994205,
            conv2_weight_scales,
            conv2_scale,
            conv2_zp,
            split1_scale,
            split1_zp,
        )
        act_nhwc = act.reshape(2, 2, 32)
        split = act_nhwc[:, :, 16:32].copy().reshape(-1)
        branch = branch_conv_corrected(split, 2, 2, weights_ohwi, branch_bias, split1_zp)
        return act, split, branch

    seeded_act, seeded_split, seeded_branch = compute_fixture(seeded_conv2_i32)
    gradient_act, gradient_split, gradient_branch = compute_fixture(gradient_conv2_i32)

    header = """#pragma once

#include "stage7_backbone_subset_fixture.h"

#include <cstddef>
#include <cstdint>

namespace y26_stage10_backbone_expansion_fixture {

struct BackboneExpansionFixture {
    const char* label;
    const char* subset_id;
    const y26_stage7_backbone_subset_fixture::BackboneSubsetFixture* stage9_fixture;
    const char* branch0_node_name;
    Y26Conv2DParams branch0_params;
    int branch0_kernel_h;
    int branch0_kernel_w;
    int branch0_activation_zero_point_u8;
    int branch0_input_storage_zero_point_s8;
    int branch0_output_zero_point_u8;
    float split_output1_scale;
    float branch0_output_scale;
    const float* branch0_weight_scales;
    std::size_t branch0_weight_scale_count;
    const std::int8_t* branch0_weights_ohwi_s8;
    std::size_t branch0_weight_count;
    const std::int32_t* branch0_bias_i32;
    std::size_t branch0_bias_count;
    const std::int8_t* expected_conv2_act_s8_nhwc;
    std::size_t expected_conv2_act_count;
    const std::int8_t* expected_split_output1_s8_nhwc;
    std::size_t expected_split_output1_count;
    const std::int32_t* expected_branch0_i32_nhwc;
    std::size_t expected_branch0_count;
};

"""
    header += c_array("kBranch0WeightsOhwiS8", weights_ohwi, "std::int8_t", 16)
    header += "\n" + c_array("kBranch0BiasI32", branch_bias, "std::int32_t", 8)
    header += "\n" + c_array("kBranch0WeightScales", branch_weight_scales, "float", 8)
    header += "\n" + c_array("kSeededExpectedConv2ActS8Nhwc", seeded_act, "std::int8_t", 16)
    header += "\n" + c_array("kSeededExpectedSplitOutput1S8Nhwc", seeded_split, "std::int8_t", 16)
    header += "\n" + c_array("kSeededExpectedBranch0I32Nhwc", seeded_branch, "std::int32_t", 8)
    header += "\n" + c_array("kGradientExpectedConv2ActS8Nhwc", gradient_act, "std::int8_t", 16)
    header += "\n" + c_array("kGradientExpectedSplitOutput1S8Nhwc", gradient_split, "std::int8_t", 16)
    header += "\n" + c_array("kGradientExpectedBranch0I32Nhwc", gradient_branch, "std::int32_t", 8)
    header += f"""
inline constexpr BackboneExpansionFixture kSyntheticSeededFixture = {{
    "synthetic_seeded",
    "candidate_E_branch1_stage9_split_model2_m0_cv1_conv",
    &y26_stage7_backbone_subset_fixture::kSyntheticSeededFixture,
    "/model.2/m.0/cv1/conv/Conv",
    Y26Conv2DParams{{2, 2, 16, 8, 1, 1, 1, 1}},
    3,
    3,
    {split1_zp},
    {split1_zp - 128},
    {branch_zp},
    {split1_scale:.12g}f,
    {branch_scale:.12g}f,
    kBranch0WeightScales,
    sizeof(kBranch0WeightScales) / sizeof(kBranch0WeightScales[0]),
    kBranch0WeightsOhwiS8,
    sizeof(kBranch0WeightsOhwiS8) / sizeof(kBranch0WeightsOhwiS8[0]),
    kBranch0BiasI32,
    sizeof(kBranch0BiasI32) / sizeof(kBranch0BiasI32[0]),
    kSeededExpectedConv2ActS8Nhwc,
    sizeof(kSeededExpectedConv2ActS8Nhwc) / sizeof(kSeededExpectedConv2ActS8Nhwc[0]),
    kSeededExpectedSplitOutput1S8Nhwc,
    sizeof(kSeededExpectedSplitOutput1S8Nhwc) / sizeof(kSeededExpectedSplitOutput1S8Nhwc[0]),
    kSeededExpectedBranch0I32Nhwc,
    sizeof(kSeededExpectedBranch0I32Nhwc) / sizeof(kSeededExpectedBranch0I32Nhwc[0]),
}};

inline constexpr BackboneExpansionFixture kSyntheticGradientFixture = {{
    "synthetic_gradient",
    "candidate_E_branch1_stage9_split_model2_m0_cv1_conv",
    &y26_stage7_backbone_subset_fixture::kSyntheticGradientFixture,
    "/model.2/m.0/cv1/conv/Conv",
    Y26Conv2DParams{{2, 2, 16, 8, 1, 1, 1, 1}},
    3,
    3,
    {split1_zp},
    {split1_zp - 128},
    {branch_zp},
    {split1_scale:.12g}f,
    {branch_scale:.12g}f,
    kBranch0WeightScales,
    sizeof(kBranch0WeightScales) / sizeof(kBranch0WeightScales[0]),
    kBranch0WeightsOhwiS8,
    sizeof(kBranch0WeightsOhwiS8) / sizeof(kBranch0WeightsOhwiS8[0]),
    kBranch0BiasI32,
    sizeof(kBranch0BiasI32) / sizeof(kBranch0BiasI32[0]),
    kGradientExpectedConv2ActS8Nhwc,
    sizeof(kGradientExpectedConv2ActS8Nhwc) / sizeof(kGradientExpectedConv2ActS8Nhwc[0]),
    kGradientExpectedSplitOutput1S8Nhwc,
    sizeof(kGradientExpectedSplitOutput1S8Nhwc) / sizeof(kGradientExpectedSplitOutput1S8Nhwc[0]),
    kGradientExpectedBranch0I32Nhwc,
    sizeof(kGradientExpectedBranch0I32Nhwc) / sizeof(kGradientExpectedBranch0I32Nhwc[0]),
}};

inline constexpr const BackboneExpansionFixture* kFixtures[] = {{
    &kSyntheticSeededFixture,
    &kSyntheticGradientFixture,
}};

}}  // namespace y26_stage10_backbone_expansion_fixture
"""
    Path(args.out_header).write_text(header)

    report = f"""# Stage 10 Fixture Oracle Metadata

model: `{args.model}`
selected_subset: `candidate_E_branch1_stage9_split_model2_m0_cv1_conv`
selected_new_boundary: `/model.2/Split` output 1 -> `/model.2/m.0/cv1/conv/Conv`

## Quantization

- conv2_output_scale: `{conv2_scale}`
- conv2_output_zero_point: `{conv2_zp}`
- split_output1_scale: `{split1_scale}`
- split_output1_zero_point: `{split1_zp}`
- branch0_output_scale: `{branch_scale}`
- branch0_output_zero_point: `{branch_zp}`
- branch0_weight_scale_count: `{branch_weight_scales.size}`
- branch0_weight_zero_points: all `0`

## Small Fixture Checksums

- seeded_conv2_act_sum: `{int(seeded_act.astype(np.int64).sum())}`
- seeded_split_output1_sum: `{int(seeded_split.astype(np.int64).sum())}`
- seeded_branch0_i32_sum: `{int(seeded_branch.astype(np.int64).sum())}`
- gradient_conv2_act_sum: `{int(gradient_act.astype(np.int64).sum())}`
- gradient_split_output1_sum: `{int(gradient_split.astype(np.int64).sum())}`
- gradient_branch0_i32_sum: `{int(gradient_branch.astype(np.int64).sum())}`

## Scope

The generated fixture is a compact deterministic scalar oracle using real ONNX
weights, scales, zero-points, and Stage 7 Conv2 corrected-int32 fixture tensors.
Large full-shape tensor dumps are intentionally not committed.
"""
    Path(args.out_report).write_text(report)
    print(f"wrote {args.out_header}")
    print(f"wrote {args.out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
