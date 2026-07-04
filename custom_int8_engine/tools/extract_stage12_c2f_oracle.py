#!/usr/bin/env python3
"""Generate Stage 12 compact C2f fixture metadata.

This host-side tool reads the accepted Q/DQ ONNX artifact plus the compact
Stage 7/10/11 fixtures. It emits a small deterministic C++ fixture for:

Stage 11 subset -> /model.2/m.0/cv2 activation -> /model.2/m.0/Add
-> /model.2/Concat -> post-Concat Q/DQ -> /model.2/cv2/conv/Conv
corrected int32 output.
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


def dequantize_u8(code: int, scale: float, zero_point: int) -> float:
    return float(code - zero_point) * scale


def accumulator_to_conv_code(acc: int, input_scale: float, weight_scale: float, conv_scale: float, conv_zp: int) -> int:
    conv_float = float(acc) * input_scale * weight_scale
    return quantize_u8(conv_float, conv_scale, conv_zp)


def accumulator_to_silu_float(
    acc: int,
    input_scale: float,
    weight_scale: float,
    conv_scale: float,
    conv_zp: int,
) -> float:
    code = accumulator_to_conv_code(acc, input_scale, weight_scale, conv_scale, conv_zp)
    x = dequantize_u8(code, conv_scale, conv_zp)
    return silu(x)


def build_merge_micro_model(path: Path, concat_scale: float, concat_zp: int) -> None:
    split0 = helper.make_tensor_value_info("split0", TensorProto.FLOAT, [1, 16, 2, 2])
    split1 = helper.make_tensor_value_info("split1", TensorProto.FLOAT, [1, 16, 2, 2])
    branch = helper.make_tensor_value_info("branch", TensorProto.FLOAT, [1, 16, 2, 2])
    concat_q = helper.make_tensor_value_info("concat_q", TensorProto.UINT8, [1, 48, 2, 2])
    inits = [
        numpy_helper.from_array(np.asarray(concat_scale, dtype=np.float32), "concat_scale"),
        numpy_helper.from_array(np.asarray(concat_zp, dtype=np.uint8), "concat_zp"),
    ]
    nodes = [
        helper.make_node("Add", ["split1", "branch"], ["add"]),
        helper.make_node("Concat", ["split0", "split1", "add"], ["concat"], axis=1),
        helper.make_node("QuantizeLinear", ["concat", "concat_scale", "concat_zp"], ["concat_q"]),
    ]
    graph = helper.make_graph(nodes, "stage12_merge_oracle", [split0, split1, branch], [concat_q], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])
    model.ir_version = 10
    onnx.save(model, path)


def run_merge_micro_model(path: Path, split0: np.ndarray, split1: np.ndarray, branch: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (concat_q,) = session.run(None, {"split0": split0, "split1": split1, "branch": branch})
    return concat_q.astype(np.uint8)


def nhwc_to_nchw(values: np.ndarray, h: int, w: int, c: int) -> np.ndarray:
    return values.reshape(h, w, c).transpose(2, 0, 1).reshape(1, c, h, w).astype(np.float32)


def nchw_to_nhwc(values: np.ndarray) -> np.ndarray:
    return values.reshape(1, values.shape[1], values.shape[2], values.shape[3]).transpose(0, 2, 3, 1).reshape(-1)


def stage9_conv2_split_floats(
    conv2_i32: np.ndarray,
    conv2_weight_scales: np.ndarray,
    act1_scale: float,
    conv2_conv_scale: float,
    conv2_conv_zp: int,
    split_scale: float,
    split_zp: int,
) -> tuple[np.ndarray, np.ndarray]:
    conv2 = conv2_i32.reshape(2, 2, 32)
    split0 = np.zeros((2, 2, 16), dtype=np.float32)
    split1 = np.zeros((2, 2, 16), dtype=np.float32)
    for h in range(2):
        for w in range(2):
            for c in range(32):
                value = accumulator_to_silu_float(
                    int(conv2[h, w, c]), act1_scale, float(conv2_weight_scales[c]), conv2_conv_scale, conv2_conv_zp
                )
                if c < 16:
                    split0[h, w, c] = np.float32(value)
                else:
                    q = quantize_u8(value, split_scale, split_zp)
                    split1[h, w, c - 16] = np.float32(dequantize_u8(q, split_scale, split_zp))
    return split0.reshape(-1), split1.reshape(-1)


def branch1_activation_float(
    branch1_i32: np.ndarray,
    branch0_act_scale: float,
    branch1_weight_scales: np.ndarray,
    branch1_conv_scale: float,
    branch1_conv_zp: int,
) -> np.ndarray:
    values = branch1_i32.reshape(2, 2, 16)
    out = np.zeros((2, 2, 16), dtype=np.float32)
    for h in range(2):
        for w in range(2):
            for c in range(16):
                out[h, w, c] = np.float32(
                    accumulator_to_silu_float(
                        int(values[h, w, c]),
                        branch0_act_scale,
                        float(branch1_weight_scales[c]),
                        branch1_conv_scale,
                        branch1_conv_zp,
                    )
                )
    return out.reshape(-1)


def concat_quantize_internal(
    split0: np.ndarray,
    split1: np.ndarray,
    branch_act: np.ndarray,
    concat_scale: float,
    concat_zp: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    add = split1.astype(np.float32) + branch_act.astype(np.float32)
    concat = np.zeros((2, 2, 48), dtype=np.float32)
    s0 = split0.reshape(2, 2, 16)
    s1 = split1.reshape(2, 2, 16)
    a = add.reshape(2, 2, 16)
    concat[:, :, 0:16] = s0
    concat[:, :, 16:32] = s1
    concat[:, :, 32:48] = a
    concat_q = np.zeros((2, 2, 48), dtype=np.uint8)
    concat_s8 = np.zeros((2, 2, 48), dtype=np.int8)
    for index, value in np.ndenumerate(concat):
        q = quantize_u8(float(value), concat_scale, concat_zp)
        concat_q[index] = q
        concat_s8[index] = signed_storage_from_u8(q)
    return add, concat_q.reshape(-1), concat_s8.reshape(-1)


def conv1x1_corrected(
    input_s8: np.ndarray,
    input_h: int,
    input_w: int,
    weights_ohwi: np.ndarray,
    bias_i32: np.ndarray,
    activation_zp_u8: int,
) -> np.ndarray:
    output_c, _, _, input_c = weights_ohwi.shape
    weight_sums = weights_ohwi.astype(np.int32).sum(axis=(1, 2, 3))
    inp = input_s8.reshape(input_h, input_w, input_c)
    raw = np.zeros((input_h, input_w, output_c), dtype=np.int32)
    for h in range(input_h):
        for w in range(input_w):
            for oc in range(output_c):
                acc = 0
                for ic in range(input_c):
                    acc += int(inp[h, w, ic]) * int(weights_ohwi[oc, 0, 0, ic])
                raw[h, w, oc] = acc
    corrected = (
        raw.astype(np.int64)
        + (128 - activation_zp_u8) * weight_sums.reshape(1, 1, output_c)
        + bias_i32.reshape(1, 1, output_c)
    )
    return corrected.astype(np.int32).reshape(-1)


def c_array(name: str, values: np.ndarray, ctype: str, per_line: int = 12) -> str:
    flat = values.reshape(-1)
    items: list[str] = []
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
    parser.add_argument("--stage11-fixture", required=True)
    parser.add_argument("--out-header", required=True)
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--out-merge-onnx", required=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    model = onnx.load(model_path)
    init = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}
    stage7_text = Path(args.stage7_fixture).read_text()
    stage11_text = Path(args.stage11_fixture).read_text()

    conv2_weight_scales = parse_array(stage7_text, "kConv2WeightScales", float).astype(np.float32)
    seeded_conv2_i32 = parse_array(stage7_text, "kSyntheticSeededExpectedConv2I32Nhwc", int).astype(np.int32)
    gradient_conv2_i32 = parse_array(stage7_text, "kSyntheticGradientExpectedConv2I32Nhwc", int).astype(np.int32)
    branch1_weight_scales = parse_array(stage11_text, "kBranch1WeightScales", float).astype(np.float32)
    seeded_branch1_i32 = parse_array(stage11_text, "kSeededExpectedBranch1I32Nhwc", int).astype(np.int32)
    gradient_branch1_i32 = parse_array(stage11_text, "kGradientExpectedBranch1I32Nhwc", int).astype(np.int32)

    act1_scale = float(init["/model.1/act/Mul_output_0_scale"])
    model2_cv1_conv_scale = float(init["/model.2/cv1/conv/Conv_output_0_scale"])
    model2_cv1_conv_zp = int(init["/model.2/cv1/conv/Conv_output_0_zero_point"])
    split1_scale = float(init["/model.2/Split_output_1_scale"])
    split1_zp = int(init["/model.2/Split_output_1_zero_point"])
    branch0_act_scale = float(init["/model.2/m.0/cv1/act/Mul_output_0_scale"])
    branch1_conv_scale = float(init["/model.2/m.0/cv2/conv/Conv_output_0_scale"])
    branch1_conv_zp = int(init["/model.2/m.0/cv2/conv/Conv_output_0_zero_point"])
    concat_scale = float(init["/model.2/Concat_output_0_scale"])
    concat_zp = int(init["/model.2/Concat_output_0_zero_point"])
    model2_cv2_scale = float(init["/model.2/cv2/conv/Conv_output_0_scale"])
    model2_cv2_zp = int(init["/model.2/cv2/conv/Conv_output_0_zero_point"])
    model2_cv2_weight_scales = init["model.2.cv2.conv.weight_scale"].astype(np.float32)
    model2_cv2_weight_zp = init["model.2.cv2.conv.weight_zero_point"].astype(np.int8)
    model2_cv2_bias = init["model.2.cv2.conv.bias_quantized"].astype(np.int32)
    model2_cv2_weights_oihw = init["model.2.cv2.conv.weight_quantized"].astype(np.int8)
    model2_cv2_weights_ohwi = np.transpose(model2_cv2_weights_oihw, (0, 2, 3, 1)).copy()

    out_merge_onnx = Path(args.out_merge_onnx)
    out_merge_onnx.parent.mkdir(parents=True, exist_ok=True)
    build_merge_micro_model(out_merge_onnx, concat_scale, concat_zp)

    def compute_fixture(conv2_i32: np.ndarray, branch1_i32: np.ndarray):
        split0, split1 = stage9_conv2_split_floats(
            conv2_i32,
            conv2_weight_scales,
            act1_scale,
            model2_cv1_conv_scale,
            model2_cv1_conv_zp,
            split1_scale,
            split1_zp,
        )
        branch_act = branch1_activation_float(
            branch1_i32,
            branch0_act_scale,
            branch1_weight_scales,
            branch1_conv_scale,
            branch1_conv_zp,
        )
        add, concat_q, concat_s8 = concat_quantize_internal(split0, split1, branch_act, concat_scale, concat_zp)
        ort_concat_q = run_merge_micro_model(
            out_merge_onnx,
            nhwc_to_nchw(split0, 2, 2, 16),
            nhwc_to_nchw(split1, 2, 2, 16),
            nhwc_to_nchw(branch_act, 2, 2, 16),
        )
        ort_concat_q_nhwc = nchw_to_nhwc(ort_concat_q).astype(np.uint8)
        concat_mismatches = int(np.count_nonzero(concat_q != ort_concat_q_nhwc))
        concat_max_abs_diff = int(np.max(np.abs(concat_q.astype(np.int16) - ort_concat_q_nhwc.astype(np.int16))))
        output = conv1x1_corrected(concat_s8, 2, 2, model2_cv2_weights_ohwi, model2_cv2_bias, concat_zp)
        return split0, split1, branch_act, add, concat_q, concat_s8, output, concat_mismatches, concat_max_abs_diff

    seeded = compute_fixture(seeded_conv2_i32, seeded_branch1_i32)
    gradient = compute_fixture(gradient_conv2_i32, gradient_branch1_i32)
    concat_mismatches = seeded[7] + gradient[7]
    concat_max_abs_diff = max(seeded[8], gradient[8])

    header = """#pragma once

#include "stage11_branch_block_fixture.h"

#include <cstddef>
#include <cstdint>

namespace y26_stage12_c2f_block_fixture {

struct C2fBlockFixture {
    const char* label;
    const char* subset_id;
    const y26_stage11_branch_block_fixture::BranchBlockFixture* stage11_fixture;
    const char* model2_cv2_node_name;
    Y26Conv2DParams model2_cv2_params;
    int model2_cv2_kernel_h;
    int model2_cv2_kernel_w;
    int concat_output_zero_point_u8;
    int concat_input_storage_zero_point_s8;
    int model2_cv2_output_zero_point_u8;
    float concat_output_scale;
    float model2_cv2_output_scale;
    const float* model2_cv2_weight_scales;
    std::size_t model2_cv2_weight_scale_count;
    const std::int8_t* model2_cv2_weights_ohwi_s8;
    std::size_t model2_cv2_weight_count;
    const std::int32_t* model2_cv2_bias_i32;
    std::size_t model2_cv2_bias_count;
    const std::int8_t* expected_concat_s8_nhwc;
    std::size_t expected_concat_count;
    const std::int32_t* expected_model2_cv2_i32_nhwc;
    std::size_t expected_model2_cv2_count;
};

"""
    header += c_array("kModel2Cv2WeightsOhwiS8", model2_cv2_weights_ohwi, "std::int8_t", 16)
    header += "\n" + c_array("kModel2Cv2BiasI32", model2_cv2_bias, "std::int32_t", 8)
    header += "\n" + c_array("kModel2Cv2WeightScales", model2_cv2_weight_scales, "float", 8)
    header += "\n" + c_array("kSeededExpectedConcatS8Nhwc", seeded[5], "std::int8_t", 16)
    header += "\n" + c_array("kSeededExpectedModel2Cv2I32Nhwc", seeded[6], "std::int32_t", 8)
    header += "\n" + c_array("kGradientExpectedConcatS8Nhwc", gradient[5], "std::int8_t", 16)
    header += "\n" + c_array("kGradientExpectedModel2Cv2I32Nhwc", gradient[6], "std::int32_t", 8)
    header += f"""
inline constexpr C2fBlockFixture kSyntheticSeededFixture = {{
    "synthetic_seeded",
    "candidate_G_model2_c2f_add_concat_cv2_conv",
    &y26_stage11_branch_block_fixture::kSyntheticSeededFixture,
    "/model.2/cv2/conv/Conv",
    Y26Conv2DParams{{2, 2, 48, 64, 1, 1, 0, 0}},
    1,
    1,
    {concat_zp},
    {concat_zp - 128},
    {model2_cv2_zp},
    {concat_scale:.12g}f,
    {model2_cv2_scale:.12g}f,
    kModel2Cv2WeightScales,
    sizeof(kModel2Cv2WeightScales) / sizeof(kModel2Cv2WeightScales[0]),
    kModel2Cv2WeightsOhwiS8,
    sizeof(kModel2Cv2WeightsOhwiS8) / sizeof(kModel2Cv2WeightsOhwiS8[0]),
    kModel2Cv2BiasI32,
    sizeof(kModel2Cv2BiasI32) / sizeof(kModel2Cv2BiasI32[0]),
    kSeededExpectedConcatS8Nhwc,
    sizeof(kSeededExpectedConcatS8Nhwc) / sizeof(kSeededExpectedConcatS8Nhwc[0]),
    kSeededExpectedModel2Cv2I32Nhwc,
    sizeof(kSeededExpectedModel2Cv2I32Nhwc) / sizeof(kSeededExpectedModel2Cv2I32Nhwc[0]),
}};

inline constexpr C2fBlockFixture kSyntheticGradientFixture = {{
    "synthetic_gradient",
    "candidate_G_model2_c2f_add_concat_cv2_conv",
    &y26_stage11_branch_block_fixture::kSyntheticGradientFixture,
    "/model.2/cv2/conv/Conv",
    Y26Conv2DParams{{2, 2, 48, 64, 1, 1, 0, 0}},
    1,
    1,
    {concat_zp},
    {concat_zp - 128},
    {model2_cv2_zp},
    {concat_scale:.12g}f,
    {model2_cv2_scale:.12g}f,
    kModel2Cv2WeightScales,
    sizeof(kModel2Cv2WeightScales) / sizeof(kModel2Cv2WeightScales[0]),
    kModel2Cv2WeightsOhwiS8,
    sizeof(kModel2Cv2WeightsOhwiS8) / sizeof(kModel2Cv2WeightsOhwiS8[0]),
    kModel2Cv2BiasI32,
    sizeof(kModel2Cv2BiasI32) / sizeof(kModel2Cv2BiasI32[0]),
    kGradientExpectedConcatS8Nhwc,
    sizeof(kGradientExpectedConcatS8Nhwc) / sizeof(kGradientExpectedConcatS8Nhwc[0]),
    kGradientExpectedModel2Cv2I32Nhwc,
    sizeof(kGradientExpectedModel2Cv2I32Nhwc) / sizeof(kGradientExpectedModel2Cv2I32Nhwc[0]),
}};

inline constexpr const C2fBlockFixture* kFixtures[] = {{
    &kSyntheticSeededFixture,
    &kSyntheticGradientFixture,
}};

}}  // namespace y26_stage12_c2f_block_fixture
"""
    Path(args.out_header).write_text(header)

    report = f"""# Stage 12 C2f Oracle Report

model: `{args.model}`
model_sha256: `{sha256_file(model_path)}`
selected_subset: `candidate_G_model2_c2f_add_concat_cv2_conv`
output_boundary: corrected int32 output of `/model.2/cv2/conv/Conv`

## Float-Domain Merge Contract

- Add node: `/model.2/m.0/Add`
- Add inputs:
  - `/model.2/Split_output_1_DequantizeLinear_Output`
  - `/model.2/m.0/cv2/act/Mul_output_0`
- Add output: `/model.2/m.0/Add_output_0`
- Concat node: `/model.2/Concat`
- Concat inputs:
  - `/model.2/Split_output_0`
  - `/model.2/Split_output_1_DequantizeLinear_Output`
  - `/model.2/m.0/Add_output_0`
- Concat output: `/model.2/Concat_output_0`
- post-Concat Q/DQ scale: `{concat_scale}`
- post-Concat Q/DQ zero_point_u8: `{concat_zp}`

## Model2 Cv2 Conv

- node: `/model.2/cv2/conv/Conv`
- compact input_shape_nhwc: `[2, 2, 48]`
- compact output_shape_nhwc: `[2, 2, 64]`
- kernel: `1x1`
- stride: `1x1`
- padding: `0`
- output_scale: `{model2_cv2_scale}`
- output_zero_point_u8: `{model2_cv2_zp}`
- weight_shape_oihw: `{list(model2_cv2_weights_oihw.shape)}`
- weight_scale_count: `{model2_cv2_weight_scales.size}`
- weight_zero_points_all_zero: `{bool(np.all(model2_cv2_weight_zp == 0))}`

## Tensor Oracle

- merge_micro_onnx: `{out_merge_onnx}`
- merge_micro_onnx_sha256: `{sha256_file(out_merge_onnx)}`
- concat_q_mismatches_against_ort: `{concat_mismatches}`
- concat_q_max_abs_diff_u8: `{concat_max_abs_diff}`

## Small Fixture Checksums

- seeded_add_float_sum: `{float(seeded[3].astype(np.float64).sum()):.12g}`
- seeded_concat_s8_sum: `{int(seeded[5].astype(np.int64).sum())}`
- seeded_model2_cv2_i32_sum: `{int(seeded[6].astype(np.int64).sum())}`
- gradient_add_float_sum: `{float(gradient[3].astype(np.float64).sum()):.12g}`
- gradient_concat_s8_sum: `{int(gradient[5].astype(np.int64).sum())}`
- gradient_model2_cv2_i32_sum: `{int(gradient[6].astype(np.int64).sum())}`

## Decision

The Add and Concat contract is float-domain in the accepted Q/DQ graph. Stage 12
therefore uses an explicit measured float fallback for Add/Concat and a post-Concat
Q/DQ handoff before `/model.2/cv2/conv/Conv`. No integer-domain Add shortcut is
accepted by this oracle.
"""
    Path(args.out_report).write_text(report)
    print(f"wrote {args.out_header}")
    print(f"wrote {args.out_report}")
    print(f"concat_mismatches={concat_mismatches} concat_max_abs_diff={concat_max_abs_diff}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
