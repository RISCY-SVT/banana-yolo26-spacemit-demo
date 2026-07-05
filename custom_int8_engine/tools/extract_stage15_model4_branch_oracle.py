#!/usr/bin/env python3
"""Generate compact Stage 15 model4 branch-entry fixtures.

This host-side tool extends the Stage 14 compact fixture through:

  /model.4/cv1 activation
  /model.4/Split_output_1 Q/DQ
  /model.4/m.0/cv1/conv corrected int32
  /model.4/m.0/cv1 activation Q/DQ

The runtime library does not depend on ONNX, ONNX Runtime, Python, or protobuf.
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
    graph = helper.make_graph(nodes, "stage15_activation_micro_oracle", [x], [y], [scale0, zp0, scale1, zp1])
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


def write_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--stage14-fixture", default="custom_int8_engine/tests/stage14_next_c2f_fixture.h")
    parser.add_argument("--out-header", default="custom_int8_engine/tests/stage15_model4_branch_fixture.h")
    parser.add_argument("--out-oracle-report", required=True)
    parser.add_argument("--out-split-report", required=True)
    parser.add_argument("--out-lut-report", required=True)
    parser.add_argument("--out-scale-report", required=True)
    parser.add_argument("--out-micro-dir", required=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    model = onnx.load(model_path)
    init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    stage14_text = Path(args.stage14_fixture).read_text()

    model4_cv1_input_scale = float(init["/model.3/act/Mul_output_0_scale"])
    model4_cv1_output_scale = float(init["/model.4/cv1/conv/Conv_output_0_scale"])
    model4_cv1_output_zp = int(init["/model.4/cv1/conv/Conv_output_0_zero_point"])
    model4_cv1_weight_scales = init["model.4.cv1.conv.weight_scale"].astype(np.float32)
    split1_scale = float(init["/model.4/Split_output_1_scale"])
    split1_zp = int(init["/model.4/Split_output_1_zero_point"])

    branch0_output_scale = float(init["/model.4/m.0/cv1/conv/Conv_output_0_scale"])
    branch0_output_zp = int(init["/model.4/m.0/cv1/conv/Conv_output_0_zero_point"])
    branch0_weight_scales = init["model.4.m.0.cv1.conv.weight_scale"].astype(np.float32)
    branch0_weight_zp = init["model.4.m.0.cv1.conv.weight_zero_point"].astype(np.int8)
    branch0_bias = init["model.4.m.0.cv1.conv.bias_quantized"].astype(np.int32)
    branch0_weights_oihw = init["model.4.m.0.cv1.conv.weight_quantized"].astype(np.int8)
    branch0_weights_ohwi = np.transpose(branch0_weights_oihw, (0, 2, 3, 1)).copy()
    branch0_act_scale = float(init["/model.4/m.0/cv1/act/Mul_output_0_scale"])
    branch0_act_zp = int(init["/model.4/m.0/cv1/act/Mul_output_0_zero_point"])

    if np.any(branch0_weight_zp != 0):
        raise RuntimeError("Stage15 branch0 requires symmetric signed weights with zero-point 0")

    seeded_model4_cv1_i32 = parse_array(stage14_text, "kSeededExpectedModel4Cv1I32Nhwc", int).astype(np.int32)
    gradient_model4_cv1_i32 = parse_array(stage14_text, "kGradientExpectedModel4Cv1I32Nhwc", int).astype(np.int32)

    def make_fixture(prefix: str, model4_cv1_i32: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        model4_cv1_act_all = activation_s8_from_i32(
            model4_cv1_i32,
            model4_cv1_input_scale,
            model4_cv1_weight_scales,
            model4_cv1_output_scale,
            model4_cv1_output_zp,
            split1_scale,
            split1_zp,
        ).reshape(-1, 64)
        split1_s8 = model4_cv1_act_all[:, 32:64].reshape(-1).astype(np.int8)
        branch0_i32 = conv_corrected_nhwc(
            split1_s8,
            1,
            1,
            branch0_weights_ohwi,
            branch0_bias,
            split1_zp,
            stride=1,
            pad=1,
        )
        branch0_act_s8 = activation_s8_from_i32(
            branch0_i32,
            split1_scale,
            branch0_weight_scales,
            branch0_output_scale,
            branch0_output_zp,
            branch0_act_scale,
            branch0_act_zp,
        ).astype(np.int8)
        if split1_s8.size != 32 or branch0_i32.size != 16 or branch0_act_s8.size != 16:
            raise RuntimeError(f"unexpected compact fixture size for {prefix}")
        return split1_s8, branch0_i32, branch0_act_s8

    seeded_split1, seeded_branch0_i32, seeded_branch0_act = make_fixture("seeded", seeded_model4_cv1_i32)
    gradient_split1, gradient_branch0_i32, gradient_branch0_act = make_fixture("gradient", gradient_model4_cv1_i32)

    micro_dir = Path(args.out_micro_dir)
    micro_dir.mkdir(parents=True, exist_ok=True)
    split_micro = micro_dir / "stage15_model4_cv1_to_split1_activation.onnx"
    branch_micro = micro_dir / "stage15_model4_m0_cv1_activation.onnx"
    build_activation_micro_model(split_micro, model4_cv1_output_scale, model4_cv1_output_zp, split1_scale, split1_zp)
    build_activation_micro_model(branch_micro, branch0_output_scale, branch0_output_zp, branch0_act_scale, branch0_act_zp)
    split_ort = run_activation_micro_model(split_micro)
    branch_ort = run_activation_micro_model(branch_micro)
    split_internal = activation_lut_internal(model4_cv1_output_scale, model4_cv1_output_zp, split1_scale, split1_zp)
    branch_internal = activation_lut_internal(branch0_output_scale, branch0_output_zp, branch0_act_scale, branch0_act_zp)
    split_lut_mismatches = int(np.count_nonzero(split_ort != split_internal))
    branch_lut_mismatches = int(np.count_nonzero(branch_ort != branch_internal))

    header_parts = [
        "#pragma once\n",
        '#include "stage14_next_c2f_fixture.h"\n\n',
        "#include <cstddef>\n#include <cstdint>\n\n",
        "namespace y26_stage15_model4_branch_fixture {\n\n",
        "struct Model4BranchFixture {\n"
        "    const char* label;\n"
        "    const char* subset_id;\n"
        "    const y26_stage14_next_c2f_fixture::NextC2fFixture* stage14_fixture;\n"
        "    const char* branch0_node_name;\n"
        "    Y26Conv2DParams branch0_params;\n"
        "    int branch0_kernel_h;\n"
        "    int branch0_kernel_w;\n"
        "    int branch0_activation_zero_point_u8;\n"
        "    int branch0_input_storage_zero_point_s8;\n"
        "    int branch0_output_zero_point_u8;\n"
        "    float split1_output_scale;\n"
        "    int split1_output_zero_point_u8;\n"
        "    float branch0_output_scale;\n"
        "    float branch0_act_output_scale;\n"
        "    int branch0_act_output_zero_point_u8;\n"
        "    const float* branch0_weight_scales;\n"
        "    std::size_t branch0_weight_scale_count;\n"
        "    const std::int8_t* branch0_weights_ohwi_s8;\n"
        "    std::size_t branch0_weight_count;\n"
        "    const std::int32_t* branch0_bias_i32;\n"
        "    std::size_t branch0_bias_count;\n"
        "    const std::int8_t* expected_split1_input_s8_nhwc;\n"
        "    std::size_t expected_split1_input_count;\n"
        "    const std::int32_t* expected_branch0_i32_nhwc;\n"
        "    std::size_t expected_branch0_count;\n"
        "    const std::int8_t* expected_branch0_act_s8_nhwc;\n"
        "    std::size_t expected_branch0_act_count;\n"
        "};\n\n",
        c_array("kBranch0WeightScales", branch0_weight_scales, "float"),
        "\n\n",
        c_array("kBranch0WeightsOhwiS8", branch0_weights_ohwi, "std::int8_t", 16),
        "\n\n",
        c_array("kBranch0BiasI32", branch0_bias, "std::int32_t", 8),
        "\n\n",
        c_array("kSeededExpectedSplit1InputS8Nhwc", seeded_split1, "std::int8_t", 16),
        "\n\n",
        c_array("kSeededExpectedBranch0I32Nhwc", seeded_branch0_i32, "std::int32_t", 8),
        "\n\n",
        c_array("kSeededExpectedBranch0ActS8Nhwc", seeded_branch0_act, "std::int8_t", 16),
        "\n\n",
        c_array("kGradientExpectedSplit1InputS8Nhwc", gradient_split1, "std::int8_t", 16),
        "\n\n",
        c_array("kGradientExpectedBranch0I32Nhwc", gradient_branch0_i32, "std::int32_t", 8),
        "\n\n",
        c_array("kGradientExpectedBranch0ActS8Nhwc", gradient_branch0_act, "std::int8_t", 16),
        "\n\n",
    ]
    branch_fields = (
        '    "/model.4/m.0/cv1/conv/Conv",\n'
        "    Y26Conv2DParams{1, 1, 32, 16, 1, 1, 1, 1},\n"
        "    3,\n"
        "    3,\n"
        f"    {split1_zp},\n"
        f"    {split1_zp - 128},\n"
        f"    {branch0_output_zp},\n"
        f"    {split1_scale:.12g}f,\n"
        f"    {split1_zp},\n"
        f"    {branch0_output_scale:.12g}f,\n"
        f"    {branch0_act_scale:.12g}f,\n"
        f"    {branch0_act_zp},\n"
        "    kBranch0WeightScales,\n"
        "    sizeof(kBranch0WeightScales) / sizeof(kBranch0WeightScales[0]),\n"
        "    kBranch0WeightsOhwiS8,\n"
        "    sizeof(kBranch0WeightsOhwiS8) / sizeof(kBranch0WeightsOhwiS8[0]),\n"
        "    kBranch0BiasI32,\n"
        "    sizeof(kBranch0BiasI32) / sizeof(kBranch0BiasI32[0]),\n"
    )
    header_parts.extend(
        [
            "inline constexpr Model4BranchFixture kSyntheticSeededFixture = {\n"
            '    "synthetic_seeded",\n'
            '    "candidate_I_model4_split_first_branch",\n'
            "    &y26_stage14_next_c2f_fixture::kSyntheticSeededFixture,\n"
            + branch_fields +
            "    kSeededExpectedSplit1InputS8Nhwc,\n"
            "    sizeof(kSeededExpectedSplit1InputS8Nhwc) / sizeof(kSeededExpectedSplit1InputS8Nhwc[0]),\n"
            "    kSeededExpectedBranch0I32Nhwc,\n"
            "    sizeof(kSeededExpectedBranch0I32Nhwc) / sizeof(kSeededExpectedBranch0I32Nhwc[0]),\n"
            "    kSeededExpectedBranch0ActS8Nhwc,\n"
            "    sizeof(kSeededExpectedBranch0ActS8Nhwc) / sizeof(kSeededExpectedBranch0ActS8Nhwc[0]),\n"
            "};\n\n",
            "inline constexpr Model4BranchFixture kSyntheticGradientFixture = {\n"
            '    "synthetic_gradient",\n'
            '    "candidate_I_model4_split_first_branch",\n'
            "    &y26_stage14_next_c2f_fixture::kSyntheticGradientFixture,\n"
            + branch_fields +
            "    kGradientExpectedSplit1InputS8Nhwc,\n"
            "    sizeof(kGradientExpectedSplit1InputS8Nhwc) / sizeof(kGradientExpectedSplit1InputS8Nhwc[0]),\n"
            "    kGradientExpectedBranch0I32Nhwc,\n"
            "    sizeof(kGradientExpectedBranch0I32Nhwc) / sizeof(kGradientExpectedBranch0I32Nhwc[0]),\n"
            "    kGradientExpectedBranch0ActS8Nhwc,\n"
            "    sizeof(kGradientExpectedBranch0ActS8Nhwc) / sizeof(kGradientExpectedBranch0ActS8Nhwc[0]),\n"
            "};\n\n",
            "inline constexpr const Model4BranchFixture* kFixtures[] = {\n"
            "    &kSyntheticSeededFixture,\n"
            "    &kSyntheticGradientFixture,\n"
            "};\n\n",
            "}  // namespace y26_stage15_model4_branch_fixture\n",
        ]
    )
    out_header = Path(args.out_header)
    out_header.write_text("".join(header_parts))

    oracle_report = f"""# Model4 Branch0 Oracle Report

model_path: `{model_path}`
model_sha256: `{sha256_file(model_path)}`
provider: `CPUExecutionProvider` for activation micro-oracles
selected_subset: `candidate_I_model4_split_first_branch`

## Compact Fixtures

| fixture | split1_count | branch0_i32_count | branch0_act_count | split1_checksum | branch0_i32_checksum | branch0_act_checksum |
|---|---:|---:|---:|---:|---:|---:|
| `synthetic_seeded` | `{seeded_split1.size}` | `{seeded_branch0_i32.size}` | `{seeded_branch0_act.size}` | `{int(seeded_split1.astype(np.int64).sum())}` | `{int(seeded_branch0_i32.astype(np.int64).sum())}` | `{int(seeded_branch0_act.astype(np.int64).sum())}` |
| `synthetic_gradient` | `{gradient_split1.size}` | `{gradient_branch0_i32.size}` | `{gradient_branch0_act.size}` | `{int(gradient_split1.astype(np.int64).sum())}` | `{int(gradient_branch0_i32.astype(np.int64).sum())}` | `{int(gradient_branch0_act.astype(np.int64).sum())}` |

## Conv Oracle

`/model.4/m.0/cv1/conv/Conv` compact oracle uses accepted ONNX quantized weights, per-channel weight scales, quantized bias, and the Stage 0-14 signed-storage correction formula.

The compact branch Conv shape is `1x1x32 -> 1x1x16`, kernel `3x3`, stride `1`, padding `1`.
"""
    write_report(Path(args.out_oracle_report), oracle_report)

    split_report = f"""# Model4 Split Contract Report

selected_subset: `candidate_I_model4_split_first_branch`

`/model.4/Split` consumes `/model.4/cv1/act/Mul_output_0` in float domain.

Split attributes:

- axis: `1`
- output0: `/model.4/Split_output_0`, shape `[1,32,80,80]`
- output1: `/model.4/Split_output_1`, shape `[1,32,80,80]`

Stage 15 uses only `/model.4/Split_output_1` for the first branch Conv.

`/model.4/Split_output_1` Q/DQ:

- scale: `{split1_scale:.12g}`
- zero_point_u8: `{split1_zp}`
- signed_storage_zero_point_s8: `{split1_zp - 128}`

`/model.4/Split_output_0` is deferred for future `/model.4/Concat`.
"""
    write_report(Path(args.out_split_report), split_report)

    lut_report = f"""# Stage 15 Boundary LUT Oracle Report

provider: `CPUExecutionProvider`

| boundary | micro_model | micro_model_sha256 | mismatches | max_abs_diff_u8 |
|---|---|---:|---:|---:|
| `/model.4/cv1/conv` -> `/model.4/Split_output_1` | `{split_micro}` | `{sha256_file(split_micro)}` | `{split_lut_mismatches}` | `{int(np.max(np.abs(split_ort.astype(np.int16) - split_internal.astype(np.int16))))}` |
| `/model.4/m.0/cv1/conv` -> `/model.4/m.0/cv1/act` | `{branch_micro}` | `{sha256_file(branch_micro)}` | `{branch_lut_mismatches}` | `{int(np.max(np.abs(branch_ort.astype(np.int16) - branch_internal.astype(np.int16))))}` |

Both accepted boundaries require `mismatches=0` and `max_abs_diff_u8=0`.
"""
    write_report(Path(args.out_lut_report), lut_report)

    scale_report = f"""# Stage 15 Scale Zero Point Report

| tensor boundary | scale | zero_point_u8 | signed_storage_zero_point_s8 | note |
|---|---:|---:|---:|---|
| `/model.4/cv1/conv/Conv_output_0` | `{model4_cv1_output_scale:.12g}` | `{model4_cv1_output_zp}` | n/a | conv output code before SiLU |
| `/model.4/Split_output_1` | `{split1_scale:.12g}` | `{split1_zp}` | `{split1_zp - 128}` | first branch Conv input |
| `/model.4/m.0/cv1/conv/Conv_output_0` | `{branch0_output_scale:.12g}` | `{branch0_output_zp}` | n/a | branch Conv output code before SiLU |
| `/model.4/m.0/cv1/act/Mul_output_0` | `{branch0_act_scale:.12g}` | `{branch0_act_zp}` | `{branch0_act_zp - 128}` | Stage 15 branch activation output |

Branch Conv weights:

- tensor: `model.4.m.0.cv1.conv.weight_quantized`
- layout in fixture: `OHWI`
- weight_zero_point: all `0`
- weight_scale_count: `{branch0_weight_scales.size}`
"""
    write_report(Path(args.out_scale_report), scale_report)

    print(f"wrote_header={out_header}")
    print(f"header_sha256={sha256_file(out_header)}")
    print(f"split_lut_mismatches={split_lut_mismatches}")
    print(f"branch_lut_mismatches={branch_lut_mismatches}")
    return 0 if split_lut_mismatches == 0 and branch_lut_mismatches == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
