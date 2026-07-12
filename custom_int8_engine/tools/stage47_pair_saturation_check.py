#!/usr/bin/env python3
"""Reconstruct one host-ORT/custom Conv mismatch with and without U8S8 pair saturation."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def rne(value: float) -> int:
    base = math.floor(value)
    fraction = value - base
    return base + int(fraction > 0.5 or (fraction == 0.5 and base % 2 != 0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--fixture", default="F2")
    parser.add_argument("--operation", type=int, default=3)
    parser.add_argument("--channel", type=int, default=25)
    parser.add_argument("--spatial-index", type=int, default=1194)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()

    with (args.package / "ops.tsv").open(newline="", encoding="utf-8") as stream:
        operation = list(csv.DictReader(stream, delimiter="\t"))[args.operation]
    with (args.package / "tensors.tsv").open(newline="", encoding="utf-8") as stream:
        tensors = {int(row["id"]): row for row in csv.DictReader(stream, delimiter="\t")}

    input_id = int(operation["input0"])
    output_id = int(operation["output0"])
    input_tensor = tensors[input_id]
    output_tensor = tensors[output_id]
    input_c = int(input_tensor["c"])
    input_h = int(input_tensor["h"])
    input_w = int(input_tensor["w"])
    output_c = int(output_tensor["c"])
    kernel_h = int(operation["kernel_h"])
    kernel_w = int(operation["kernel_w"])

    source = np.fromfile(
        args.package / "oracles" / args.fixture / f"tensor_{input_id:03d}_nchw_u8.bin",
        dtype=np.uint8,
    ).reshape(1, input_c, input_h, input_w)
    weights = np.fromfile(args.package / operation["weights_file"], dtype=np.int8).reshape(
        output_c, kernel_h, kernel_w, input_c
    )
    bias = np.fromfile(args.package / operation["bias_file"], dtype=np.int32)
    weight_scales = np.fromfile(args.package / operation["weight_scales_file"], dtype=np.float32)
    expected = np.fromfile(
        args.package / "oracles" / args.fixture / f"tensor_{output_id:03d}_nchw_u8.bin",
        dtype=np.uint8,
    ).reshape(1, output_c, int(output_tensor["h"]), int(output_tensor["w"]))

    y, x = divmod(args.spatial_index, int(output_tensor["w"]))
    values = source[0, :, y, x].astype(np.int32)
    channel_weights = weights[args.channel, 0, 0, :].astype(np.int32)
    input_zero_point = int(input_tensor["zero_point"])
    normal = int(((values - input_zero_point) * channel_weights).sum() + int(bias[args.channel]))

    pair_sums = []
    unsaturated_pair_sums = []
    for index in range(0, input_c, 2):
        pair = int(values[index] * channel_weights[index] + values[index + 1] * channel_weights[index + 1])
        unsaturated_pair_sums.append(pair)
        pair_sums.append(max(-32768, min(32767, pair)))
    saturated = int(sum(pair_sums) - input_zero_point * int(channel_weights.sum()) + int(bias[args.channel]))

    input_scale = np.float32(float(input_tensor["scale"]))
    conv_scale = np.float32(float(operation["conv_output_scale"]))
    conv_zero_point = int(operation["conv_output_zero_point"])
    output_scale = np.float32(float(output_tensor["scale"]))
    output_zero_point = int(output_tensor["zero_point"])

    def output_codes(accumulator: int) -> tuple[int, int]:
        accumulator_scale = np.float32(input_scale * weight_scales[args.channel])
        dequantized = np.float32(np.float32(accumulator) * accumulator_scale)
        conv_code = max(0, min(255, rne(float(dequantized) / float(conv_scale)) + conv_zero_point))
        conv_value = np.float32(np.float32(conv_code - conv_zero_point) * conv_scale)
        activated = np.float32(conv_value / np.float32(1.0 + np.exp(-conv_value, dtype=np.float32)))
        output_code = max(0, min(255, rne(float(activated) / float(output_scale)) + output_zero_point))
        return conv_code, output_code

    normal_codes = output_codes(normal)
    saturated_codes = output_codes(saturated)
    expected_code = int(expected[0, args.channel, y, x])
    saturated_pairs = sum(left != right for left, right in zip(pair_sums, unsaturated_pair_sums, strict=True))
    print(f"fixture={args.fixture}")
    print(f"operation={args.operation}")
    print(f"channel={args.channel}")
    print(f"spatial_index={args.spatial_index}")
    print(f"normal_accumulator={normal}")
    print(f"pair_saturated_accumulator={saturated}")
    print(f"pair_saturation_delta={saturated - normal}")
    print(f"saturated_pairs={saturated_pairs}")
    print(f"normal_conv_code={normal_codes[0]}")
    print(f"normal_output_code={normal_codes[1]}")
    print(f"pair_saturated_conv_code={saturated_codes[0]}")
    print(f"pair_saturated_output_code={saturated_codes[1]}")
    print(f"host_ort_output_code={expected_code}")
    print(f"host_matches_normal={int(expected_code == normal_codes[1])}")
    print(f"host_matches_pair_saturated={int(expected_code == saturated_codes[1])}")
    scale_product_f32 = np.float32(input_scale * weight_scales[args.channel])
    route_acc_then_div = float(np.float32(np.float32(normal) * scale_product_f32)) / float(conv_scale)
    route_ratio_then_acc_f32 = float(
        np.float32(np.float32(normal) * np.float32(scale_product_f32 / conv_scale))
    )
    route_ratio_f64 = float(normal) * (
        float(input_scale) * float(weight_scales[args.channel]) / float(conv_scale)
    )
    print(f"scaled_acc_then_div={route_acc_then_div:.17g}")
    print(f"scaled_ratio_then_acc_f32={route_ratio_then_acc_f32:.17g}")
    print(f"scaled_ratio_f64={route_ratio_f64:.17g}")
    print(f"code_acc_then_div={rne(route_acc_then_div) + conv_zero_point}")
    print(f"code_ratio_then_acc_f32={rne(route_ratio_then_acc_f32) + conv_zero_point}")
    print(f"code_ratio_f64={rne(route_ratio_f64) + conv_zero_point}")
    if args.model is not None:
        conv_output_name = operation["name"] + "_output_0_QuantizeLinear_Output"
        cut_path = args.package / "cuts" / f"diagnostic_op_{args.operation:03d}_conv_and_act.onnx"
        onnx.utils.extract_model(
            str(args.model),
            str(cut_path),
            [input_tensor["logical_name"]],
            [conv_output_name, output_tensor["logical_name"]],
        )
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(cut_path), sess_options=session_options, providers=["CPUExecutionProvider"])
        conv_codes, activation_codes = session.run(
            None, {input_tensor["logical_name"]: np.ascontiguousarray(source)}
        )
        print(f"host_ort_conv_code={int(conv_codes[0, args.channel, y, x])}")
        print(f"host_ort_activation_code={int(activation_codes[0, args.channel, y, x])}")
        print(f"diagnostic_cut={cut_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
