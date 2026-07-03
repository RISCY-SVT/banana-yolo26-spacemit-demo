#!/usr/bin/env python3
"""Extract small real-graph Conv fixtures for Stage 3 tests.

This is an offline oracle/converter helper. The runtime library must not depend
on Python, ONNX, protobuf, or ONNX Runtime.
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
from onnx import helper, numpy_helper, shape_inference


SELECTED = [
    {
        "label": "conv1x1_model_2_cv1",
        "node": "/model.2/cv1/conv/Conv",
        "compare_h": 4,
        "compare_w": 4,
        "input_crop_h": 4,
        "input_crop_w": 4,
    },
    {
        "label": "conv3x3_model_2_m0_cv1",
        "node": "/model.2/m.0/cv1/conv/Conv",
        "compare_h": 4,
        "compare_w": 4,
        "input_crop_h": 5,
        "input_crop_w": 5,
    },
]


def c_array(name: str, values: np.ndarray, ctype: str, cols: int = 16) -> str:
    flat = values.reshape(-1).tolist()
    lines = [f"inline constexpr {ctype} {name}[] = {{"]
    for offset in range(0, len(flat), cols):
        row = ", ".join(str(int(v)) for v in flat[offset : offset + cols])
        suffix = "," if offset + cols < len(flat) else ""
        lines.append(f"    {row}{suffix}")
    lines.append("};")
    return "\n".join(lines)


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


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


def attrs(node: onnx.NodeProto) -> dict[str, Any]:
    return {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}


def selected_outputs_model(model: onnx.ModelProto, outputs: list[str]) -> onnx.ModelProto:
    inferred = shape_inference.infer_shapes(model)
    values = value_info_map(inferred)
    clone = copy.deepcopy(inferred)
    existing = {output.name for output in clone.graph.output}
    for name in outputs:
        if name not in existing:
            clone.graph.output.append(copy.deepcopy(values[name]))
            existing.add(name)
    return clone


def compute_expected_i32(
    activation_q: np.ndarray,
    weight_q: np.ndarray,
    bias_q: np.ndarray,
    activation_zero_point: int,
    weight_zero_point: np.ndarray,
    attrs_map: dict[str, Any],
    compare_h: int,
    compare_w: int,
) -> np.ndarray:
    kernel_h, kernel_w = attrs_map["kernel_shape"]
    stride_h, stride_w = attrs_map["strides"]
    pad_top, pad_left = attrs_map["pads"][0], attrs_map["pads"][1]
    output_c, input_c = weight_q.shape[0], weight_q.shape[1]
    expected = np.zeros((compare_h, compare_w, output_c), dtype=np.int32)
    for oh in range(compare_h):
        for ow in range(compare_w):
            for oc in range(output_c):
                acc = int(bias_q[oc])
                wzp = int(weight_zero_point[oc] if weight_zero_point.ndim else weight_zero_point.item())
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        ih = oh * stride_h + kh - pad_top
                        iw = ow * stride_w + kw - pad_left
                        for ic in range(input_c):
                            if ih < 0 or iw < 0 or ih >= activation_q.shape[2] or iw >= activation_q.shape[3]:
                                aq = activation_zero_point
                            else:
                                aq = int(activation_q[0, ic, ih, iw])
                            acc += (aq - activation_zero_point) * (int(weight_q[oc, ic, kh, kw]) - wzp)
                expected[oh, ow, oc] = acc
    return expected


def crop_activation_s8_nhwc(
    activation_q: np.ndarray,
    input_crop_h: int,
    input_crop_w: int,
) -> np.ndarray:
    cropped = activation_q[0, :, 0:input_crop_h, 0:input_crop_w]
    nhwc = np.transpose(cropped, (1, 2, 0)).astype(np.int16) - 128
    return nhwc.astype(np.int8)


def weight_ohwi(weight_q: np.ndarray) -> np.ndarray:
    return np.transpose(weight_q, (0, 2, 3, 1)).astype(np.int8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--header-out", required=True, type=Path)
    parser.add_argument("--report-out", required=True, type=Path)
    parser.add_argument("--dump-dir", required=True, type=Path)
    args = parser.parse_args()

    args.dump_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(args.model), load_external_data=True)
    inferred = shape_inference.infer_shapes(model)
    producers = producer_map(inferred)
    initializers = {tensor.name: numpy_helper.to_array(tensor) for tensor in inferred.graph.initializer}
    nodes = {node.name: node for node in inferred.graph.node}

    extra_outputs: list[str] = []
    for spec in SELECTED:
        node = nodes[spec["node"]]
        activation_dq = producers[node.input[0]]
        extra_outputs.append(activation_dq.input[0])
        extra_outputs.append(node.output[0])

    oracle_model = selected_outputs_model(model, extra_outputs)
    oracle_model_path = args.dump_dir / "stage3_selected_conv_outputs.onnx"
    onnx.save(oracle_model, str(oracle_model_path))

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(oracle_model_path), sess_options=session_options, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(20260703)
    images = rng.random((1, 3, 640, 640), dtype=np.float32)
    outputs = session.run(None, {"images": images})
    output_map = {output.name: value for output, value in zip(session.get_outputs(), outputs)}

    header_parts = [
        "#pragma once",
        "",
        "#include \"y26_k1x_conv_kernels.h\"",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace y26_stage3_real_fixture {",
        "",
        "struct RealConvFixture {",
        "    const char* label;",
        "    const char* node_name;",
        "    Y26Conv2DParams params;",
        "    int kernel_h;",
        "    int kernel_w;",
        "    int compare_h;",
        "    int compare_w;",
        "    int activation_zero_point_u8;",
        "    int input_storage_zero_point_s8;",
        "    const std::int8_t* input_nhwc_s8;",
        "    std::size_t input_count;",
        "    const std::int8_t* weights_ohwi_s8;",
        "    std::size_t weight_count;",
        "    const std::int32_t* bias_i32;",
        "    std::size_t bias_count;",
        "    const std::int32_t* expected_i32_nhwc;",
        "    std::size_t expected_count;",
        "};",
        "",
    ]
    report: dict[str, Any] = {
        "model": str(args.model.resolve()),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "oracle_model": str(oracle_model_path.resolve()),
        "python": {
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
        },
        "selected": [],
    }

    fixture_names: list[str] = []
    for spec in SELECTED:
        label = spec["label"]
        node = nodes[spec["node"]]
        node_attrs = attrs(node)
        activation_dq = producers[node.input[0]]
        weight_dq = producers[node.input[1]]
        bias_dq = producers[node.input[2]]
        activation_q = output_map[activation_dq.input[0]]
        conv_float = output_map[node.output[0]]
        weight_q = initializers[weight_dq.input[0]]
        weight_scale = initializers[weight_dq.input[1]]
        weight_zero_point = initializers[weight_dq.input[2]]
        activation_scale = float(initializers[activation_dq.input[1]].item())
        activation_zero_point = int(initializers[activation_dq.input[2]].item())
        bias_q = initializers[bias_dq.input[0]].astype(np.int32)
        compare_h = int(spec["compare_h"])
        compare_w = int(spec["compare_w"])
        expected_i32 = compute_expected_i32(
            activation_q,
            weight_q,
            bias_q,
            activation_zero_point,
            weight_zero_point,
            node_attrs,
            compare_h,
            compare_w,
        )
        dequant = np.empty(expected_i32.shape, dtype=np.float32)
        for oc in range(weight_q.shape[0]):
            dequant[:, :, oc] = expected_i32[:, :, oc] * (activation_scale * float(weight_scale[oc]))
        ort_roi = np.transpose(conv_float[0, :, 0:compare_h, 0:compare_w], (1, 2, 0))
        max_abs_diff = float(np.max(np.abs(dequant - ort_roi)))

        input_s8 = crop_activation_s8_nhwc(activation_q, int(spec["input_crop_h"]), int(spec["input_crop_w"]))
        weights = weight_ohwi(weight_q)
        output_h = (int(spec["input_crop_h"]) + node_attrs["pads"][0] + node_attrs["pads"][2] - node_attrs["kernel_shape"][0]) // node_attrs["strides"][0] + 1
        output_w = (int(spec["input_crop_w"]) + node_attrs["pads"][1] + node_attrs["pads"][3] - node_attrs["kernel_shape"][1]) // node_attrs["strides"][1] + 1

        args.dump_dir.joinpath(f"{label}_input_s8.bin").write_bytes(np.ascontiguousarray(input_s8).tobytes())
        args.dump_dir.joinpath(f"{label}_weights_ohwi_s8.bin").write_bytes(np.ascontiguousarray(weights).tobytes())
        args.dump_dir.joinpath(f"{label}_bias_i32.bin").write_bytes(np.ascontiguousarray(bias_q).tobytes())
        args.dump_dir.joinpath(f"{label}_expected_i32.bin").write_bytes(np.ascontiguousarray(expected_i32).tobytes())

        prefix = "k" + "".join(part.capitalize() for part in label.split("_"))
        header_parts.append(c_array(prefix + "InputS8", input_s8, "std::int8_t"))
        header_parts.append("")
        header_parts.append(c_array(prefix + "WeightsOhwiS8", weights, "std::int8_t"))
        header_parts.append("")
        header_parts.append(c_array(prefix + "BiasI32", bias_q, "std::int32_t", cols=8))
        header_parts.append("")
        header_parts.append(c_array(prefix + "ExpectedI32Nhwc", expected_i32, "std::int32_t", cols=8))
        header_parts.append("")
        fixture_name = prefix + "Fixture"
        fixture_names.append(fixture_name)
        header_parts.extend(
            [
                f"inline constexpr RealConvFixture {fixture_name} = {{",
                f"    \"{label}\",",
                f"    \"{spec['node']}\",",
                "    Y26Conv2DParams{"
                f"{int(spec['input_crop_h'])}, {int(spec['input_crop_w'])}, {int(weight_q.shape[1])}, {int(weight_q.shape[0])}, "
                f"{int(node_attrs['strides'][0])}, {int(node_attrs['strides'][1])}, {int(node_attrs['pads'][0])}, {int(node_attrs['pads'][1])}"
                "},",
                f"    {int(node_attrs['kernel_shape'][0])},",
                f"    {int(node_attrs['kernel_shape'][1])},",
                f"    {compare_h},",
                f"    {compare_w},",
                f"    {activation_zero_point},",
                f"    {activation_zero_point - 128},",
                f"    {prefix}InputS8,",
                f"    sizeof({prefix}InputS8) / sizeof({prefix}InputS8[0]),",
                f"    {prefix}WeightsOhwiS8,",
                f"    sizeof({prefix}WeightsOhwiS8) / sizeof({prefix}WeightsOhwiS8[0]),",
                f"    {prefix}BiasI32,",
                f"    sizeof({prefix}BiasI32) / sizeof({prefix}BiasI32[0]),",
                f"    {prefix}ExpectedI32Nhwc,",
                f"    sizeof({prefix}ExpectedI32Nhwc) / sizeof({prefix}ExpectedI32Nhwc[0]),",
                "};",
                "",
            ]
        )

        report["selected"].append(
            {
                "label": label,
                "node": spec["node"],
                "attrs": {k: list(v) if isinstance(v, (list, tuple)) else v for k, v in node_attrs.items()},
                "input_q_tensor": activation_dq.input[0],
                "activation_q_shape": list(activation_q.shape),
                "activation_zero_point_u8": activation_zero_point,
                "activation_scale": activation_scale,
                "weight_q_shape": list(weight_q.shape),
                "weight_scale_shape": list(weight_scale.shape),
                "weight_zero_point_min": int(weight_zero_point.min()),
                "weight_zero_point_max": int(weight_zero_point.max()),
                "bias_q_shape": list(bias_q.shape),
                "fixture_params": {
                    "input_h": int(spec["input_crop_h"]),
                    "input_w": int(spec["input_crop_w"]),
                    "input_c": int(weight_q.shape[1]),
                    "output_c": int(weight_q.shape[0]),
                    "output_h": int(output_h),
                    "output_w": int(output_w),
                    "compare_h": compare_h,
                    "compare_w": compare_w,
                },
                "dequant_max_abs_diff_vs_ort": max_abs_diff,
                "sha256": {
                    "input_s8": sha256_array(input_s8),
                    "weights_ohwi_s8": sha256_array(weights),
                    "bias_i32": sha256_array(bias_q),
                    "expected_i32": sha256_array(expected_i32),
                },
            }
        )

    header_parts.append("inline constexpr const RealConvFixture* kFixtures[] = {")
    for name in fixture_names:
        header_parts.append(f"    &{name},")
    header_parts.append("};")
    header_parts.append("")
    header_parts.append("}  // namespace y26_stage3_real_fixture")
    args.header_out.parent.mkdir(parents=True, exist_ok=True)
    args.header_out.write_text("\n".join(header_parts) + "\n", encoding="utf-8")
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
