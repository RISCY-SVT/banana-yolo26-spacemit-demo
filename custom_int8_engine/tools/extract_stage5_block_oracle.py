#!/usr/bin/env python3
"""Extract Stage 5 block0 Conv-only oracle fixtures.

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
from onnx import helper, numpy_helper, shape_inference


TARGET_NODE = "/model.0/conv/Conv"
BLOCK_ID = "block0_conv_only"


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def c_array(name: str, values: np.ndarray, ctype: str, cols: int = 16) -> str:
    flat = values.reshape(-1).tolist()
    lines = [f"inline constexpr {ctype} {name}[] = {{"]
    for offset in range(0, len(flat), cols):
        row = ", ".join(str(int(v)) for v in flat[offset : offset + cols])
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
    rng = np.random.default_rng(20260703)
    return rng.random((1, 3, 640, 640), dtype=np.float32)


def synthetic_gradient_input() -> np.ndarray:
    x = np.linspace(0.0, 1.0, 640, dtype=np.float32)
    y = np.linspace(0.0, 1.0, 640, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    img = np.stack([xx, yy, 0.5 * (xx + yy)], axis=0)
    return img.reshape(1, 3, 640, 640).astype(np.float32)


def crop_activation_s8_nhwc(activation_q: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    cropped = activation_q[0, :, 0:crop_h, 0:crop_w]
    nhwc_i16 = np.transpose(cropped, (1, 2, 0)).astype(np.int16) - 128
    return nhwc_i16.astype(np.int8)


def weight_ohwi(weight_q: np.ndarray) -> np.ndarray:
    return np.transpose(weight_q, (0, 2, 3, 1)).astype(np.int8)


def compute_expected_i32(
    activation_q: np.ndarray,
    weight_q: np.ndarray,
    bias_q: np.ndarray,
    activation_zero_point: int,
    weight_zero_point: np.ndarray,
    node_attrs: dict[str, Any],
    compare_h: int,
    compare_w: int,
) -> np.ndarray:
    kernel_h, kernel_w = node_attrs["kernel_shape"]
    stride_h, stride_w = node_attrs["strides"]
    pad_top, pad_left = node_attrs["pads"][0], node_attrs["pads"][1]
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dump-dir", required=True, type=Path)
    parser.add_argument("--fixture-header-out", required=True, type=Path)
    parser.add_argument("--metadata-out", required=True, type=Path)
    parser.add_argument("--real-input-npy", type=Path)
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
    node = nodes[TARGET_NODE]
    node_attrs = attrs(node)
    activation_dq = producers[node.input[0]]
    weight_dq = producers[node.input[1]]
    bias_dq = producers[node.input[2]]
    conv_q = consumers[node.output[0]][0]
    conv_dq = consumers[conv_q.output[0]][0]

    outputs = [
        activation_dq.input[0],
        node.input[0],
        node.output[0],
        conv_q.output[0],
        conv_dq.output[0],
    ]
    oracle_model = selected_outputs_model(model, outputs)
    oracle_model_path = args.dump_dir / "stage5_block0_outputs.onnx"
    onnx.save(oracle_model, str(oracle_model_path))

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(oracle_model_path), sess_options=session_options, providers=["CPUExecutionProvider"])

    input_cases: list[tuple[str, np.ndarray, str]] = [
        ("synthetic_seeded", synthetic_seeded_input(), "seed=20260703 random float32 [0,1)"),
        ("synthetic_gradient", synthetic_gradient_input(), "deterministic H/W gradient float32 [0,1]"),
    ]
    if args.real_input_npy is not None and args.real_input_npy.is_file():
        input_cases.append(("real_preprocessed", np.load(args.real_input_npy).astype(np.float32), str(args.real_input_npy)))

    weight_q = initializers[weight_dq.input[0]]
    weight_scale = initializers[weight_dq.input[1]]
    weight_zero_point = initializers[weight_dq.input[2]]
    bias_q = initializers[bias_dq.input[0]].astype(np.int32)
    activation_scale = float(initializers[activation_dq.input[1]].item())
    activation_zero_point = int(initializers[activation_dq.input[2]].item())
    output_scale = float(initializers[conv_q.input[1]].item())
    output_zero_point = int(initializers[conv_q.input[2]].item())
    weights = weight_ohwi(weight_q)
    compare_h = 4
    compare_w = 4
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
        "namespace y26_stage5_block0_fixture {",
        "",
        "struct Block0Fixture {",
        "    const char* label;",
        "    const char* node_name;",
        "    Y26Conv2DParams params;",
        "    int kernel_h;",
        "    int kernel_w;",
        "    int compare_h;",
        "    int compare_w;",
        "    int activation_zero_point_u8;",
        "    int input_storage_zero_point_s8;",
        "    int output_zero_point_u8;",
        "    float input_scale;",
        "    float output_scale;",
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
        "block_id": BLOCK_ID,
        "target_node": TARGET_NODE,
        "model": str(args.model.resolve()),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "oracle_model": str(oracle_model_path.resolve()),
        "python": {
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
        },
        "xslim_used": False,
        "node": {
            "inputs": list(node.input),
            "outputs": list(node.output),
            "attrs": {k: list(v) if isinstance(v, (list, tuple)) else v for k, v in node_attrs.items()},
            "input_shape": tensor_shape(inferred, node.input[0]),
            "output_shape": tensor_shape(inferred, node.output[0]),
            "downstream": [
                {"name": c.name, "op_type": c.op_type, "inputs": list(c.input), "outputs": list(c.output)}
                for c in consumers[node.output[0]]
            ],
        },
        "quantization": {
            "activation_scale": activation_scale,
            "activation_zero_point_u8": activation_zero_point,
            "input_storage_zero_point_s8": activation_zero_point - 128,
            "weight_scale_shape": list(weight_scale.shape),
            "weight_scale_min": float(np.min(weight_scale)),
            "weight_scale_max": float(np.max(weight_scale)),
            "weight_zero_point_min": int(np.min(weight_zero_point)),
            "weight_zero_point_max": int(np.max(weight_zero_point)),
            "output_scale": output_scale,
            "output_zero_point_u8": output_zero_point,
        },
        "real_preprocessed_input": "not-provided" if args.real_input_npy is None else str(args.real_input_npy),
        "cases": [],
        "dumps": {},
    }

    fixture_names: list[str] = []
    header_parts.append(c_array("kBlock0WeightsOhwiS8", weights, "std::int8_t"))
    header_parts.append("")
    header_parts.append(c_array("kBlock0BiasI32", bias_q, "std::int32_t", cols=8))
    header_parts.append("")

    for case_name, input_tensor, source in input_cases:
        outputs_values = session.run(None, {"images": input_tensor})
        output_map = {output.name: value for output, value in zip(session.get_outputs(), outputs_values)}
        activation_q = output_map[activation_dq.input[0]]
        conv_float = output_map[node.output[0]]
        conv_q_tensor = output_map[conv_q.output[0]]
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
        input_s8 = crop_activation_s8_nhwc(activation_q, input_crop_h, input_crop_w)

        safe = case_name.replace("-", "_")
        camel = "".join(part.capitalize() for part in safe.split("_"))
        input_name = f"kBlock0{camel}InputS8"
        expected_name = f"kBlock0{camel}ExpectedI32Nhwc"
        fixture_name = f"k{camel}Fixture"
        fixture_names.append(fixture_name)

        header_parts.append(c_array(input_name, input_s8, "std::int8_t"))
        header_parts.append("")
        header_parts.append(c_array(expected_name, expected_i32, "std::int32_t", cols=8))
        header_parts.append("")
        header_parts.append(f"inline constexpr Block0Fixture {fixture_name} = {{")
        header_parts.append(f'    "{case_name}",')
        header_parts.append(f'    "{TARGET_NODE}",')
        header_parts.append(f"    Y26Conv2DParams{{{input_crop_h}, {input_crop_w}, 3, 16, 2, 2, 1, 1}},")
        header_parts.append("    3,")
        header_parts.append("    3,")
        header_parts.append(f"    {compare_h},")
        header_parts.append(f"    {compare_w},")
        header_parts.append(f"    {activation_zero_point},")
        header_parts.append(f"    {activation_zero_point - 128},")
        header_parts.append(f"    {output_zero_point},")
        header_parts.append(f"    {activation_scale:.12g}f,")
        header_parts.append(f"    {output_scale:.12g}f,")
        header_parts.append(f"    {input_name},")
        header_parts.append(f"    sizeof({input_name}) / sizeof({input_name}[0]),")
        header_parts.append("    kBlock0WeightsOhwiS8,")
        header_parts.append("    sizeof(kBlock0WeightsOhwiS8) / sizeof(kBlock0WeightsOhwiS8[0]),")
        header_parts.append("    kBlock0BiasI32,")
        header_parts.append("    sizeof(kBlock0BiasI32) / sizeof(kBlock0BiasI32[0]),")
        header_parts.append(f"    {expected_name},")
        header_parts.append(f"    sizeof({expected_name}) / sizeof({expected_name}[0]),")
        header_parts.append("};")
        header_parts.append("")

        case_dir = args.dump_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        dump_info = {
            "model_input_float": write_binary(case_dir / "model_input_float32.bin", input_tensor),
            "block_input_q_u8": write_binary(case_dir / "block_input_q_u8.bin", activation_q),
            "block_input_s8_roi_nhwc": write_binary(case_dir / "block_input_s8_roi_nhwc.bin", input_s8),
            "block_conv_float": write_binary(case_dir / "block_conv_float32.bin", conv_float),
            "block_conv_q_u8": write_binary(case_dir / "block_conv_q_u8.bin", conv_q_tensor),
            "block_expected_i32_roi_nhwc": write_binary(case_dir / "block_expected_i32_roi_nhwc.bin", expected_i32),
        }
        report["cases"].append(
            {
                "label": case_name,
                "source": source,
                "input_crop_h": input_crop_h,
                "input_crop_w": input_crop_w,
                "compare_h": compare_h,
                "compare_w": compare_w,
                "conv_dequant_max_abs_diff_vs_ort_roi": max_abs_diff,
                "input_s8_sha256": sha256_array(input_s8),
                "expected_i32_sha256": sha256_array(expected_i32),
            }
        )
        report["dumps"][case_name] = dump_info

    header_parts.append("inline constexpr const Block0Fixture* kFixtures[] = {")
    for fixture_name in fixture_names:
        header_parts.append(f"    &{fixture_name},")
    header_parts.append("};")
    header_parts.append("")
    if "kSyntheticSeededFixture" in fixture_names:
        header_parts.append("inline constexpr const Block0Fixture& kSyntheticSeededFixture = kSyntheticSeededFixture;")
    header_parts.append("")
    header_parts.append("}  // namespace y26_stage5_block0_fixture")
    header_parts.append("")

    # Avoid a self-referential alias with the same identifier.
    text = "\n".join(header_parts).replace(
        "inline constexpr const Block0Fixture& kSyntheticSeededFixture = kSyntheticSeededFixture;\n",
        "",
    )
    args.fixture_header_out.write_text(text)
    args.metadata_out.write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
