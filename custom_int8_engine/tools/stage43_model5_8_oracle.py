#!/usr/bin/env python3
"""Generate graph-derived Stage43 model5-8 cuts, fixtures, and fixed-host oracles."""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import platform
import struct
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnx.utils
import onnxruntime as ort
from onnx import TensorProto, numpy_helper, shape_inference


MODEL4_PREACT = "/model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output"
MODEL4_POSTACT = "/model.4/cv2/act/Mul_output_0_QuantizeLinear_Output"
BLOCKS = {
    "model5": (MODEL4_POSTACT, "/model.5/act/Mul_output_0_QuantizeLinear_Output"),
    "model6": (
        "/model.5/act/Mul_output_0_QuantizeLinear_Output",
        "/model.6/cv2/act/Mul_output_0_QuantizeLinear_Output",
    ),
    "model7": (
        "/model.6/cv2/act/Mul_output_0_QuantizeLinear_Output",
        "/model.7/act/Mul_output_0_QuantizeLinear_Output",
    ),
    "model8": (
        "/model.7/act/Mul_output_0_QuantizeLinear_Output",
        "/model.8/cv2/act/Mul_output_0_QuantizeLinear_Output",
    ),
}
BOUNDARIES = [MODEL4_PREACT, MODEL4_POSTACT, *(output for _, output in BLOCKS.values())]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def safe_name(name: str) -> str:
    return name.strip("/").replace("/", "__").replace(":", "_")


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def session_options(opt_level: str, optimized_path: Path | None = None) -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.enable_mem_pattern = True
    options.enable_cpu_mem_arena = True
    options.add_session_config_entry("session.intra_op.allow_spinning", "1")
    options.add_session_config_entry("session.inter_op.allow_spinning", "1")
    options.graph_optimization_level = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }[opt_level]
    if optimized_path is not None:
        optimized_path.parent.mkdir(parents=True, exist_ok=True)
        options.optimized_model_filepath = str(optimized_path)
    return options


def make_session(path: Path, opt_level: str = "all", optimized_path: Path | None = None) -> ort.InferenceSession:
    return ort.InferenceSession(
        str(path),
        sess_options=session_options(opt_level, optimized_path),
        providers=["CPUExecutionProvider"],
    )


def inferred_metadata(model: onnx.ModelProto) -> dict[str, dict[str, object]]:
    inferred = shape_inference.infer_shapes(model)
    metadata: dict[str, dict[str, object]] = {}
    for value in [*inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info]:
        tensor_type = value.type.tensor_type
        dims: list[int | str] = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(dim.dim_value)
            elif dim.HasField("dim_param"):
                dims.append(dim.dim_param)
            else:
                dims.append("?")
        metadata[value.name] = {
            "dtype": TensorProto.DataType.Name(tensor_type.elem_type),
            "shape": dims,
        }
    return metadata


def producer_and_consumers(model: onnx.ModelProto, tensor_name: str) -> tuple[onnx.NodeProto | None, list[str]]:
    producer = next((node for node in model.graph.node if tensor_name in node.output), None)
    consumers = [node.name or node.op_type for node in model.graph.node if tensor_name in node.input]
    return producer, consumers


def quant_metadata(
    model: onnx.ModelProto, tensor_name: str, initializers: dict[str, np.ndarray]
) -> dict[str, object]:
    producer, _ = producer_and_consumers(model, tensor_name)
    record: dict[str, object] = {
        "scale_name": "",
        "scale": "",
        "scale_sha256": "",
        "zero_point_name": "",
        "zero_point": "",
        "zero_point_sha256": "",
        "axis": "",
    }
    if producer is None or producer.op_type != "QuantizeLinear" or len(producer.input) < 3:
        return record
    scale_name, zero_name = producer.input[1:3]
    scale = initializers[scale_name]
    zero = initializers[zero_name]
    axis = next((onnx.helper.get_attribute_value(attr) for attr in producer.attribute if attr.name == "axis"), "")
    record.update(
        {
            "scale_name": scale_name,
            "scale": ",".join(f"{float(value):.12g}" for value in scale.reshape(-1)),
            "scale_sha256": sha256_array(scale),
            "zero_point_name": zero_name,
            "zero_point": ",".join(str(int(value)) for value in zero.reshape(-1)),
            "zero_point_sha256": sha256_array(zero),
            "axis": axis,
        }
    )
    return record


def tensor_contract(
    model: onnx.ModelProto,
    metadata: dict[str, dict[str, object]],
    initializers: dict[str, np.ndarray],
    block: str,
    role: str,
    tensor_name: str,
) -> dict[str, object]:
    producer, consumers = producer_and_consumers(model, tensor_name)
    meta = metadata[tensor_name]
    quant = quant_metadata(model, tensor_name, initializers)
    return {
        "block": block,
        "role": role,
        "tensor_name": tensor_name,
        "producer": "graph_input" if producer is None else producer.name or producer.op_type,
        "consumers": ",".join(consumers),
        "dtype": meta["dtype"],
        "shape": "x".join(map(str, meta["shape"])),
        "logical_layout": "NCHW" if len(meta["shape"]) == 4 else "graph-native",
        **quant,
    }


def extract_cuts(
    model_path: Path, model: onnx.ModelProto, out_dir: Path
) -> tuple[dict[str, Path], Path, Path, Path]:
    cuts_dir = out_dir / "cuts"
    cuts_dir.mkdir(parents=True, exist_ok=True)
    block_cuts: dict[str, Path] = {}
    for block, (input_name, output_name) in BLOCKS.items():
        cut_path = cuts_dir / f"{block}_isolated_quantized.onnx"
        onnx.utils.extract_model(str(model_path), str(cut_path), [input_name], [output_name])
        block_cuts[block] = cut_path
    postact_cut = cuts_dir / "model4_final_activation_quantized.onnx"
    onnx.utils.extract_model(str(model_path), str(postact_cut), [MODEL4_PREACT], [MODEL4_POSTACT])
    island_cut = cuts_dir / "model4_preact_to_model8_quantized.onnx"
    onnx.utils.extract_model(
        str(model_path),
        str(island_cut),
        [MODEL4_PREACT],
        [MODEL4_POSTACT, *(output for _, output in BLOCKS.values())],
    )
    full_boundary_cut = cuts_dir / "images_to_model4_model8_boundaries.onnx"
    onnx.utils.extract_model(str(model_path), str(full_boundary_cut), ["images"], BOUNDARIES)
    return block_cuts, postact_cut, island_cut, full_boundary_cut


def letterbox_image(path: Path, size: int = 640) -> tuple[np.ndarray, dict[str, object]]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    src_h, src_w = image.shape[:2]
    ratio = min(size / src_h, size / src_w)
    new_w = int(round(src_w * ratio))
    new_h = int(round(src_h * ratio))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw = (size - new_w) / 2.0
    dh = (size - new_h) / 2.0
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    if padded.shape[:2] != (size, size):
        raise ValueError(f"letterbox shape error: {padded.shape}")
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(np.transpose(rgb.astype(np.float32) / np.float32(255.0), (2, 0, 1))[None])
    return tensor, {
        "decode_library": f"opencv-python {cv2.__version__}",
        "source_shape": f"{src_h}x{src_w}x3",
        "resize": f"{new_h}x{new_w}",
        "ratio": f"{ratio:.12g}",
        "pad_left": left,
        "pad_right": right,
        "pad_top": top,
        "pad_bottom": bottom,
        "pad_value": 114,
        "channel_order": "BGR-decode to RGB",
        "normalization": "float32 / 255.0",
        "output_layout": "NCHW",
    }


def structured_fixture(kind: str, shape: tuple[int, ...], zero_point: int) -> np.ndarray:
    count = math.prod(shape)
    if kind == "random_4301":
        return np.random.default_rng(4301).integers(0, 256, size=shape, dtype=np.uint8)
    if kind == "random_4302":
        return np.random.default_rng(4302).integers(0, 256, size=shape, dtype=np.uint8)
    if kind == "zero_point_structured":
        offsets = np.asarray([-8, -4, -2, -1, 0, 1, 2, 4, 8], dtype=np.int16)
        values = zero_point + offsets[np.arange(count) % offsets.size]
        return np.clip(values, 0, 255).astype(np.uint8).reshape(shape)
    if kind == "edge_saturation":
        values = np.asarray([0, 1, zero_point - 1, zero_point, zero_point + 1, 127, 128, 254, 255], dtype=np.int16)
        return np.clip(values[np.arange(count) % values.size], 0, 255).astype(np.uint8).reshape(shape)
    raise ValueError(kind)


def save_tensor(base: Path, array: np.ndarray) -> dict[str, object]:
    npy = base.parent / f"{base.name}.npy"
    raw = base.parent / f"{base.name}.bin"
    npy.parent.mkdir(parents=True, exist_ok=True)
    contiguous = np.ascontiguousarray(array)
    np.save(npy, contiguous, allow_pickle=False)
    contiguous.tofile(raw)
    return {
        "npy_path": str(npy),
        "npy_sha256": sha256_file(npy),
        "raw_path": str(raw),
        "raw_sha256": sha256_file(raw),
        "dtype": str(contiguous.dtype),
        "shape": "x".join(map(str, contiguous.shape)),
        "element_count": contiguous.size,
        "min": float(contiguous.min()),
        "max": float(contiguous.max()),
        "mean": float(contiguous.astype(np.float64).mean()),
        "sum": float(contiguous.astype(np.float64).sum()),
    }


def compare_exact(actual: np.ndarray, expected: np.ndarray, context: str) -> None:
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise ValueError(f"{context}: structural mismatch {actual.dtype}/{actual.shape} vs {expected.dtype}/{expected.shape}")
    mismatches = int(np.count_nonzero(actual != expected))
    if mismatches:
        diff = np.abs(actual.astype(np.int64) - expected.astype(np.int64))
        raise ValueError(f"{context}: mismatches={mismatches} max_abs_diff={int(diff.max())}")


def block_inventory(block: str, cut_path: Path) -> dict[str, object]:
    cut = onnx.load(cut_path, load_external_data=False)
    ops = Counter(node.op_type for node in cut.graph.node)
    conv_nodes = [node for node in cut.graph.node if node.op_type == "Conv"]
    return {
        "block": block,
        "cut_path": str(cut_path),
        "cut_sha256": sha256_file(cut_path),
        "node_count": len(cut.graph.node),
        "first_node": cut.graph.node[0].name if cut.graph.node else "",
        "last_node": cut.graph.node[-1].name if cut.graph.node else "",
        "operator_mix": ",".join(f"{key}:{ops[key]}" for key in sorted(ops)),
        "conv_count": len(conv_nodes),
        "conv_nodes": ",".join(node.name for node in conv_nodes),
        "input_name": BLOCKS[block][0],
        "output_name": BLOCKS[block][1],
    }


def write_model5_assets(model: onnx.ModelProto, initializers: dict[str, np.ndarray], out_dir: Path) -> dict[str, object]:
    assets = out_dir / "model5_runtime_assets"
    assets.mkdir(parents=True, exist_ok=True)
    weights_oihw = np.asarray(initializers["model.5.conv.weight_quantized"], dtype=np.int8)
    weights_ohwi = np.ascontiguousarray(np.transpose(weights_oihw, (0, 2, 3, 1)))
    weight_scales = np.ascontiguousarray(initializers["model.5.conv.weight_scale"], dtype=np.float32)
    bias = np.ascontiguousarray(initializers["model.5.conv.bias_quantized"], dtype=np.int32)
    records = {}
    for name, array in (("weights_ohwi_s8", weights_ohwi), ("weight_scales_f32", weight_scales), ("bias_i32", bias)):
        path = assets / f"{name}.bin"
        array.tofile(path)
        records[name] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    init = initializers
    model4_conv_scale = float(init["/model.4/cv2/conv/Conv_output_0_scale"])
    model4_conv_zp = int(init["/model.4/cv2/conv/Conv_output_0_zero_point"])
    model4_act_scale = float(init["/model.4/cv2/act/Mul_output_0_scale"])
    model4_act_zp = int(init["/model.4/cv2/act/Mul_output_0_zero_point"])
    model5_conv_scale = float(init["/model.5/conv/Conv_output_0_scale"])
    model5_conv_zp = int(init["/model.5/conv/Conv_output_0_zero_point"])
    model5_act_scale = float(init["/model.5/act/Mul_output_0_scale"])
    model5_act_zp = int(init["/model.5/act/Mul_output_0_zero_point"])
    config = {
        "node_name": "/model.5/conv/Conv",
        "input_h": 80,
        "input_w": 80,
        "input_c": 128,
        "output_c": 128,
        "kernel_h": 3,
        "kernel_w": 3,
        "stride_h": 2,
        "stride_w": 2,
        "pad_h": 1,
        "pad_w": 1,
        "model4_conv_scale": model4_conv_scale,
        "model4_conv_zero_point_u8": model4_conv_zp,
        "model4_act_scale": model4_act_scale,
        "model4_act_zero_point_u8": model4_act_zp,
        "model5_activation_zero_point_u8": model4_act_zp,
        "model5_input_storage_zero_point_s8": model4_act_zp - 128,
        "model5_input_scale": model4_act_scale,
        "model5_conv_output_scale": model5_conv_scale,
        "model5_conv_output_zero_point_u8": model5_conv_zp,
        "model5_act_output_scale": model5_act_scale,
        "model5_act_output_zero_point_u8": model5_act_zp,
        "weights_ohwi_s8": records["weights_ohwi_s8"],
        "weight_scales_f32": records["weight_scales_f32"],
        "bias_i32": records["bias_i32"],
    }
    config_path = assets / "model5_runtime_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"config_path": str(config_path), "config_sha256": sha256_file(config_path), **config}


def generate_oracles(args: argparse.Namespace) -> None:
    model_path = Path(args.model).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(model_path)
    metadata = inferred_metadata(model)
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    missing = [name for name in BOUNDARIES if name not in metadata]
    if missing:
        raise ValueError(f"missing graph metadata: {missing}")
    block_cuts, postact_cut, island_cut, full_boundary_cut = extract_cuts(model_path, model, out_dir)

    contract_rows: list[dict[str, object]] = []
    for block, (input_name, output_name) in BLOCKS.items():
        contract_rows.append(tensor_contract(model, metadata, initializers, block, "quantized_input", input_name))
        contract_rows.append(tensor_contract(model, metadata, initializers, block, "quantized_output", output_name))
        input_dq = next(node for node in model.graph.node if input_name in node.input and node.op_type == "DequantizeLinear")
        output_dq = next(node for node in model.graph.node if output_name in node.input and node.op_type == "DequantizeLinear")
        contract_rows.append(
            tensor_contract(model, metadata, initializers, block, "semantic_input", input_dq.output[0])
        )
        contract_rows.append(
            tensor_contract(model, metadata, initializers, block, "semantic_output", output_dq.output[0])
        )
    contract_rows.insert(0, tensor_contract(model, metadata, initializers, "model4", "preactivation", MODEL4_PREACT))
    contract_rows.insert(1, tensor_contract(model, metadata, initializers, "model4", "postactivation", MODEL4_POSTACT))
    write_tsv(out_dir / "model5_8_contract_manifest.tsv", contract_rows)

    inventory_rows = [block_inventory(block, cut) for block, cut in block_cuts.items()]
    write_tsv(out_dir / "model5_8_operator_inventory.tsv", inventory_rows)

    weight_rows: list[dict[str, object]] = []
    for block in BLOCKS:
        prefix = f"model.{block.removeprefix('model')}"
        for name, array in sorted(initializers.items()):
            if not name.startswith(prefix) or not ("weight" in name or "bias" in name):
                continue
            weight_rows.append(
                {
                    "block": block,
                    "initializer": name,
                    "dtype": str(array.dtype),
                    "shape": "x".join(map(str, array.shape)) if array.shape else "scalar",
                    "element_count": array.size,
                    "bytes": array.nbytes,
                    "raw_sha256": sha256_array(array),
                }
            )
    write_tsv(out_dir / "model5_8_weight_manifest.tsv", weight_rows)
    runtime_config = write_model5_assets(model, initializers, out_dir)

    optimized_path = out_dir / "host_optimized_full_model_ort_all.onnx"
    optimized_session = make_session(model_path, "all", optimized_path)
    del optimized_session

    full_session = make_session(full_boundary_cut, "all")
    island_session = make_session(island_cut, "all")
    postact_session = make_session(postact_cut, "all")
    block_sessions = {block: make_session(path, "all") for block, path in block_cuts.items()}

    fixture_rows: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []
    fixture_inputs: list[tuple[str, str, np.ndarray, dict[str, object], Path | None]] = []
    stage42_input = Path(args.stage42_input).resolve()
    fixture_inputs.append(
        ("F0", "accepted_stage42_synthetic_seeded", np.load(stage42_input, allow_pickle=False), {}, stage42_input)
    )
    preact_shape = tuple(int(value) for value in metadata[MODEL4_PREACT]["shape"])
    model4_zp = int(initializers["/model.4/cv2/conv/Conv_output_0_zero_point"])
    for fixture_id, kind in (
        ("F1", "random_4301"),
        ("F2", "random_4302"),
        ("F3", "zero_point_structured"),
        ("F4", "edge_saturation"),
    ):
        fixture_inputs.append((fixture_id, kind, structured_fixture(kind, preact_shape, model4_zp), {}, None))
    for offset, spec in enumerate(args.image, start=5):
        if "=" not in spec:
            raise ValueError(f"--image requires NAME=PATH: {spec}")
        label, path_text = spec.split("=", 1)
        path = Path(path_text).resolve()
        tensor, preprocessing = letterbox_image(path)
        fixture_inputs.append((f"F{offset}", label, tensor, preprocessing, path))

    tensors_dir = out_dir / "fixtures"
    for fixture_id, label, source_tensor, preprocessing, source_path in fixture_inputs:
        is_full_input = source_tensor.dtype == np.float32 and source_tensor.shape == (1, 3, 640, 640)
        fixture_dir = tensors_dir / fixture_id
        fixture_dir.mkdir(parents=True, exist_ok=True)
        source_record = save_tensor(fixture_dir / ("images" if is_full_input else "model4_preact_input"), source_tensor)
        if is_full_input:
            outputs = full_session.run(BOUNDARIES, {"images": source_tensor})
        else:
            island_outputs = island_session.run(
                [MODEL4_POSTACT, *(output for _, output in BLOCKS.values())], {MODEL4_PREACT: source_tensor}
            )
            outputs = [source_tensor, *island_outputs]
        boundary_values = dict(zip(BOUNDARIES, map(np.asarray, outputs), strict=True))

        postact_check = np.asarray(postact_session.run([MODEL4_POSTACT], {MODEL4_PREACT: boundary_values[MODEL4_PREACT]})[0])
        compare_exact(postact_check, boundary_values[MODEL4_POSTACT], f"{fixture_id} model4 postactivation cut")
        current = boundary_values[MODEL4_POSTACT]
        for block, (input_name, output_name) in BLOCKS.items():
            if input_name != MODEL4_POSTACT:
                compare_exact(current, boundary_values[input_name], f"{fixture_id} {block} chained input")
            cut_output = np.asarray(block_sessions[block].run([output_name], {input_name: current})[0])
            compare_exact(cut_output, boundary_values[output_name], f"{fixture_id} {block} isolated cut")
            current = cut_output

        fixture_rows.append(
            {
                "fixture_id": fixture_id,
                "label": label,
                "fixture_class": "full_model_input" if is_full_input else "direct_model4_preact_uint8",
                "source_path": "" if source_path is None else str(source_path),
                "source_sha256": "" if source_path is None else sha256_file(source_path),
                "input_npy": source_record["npy_path"],
                "input_npy_sha256": source_record["npy_sha256"],
                "input_raw_sha256": source_record["raw_sha256"],
                "input_dtype": source_record["dtype"],
                "input_shape": source_record["shape"],
                "decode_library": preprocessing.get("decode_library", "not-applicable"),
                "source_shape": preprocessing.get("source_shape", "not-applicable"),
                "resize": preprocessing.get("resize", "not-applicable"),
                "ratio": preprocessing.get("ratio", "not-applicable"),
                "padding": (
                    "not-applicable"
                    if not preprocessing
                    else f"{preprocessing['pad_left']},{preprocessing['pad_top']},"
                    f"{preprocessing['pad_right']},{preprocessing['pad_bottom']}"
                ),
                "channel_order": preprocessing.get("channel_order", "not-applicable"),
                "normalization": preprocessing.get("normalization", "not-applicable"),
            }
        )
        for boundary in BOUNDARIES:
            record = save_tensor(fixture_dir / safe_name(boundary), boundary_values[boundary])
            contract = next(row for row in contract_rows if row["tensor_name"] == boundary)
            oracle_rows.append(
                {
                    "fixture_id": fixture_id,
                    "tensor_name": boundary,
                    "dtype": record["dtype"],
                    "shape": record["shape"],
                    "layout": "NCHW",
                    "scale": contract["scale"],
                    "zero_point": contract["zero_point"],
                    "npy_path": record["npy_path"],
                    "npy_sha256": record["npy_sha256"],
                    "raw_path": record["raw_path"],
                    "raw_sha256": record["raw_sha256"],
                    "element_count": record["element_count"],
                    "min": record["min"],
                    "max": record["max"],
                    "mean": f"{record['mean']:.12g}",
                    "sum": f"{record['sum']:.12g}",
                    "isolated_cut_replay": "exact",
                }
            )
    write_tsv(out_dir / "model5_8_fixture_manifest.tsv", fixture_rows)
    write_tsv(out_dir / "model5_8_oracle_manifest.tsv", oracle_rows)

    cut_manifest = []
    for block, path in block_cuts.items():
        cut_manifest.append({"cut": block, "path": str(path), "sha256": sha256_file(path)})
    for label, path in (
        ("model4_final_activation", postact_cut),
        ("model4_to_model8_island", island_cut),
        ("images_to_model8_boundaries", full_boundary_cut),
        ("host_optimized_full_model", optimized_path),
    ):
        cut_manifest.append({"cut": label, "path": str(path), "sha256": sha256_file(path)})
    write_tsv(out_dir / "cut_and_optimized_graph_manifest.tsv", cut_manifest)

    summary = {
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "ort_version": ort.__version__,
        "onnx_version": onnx.__version__,
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "provider": "CPUExecutionProvider",
        "optimization": "ORT_ENABLE_ALL",
        "execution_mode": "sequential",
        "intra_threads": 1,
        "inter_threads": 1,
        "fixture_count": len(fixture_rows),
        "oracle_tensor_count": len(oracle_rows),
        "runtime_config": runtime_config,
    }
    (out_dir / "oracle_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


def round_fraction_ties_even(value: Fraction) -> int:
    floor_value = value.numerator // value.denominator
    fraction = value - floor_value
    half = Fraction(1, 2)
    if fraction < half:
        return floor_value
    if fraction > half:
        return floor_value + 1
    return floor_value if floor_value % 2 == 0 else floor_value + 1


def float32_bits(value: np.float32) -> str:
    return f"0x{struct.unpack('<I', struct.pack('<f', float(value)))[0]:08x}"


def quantize_five_case(args: argparse.Namespace) -> None:
    host_float = np.fromfile(args.host_float, dtype=np.float32)
    host_q = np.fromfile(args.host_q, dtype=np.uint8)
    board_q = np.fromfile(args.board_q, dtype=np.uint8)
    if host_float.size != host_q.size or host_q.size != board_q.size:
        raise ValueError("five-case tensors have different element counts")
    mismatch_indices = np.flatnonzero(host_q != board_q)
    if mismatch_indices.size != 5:
        raise ValueError(f"expected five mismatches, got {mismatch_indices.size}")
    scale = np.float32(args.scale)
    zero_point = int(args.zero_point)
    libc = ctypes.CDLL(None)
    host_fenv = int(libc.fegetround()) if hasattr(libc, "fegetround") else -1
    rows: list[dict[str, object]] = []
    for index in mismatch_indices:
        x = np.float32(host_float[index])
        exact_quotient = Fraction.from_float(float(x)) / Fraction.from_float(float(scale))
        rounded = round_fraction_ties_even(exact_quotient)
        independent = min(255, max(0, rounded + zero_point))
        quotient_f32 = np.float32(x / scale)
        quotient_f64 = float(x) / float(scale)
        nearest_half_distance = abs(float(exact_quotient) - (math.floor(float(exact_quotient)) + 0.5))
        rows.append(
            {
                "tensor_index": int(index),
                "x_bits": float32_bits(x),
                "x_decimal": f"{float(x):.17g}",
                "scale_bits": float32_bits(scale),
                "scale_decimal": f"{float(scale):.17g}",
                "zero_point": zero_point,
                "x_over_scale_float32": f"{float(quotient_f32):.17g}",
                "x_over_scale_float64": f"{quotient_f64:.17g}",
                "x_over_scale_exact_fraction": f"{exact_quotient.numerator}/{exact_quotient.denominator}",
                "nearest_half_distance": f"{nearest_half_distance:.17g}",
                "independent_rne_code": independent,
                "host_ort_code": int(host_q[index]),
                "board_ort_code": int(board_q[index]),
                "host_matches_independent": int(host_q[index]) == independent,
                "board_matches_independent": int(board_q[index]) == independent,
                "host_fenv": host_fenv,
                "board_fenv_frm": "not-captured-by-vendor-ORT-session",
            }
        )
    write_tsv(Path(args.output), rows)
    print(json.dumps({"cases": len(rows), "host_fenv": host_fenv, "output": args.output}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--model", required=True)
    generate.add_argument("--stage42-input", required=True)
    generate.add_argument("--image", action="append", default=[])
    generate.add_argument("--out-dir", required=True)
    generate.set_defaults(func=generate_oracles)
    five = subparsers.add_parser("quantize-five")
    five.add_argument("--host-float", required=True)
    five.add_argument("--host-q", required=True)
    five.add_argument("--board-q", required=True)
    five.add_argument("--scale", required=True, type=float)
    five.add_argument("--zero-point", required=True, type=int)
    five.add_argument("--output", required=True)
    five.set_defaults(func=quantize_five_case)
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
