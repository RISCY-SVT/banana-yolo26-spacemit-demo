#!/usr/bin/env python3
"""Generate explicit-contract host ORT boundary and model16 oracle artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnx.utils
import onnxruntime as ort
from onnx import TensorProto, numpy_helper, shape_inference


DEFAULT_BOUNDARIES = [
    "/model.0/conv/Conv_output_0_QuantizeLinear_Output",
    "/model.0/act/Mul_output_0_QuantizeLinear_Output",
    "/model.1/conv/Conv_output_0_QuantizeLinear_Output",
    "/model.1/act/Mul_output_0_QuantizeLinear_Output",
    "/model.3/act/Mul_output_0_QuantizeLinear_Output",
    "/model.4/cv1/conv/Conv_output_0_QuantizeLinear_Output",
    "/model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output",
    "/model.4/cv2/act/Mul_output_0_QuantizeLinear_Output",
    "/model.16/cv2/act/Mul_output_0_QuantizeLinear_Output",
    "/model.22/cv2/act/Mul_output_0_QuantizeLinear_Output",
    "output0",
]

MODEL16_SEMANTIC_INPUT = "/model.15/Concat_output_0_DequantizeLinear_Output"
MODEL16_SEMANTIC_OUTPUT = "/model.16/cv2/act/Mul_output_0_DequantizeLinear_Output"
MODEL16_QUANT_INPUT = "/model.15/Concat_output_0_QuantizeLinear_Output"
MODEL16_QUANT_OUTPUT = "/model.16/cv2/act/Mul_output_0_QuantizeLinear_Output"


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


def session_options(args: argparse.Namespace) -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.enable_mem_pattern = bool(args.memory_pattern)
    options.enable_cpu_mem_arena = bool(args.cpu_arena)
    options.add_session_config_entry("session.intra_op.allow_spinning", str(args.thread_spinning))
    options.add_session_config_entry("session.inter_op.allow_spinning", str(args.thread_spinning))
    levels = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    options.graph_optimization_level = levels[args.opt_level]
    options.log_severity_level = args.log_severity
    options.log_verbosity_level = args.log_verbosity
    return options


def make_session(path: Path, args: argparse.Namespace) -> ort.InferenceSession:
    return ort.InferenceSession(
        str(path), sess_options=session_options(args), providers=["CPUExecutionProvider"]
    )


def extract_cut(model_path: Path, output: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.utils.extract_model(str(model_path), str(out_path), ["images"], [output])


def value_metadata(model: onnx.ModelProto) -> dict[str, tuple[str, str]]:
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass
    records: dict[str, tuple[str, str]] = {}
    tensors = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    for value in tensors:
        tensor_type = value.type.tensor_type
        dtype = TensorProto.DataType.Name(tensor_type.elem_type) if tensor_type.elem_type else "UNDEFINED"
        dims: list[str] = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(str(dim.dim_value))
            elif dim.HasField("dim_param"):
                dims.append(dim.dim_param)
            else:
                dims.append("?")
        records[value.name] = (dtype, "x".join(dims))
    return records


def quant_metadata(model: onnx.ModelProto, output: str) -> tuple[str, str, str, str]:
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    producer = next((node for node in model.graph.node if output in node.output), None)
    if producer is None or producer.op_type != "QuantizeLinear" or len(producer.input) < 3:
        return "", "", "", ""
    scale_name, zero_name = producer.input[1], producer.input[2]
    scale = initializers.get(scale_name)
    zero = initializers.get(zero_name)
    scale_value = "" if scale is None or scale.size != 1 else f"{float(scale.reshape(-1)[0]):.12g}"
    zero_value = "" if zero is None or zero.size != 1 else str(int(zero.reshape(-1)[0]))
    return scale_name, scale_value, zero_name, zero_value


def producer_consumers(model: onnx.ModelProto, name: str) -> tuple[str, str]:
    producer = next((node.name or node.op_type for node in model.graph.node if name in node.output), "graph_input")
    consumers = [node.name or node.op_type for node in model.graph.node if name in node.input]
    return producer, ",".join(consumers)


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def run_cut(cut_path: Path, output: str, images: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    session = make_session(cut_path, args)
    return np.asarray(session.run([output], {"images": images})[0])


def generate_model16(
    model_path: Path,
    model: onnx.ModelProto,
    images: np.ndarray,
    out_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    contracts = [
        ("semantic", MODEL16_SEMANTIC_INPUT, MODEL16_SEMANTIC_OUTPUT),
        ("quantized", MODEL16_QUANT_INPUT, MODEL16_QUANT_OUTPUT),
    ]
    metadata = value_metadata(model)
    for label, input_name, output_name in contracts:
        prefix_path = out_dir / f"model16_{label}_prefix.onnx"
        cut_path = out_dir / f"model16_{label}_cut.onnx"
        onnx.utils.extract_model(str(model_path), str(prefix_path), ["images"], [input_name])
        onnx.utils.extract_model(str(model_path), str(cut_path), [input_name], [output_name])
        input_value = run_cut(prefix_path, input_name, images, args)
        cut_session = make_session(cut_path, args)
        output_value = np.asarray(cut_session.run([output_name], {input_name: input_value})[0])
        input_path = out_dir / f"model16_{label}_input.npy"
        output_path = out_dir / f"model16_{label}_output.npy"
        np.save(input_path, input_value)
        np.save(output_path, output_value)
        input_dtype, input_shape = metadata.get(input_name, (str(input_value.dtype), "x".join(map(str, input_value.shape))))
        output_dtype, output_shape = metadata.get(
            output_name, (str(output_value.dtype), "x".join(map(str, output_value.shape)))
        )
        in_scale_name, in_scale, in_zero_name, in_zero = quant_metadata(model, input_name)
        out_scale_name, out_scale, out_zero_name, out_zero = quant_metadata(model, output_name)
        records.append(
            {
                "contract": label,
                "model_sha256": sha256_file(model_path),
                "cut_path": str(cut_path),
                "cut_sha256": sha256_file(cut_path),
                "input_name": input_name,
                "input_dtype": input_dtype,
                "input_shape": input_shape,
                "input_layout": "NCHW" if input_value.ndim == 4 else "graph-native",
                "input_scale_name": in_scale_name,
                "input_scale": in_scale,
                "input_zero_point_name": in_zero_name,
                "input_zero_point": in_zero,
                "input_npy": str(input_path),
                "input_npy_sha256": sha256_file(input_path),
                "input_raw_sha256": sha256_array(input_value),
                "output_name": output_name,
                "output_dtype": output_dtype,
                "output_shape": output_shape,
                "output_layout": "NCHW" if output_value.ndim == 4 else "graph-native",
                "output_scale_name": out_scale_name,
                "output_scale": out_scale,
                "output_zero_point_name": out_zero_name,
                "output_zero_point": out_zero,
                "output_npy": str(output_path),
                "output_npy_sha256": sha256_file(output_path),
                "output_raw_sha256": sha256_array(output_value),
                "ort_version": ort.__version__,
                "provider": "CPUExecutionProvider",
                "opt_level": args.opt_level,
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-npy", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--opt-level", choices=["disable", "basic", "extended", "all"], required=True)
    parser.add_argument("--memory-pattern", type=int, choices=[0, 1], default=0)
    parser.add_argument("--cpu-arena", type=int, choices=[0, 1], default=0)
    parser.add_argument("--thread-spinning", type=int, choices=[0, 1], default=0)
    parser.add_argument("--log-severity", type=int, default=2)
    parser.add_argument("--log-verbosity", type=int, default=0)
    parser.add_argument("--boundary", action="append", default=[])
    parser.add_argument("--include-model16", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    input_path = Path(args.input_npy).resolve()
    out_dir = Path(args.out_dir).resolve()
    cut_dir = out_dir / "cuts"
    tensor_dir = out_dir / "tensors"
    cut_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(model_path)
    images = np.load(input_path, allow_pickle=False)
    if images.dtype != np.float32 or images.shape != (1, 3, 640, 640):
        raise ValueError(f"expected float32 1x3x640x640 images, got {images.dtype} {images.shape}")
    all_outputs = {output for node in model.graph.node for output in node.output}
    all_outputs.update(output.name for output in model.graph.output)
    boundaries = args.boundary or DEFAULT_BOUNDARIES
    missing = [name for name in boundaries if name not in all_outputs]
    if missing:
        raise ValueError(f"missing graph boundaries: {missing}")

    metadata = value_metadata(model)
    rows: list[dict[str, object]] = []
    for index, boundary in enumerate(boundaries):
        cut_path = cut_dir / f"{index:02d}_{safe_name(boundary)}.onnx"
        extract_cut(model_path, boundary, cut_path)
        output = run_cut(cut_path, boundary, images, args)
        npy_path = tensor_dir / f"{index:02d}_{safe_name(boundary)}.npy"
        raw_path = tensor_dir / f"{index:02d}_{safe_name(boundary)}.bin"
        np.save(npy_path, output)
        np.ascontiguousarray(output).tofile(raw_path)
        dtype, shape = metadata.get(boundary, (str(output.dtype), "x".join(map(str, output.shape))))
        producer, consumers = producer_consumers(model, boundary)
        scale_name, scale, zero_name, zero = quant_metadata(model, boundary)
        cut = onnx.load(cut_path)
        rows.append(
            {
                "index": index,
                "tensor_name": boundary,
                "producer": producer,
                "consumers": consumers,
                "dtype": dtype,
                "shape": shape,
                "layout": "NCHW" if output.ndim == 4 else "graph-native",
                "scale_name": scale_name,
                "scale": scale,
                "zero_point_name": zero_name,
                "zero_point": zero,
                "element_count": output.size,
                "min": float(np.nanmin(output)),
                "max": float(np.nanmax(output)),
                "mean": float(np.nanmean(output.astype(np.float64))),
                "sum": float(np.nansum(output.astype(np.float64))),
                "nonfinite_count": int(np.count_nonzero(~np.isfinite(output)))
                if np.issubdtype(output.dtype, np.floating)
                else 0,
                "cut_path": str(cut_path),
                "cut_sha256": sha256_file(cut_path),
                "cut_nodes": len(cut.graph.node),
                "output_npy": str(npy_path),
                "output_npy_sha256": sha256_file(npy_path),
                "output_raw": str(raw_path),
                "output_raw_sha256": sha256_file(raw_path),
            }
        )

    write_tsv(out_dir / "boundary_manifest.tsv", rows)
    model16_rows = generate_model16(model_path, model, images, out_dir / "model16", args) if args.include_model16 else []
    if model16_rows:
        write_tsv(out_dir / "model16_oracle_manifest.tsv", model16_rows)
    summary = {
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "input": str(input_path),
        "input_npy_sha256": sha256_file(input_path),
        "input_raw_sha256": sha256_array(images),
        "ort_version": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "selected_provider": "CPUExecutionProvider",
        "opt_level": args.opt_level,
        "execution_mode": "sequential",
        "intra_threads": 1,
        "inter_threads": 1,
        "memory_pattern": args.memory_pattern,
        "cpu_arena": args.cpu_arena,
        "thread_spinning": args.thread_spinning,
        "boundary_count": len(rows),
        "model16_contract_count": len(model16_rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
