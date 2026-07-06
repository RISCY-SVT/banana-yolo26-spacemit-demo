#!/usr/bin/env python3
"""Extract full-shape ONNX Runtime oracle tensors for selected YOLO26 boundaries.

This is a host-side tooling script only. The runtime C++ library must not depend
on ONNX, ONNX Runtime, Python, or protobuf.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


MODEL4_C2F_BOUNDARIES = [
    "/model.4/cv1/conv/Conv_output_0",
    "/model.4/cv1/conv/Conv_output_0_QuantizeLinear_Output",
    "/model.4/cv1/conv/Conv_output_0_DequantizeLinear_Output",
    "/model.4/cv1/act/Mul_output_0",
    "/model.4/Split_output_0",
    "/model.4/Split_output_1",
    "/model.4/Split_output_1_QuantizeLinear_Output",
    "/model.4/Split_output_1_DequantizeLinear_Output",
    "/model.4/m.0/cv1/conv/Conv_output_0",
    "/model.4/m.0/cv1/conv/Conv_output_0_QuantizeLinear_Output",
    "/model.4/m.0/cv1/conv/Conv_output_0_DequantizeLinear_Output",
    "/model.4/m.0/cv1/act/Mul_output_0",
    "/model.4/m.0/cv1/act/Mul_output_0_QuantizeLinear_Output",
    "/model.4/m.0/cv1/act/Mul_output_0_DequantizeLinear_Output",
    "/model.4/m.0/cv2/conv/Conv_output_0",
    "/model.4/m.0/cv2/conv/Conv_output_0_QuantizeLinear_Output",
    "/model.4/m.0/cv2/conv/Conv_output_0_DequantizeLinear_Output",
    "/model.4/m.0/cv2/act/Mul_output_0",
    "/model.4/m.0/Add_output_0",
    "/model.4/Concat_output_0",
    "/model.4/Concat_output_0_QuantizeLinear_Output",
    "/model.4/Concat_output_0_DequantizeLinear_Output",
    "/model.4/cv2/conv/Conv_output_0",
    "/model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output",
    "/model.4/cv2/conv/Conv_output_0_DequantizeLinear_Output",
]


@dataclass(frozen=True)
class TensorRecord:
    name: str
    safe_name: str
    path: str
    sha256: str
    dtype: str
    shape: str
    min_value: str
    max_value: str
    sum_value: str
    scale_name: str
    scale_value: str
    zero_point_name: str
    zero_point_value: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(name: str) -> str:
    return name.strip("/").replace("/", "__").replace(":", "_")


def input_array(shape: list[int], mode: str) -> np.ndarray:
    if mode == "synthetic_seeded":
        rng = np.random.default_rng(20260706)
        return rng.uniform(0.0, 1.0, size=tuple(shape)).astype(np.float32)
    if mode == "synthetic_gradient":
        total = int(np.prod(shape))
        return np.linspace(0.0, 1.0, num=total, dtype=np.float32).reshape(shape)
    if mode == "zeros":
        return np.zeros(shape, dtype=np.float32)
    raise ValueError(f"unsupported input mode: {mode}")


def tensor_elem_type(name: str) -> int:
    if name.endswith("_QuantizeLinear_Output"):
        return TensorProto.UINT8
    return TensorProto.FLOAT


def add_outputs(model: onnx.ModelProto, boundaries: list[str]) -> None:
    existing = {o.name for o in model.graph.output}
    for name in boundaries:
        if name in existing:
            continue
        model.graph.output.append(helper.make_tensor_value_info(name, tensor_elem_type(name), None))


def scalar_initializer(init: dict[str, np.ndarray], name: str) -> str:
    value = init.get(name)
    if value is None:
        return ""
    flat = value.reshape(-1)
    if flat.size != 1:
        return ""
    item = flat[0]
    if np.issubdtype(value.dtype, np.integer):
        return str(int(item))
    return f"{float(item):.12g}"


def qdq_metadata(init: dict[str, np.ndarray], tensor_name: str) -> tuple[str, str, str, str]:
    candidates = []
    if tensor_name.endswith("_QuantizeLinear_Output"):
        base = tensor_name.removesuffix("_QuantizeLinear_Output")
        candidates.append(base)
    if tensor_name.endswith("_DequantizeLinear_Output"):
        base = tensor_name.removesuffix("_DequantizeLinear_Output")
        candidates.append(base)
    candidates.append(tensor_name)
    for base in candidates:
        scale_name = f"{base}_scale"
        zp_name = f"{base}_zero_point"
        scale_value = scalar_initializer(init, scale_name)
        zp_value = scalar_initializer(init, zp_name)
        if scale_value or zp_value:
            return scale_name, scale_value, zp_name, zp_value
    return "", "", "", ""


def format_float(value: np.generic | float | int) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.12g}"


def summarize_tensor(out_dir: Path, init: dict[str, np.ndarray], name: str, arr: np.ndarray) -> TensorRecord:
    file_name = f"{safe_name(name)}.npy"
    out_path = out_dir / file_name
    np.save(out_path, arr)
    flat = arr.reshape(-1)
    scale_name, scale_value, zp_name, zp_value = qdq_metadata(init, name)
    return TensorRecord(
        name=name,
        safe_name=safe_name(name),
        path=str(out_path),
        sha256=sha256_file(out_path),
        dtype=str(arr.dtype),
        shape="x".join(str(v) for v in arr.shape),
        min_value=format_float(flat.min()) if flat.size else "",
        max_value=format_float(flat.max()) if flat.size else "",
        sum_value=format_float(flat.astype(np.float64).sum()) if flat.size else "",
        scale_name=scale_name,
        scale_value=scale_value,
        zero_point_name=zp_name,
        zero_point_value=zp_value,
    )


def write_tsv(path: Path, records: list[TensorRecord]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TensorRecord.__dataclass_fields__.keys()), delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--input-mode", default="synthetic_seeded", choices=["synthetic_seeded", "synthetic_gradient", "zeros"])
    parser.add_argument("--boundary-set", default="model4_c2f", choices=["model4_c2f"])
    parser.add_argument("--boundary", action="append", default=[])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checksums", required=True)
    parser.add_argument("--metadata-json", required=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    boundaries = list(MODEL4_C2F_BOUNDARIES if args.boundary_set == "model4_c2f" else [])
    boundaries.extend(args.boundary)
    boundaries = list(dict.fromkeys(boundaries))

    model = onnx.load(model_path)
    init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    add_outputs(model, boundaries)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        temp_model = Path(f.name)
    onnx.save(model, temp_model)
    try:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        session = ort.InferenceSession(str(temp_model), sess_options=so, providers=["CPUExecutionProvider"])
        model_input = session.get_inputs()[0]
        input_shape = [int(v) for v in model_input.shape]
        x = input_array(input_shape, args.input_mode)
        input_path = out_dir / f"input__{args.input_mode}.npy"
        np.save(input_path, x)
        outputs = session.run(boundaries, {model_input.name: x})
    finally:
        temp_model.unlink(missing_ok=True)

    records = [summarize_tensor(out_dir, init, name, arr) for name, arr in zip(boundaries, outputs)]
    write_tsv(Path(args.manifest), records)
    write_tsv(Path(args.checksums), records)

    metadata = {
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "provider": "CPUExecutionProvider",
        "input_name": model_input.name,
        "input_shape": input_shape,
        "input_mode": args.input_mode,
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "boundary_set": args.boundary_set,
        "boundary_count": len(records),
        "output_dir": str(out_dir),
        "opsets": [{"domain": o.domain, "version": o.version} for o in model.opset_import],
        "ir_version": model.ir_version,
    }
    Path(args.metadata_json).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, sort_keys=True))
    for record in records:
        print(f"{record.name}\t{record.dtype}\t{record.shape}\t{record.sha256}\t{record.sum_value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
