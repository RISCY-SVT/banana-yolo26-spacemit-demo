#!/usr/bin/env python3
"""Build and run a same-input ONNX Runtime cut for the Stage22 model4 C2f gate.

This is host-side tooling only. The runtime C++ library must not depend on
ONNX, ONNX Runtime, Python, or protobuf.
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
import onnx.utils
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


CUT_INPUT = "/model.4/cv1/conv/Conv_output_0_QuantizeLinear_Output"
CUT_OUTPUT = "/model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output"


@dataclass(frozen=True)
class TensorRecord:
    name: str
    npy_path: str
    bin_path: str
    sha256_npy: str
    sha256_bin: str
    dtype: str
    shape_nchw: str
    shape_nhwc: str
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


def add_outputs(model: onnx.ModelProto, outputs: list[str]) -> None:
    existing = {o.name for o in model.graph.output}
    for name in outputs:
        if name not in existing:
            model.graph.output.append(helper.make_tensor_value_info(name, TensorProto.UINT8, None))


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
        candidates.append(tensor_name.removesuffix("_QuantizeLinear_Output"))
    if tensor_name.endswith("_DequantizeLinear_Output"):
        candidates.append(tensor_name.removesuffix("_DequantizeLinear_Output"))
    candidates.append(tensor_name)
    for base in candidates:
        scale_name = f"{base}_scale"
        zp_name = f"{base}_zero_point"
        scale_value = scalar_initializer(init, scale_name)
        zp_value = scalar_initializer(init, zp_name)
        if scale_value or zp_value:
            return scale_name, scale_value, zp_name, zp_value
    return "", "", "", ""


def format_value(value: np.generic | float | int) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.12g}"


def nchw_to_nhwc(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 4:
        raise ValueError(f"expected NCHW 4D tensor, got shape {arr.shape}")
    return np.ascontiguousarray(np.transpose(arr, (0, 2, 3, 1)))


def write_tensor(out_dir: Path, init: dict[str, np.ndarray], name: str, arr: np.ndarray, suffix: str) -> TensorRecord:
    npy_path = out_dir / f"{suffix}.npy"
    bin_path = out_dir / f"{suffix}_nhwc.bin"
    np.save(npy_path, arr)
    nhwc = nchw_to_nhwc(arr)
    nhwc.tofile(bin_path)
    flat = arr.reshape(-1)
    scale_name, scale_value, zp_name, zp_value = qdq_metadata(init, name)
    return TensorRecord(
        name=name,
        npy_path=str(npy_path),
        bin_path=str(bin_path),
        sha256_npy=sha256_file(npy_path),
        sha256_bin=sha256_file(bin_path),
        dtype=str(arr.dtype),
        shape_nchw="x".join(str(v) for v in arr.shape),
        shape_nhwc="x".join(str(v) for v in nhwc.shape),
        min_value=format_value(flat.min()) if flat.size else "",
        max_value=format_value(flat.max()) if flat.size else "",
        sum_value=format_value(flat.astype(np.float64).sum()) if flat.size else "",
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


def run_session(model_path: Path, output_names: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=so, providers=["CPUExecutionProvider"])
    return session.run(output_names, feed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--input-mode", default="synthetic_seeded", choices=["synthetic_seeded", "synthetic_gradient", "zeros"])
    parser.add_argument("--cut-model", required=True)
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--manifest-tsv", required=True)
    parser.add_argument("--report-md", required=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cut_model_path = Path(args.cut_model)
    cut_model_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(model_path)
    init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    add_outputs(model, [CUT_INPUT, CUT_OUTPUT])

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        full_outputs_model = Path(f.name)
    onnx.save(model, full_outputs_model)
    try:
        session = ort.InferenceSession(str(full_outputs_model), providers=["CPUExecutionProvider"])
        model_input = session.get_inputs()[0]
        input_shape = [int(v) for v in model_input.shape]
        full_input = input_array(input_shape, args.input_mode)
        full_outputs = session.run([CUT_INPUT, CUT_OUTPUT], {model_input.name: full_input})
    finally:
        full_outputs_model.unlink(missing_ok=True)

    cut_input, full_output = full_outputs
    onnx.utils.extract_model(str(model_path), str(cut_model_path), [CUT_INPUT], [CUT_OUTPUT])
    cut_output = run_session(cut_model_path, [CUT_OUTPUT], {CUT_INPUT: cut_input})[0]

    cut_mismatch = int(np.count_nonzero(cut_output != full_output))
    max_abs_diff = int(np.max(np.abs(cut_output.astype(np.int16) - full_output.astype(np.int16)))) if full_output.size else 0

    input_record = write_tensor(out_dir, init, CUT_INPUT, cut_input, "model4_cv1_conv_q_u8")
    output_record = write_tensor(out_dir, init, CUT_OUTPUT, cut_output, "model4_cv2_conv_q_u8_expected")
    full_output_record = write_tensor(out_dir, init, CUT_OUTPUT, full_output, "model4_cv2_conv_q_u8_full_model")
    records = [input_record, output_record, full_output_record]
    write_tsv(Path(args.manifest_tsv), records)

    metadata = {
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "cut_model": str(cut_model_path),
        "cut_model_sha256": sha256_file(cut_model_path),
        "provider": "CPUExecutionProvider",
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
        "full_model_input_name": model_input.name,
        "full_model_input_shape": input_shape,
        "input_mode": args.input_mode,
        "cut_input": CUT_INPUT,
        "cut_output": CUT_OUTPUT,
        "cut_input_nhwc_bin": input_record.bin_path,
        "cut_input_nhwc_bin_sha256": input_record.sha256_bin,
        "cut_output_nhwc_bin": output_record.bin_path,
        "cut_output_nhwc_bin_sha256": output_record.sha256_bin,
        "cut_vs_full_model_mismatches": cut_mismatch,
        "cut_vs_full_model_max_abs_diff": max_abs_diff,
        "opsets": [{"domain": o.domain, "version": o.version} for o in model.opset_import],
        "ir_version": model.ir_version,
    }
    Path(args.metadata_json).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    Path(args.report_md).write_text(
        "# ONNX Cut Oracle Report\n\n"
        f"- model: `{model_path}`\n"
        f"- model_sha256: `{metadata['model_sha256']}`\n"
        f"- cut_model: `{cut_model_path}`\n"
        f"- cut_model_sha256: `{metadata['cut_model_sha256']}`\n"
        f"- provider: `CPUExecutionProvider`\n"
        f"- onnx: `{onnx.__version__}`\n"
        f"- onnxruntime: `{ort.__version__}`\n"
        f"- cut_input: `{CUT_INPUT}`\n"
        f"- cut_output: `{CUT_OUTPUT}`\n"
        f"- input_nhwc_bin_sha256: `{input_record.sha256_bin}`\n"
        f"- output_nhwc_bin_sha256: `{output_record.sha256_bin}`\n"
        f"- cut_vs_full_model_mismatches: `{cut_mismatch}`\n"
        f"- cut_vs_full_model_max_abs_diff: `{max_abs_diff}`\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, sort_keys=True))
    return 0 if cut_mismatch == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
