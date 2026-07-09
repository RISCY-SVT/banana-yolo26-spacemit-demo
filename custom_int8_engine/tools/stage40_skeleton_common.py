#!/usr/bin/env python3
"""Shared helpers for Stage40 YOLO26 full-model skeleton tooling."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


MODEL4_CUT_INPUT = "/model.4/cv1/conv/Conv_output_0_QuantizeLinear_Output"
MODEL4_CUT_OUTPUT = "/model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output"
DEFAULT_BOUNDARIES = [MODEL4_CUT_INPUT, MODEL4_CUT_OUTPUT]


@dataclass(frozen=True)
class TensorSummary:
    name: str
    path: str
    sha256: str
    dtype: str
    shape: str
    min_value: str
    max_value: str
    sum_value: str


@dataclass(frozen=True)
class CompareSummary:
    name: str
    status: str
    lhs_shape: str
    rhs_shape: str
    lhs_dtype: str
    rhs_dtype: str
    lhs_sha256: str
    rhs_sha256: str
    total: int
    mismatches: int
    max_abs_diff: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def safe_name(name: str) -> str:
    return name.strip("/").replace("/", "__").replace(":", "_")


def input_array(shape: Iterable[int], mode: str) -> np.ndarray:
    shape = tuple(int(v) for v in shape)
    if mode == "synthetic_seeded":
        rng = np.random.default_rng(20260709)
        return rng.uniform(0.0, 1.0, size=shape).astype(np.float32)
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


def add_outputs(model: onnx.ModelProto, outputs: Iterable[str]) -> None:
    existing = {o.name for o in model.graph.output}
    for name in outputs:
        if name not in existing:
            model.graph.output.append(helper.make_tensor_value_info(name, tensor_elem_type(name), None))


def session_options() -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return so


def make_session(model_path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(model_path), sess_options=session_options(), providers=["CPUExecutionProvider"])


def run_session(model_path: Path, output_names: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
    return make_session(model_path).run(output_names, feed)


def time_session(model_path: Path,
                 output_names: list[str],
                 feed: dict[str, np.ndarray],
                 warmup: int,
                 runs: int) -> tuple[list[np.ndarray], float]:
    session = make_session(model_path)
    for _ in range(max(0, warmup)):
        session.run(output_names, feed)
    start = time.perf_counter()
    result: list[np.ndarray] = []
    for _ in range(max(1, runs)):
        result = session.run(output_names, feed)
    elapsed_us = (time.perf_counter() - start) * 1_000_000.0 / max(1, runs)
    return result, elapsed_us


def summarize_array(name: str, arr: np.ndarray, path: Path) -> TensorSummary:
    flat = arr.reshape(-1)
    if np.issubdtype(arr.dtype, np.integer):
        min_value = str(int(flat.min())) if flat.size else ""
        max_value = str(int(flat.max())) if flat.size else ""
    else:
        min_value = f"{float(flat.min()):.12g}" if flat.size else ""
        max_value = f"{float(flat.max()):.12g}" if flat.size else ""
    return TensorSummary(
        name=name,
        path=str(path),
        sha256=sha256_file(path),
        dtype=str(arr.dtype),
        shape="x".join(str(v) for v in arr.shape),
        min_value=min_value,
        max_value=max_value,
        sum_value=f"{float(flat.astype(np.float64).sum()):.12g}" if flat.size else "",
    )


def save_npy(name: str, arr: np.ndarray, out_dir: Path) -> TensorSummary:
    path = out_dir / f"{safe_name(name)}.npy"
    np.save(path, arr)
    return summarize_array(name, arr, path)


def save_nhwc_bin(name: str, arr_nchw: np.ndarray, out_dir: Path, suffix: str | None = None) -> TensorSummary:
    if arr_nchw.ndim != 4:
        raise ValueError(f"expected 4D NCHW tensor for NHWC bin, got {arr_nchw.shape}")
    arr_nhwc = np.ascontiguousarray(np.transpose(arr_nchw, (0, 2, 3, 1)))
    path = out_dir / f"{suffix or safe_name(name)}_nhwc.bin"
    arr_nhwc.tofile(path)
    return summarize_array(name, arr_nhwc, path)


def load_nhwc_bin(path: Path, shape_nchw: Iterable[int], dtype: np.dtype) -> np.ndarray:
    shape = tuple(int(v) for v in shape_nchw)
    if len(shape) != 4:
        raise ValueError(f"expected 4D NCHW shape, got {shape}")
    nhwc_shape = (shape[0], shape[2], shape[3], shape[1])
    arr = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(nhwc_shape))
    if arr.size != expected:
        raise ValueError(f"{path} has {arr.size} elements, expected {expected}")
    return np.ascontiguousarray(np.transpose(arr.reshape(nhwc_shape), (0, 3, 1, 2)))


def compare_arrays(name: str, lhs: np.ndarray, rhs: np.ndarray) -> CompareSummary:
    same_shape = lhs.shape == rhs.shape
    lhs_sha = sha256_array(lhs)
    rhs_sha = sha256_array(rhs)
    if not same_shape:
        return CompareSummary(name, "shape_mismatch", "x".join(map(str, lhs.shape)), "x".join(map(str, rhs.shape)),
                              str(lhs.dtype), str(rhs.dtype), lhs_sha, rhs_sha, 0, -1, "")
    if np.issubdtype(lhs.dtype, np.floating) or np.issubdtype(rhs.dtype, np.floating):
        diff = np.abs(lhs.astype(np.float64) - rhs.astype(np.float64))
        mismatches = int(np.count_nonzero(diff != 0.0))
        max_abs = f"{float(diff.max()):.12g}" if diff.size else "0"
    else:
        diff = np.abs(lhs.astype(np.int64) - rhs.astype(np.int64))
        mismatches = int(np.count_nonzero(diff))
        max_abs = str(int(diff.max())) if diff.size else "0"
    status = "pass" if mismatches == 0 and lhs_sha == rhs_sha else "fail"
    return CompareSummary(name, status, "x".join(map(str, lhs.shape)), "x".join(map(str, rhs.shape)),
                          str(lhs.dtype), str(rhs.dtype), lhs_sha, rhs_sha, int(lhs.size), mismatches, max_abs)


def write_tsv(path: Path, records: Iterable[object]) -> None:
    rows = [asdict(record) for record in records]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def initializer_metadata(model: onnx.ModelProto, tensor_name: str) -> dict[str, str]:
    init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    candidates = []
    if tensor_name.endswith("_QuantizeLinear_Output"):
        candidates.append(tensor_name.removesuffix("_QuantizeLinear_Output"))
    if tensor_name.endswith("_DequantizeLinear_Output"):
        candidates.append(tensor_name.removesuffix("_DequantizeLinear_Output"))
    candidates.append(tensor_name)
    for base in candidates:
        scale_name = f"{base}_scale"
        zp_name = f"{base}_zero_point"
        if scale_name in init or zp_name in init:
            scale = init.get(scale_name)
            zp = init.get(zp_name)
            return {
                "scale_name": scale_name if scale is not None else "",
                "scale_value": str(float(scale.reshape(-1)[0])) if scale is not None and scale.size == 1 else "",
                "zero_point_name": zp_name if zp is not None else "",
                "zero_point_value": str(int(zp.reshape(-1)[0])) if zp is not None and zp.size == 1 else "",
            }
    return {"scale_name": "", "scale_value": "", "zero_point_name": "", "zero_point_value": ""}
