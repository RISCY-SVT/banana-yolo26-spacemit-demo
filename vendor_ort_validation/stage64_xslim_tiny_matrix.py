#!/usr/bin/env python3
"""Run hermetic XSlim static-PTQ smoke tests on tiny ONNX models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", required=True)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--models", nargs="+", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dtype_for_onnx(element_type: int) -> np.dtype[Any]:
    return np.dtype(onnx.helper.tensor_dtype_to_np_dtype(element_type))


def shape_for(value: onnx.ValueInfoProto) -> list[int]:
    return [
        int(item.dim_value) if item.HasField("dim_value") and item.dim_value > 0 else 1
        for item in value.type.tensor_type.shape.dim
    ]


def make_data(
    model: onnx.ModelProto, root: Path
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(640)
    source_inputs: dict[str, np.ndarray] = {}
    parameters: list[dict[str, Any]] = []
    for input_index, value in enumerate(model.graph.input):
        shape = shape_for(value)
        dtype = dtype_for_onnx(value.type.tensor_type.elem_type)
        arrays = []
        for sample in range(10):
            if np.issubdtype(dtype, np.floating):
                array = rng.uniform(-1.0, 1.0, size=shape).astype(dtype)
            elif np.issubdtype(dtype, np.integer):
                array = rng.integers(-8, 9, size=shape, dtype=dtype)
            else:
                raise TypeError(f"unsupported input dtype {dtype}")
            arrays.append(array)
            sample_path = root / f"input{input_index}_{sample:02d}.npy"
            np.save(sample_path, array)
        source_inputs[value.name] = arrays[0]
        list_path = root / f"input{input_index}.txt"
        list_path.write_text(
            "".join(f"{root / f'input{input_index}_{sample:02d}.npy'}\n" for sample in range(10)),
            encoding="utf-8",
        )
        parameters.append(
            {
                "input_name": value.name,
                "input_shape": shape,
                "file_type": "npy",
                "data_list_path": str(list_path),
            }
        )
    return source_inputs, parameters


def cpu_run(model_path: Path, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    return session.run(None, inputs)


def classify(
    returncode: int,
    output_exists: bool,
    checker: str,
    semantic: str,
    log_text: str,
) -> str:
    if returncode == 124:
        return "new-failure-timeout"
    if "too many values to unpack" in log_text:
        return "still-broken-reducemax-two-input"
    if returncode != 0:
        return "new-failure"
    if not output_exists or checker != "pass":
        return "new-failure-invalid-output"
    if semantic != "pass":
        return "new-failure-host-semantics"
    return "fixed"


def main() -> int:
    options = parse_args()
    rows: list[dict[str, Any]] = []
    for model_path in sorted(options.models):
        case_root = options.output_root / options.lane / model_path.stem
        case_root.mkdir(parents=True, exist_ok=True)
        model = onnx.load(model_path)
        inputs, input_parameters = make_data(model, case_root / "calibration")
        output_prefix = f"{model_path.stem}.xslim"
        config = {
            "model_parameters": {
                "onnx_model": str(model_path.resolve()),
                "working_dir": str((case_root / "output").resolve()),
                "output_prefix": output_prefix,
                "skip_onnxsim": False,
            },
            "calibration_parameters": {
                "calibration_step": 10,
                "calibration_batch_size": 1,
                "calibration_device": "cpu",
                "calibration_type": "default",
                "input_parameters": input_parameters,
            },
            "quantization_parameters": {
                "precision_level": 0,
                "finetune_level": 0,
                "analysis_enable": False,
            },
        }
        config_path = case_root / "config.json"
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        output_model = case_root / "output" / f"{output_prefix}.onnx"
        log_path = case_root / "xslim.log"
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "0"
        start = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            try:
                process = subprocess.run(
                    [str(options.python), "-m", "xslim", "-c", str(config_path)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    timeout=options.timeout_seconds,
                    check=False,
                )
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                returncode = 124
                log.write(
                    f"\nSTAGE64_TIMEOUT seconds={options.timeout_seconds}\n"
                )
        elapsed = time.monotonic() - start
        checker = "not-run"
        checker_error = ""
        semantic = "not-run"
        max_abs = float("nan")
        output_hash = ""
        if output_model.exists():
            output_hash = sha256(output_model)
            try:
                onnx.checker.check_model(onnx.load(output_model))
                checker = "pass"
            except Exception as exc:  # noqa: BLE001
                checker = "fail"
                checker_error = f"{type(exc).__name__}: {exc}".replace("\n", " ")
            if checker == "pass":
                try:
                    reference = cpu_run(model_path, inputs)
                    candidate = cpu_run(output_model, inputs)
                    if len(reference) != len(candidate):
                        semantic = "fail-output-count"
                    else:
                        differences = [
                            float(
                                np.max(
                                    np.abs(
                                        expected.astype(np.float64)
                                        - actual.astype(np.float64)
                                    )
                                )
                            )
                            for expected, actual in zip(reference, candidate)
                        ]
                        max_abs = max(differences, default=0.0)
                        semantic = (
                            "pass"
                            if all(
                                expected.shape == actual.shape
                                and np.isfinite(actual).all()
                                for expected, actual in zip(reference, candidate)
                            )
                            else "fail"
                        )
                except Exception as exc:  # noqa: BLE001
                    semantic = f"fail:{type(exc).__name__}:{exc}".replace(
                        "\n", " "
                    )
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        rows.append(
            {
                "lane": options.lane,
                "model": model_path.name,
                "model_sha256": sha256(model_path),
                "returncode": returncode,
                "elapsed_seconds": f"{elapsed:.6f}",
                "output_exists": int(output_model.exists()),
                "output_sha256": output_hash,
                "checker": checker,
                "checker_error": checker_error,
                "host_semantic": semantic,
                "max_abs_difference": max_abs,
                "classification": classify(
                    returncode,
                    output_model.exists(),
                    checker,
                    semantic,
                    log_text,
                ),
                "log": str(log_path),
            }
        )

    options.summary.parent.mkdir(parents=True, exist_ok=True)
    with options.summary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
