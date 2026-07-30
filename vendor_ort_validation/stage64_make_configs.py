#!/usr/bin/env python3
"""Generate the bounded Stage64 XSlim configuration matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SPLIT_NAMES = [
    "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0",
    "/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0",
    "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0",
    "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0",
    "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0",
    "/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--calibration-list", required=True, type=Path)
    parser.add_argument("--preprocess-file", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config-root", required=True, type=Path)
    return parser.parse_args()


def config(
    model: Path,
    calibration_list: Path,
    output_dir: Path,
    prefix: str,
    preprocess: Path | None,
    split: bool,
) -> dict[str, object]:
    input_parameters: dict[str, object] = {
        "input_name": "images",
        "input_shape": [1, 3, 640, 640],
        "file_type": "img",
        "color_format": "rgb",
        "mean_value": [0.0, 0.0, 0.0],
        "std_value": [255.0, 255.0, 255.0],
        "data_list_path": str(calibration_list.resolve()),
    }
    if preprocess is not None:
        input_parameters["std_value"] = [1.0, 1.0, 1.0]
        input_parameters["preprocess_file"] = (
            f"{preprocess.resolve()}:preprocess_impl"
        )
    quantization_parameters: dict[str, object] = {
        "precision_level": 0,
        "finetune_level": 1,
        "analysis_enable": True,
    }
    if split:
        quantization_parameters["truncate_var_names"] = SPLIT_NAMES
    return {
        "model_parameters": {
            "onnx_model": str(model.resolve()),
            "output_prefix": prefix,
            "working_dir": str(output_dir.resolve()),
            "skip_onnxsim": False,
        },
        "calibration_parameters": {
            "calibration_step": 50,
            "calibration_batch_size": 1,
            "calibration_device": "cpu",
            "calibration_type": "default",
            "input_parameters": [input_parameters],
        },
        "quantization_parameters": quantization_parameters,
    }


def main() -> int:
    options = parse_args()
    options.config_root.mkdir(parents=True, exist_ok=True)
    lanes = [
        ("R211_VENDOR_LITERAL_SPLIT", None, True),
        ("R211_PROJECT_EXACT_SPLIT", options.preprocess_file, True),
        ("V9A33_VENDOR_LITERAL_SPLIT", None, True),
        ("V9A33_PROJECT_EXACT_SPLIT", options.preprocess_file, True),
        ("R211_DIRECT_E2E", options.preprocess_file, False),
        ("V9A33_DIRECT_E2E", options.preprocess_file, False),
    ]
    for name, preprocess, split in lanes:
        output_dir = options.output_root / name
        payload = config(
            options.model,
            options.calibration_list,
            output_dir,
            name.lower(),
            preprocess,
            split,
        )
        (options.config_root / f"{name}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
