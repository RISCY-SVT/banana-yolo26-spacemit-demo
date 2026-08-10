#!/usr/bin/env python3
"""Generate the factor-isolating Stage65B-R1 XSlim PTQ matrix."""

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

LANES = {
    "B1": ("selection_C50.txt", 50, 0, 1),
    "B2": ("selection_C50.txt", 50, 1, 2),
    "B3": ("selection_C200.txt", 200, 1, 2),
    "B4": ("selection_C500.txt", 500, 1, 2),
    "B5": ("selection_C1000.txt", 1000, 1, 2),
    "B6": ("selection_C500_size_balanced.txt", 500, 1, 2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--list-root", required=True, type=Path)
    parser.add_argument("--preprocess-file", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config-root", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    options = parse_args()
    options.config_root.mkdir(parents=True, exist_ok=True)
    for lane, (list_name, count, precision, finetune) in LANES.items():
        image_list = (options.list_root / list_name).resolve()
        if len(image_list.read_text(encoding="utf-8").splitlines()) != count:
            raise RuntimeError(f"{lane} list does not contain {count} images")
        payload = {
            "model_parameters": {
                "onnx_model": str(options.model.resolve()),
                "output_prefix": f"stage65b_r1_{lane.lower()}_split_s8_qdq",
                "working_dir": str((options.output_root / lane).resolve()),
                "skip_onnxsim": False,
            },
            "calibration_parameters": {
                "calibration_step": count,
                "calibration_batch_size": 1,
                "calibration_device": "cpu",
                "calibration_type": "default",
                "input_parameters": [
                    {
                        "input_name": "images",
                        "input_shape": [1, 3, 640, 640],
                        "file_type": "img",
                        "color_format": "rgb",
                        "mean_value": [0.0, 0.0, 0.0],
                        "std_value": [1.0, 1.0, 1.0],
                        "preprocess_file": (
                            f"{options.preprocess_file.resolve()}:preprocess_impl"
                        ),
                        "data_list_path": str(image_list),
                    }
                ],
            },
            "quantization_parameters": {
                "precision_level": precision,
                "finetune_level": finetune,
                "analysis_enable": True,
                "truncate_var_names": SPLIT_NAMES,
            },
        }
        (options.config_root / f"{lane}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
