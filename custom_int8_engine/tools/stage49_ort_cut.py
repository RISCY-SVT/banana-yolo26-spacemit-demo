#!/usr/bin/env python3
"""Build the Stage49 B120 timing cut without changing the accepted model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx


INPUT_NAME = "/model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output"
OUTPUT_NAME = "/model.6/cv2/act/Mul_output_0_QuantizeLinear_Output"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--fixture", default="F0")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    accepted = onnx.load(args.model)
    inferred = onnx.shape_inference.infer_shapes(accepted)
    cut = onnx.utils.Extractor(inferred).extract_model([INPUT_NAME], [OUTPUT_NAME])
    onnx.checker.check_model(cut)
    cut_path = args.out_dir / "model4_preact_to_model6_output_b120_timing.onnx"
    onnx.save(cut, cut_path)

    input_raw = args.package / "oracles" / args.fixture / "tensor_000_nchw_u8.bin"
    output_raw = args.package / "oracles" / args.fixture / "tensor_016_nchw_u8.bin"
    input_values = np.fromfile(input_raw, dtype=np.uint8)
    output_values = np.fromfile(output_raw, dtype=np.uint8)
    if input_values.size != 1 * 128 * 80 * 80 or output_values.size != 1 * 128 * 40 * 40:
        raise RuntimeError("Stage49 oracle byte count mismatch")
    input_path = args.out_dir / f"{args.fixture}_model4_preact_nchw_u8.npy"
    integer_output_path = args.out_dir / f"{args.fixture}_model6_integer_contract_nchw_u8.npy"
    np.save(input_path, input_values.reshape(1, 128, 80, 80), allow_pickle=False)
    np.save(integer_output_path, output_values.reshape(1, 128, 40, 40), allow_pickle=False)

    manifest = {
        "accepted_model": str(args.model),
        "accepted_model_sha256": sha256(args.model),
        "cut": str(cut_path),
        "cut_sha256": sha256(cut_path),
        "input_name": INPUT_NAME,
        "input_shape": [1, 128, 80, 80],
        "input_dtype": "uint8",
        "input_npy": str(input_path),
        "input_npy_sha256": sha256(input_path),
        "output_name": OUTPUT_NAME,
        "output_shape": [1, 128, 40, 40],
        "output_dtype": "uint8",
        "integer_contract_output_npy": str(integer_output_path),
        "integer_contract_output_npy_sha256": sha256(integer_output_path),
        "ort_role": "B120 performance diagnostic only",
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**manifest, "manifest_sha256": sha256(manifest_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
