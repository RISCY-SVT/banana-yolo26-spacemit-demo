#!/usr/bin/env python3
"""Build B120 performance-diagnostic cuts for the Stage51 model9 region."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx


COMBINED_INPUT = "/model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output"
REGION_INPUT = "/model.8/cv2/act/Mul_output_0_QuantizeLinear_Output"
REGION_OUTPUT = "/model.9/Add_output_0_QuantizeLinear_Output"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_cut(model: onnx.ModelProto, input_name: str, output_name: str, path: Path) -> None:
    cut = onnx.utils.Extractor(model).extract_model([input_name], [output_name])
    onnx.checker.check_model(cut)
    onnx.save(cut, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--fixture", default="F0")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    inferred = onnx.shape_inference.infer_shapes(onnx.load(args.model))
    region_cut = args.out_dir / "model8_to_model9_b120_timing.onnx"
    combined_cut = args.out_dir / "model4_preact_to_model9_b120_timing.onnx"
    save_cut(inferred, REGION_INPUT, REGION_OUTPUT, region_cut)
    save_cut(inferred, COMBINED_INPUT, REGION_OUTPUT, combined_cut)

    package_meta = json.loads((args.package / "package.json").read_text(encoding="utf-8"))
    region_input_id = int(package_meta["next_region_input_tensor_id"])
    combined_input_id = int(package_meta["input_tensor_id"])
    output_id = int(package_meta["next_region_output_tensor_id"])

    def tensor_npy(tensor_id: int, shape: tuple[int, ...], label: str) -> Path:
        raw = args.package / "oracles" / args.fixture / f"tensor_{tensor_id:03d}_nchw_u8.bin"
        values = np.fromfile(raw, dtype=np.uint8)
        if values.size != int(np.prod(shape)):
            raise RuntimeError(f"{label} oracle byte count mismatch")
        path = args.out_dir / f"{args.fixture}_{label}_nchw_u8.npy"
        np.save(path, values.reshape(shape), allow_pickle=False)
        return path

    region_input = tensor_npy(region_input_id, (1, 256, 20, 20), "model8_output")
    combined_input = tensor_npy(combined_input_id, (1, 128, 80, 80), "model4_preact")
    integer_output = tensor_npy(output_id, (1, 256, 20, 20), "model9_integer_output")
    manifest = {
        "accepted_model": str(args.model.resolve()),
        "accepted_model_sha256": sha256(args.model),
        "combined_cut": str(combined_cut),
        "combined_cut_sha256": sha256(combined_cut),
        "combined_input_name": COMBINED_INPUT,
        "combined_input_npy": str(combined_input),
        "combined_input_npy_sha256": sha256(combined_input),
        "integer_output_npy": str(integer_output),
        "integer_output_npy_sha256": sha256(integer_output),
        "ort_role": "B120 performance diagnostic only",
        "output_name": REGION_OUTPUT,
        "region_cut": str(region_cut),
        "region_cut_sha256": sha256(region_cut),
        "region_input_name": REGION_INPUT,
        "region_input_npy": str(region_input),
        "region_input_npy_sha256": sha256(region_input),
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**manifest, "manifest_sha256": sha256(manifest_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
