#!/usr/bin/env python3
"""Run the accepted YOLO26 Q/DQ ONNX model with ORT CPU and dump references."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import onnx

from stage40_skeleton_common import (
    DEFAULT_BOUNDARIES,
    add_outputs,
    input_array,
    make_session,
    save_nhwc_bin,
    save_npy,
    sha256_file,
    write_json,
    write_tsv,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--input-mode", default="synthetic_seeded",
                        choices=["synthetic_seeded", "synthetic_gradient", "zeros"])
    parser.add_argument("--boundary", action="append", default=[])
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--tensor-manifest", required=True)
    parser.add_argument("--report-md", required=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(model_path)
    original_outputs = [o.name for o in model.graph.output]
    boundaries = list(dict.fromkeys(DEFAULT_BOUNDARIES + args.boundary))
    add_outputs(model, boundaries)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        temp_model = Path(f.name)
    onnx.save(model, temp_model)
    try:
        session = make_session(temp_model)
        model_input = session.get_inputs()[0]
        shape = [int(v) for v in model_input.shape]
        x = input_array(shape, args.input_mode)
        input_path = out_dir / "full_model_input.npy"
        import numpy as np
        np.save(input_path, x)
        output_names = original_outputs + boundaries
        arrays = session.run(output_names, {model_input.name: x})
    finally:
        temp_model.unlink(missing_ok=True)

    records = [save_npy("images", x, out_dir)]
    for name, arr in zip(output_names, arrays):
        records.append(save_npy(name, arr, out_dir))
        if arr.ndim == 4 and name.endswith("_QuantizeLinear_Output"):
            records.append(save_nhwc_bin(name, arr, out_dir, safe_suffix(name)))
    write_tsv(Path(args.tensor_manifest), records)

    metadata = {
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "provider": "CPUExecutionProvider",
        "input_name": model_input.name,
        "input_shape": shape,
        "input_mode": args.input_mode,
        "input_path": str(input_path),
        "output_names": output_names,
    }
    write_json(Path(args.metadata_json), metadata)
    Path(args.report_md).write_text(
        "# Full ONNX Runtime CPU Reference Report\n\n"
        f"- model: `{model_path}`\n"
        f"- model_sha256: `{metadata['model_sha256']}`\n"
        "- provider: `CPUExecutionProvider`\n"
        f"- input_name: `{model_input.name}`\n"
        f"- input_shape: `{shape}`\n"
        f"- input_mode: `{args.input_mode}`\n"
        f"- outputs: `{', '.join(output_names)}`\n",
        encoding="utf-8",
    )
    print(metadata)
    return 0


def safe_suffix(name: str) -> str:
    return name.strip("/").replace("/", "__").replace(":", "_")


if __name__ == "__main__":
    raise SystemExit(main())
