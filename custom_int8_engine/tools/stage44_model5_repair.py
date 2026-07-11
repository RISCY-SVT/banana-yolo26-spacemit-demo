#!/usr/bin/env python3
"""Generate and validate Stage44 model4/model5 paired-island ONNX cuts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnx.utils
import onnxruntime as ort


MODEL4_INPUT = "/model.4/cv1/conv/Conv_output_0_QuantizeLinear_Output"
MODEL4_PREACT = "/model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output"
MODEL4_POSTACT = "/model.4/cv2/act/Mul_output_0_QuantizeLinear_Output"
MODEL5_POSTACT = "/model.5/act/Mul_output_0_QuantizeLinear_Output"
FINAL_OUTPUT = "output0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.enable_mem_pattern = True
    options.enable_cpu_mem_arena = True
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def extract(model: Path, output: Path, inputs: list[str], outputs: list[str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.utils.extract_model(str(model), str(output), inputs, outputs)
    onnx.checker.check_model(onnx.load(output, load_external_data=False))


def generate(args: argparse.Namespace) -> int:
    model = args.model.resolve()
    output = args.output.resolve()
    cuts = output / "cuts"
    prefix = cuts / "prefix_images_to_model4_input.onnx"
    suffix_a = cuts / "suffix_model4_output_to_output0.onnx"
    suffix_b = cuts / "suffix_model4_postact_model5_output_to_output0.onnx"
    boundary = cuts / "images_to_stage44_boundaries.onnx"

    extract(model, prefix, ["images"], [MODEL4_INPUT])
    extract(model, suffix_a, [MODEL4_PREACT], [FINAL_OUTPUT])
    extract(model, suffix_b, [MODEL4_POSTACT, MODEL5_POSTACT], [FINAL_OUTPUT])
    extract(model, boundary, ["images"], [MODEL4_PREACT, MODEL4_POSTACT, MODEL5_POSTACT, FINAL_OUTPUT])

    image = np.load(args.input, allow_pickle=False)
    image = np.ascontiguousarray(image)
    boundary_session = make_session(boundary)
    model4_preact, model4_postact, model5_postact, full_output = boundary_session.run(
        [MODEL4_PREACT, MODEL4_POSTACT, MODEL5_POSTACT, FINAL_OUTPUT], {"images": image}
    )
    suffix_a_output = make_session(suffix_a).run([FINAL_OUTPUT], {MODEL4_PREACT: model4_preact})[0]
    suffix_b_output = make_session(suffix_b).run(
        [FINAL_OUTPUT], {MODEL4_POSTACT: model4_postact, MODEL5_POSTACT: model5_postact}
    )[0]

    comparisons = {
        "suffix_a_mismatches": int(np.count_nonzero(suffix_a_output.view(np.uint8) != full_output.view(np.uint8))),
        "suffix_b_mismatches": int(np.count_nonzero(suffix_b_output.view(np.uint8) != full_output.view(np.uint8))),
        "suffix_a_max_abs_diff": float(np.max(np.abs(suffix_a_output.astype(np.float64) - full_output))),
        "suffix_b_max_abs_diff": float(np.max(np.abs(suffix_b_output.astype(np.float64) - full_output))),
    }
    manifest = {
        "model": str(model),
        "model_sha256": sha256_file(model),
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input.resolve()),
        "runtime": ort.__version__,
        "provider": "CPUExecutionProvider",
        "optimization": "ORT_ENABLE_ALL",
        "cuts": {
            path.name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "inputs": [value.name for value in onnx.load(path, load_external_data=False).graph.input],
                "outputs": [value.name for value in onnx.load(path, load_external_data=False).graph.output],
                "nodes": len(onnx.load(path, load_external_data=False).graph.node),
            }
            for path in (prefix, suffix_a, suffix_b, boundary)
        },
        "comparisons": comparisons,
    }
    (output / "stage44_cut_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0 if comparisons["suffix_a_mismatches"] == 0 and comparisons["suffix_b_mismatches"] == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return generate(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
