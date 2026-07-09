#!/usr/bin/env python3
"""Create Stage40 block cuts and run all-ORT/custom-model4 skeleton checks."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnx.utils

from stage40_skeleton_common import (
    MODEL4_CUT_INPUT,
    MODEL4_CUT_OUTPUT,
    add_outputs,
    compare_arrays,
    input_array,
    load_nhwc_bin,
    make_session,
    run_session,
    save_nhwc_bin,
    save_npy,
    sha256_file,
    time_session,
    write_json,
    write_tsv,
)


def extract_cut(model: Path, out_path: Path, inputs: list[str], outputs: list[str]) -> dict[str, object]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.utils.extract_model(str(model), str(out_path), inputs, outputs)
    cut = onnx.load(out_path)
    return {
        "path": str(out_path),
        "sha256": sha256_file(out_path),
        "nodes": len(cut.graph.node),
        "inputs": [i.name for i in cut.graph.input],
        "outputs": [o.name for o in cut.graph.output],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--input-mode", default="synthetic_seeded",
                        choices=["synthetic_seeded", "synthetic_gradient", "zeros"])
    parser.add_argument("--custom-model4-output-bin")
    parser.add_argument("--profile-warmup", type=int, default=2)
    parser.add_argument("--profile-runs", type=int, default=5)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--boundary-tsv", required=True)
    parser.add_argument("--compare-tsv", required=True)
    parser.add_argument("--profile-tsv", required=True)
    parser.add_argument("--report-md", required=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    out_dir = Path(args.out_dir)
    tensor_dir = out_dir / "tensors"
    cut_dir = out_dir / "cuts"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    cut_dir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(model_path)
    final_outputs = [o.name for o in model.graph.output]
    if len(final_outputs) != 1:
        raise ValueError(f"expected one final output, got {final_outputs}")
    final_output = final_outputs[0]
    model_input = model.graph.input[0].name

    cuts = {
        "prefix": extract_cut(model_path, cut_dir / "prefix_images_to_model4_input.onnx",
                              [model_input], [MODEL4_CUT_INPUT]),
        "model4": extract_cut(model_path, cut_dir / "model4_input_to_model4_output.onnx",
                              [MODEL4_CUT_INPUT], [MODEL4_CUT_OUTPUT]),
        "suffix": extract_cut(model_path, cut_dir / "suffix_model4_output_to_output0.onnx",
                              [MODEL4_CUT_OUTPUT], [final_output]),
    }

    full_model_with_boundaries = onnx.load(model_path)
    add_outputs(full_model_with_boundaries, [MODEL4_CUT_INPUT, MODEL4_CUT_OUTPUT])
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        full_with_boundaries_path = Path(f.name)
    onnx.save(full_model_with_boundaries, full_with_boundaries_path)
    try:
        full_session = make_session(full_with_boundaries_path)
        input_info = full_session.get_inputs()[0]
        input_shape = [int(v) for v in input_info.shape]
        x = input_array(input_shape, args.input_mode)
        full_outputs = full_session.run([final_output, MODEL4_CUT_INPUT, MODEL4_CUT_OUTPUT], {input_info.name: x})
    finally:
        full_with_boundaries_path.unlink(missing_ok=True)
    full_final, full_model4_input, full_model4_output = full_outputs

    prefix_output = run_session(Path(cuts["prefix"]["path"]), [MODEL4_CUT_INPUT], {model_input: x})[0]
    ort_model4_output = run_session(Path(cuts["model4"]["path"]), [MODEL4_CUT_OUTPUT],
                                    {MODEL4_CUT_INPUT: prefix_output})[0]
    all_ort_final = run_session(Path(cuts["suffix"]["path"]), [final_output],
                                {MODEL4_CUT_OUTPUT: ort_model4_output})[0]

    tensor_records = [
        save_npy("images", x, tensor_dir),
        save_npy(final_output, full_final, tensor_dir),
        save_npy(MODEL4_CUT_INPUT, full_model4_input, tensor_dir),
        save_npy(MODEL4_CUT_OUTPUT, full_model4_output, tensor_dir),
        save_npy("prefix__" + MODEL4_CUT_INPUT, prefix_output, tensor_dir),
        save_npy("all_ort_model4__" + MODEL4_CUT_OUTPUT, ort_model4_output, tensor_dir),
        save_npy("all_ort_skeleton__" + final_output, all_ort_final, tensor_dir),
        save_nhwc_bin(MODEL4_CUT_INPUT, full_model4_input, tensor_dir, "model4_cv1_conv_q_u8"),
        save_nhwc_bin(MODEL4_CUT_OUTPUT, full_model4_output, tensor_dir, "model4_cv2_conv_q_u8_expected"),
    ]

    compares = [
        compare_arrays("prefix_vs_full_model4_input", prefix_output, full_model4_input),
        compare_arrays("all_ort_model4_vs_full_model4_output", ort_model4_output, full_model4_output),
        compare_arrays("all_ort_final_vs_full_reference", all_ort_final, full_final),
    ]

    custom_final_status = "not_run"
    custom_boundary_status = "not_run"
    if args.custom_model4_output_bin:
        custom_output = load_nhwc_bin(Path(args.custom_model4_output_bin), full_model4_output.shape, full_model4_output.dtype)
        custom_final = run_session(Path(cuts["suffix"]["path"]), [final_output], {MODEL4_CUT_OUTPUT: custom_output})[0]
        tensor_records.append(save_npy("custom_model4__" + MODEL4_CUT_OUTPUT, custom_output, tensor_dir))
        tensor_records.append(save_npy("custom_model4_skeleton__" + final_output, custom_final, tensor_dir))
        custom_boundary = compare_arrays("custom_model4_output_vs_full_model4_output", custom_output, full_model4_output)
        custom_final_cmp = compare_arrays("custom_model4_skeleton_final_vs_full_reference", custom_final, full_final)
        compares.extend([custom_boundary, custom_final_cmp])
        custom_boundary_status = custom_boundary.status
        custom_final_status = custom_final_cmp.status

    write_tsv(Path(args.boundary_tsv), tensor_records)
    write_tsv(Path(args.compare_tsv), compares)

    _, full_us = time_session(model_path, [final_output], {model_input: x}, args.profile_warmup, args.profile_runs)
    _, prefix_us = time_session(Path(cuts["prefix"]["path"]), [MODEL4_CUT_INPUT],
                                {model_input: x}, args.profile_warmup, args.profile_runs)
    _, model4_us = time_session(Path(cuts["model4"]["path"]), [MODEL4_CUT_OUTPUT],
                                {MODEL4_CUT_INPUT: prefix_output}, args.profile_warmup, args.profile_runs)
    _, suffix_us = time_session(Path(cuts["suffix"]["path"]), [final_output],
                                {MODEL4_CUT_OUTPUT: ort_model4_output}, args.profile_warmup, args.profile_runs)
    profile_rows = [
        {"block": "full_ort_reference", "fallback_vs_custom": "ORT_CPU", "mean_us": f"{full_us:.3f}",
         "boundary_bytes": str(x.nbytes), "note": "skeleton_total_latency_not_model_fps"},
        {"block": "prefix_images_to_model4_input", "fallback_vs_custom": "ORT_CPU", "mean_us": f"{prefix_us:.3f}",
         "boundary_bytes": str(full_model4_input.nbytes), "note": "cut timing"},
        {"block": "model4_cut_all_ort", "fallback_vs_custom": "ORT_CPU", "mean_us": f"{model4_us:.3f}",
         "boundary_bytes": str(full_model4_output.nbytes), "note": "cut timing"},
        {"block": "suffix_model4_output_to_output0", "fallback_vs_custom": "ORT_CPU", "mean_us": f"{suffix_us:.3f}",
         "boundary_bytes": str(full_model4_output.nbytes), "note": "cut timing"},
    ]
    write_tsv_generic(Path(args.profile_tsv), profile_rows)

    summary = {
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "provider": "CPUExecutionProvider",
        "input_name": model_input,
        "input_shape": input_shape,
        "input_mode": args.input_mode,
        "final_output": final_output,
        "model4_cut_input": MODEL4_CUT_INPUT,
        "model4_cut_output": MODEL4_CUT_OUTPUT,
        "cuts": cuts,
        "all_ort_status": "pass" if all(c.status == "pass" for c in compares[:3]) else "fail",
        "custom_model4_boundary_status": custom_boundary_status,
        "custom_model4_final_status": custom_final_status,
        "profile_mean_us": {row["block"]: row["mean_us"] for row in profile_rows},
    }
    write_json(Path(args.summary_json), summary)
    Path(args.report_md).write_text(render_report(summary, compares), encoding="utf-8")
    print(summary)
    return 0 if summary["all_ort_status"] == "pass" and custom_final_status in ("not_run", "pass") else 1


def write_tsv_generic(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    import csv
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def render_report(summary: dict[str, object], compares: list[object]) -> str:
    lines = [
        "# Stage40 Block Cut Skeleton Report",
        "",
        f"- model: `{summary['model']}`",
        f"- model_sha256: `{summary['model_sha256']}`",
        "- provider: `CPUExecutionProvider`",
        f"- model4_cut_input: `{summary['model4_cut_input']}`",
        f"- model4_cut_output: `{summary['model4_cut_output']}`",
        f"- all_ort_status: `{summary['all_ort_status']}`",
        f"- custom_model4_boundary_status: `{summary['custom_model4_boundary_status']}`",
        f"- custom_model4_final_status: `{summary['custom_model4_final_status']}`",
        "",
        "| comparison | status | mismatches | max_abs_diff |",
        "|---|---:|---:|---:|",
    ]
    for cmp in compares:
        lines.append(f"| {cmp.name} | {cmp.status} | {cmp.mismatches} | {cmp.max_abs_diff} |")
    lines.append("")
    lines.append("Timing is skeleton profiling only, not model FPS.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
