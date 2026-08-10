#!/usr/bin/env python3
"""Resume-safe Stage65B-R1 PTQ, host-gate, causal, and COCO pipeline."""

from __future__ import annotations

import argparse
import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path


LANES = ("B1", "B2", "B3", "B4", "B5", "B6")
TIMEOUTS = {
    "B1": 28_800,
    "B2": 28_800,
    "B3": 86_400,
    "B4": 172_800,
    "B5": 345_600,
    "B6": 172_800,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--fp32-inference", required=True, type=Path)
    parser.add_argument("--fp32-tail", required=True, type=Path)
    parser.add_argument("--expected-tail-sha256", required=True)
    parser.add_argument("--h500-list", required=True, type=Path)
    parser.add_argument("--scout-list", required=True, type=Path)
    parser.add_argument("--val-list", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--stage-dir", required=True, type=Path)
    parser.add_argument("--stage64-derived", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=65001)
    return parser.parse_args()


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def emit(event: str, *values: object) -> None:
    print("\t".join((event, *(str(value) for value in values), stamp())), flush=True)


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def summary_passes(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    return len(rows) == 1 and all(
        (
            rows[0].get("returncode") == "0",
            rows[0].get("output_exists") == "1",
            rows[0].get("checker") == "pass",
        )
    )


def generation_identity(options: argparse.Namespace, lane: str, summary: Path) -> None:
    run(
        [
            str(options.python),
            str(options.repo / "vendor_ort_validation/stage65b_r1_repro_check.py"),
            "--lane",
            lane,
            "--run1-summary",
            str(summary),
            "--output",
            str(options.raw_root / "reproducibility" / f"{lane}-single.tsv"),
        ]
    )


def generate(
    options: argparse.Namespace, lane: str, run_id: str, summary: Path
) -> None:
    if summary_passes(summary):
        generation_identity(options, lane, summary)
        emit("generation-resume-identity-pass", lane, run_id)
        return
    run_root = options.raw_root / "quantization" / lane / run_id
    if run_root.exists() or summary.exists():
        raise RuntimeError(
            f"incomplete generation artifacts require explicit audit: {run_root}"
        )
    emit("generation-start", lane, run_id, TIMEOUTS[lane])
    run(
        [
            str(options.python),
            str(options.repo / "vendor_ort_validation/stage64_run_quantization.py"),
            "--lane",
            lane,
            "--python",
            str(options.python),
            "--config",
            str(options.raw_root / "configs/effective_configs" / f"{lane}.json"),
            "--run-id",
            run_id,
            "--output-root",
            str(options.raw_root / "quantization"),
            "--summary",
            str(summary),
            "--timeout-seconds",
            str(TIMEOUTS[lane]),
            "--launcher",
            str(options.repo / "vendor_ort_validation/stage65b_r1_seeded_xslim.py"),
            "--random-seed",
            str(options.random_seed),
        ]
    )
    generation_identity(options, lane, summary)
    emit("generation-pass", lane, run_id)


def first_run(lane: str) -> tuple[str, str]:
    return ("run1-long", "B5-run1-long.tsv") if lane == "B5" else (
        "run1",
        f"{lane}-run1.tsv",
    )


def postprocess(options: argparse.Namespace, lane: str) -> None:
    summary = options.raw_root / "postprocess" / lane / "postprocess-summary.tsv"
    if summary.is_file():
        with summary.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        if len(rows) == 1 and rows[0].get("status") == "pass":
            emit("postprocess-resume-pass", lane)
            return
    root = options.raw_root / "postprocess" / lane
    if root.exists():
        raise RuntimeError(f"incomplete postprocess root requires explicit audit: {root}")
    run_id, _ = first_run(lane)
    emit("postprocess-start", lane)
    run(
        [
            str(options.python),
            str(options.repo / "vendor_ort_validation/stage65b_r1_postprocess.py"),
            "--lane",
            lane,
            "--run-root",
            str(options.raw_root / "quantization" / lane / run_id),
            "--python",
            str(options.python),
            "--repo",
            str(options.repo),
            "--source-model",
            str(options.source_model),
            "--fp32-inference",
            str(options.fp32_inference),
            "--fp32-tail",
            str(options.fp32_tail),
            "--expected-tail-sha256",
            options.expected_tail_sha256,
            "--h500-list",
            str(options.h500_list),
            "--scout-list",
            str(options.scout_list),
            "--annotations",
            str(options.annotations),
            "--tensor-list",
            str(options.repo / "vendor_ort_validation/stage65b_r1_boundaries.txt"),
            "--preprocess",
            str(options.repo / "vendor_ort_validation/stage64_preprocess.py")
            + ":boundary_audit_preprocess",
            "--output-root",
            str(options.raw_root / "postprocess"),
            "--threads",
            str(options.threads),
            "--boundary-audit",
        ]
    )
    emit("postprocess-pass", lane)


def main() -> int:
    options = parse_args()
    if options.threads < 1:
        raise ValueError("--threads must be positive")
    emit("resume-pipeline-start")
    for lane in LANES:
        run_id, filename = first_run(lane)
        generate(
            options,
            lane,
            run_id,
            options.raw_root / "quantization" / filename,
        )
    for lane in LANES:
        generate(
            options,
            lane,
            "run2",
            options.raw_root / "quantization" / f"{lane}-run2.tsv",
        )
        _, first_name = first_run(lane)
        run(
            [
                str(options.python),
                str(options.repo / "vendor_ort_validation/stage65b_r1_repro_check.py"),
                "--lane",
                lane,
                "--run1-summary",
                str(options.raw_root / "quantization" / first_name),
                "--run2-summary",
                str(options.raw_root / "quantization" / f"{lane}-run2.tsv"),
                "--output",
                str(options.raw_root / "reproducibility" / f"{lane}.tsv"),
            ]
        )
        emit("reproducibility-pass", lane)
    for lane in LANES:
        postprocess(options, lane)

    matrix_summary = options.raw_root / "full-matrix/full-matrix-summary.tsv"
    if not matrix_summary.is_file():
        if (options.raw_root / "full-matrix").exists():
            raise RuntimeError("incomplete full-matrix root requires explicit audit")
        emit("full-matrix-start")
        run(
            [
                str(options.python),
                str(options.repo / "vendor_ort_validation/stage65b_r1_full_matrix.py"),
                "--python",
                str(options.python),
                "--repo",
                str(options.repo),
                "--postprocess-root",
                str(options.raw_root / "postprocess"),
                "--fp32-inference",
                str(options.fp32_inference),
                "--fp32-tail",
                str(options.fp32_tail),
                "--source-model",
                str(options.source_model),
                "--h500-list",
                str(options.h500_list),
                "--scout-list",
                str(options.scout_list),
                "--val-list",
                str(options.val_list),
                "--annotations",
                str(options.annotations),
                "--output-root",
                str(options.raw_root / "full-matrix"),
                "--threads",
                str(options.threads),
            ]
        )
        emit("full-matrix-pass")
    else:
        emit("full-matrix-resume-pass")

    run(
        [
            str(options.python),
            str(options.repo / "vendor_ort_validation/stage65b_r1_finalize.py"),
            "--raw-root",
            str(options.raw_root),
            "--stage-dir",
            str(options.stage_dir),
            "--stage64-derived",
            str(options.stage64_derived),
        ]
    )
    emit("finalize-pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
