#!/usr/bin/env python3
"""Run one immutable DEV-001A candidate from an accepted base configuration."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--launcher", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=65001)
    parser.add_argument("--timeout-seconds", type=int, default=28_800)
    return parser.parse_args()


def main() -> int:
    options = parse_args()
    lane = options.lane.upper()
    run_root = options.raw_root / "quantization" / lane / options.run_id
    runtime_config = (
        options.raw_root / "runtime-configs" / f"{lane}-{options.run_id}.json"
    )
    summary = options.raw_root / "quantization" / f"{lane}-{options.run_id}.tsv"
    if run_root.exists() or runtime_config.exists() or summary.exists():
        raise RuntimeError(
            "refusing to reuse candidate run state: "
            f"{run_root}, {runtime_config}, {summary}"
        )
    config = json.loads(options.base_config.read_text(encoding="utf-8"))
    config["model_parameters"]["working_dir"] = "__SET_BY_RUNNER__"
    config["quantization_parameters"]["range_policy_manifest_path"] = str(
        (run_root / "range-policy-manifest.json").resolve()
    )
    runtime_config.parent.mkdir(parents=True, exist_ok=True)
    runtime_config.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    command = [
        str(options.python),
        str(options.repo / "vendor_ort_validation/stage64_run_quantization.py"),
        "--lane",
        lane,
        "--python",
        str(options.python),
        "--config",
        str(runtime_config),
        "--run-id",
        options.run_id,
        "--output-root",
        str(options.raw_root / "quantization"),
        "--summary",
        str(summary),
        "--timeout-seconds",
        str(options.timeout_seconds),
        "--launcher",
        str(options.launcher),
        "--random-seed",
        str(options.seed),
    ]
    process = subprocess.run(command, check=False)
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
