#!/usr/bin/env python3
"""Execute one immutable XSlim configuration with bounded evidence capture."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", required=True)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument("--launcher", type=Path)
    parser.add_argument("--random-seed", type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_manifest(root: Path, output: Path) -> tuple[int, int, str]:
    rows: list[dict[str, Any]] = []
    tree_digest = hashlib.sha256()
    byte_count = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        file_hash = sha256(path)
        size = path.stat().st_size
        byte_count += size
        rows.append(
            {
                "relative_path": relative,
                "bytes": size,
                "sha256": file_hash,
                "mode": oct(path.stat().st_mode & 0o777),
            }
        )
        tree_digest.update(relative.encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(str(size).encode("ascii"))
        tree_digest.update(b"\0")
        tree_digest.update(file_hash.encode("ascii"))
        tree_digest.update(b"\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        fields = ["relative_path", "bytes", "sha256", "mode"]
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows), byte_count, tree_digest.hexdigest()


def main() -> int:
    options = parse_args()
    run_root = options.output_root / options.lane / options.run_id
    if run_root.exists():
        raise RuntimeError(f"refusing to reuse existing run root: {run_root}")
    run_root.mkdir(parents=True)
    output_dir = run_root / "output"
    config = json.loads(options.config.read_text(encoding="utf-8"))
    config["model_parameters"]["working_dir"] = str(output_dir.resolve())
    effective_config = run_root / "effective-config.json"
    effective_config.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_model = output_dir / (
        str(config["model_parameters"]["output_prefix"]) + ".onnx"
    )
    log_path = run_root / "xslim.log"
    time_path = run_root / "time-v.txt"
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "0"
    environment_roots = {
        "TMPDIR": run_root / "tmp",
        "XDG_CACHE_HOME": run_root / "cache",
        "PIP_CACHE_DIR": run_root / "pip-cache",
        "TORCH_HOME": run_root / "torch-home",
        "CCACHE_DIR": run_root / "ccache",
    }
    for name, path in environment_roots.items():
        path.mkdir(parents=True, exist_ok=True)
        environment[name] = str(path.resolve())
    xslim_command = [str(options.python)]
    if options.launcher is not None:
        if options.random_seed is None:
            raise ValueError("--launcher requires --random-seed")
        if not options.launcher.is_file():
            raise FileNotFoundError(options.launcher)
        environment["STAGE65B_R1_RANDOM_SEED"] = str(options.random_seed)
        xslim_command.append(str(options.launcher.resolve()))
    else:
        if options.random_seed is not None:
            raise ValueError("--random-seed requires --launcher")
        xslim_command.extend(["-m", "xslim"])
    time_binary = shutil.which("time")
    if time_binary:
        command = [
            time_binary,
            "-v",
            "-o",
            str(time_path),
            *xslim_command,
            "-c",
            str(effective_config),
        ]
    else:
        time_path.write_text(
            "GNU time unavailable; elapsed wall time captured by Python.\n",
            encoding="utf-8",
        )
        command = [
            *xslim_command,
            "-c",
            str(effective_config),
        ]
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        try:
            process = subprocess.run(
                command,
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
    model_hash = ""
    node_count = ""
    qdq_count = ""
    qlinear_count = ""
    if output_model.exists():
        model_hash = sha256(output_model)
        try:
            model = onnx.load(output_model, load_external_data=False)
            onnx.checker.check_model(model)
            checker = "pass"
            node_count = len(model.graph.node)
            qdq_count = sum(
                node.op_type in {"QuantizeLinear", "DequantizeLinear"}
                for node in model.graph.node
            )
            qlinear_count = sum(
                node.op_type.startswith("QLinear") for node in model.graph.node
            )
        except Exception as exc:  # noqa: BLE001
            checker = "fail"
            checker_error = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    file_count, byte_count, tree_hash = tree_manifest(
        run_root, run_root / "output-tree-manifest.tsv"
    )
    output_file_count, output_byte_count, output_tree_hash = tree_manifest(
        output_dir, run_root / "generated-output-tree-manifest.tsv"
    )
    row = {
        "lane": options.lane,
        "run_id": options.run_id,
        "config": str(options.config.resolve()),
        "effective_config_sha256": sha256(effective_config),
        "python": str(options.python.absolute()),
        "python_realpath": str(options.python.resolve()),
        "launcher": str(options.launcher.resolve()) if options.launcher else "",
        "random_seed": "" if options.random_seed is None else options.random_seed,
        "returncode": returncode,
        "elapsed_seconds": f"{elapsed:.6f}",
        "output_model": str(output_model),
        "output_exists": int(output_model.exists()),
        "output_sha256": model_hash,
        "checker": checker,
        "checker_error": checker_error,
        "node_count": node_count,
        "qdq_count": qdq_count,
        "qlinear_count": qlinear_count,
        "tree_file_count": file_count,
        "tree_byte_count": byte_count,
        "tree_sha256": tree_hash,
        "output_tree_file_count": output_file_count,
        "output_tree_byte_count": output_byte_count,
        "output_tree_sha256": output_tree_hash,
        "log": str(log_path),
        "time_v": str(time_path),
    }
    options.summary.parent.mkdir(parents=True, exist_ok=True)
    with options.summary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=list(row),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(row)
    return 0 if returncode == 0 and checker == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
