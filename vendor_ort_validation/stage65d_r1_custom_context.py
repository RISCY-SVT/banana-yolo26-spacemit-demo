#!/usr/bin/env python3
"""Normalize the read-only accepted custom-engine application context."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import statistics
import subprocess
from pathlib import Path

PROTECTED = Path("/data/banana-yolo26-spacemit-demo")
CUSTOM = Path(
    "/data/releases/banana-yolo26-k1x-int8-executor/"
    "0.10.0-internal-rd.1-stage62-sdk"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty TSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def stats(values: list[float]) -> dict[str, object]:
    return {
        "samples": len(values),
        "mean_us": statistics.fmean(values),
        "median_us": statistics.median(values),
        "p95_us": percentile(values, 0.95),
        "p99_us": percentile(values, 0.99),
        "min_us": min(values),
        "max_us": max(values),
    }


def parse_custom_raw(path: Path) -> list[dict[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        fields = line.split("\t")
        if not fields or fields[0] != "raw":
            continue
        row = {}
        for field in fields[1:]:
            key, separator, value = field.partition("=")
            if not separator or not key:
                raise ValueError(f"invalid custom raw field: {field}")
            row[key] = value
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--tracked-root", required=True, type=Path)
    options = parser.parse_args()

    board_identity = read_tsv(options.raw_root / "identity.raw.tsv")
    if any(row["status"] != "pass" for row in board_identity):
        raise RuntimeError("board custom-release identity failed")
    expected = {
        "tag_object": "bf1043d669ff38461a62d116f383ad530128d9b5",
        "tag_peeled": "1fd2e71bb1d5a924e7c0444cada94f681b73aa91",
        "release_manifest": "dced8ddfc540ab5b7fd72ecfe7a16021338ea56258fb33d09c5e023ba3d98b98",
        "sha256sums": "f9c604d7a3167664a86c48dd101e4f4935a243bde726c6853a9f9390aa278341",
        "executor": "34da155ed02a83a74babbec30aff960bdccfb6cc16018230ae7bc030462f7187",
        "source_model": "30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c",
    }
    actual = {
        "tag_object": subprocess.check_output(
            ["git", "rev-parse", "v0.10.0-internal-rd.1^{tag}"],
            cwd=PROTECTED,
            text=True,
        ).strip(),
        "tag_peeled": subprocess.check_output(
            ["git", "rev-parse", "v0.10.0-internal-rd.1^{}"],
            cwd=PROTECTED,
            text=True,
        ).strip(),
        "release_manifest": sha256(CUSTOM / "release_manifest.json"),
        "sha256sums": sha256(CUSTOM / "SHA256SUMS"),
        "executor": sha256(CUSTOM / "bin/yolo26_k1x_int8"),
        "source_model": (CUSTOM / "model-evidence/SOURCE_MODEL_SHA256")
        .read_text(encoding="utf-8")
        .split()[0],
    }
    binding = [
        {
            "field": "version",
            "actual": "0.10.0-internal-rd.1",
            "expected": "0.10.0-internal-rd.1",
            "status": "pass",
        },
        {
            "field": "integer_contract",
            "actual": "K1X_INT8_V1",
            "expected": "K1X_INT8_V1",
            "status": "pass",
        },
    ]
    binding.extend(
        {
            "field": field,
            "actual": value,
            "expected": expected[field],
            "status": "pass" if value == expected[field] else "fail",
        }
        for field, value in actual.items()
    )
    binding.extend(
        {
            "field": f"board_{row['field']}",
            "actual": row["actual"],
            "expected": row["expected"],
            "status": row["status"],
        }
        for row in board_identity
    )
    write_tsv(options.tracked_root / "custom_release_binding.tsv", binding)
    if any(row["status"] != "pass" for row in binding):
        raise RuntimeError("accepted custom package binding failed")

    rows: list[dict[str, object]] = []
    for model in ("B2", "C2"):
        samples = read_tsv(options.raw_root / f"vendor-{model}/samples.tsv")
        if len(samples) != 500 or len({row["output_fnv1a64"] for row in samples}) != 1:
            raise RuntimeError(f"vendor {model} timing count or output hash changed")
        run_log = (options.raw_root / f"vendor-{model}/run.log").read_text(
            encoding="utf-8", errors="replace"
        )
        if "stage64_result status=pass" not in run_log:
            raise RuntimeError(f"vendor {model} context run did not pass")
        for field, role in (("inference_us", "vendor-ep-inference"), ("total_us", "vendor-ep-two-stage")):
            rows.append(
                {
                    "surface": model,
                    "role": role,
                    "input_sha256": "64d11ef4c1e470282a385f7d293607b639da2f40405c92238897253dd1e23f14",
                    **stats([float(row[field]) for row in samples]),
                    "caveat": "same-source vendor S8-QDQ; common FP32 tail is separate",
                }
            )
    custom_rows = parse_custom_raw(options.raw_root / "custom/samples.raw.tsv")
    required_fields = {"repeat", "run", "wall_us", "input_us", "affinity_ok", "hash"}
    if len(custom_rows) != 500 or any(not required_fields.issubset(row) for row in custom_rows):
        raise RuntimeError("custom benchmark did not produce the expected 500-row contract")
    hashes = {row["hash"] for row in custom_rows}
    affinities = {row["affinity_ok"] for row in custom_rows}
    if hashes != {"0xd43f5e018b415631"} or affinities != {"1"}:
        raise RuntimeError("custom output hash or affinity contract changed")
    wall_values = [float(row["wall_us"]) for row in custom_rows]
    input_values = [float(row["input_us"]) for row in custom_rows]
    pure_values = [wall - input_time for wall, input_time in zip(wall_values, input_values)]
    if any(value < 0 for value in pure_values):
        raise RuntimeError("custom pure-executor timing is negative")
    for role, values in (
        ("custom-pure-executor", pure_values),
        ("custom-total-model-to-1x300x6", wall_values),
        ("custom-input-quantize-layout", input_values),
    ):
        rows.append(
            {
                "surface": "accepted-custom-engine",
                "role": role,
                "input_sha256": "64d11ef4c1e470282a385f7d293607b639da2f40405c92238897253dd1e23f14",
                **stats(values),
                "caveat": (
                    "K1X_INT8_V1 native low-latency profile; pure executor is wall_us-input_us; "
                    "custom graph has no separate comparable FP32 tail; cross-surface context only"
                ),
            }
        )
    write_tsv(options.tracked_root / "cross_surface_performance_context.tsv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
