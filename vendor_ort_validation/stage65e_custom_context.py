#!/usr/bin/env python3
"""Normalize the read-only Stage65E custom-engine application context."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

PROTECTED = Path("/data/banana-yolo26-spacemit-demo")
CUSTOM = Path(
    "/data/releases/banana-yolo26-k1x-int8-executor/"
    "0.10.0-internal-rd.1-stage62-sdk"
)
ACCEPTED_CUSTOM_ACCURACY = Path(
    "/data/worktrees/banana-yolo26-xslim211-s8-qdq-validation/stages/"
    "BANANA-YOLO26-XSLIM-STAGE65D-C2-FROZEN-K1X-SPACEMIT-EP-BOARD-"
    "PASSPORT-FP32-FP16-AND-CUSTOM-ENGINE-APPLICATION-COMPARISON-001/"
    "cross_surface_application_accuracy_table.tsv"
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
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
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


def state_contract(root: Path) -> tuple[str, str]:
    paths = [
        root / arm / f"state-{moment}.tsv"
        for arm in ("vendor-B2", "vendor-C2", "custom")
        for moment in ("before", "after")
    ]
    governors: set[str] = set()
    frequencies: list[int] = []
    temperatures: list[int] = []
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            return f"missing-or-empty={path}", "fail"
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) != 3:
                continue
            if fields[0] == "governor":
                governors.add(fields[2])
            elif fields[0] == "frequency_khz":
                frequencies.append(int(fields[2]))
            elif fields[0] == "temperature_millic":
                temperatures.append(int(fields[2]))
    passed = (
        governors == {"performance"}
        and bool(frequencies)
        and min(frequencies) >= 0.95 * max(frequencies)
        and bool(temperatures)
        and max(temperatures) <= 85000
    )
    actual = (
        f"snapshots={len(paths)};governors={','.join(sorted(governors))};"
        f"frequency_khz={min(frequencies) if frequencies else 'missing'}.."
        f"{max(frequencies) if frequencies else 'missing'};"
        f"temperature_millic_max={max(temperatures) if temperatures else 'missing'}"
    )
    return actual, "pass" if passed else "fail"


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
        "tag_object": subprocess.check_output(["git", "rev-parse", "v0.10.0-internal-rd.1^{tag}"], cwd=PROTECTED, text=True).strip(),
        "tag_peeled": subprocess.check_output(["git", "rev-parse", "v0.10.0-internal-rd.1^{}"], cwd=PROTECTED, text=True).strip(),
        "release_manifest": sha256(CUSTOM / "release_manifest.json"),
        "sha256sums": sha256(CUSTOM / "SHA256SUMS"),
        "executor": sha256(CUSTOM / "bin/yolo26_k1x_int8"),
        "source_model": (CUSTOM / "model-evidence/SOURCE_MODEL_SHA256").read_text(encoding="utf-8").split()[0],
    }
    binding: list[dict[str, object]] = [
        {"field": "version", "actual": "0.10.0-internal-rd.1", "expected": "0.10.0-internal-rd.1", "status": "pass"},
        {"field": "integer_contract", "actual": "K1X_INT8_V1", "expected": "K1X_INT8_V1", "status": "pass"},
    ]
    binding.extend(
        {"field": field, "actual": value, "expected": expected[field], "status": "pass" if value == expected[field] else "fail"}
        for field, value in actual.items()
    )
    binding.extend(
        {"field": f"board_{row['field']}", "actual": row["actual"], "expected": row["expected"], "status": row["status"]}
        for row in board_identity
    )
    state_actual, state_status = state_contract(options.raw_root)
    binding.append({
        "field": "same_boot_system_state",
        "actual": state_actual,
        "expected": "performance governor; <=5% frequency spread; <=85000 millic",
        "status": state_status,
    })
    write_tsv(options.tracked_root / "custom_release_binding.tsv", binding)
    if any(row["status"] != "pass" for row in binding):
        raise RuntimeError("accepted custom package binding failed")

    vendor_rows: list[dict[str, object]] = []
    for model in ("B2", "C2"):
        samples = read_tsv(options.raw_root / f"vendor-{model}/samples.tsv")
        required_sample_fields = {"repeat", "run", "inference_us", "tail_us", "total_us"}
        if len(samples) != 500 or not required_sample_fields.issubset(samples[0]):
            raise RuntimeError(f"vendor {model} timing contract changed")
        per_run_hashes = "output_fnv1a64" in samples[0]
        if per_run_hashes and len({row["output_fnv1a64"] for row in samples}) != 1:
            raise RuntimeError(f"vendor {model} per-run output hash changed")
        output_path = options.raw_root / f"vendor-{model}/output.bin"
        if output_path.stat().st_size != 1800 * 4:
            raise RuntimeError(f"vendor {model} output-size contract changed")
        log = (options.raw_root / f"vendor-{model}/run.log").read_text(encoding="utf-8", errors="replace")
        if "stage64_result status=pass" not in log:
            raise RuntimeError(f"vendor {model} context run did not pass")
        for field, role in (
            ("inference_us", "inference"),
            ("tail_us", "tail"),
            ("total_us", "two-stage"),
        ):
            vendor_rows.append({
                "surface": model,
                "runtime": "SpaceMIT EP 2.0.6",
                "role": role,
                "input_sha256": "64d11ef4c1e470282a385f7d293607b639da2f40405c92238897253dd1e23f14",
                **stats([float(row[field]) for row in samples]),
                "caveat": "same-source vendor S8-QDQ; common FP32 tail is separate; "
                + ("per-run FNV stable" if per_run_hashes else "bound runner does not emit per-run FNV; final output SHA is identity-bound"),
            })
    write_tsv(options.tracked_root / "same_source_vendor_performance_table.tsv", vendor_rows)

    custom_rows = parse_custom_raw(options.raw_root / "custom/samples.raw.tsv")
    required_fields = {"repeat", "run", "wall_us", "input_us", "affinity_ok", "hash"}
    if len(custom_rows) != 500 or any(not required_fields.issubset(row) for row in custom_rows):
        raise RuntimeError("custom benchmark did not produce the expected 500-row contract")
    sample_grid = {(int(row["repeat"]), int(row["run"])) for row in custom_rows}
    if sample_grid != {(repeat, run) for repeat in range(5) for run in range(100)}:
        raise RuntimeError("custom benchmark repeat/run grid is incomplete or duplicated")
    if {row["hash"] for row in custom_rows} != {"0xd43f5e018b415631"} or {row["affinity_ok"] for row in custom_rows} != {"1"}:
        raise RuntimeError("custom output hash or affinity contract changed")
    detections = json.loads((options.raw_root / "custom/output.json").read_text(encoding="utf-8"))
    if len(detections) != 300:
        raise RuntimeError("custom output JSON does not contain exactly 300 rows")
    for detection in detections:
        values = [*detection.get("box", []), detection.get("score"), detection.get("class")]
        if len(values) != 6 or any(not isinstance(value, (int, float)) for value in values):
            raise RuntimeError("custom output JSON row contract changed")
        if any(not math.isfinite(float(value)) for value in values):
            raise RuntimeError("custom output JSON contains a non-finite value")
    wall = [float(row["wall_us"]) for row in custom_rows]
    input_values = [float(row["input_us"]) for row in custom_rows]
    pure = [total - input_time for total, input_time in zip(wall, input_values)]
    if any(value < 0 for value in pure):
        raise RuntimeError("custom pure-executor timing is negative")
    application_rows = list(vendor_rows)
    for role, values in (
        ("custom-pure-executor", pure),
        ("custom-total-model-to-1x300x6", wall),
        ("custom-input-quantize-layout", input_values),
    ):
        application_rows.append({
            "surface": "accepted-custom-engine",
            "runtime": "K1X_INT8_V1 0.10.0-internal-rd.1",
            "role": role,
            "input_sha256": "64d11ef4c1e470282a385f7d293607b639da2f40405c92238897253dd1e23f14",
            **stats(values),
            "caveat": "different model/export/quantization/runtime surface; application context only",
        })
    write_tsv(options.tracked_root / "cross_surface_application_performance_table.tsv", application_rows)

    accepted_rows = read_tsv(ACCEPTED_CUSTOM_ACCURACY)
    custom_accuracy = next(row for row in accepted_rows if row["surface"] == "accepted-custom-engine")
    accuracy_rows = read_tsv(options.tracked_root / "accuracy_absolute.tsv")
    result_accuracy: list[dict[str, object]] = [custom_accuracy]
    model_sha = {
        "B2_BOARD_EP": "40ba6a7f9aebaa98a1c3abe5fce1f66f1bebcd0b10b7af3d26d30414a331d853",
        "C2_BOARD_EP": "281f4acd1261e7ee2c38b6e3bdecbf61c3d91cf710c63e6bc6cdaf257a52669b",
    }
    for surface in ("B2_BOARD_EP", "C2_BOARD_EP"):
        row = next(item for item in accuracy_rows if item["surface"] == surface)
        result_accuracy.append({
            "surface": surface,
            "dataset": "val2017-5000",
            "model_sha256": model_sha[surface],
            "lineage": "same-source YOLO26 split plus common FP32 tail",
            "quantization": "signed S8-QDQ",
            "runtime": "SpaceMIT EP 2.0.6",
            "map50_95": row["map50_95"],
            "map75": row["map75"],
            "ap_small": row["ap_small"],
            "ap_medium": row["ap_medium"],
            "ap_large": row["ap_large"],
            "ar_large": row["ar_large"],
            "ar100": row["ar_100"],
            "prediction_count": row["prediction_count"],
            "caveat": "same-source vendor comparison; frozen Stage65D-R1 prediction evidence",
        })
    write_tsv(options.tracked_root / "cross_surface_application_accuracy_table.tsv", result_accuracy)
    (options.tracked_root / "cross_surface_comparison_caveats.md").write_text(
        "# Cross-surface comparison caveats\n\n"
        "The accepted custom executor and the B2/C2 vendor lane use different model/export "
        "lineages, quantization formats, runtime backends, and output implementations. Their "
        "same-boot fixed-input timings and accepted COCO metrics are application-level context, "
        "not an engine-only or quantizer-only comparison. The frozen vendor timing contract uses "
        "CPU0-3, while the accepted custom low-latency profile natively uses CPU0-4; rows remain "
        "separate and no direct speedup ratio is claimed. No custom model or executable was rebuilt.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
