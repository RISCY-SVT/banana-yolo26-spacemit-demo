#!/usr/bin/env python3
"""Normalize Stage65D-R1 matched ABBA timing with an empirical B2/B2 noise floor."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

SESSION_RE = re.compile(r"stage64_session inference_create_us=(?P<inference>[0-9.]+) tail_create_us=(?P<tail>[0-9.]+)")
FIRST_RE = re.compile(r"stage64_first inference_us=(?P<inference>[0-9.]+) tail_us=(?P<tail>[0-9.]+) total_us=(?P<total>[0-9.]+)")
LABEL_RE = re.compile(r"(noise|abba)-b([0-9]+)-s([0-9]+)-(C2|B2)")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stats(values: list[float]) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    standard = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "samples": len(values),
        "mean_us": mean,
        "stddev_us": standard,
        "cv_pct": 0.0 if mean == 0 else 100.0 * standard / mean,
        "median_us": statistics.median(values),
        "p90_us": percentile(values, 0.90),
        "p95_us": percentile(values, 0.95),
        "p99_us": percentile(values, 0.99),
        "min_us": min(values),
        "max_us": max(values),
    }


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty TSV: {path}")
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_samples(path: Path) -> dict[str, list[float]]:
    result = {"inference": [], "tail": [], "two_stage": []}
    for row in read_tsv(path):
        result["inference"].append(float(row["inference_us"]))
        result["tail"].append(float(row["tail_us"]))
        result["two_stage"].append(float(row["total_us"]))
    return result


def bootstrap_mean(values: list[float], seed: int, replicates: int = 10000) -> tuple[float, float, float]:
    rng = random.Random(seed)
    draws = [statistics.fmean(rng.choice(values) for _ in values) for _ in range(replicates)]
    return statistics.fmean(values), percentile(draws, 0.025), percentile(draws, 0.975)


def parse_pipeline(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise ValueError(f"invalid --pipeline: {value}")
    return name, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--performance-root", required=True, type=Path)
    parser.add_argument("--tracked-root", required=True, type=Path)
    parser.add_argument("--pipeline", action="append", default=[])
    parser.add_argument("--file-components", type=Path)
    parser.add_argument("--seed", type=int, default=65013)
    options = parser.parse_args()

    slot_rows: list[dict[str, object]] = []
    creation_rows: list[dict[str, object]] = []
    first_rows: list[dict[str, object]] = []
    values_by_slot: dict[tuple[str, int, int, str, str], list[float]] = {}
    thermal_rows: list[dict[str, object]] = []
    status_rows = read_tsv(options.performance_root / "status.raw.tsv")
    status_by_key: dict[tuple[str, int, int, str], dict[str, str]] = {}
    for row in status_rows:
        key = (row["phase"], int(row["block"]), int(row["slot"]), row["model"])
        if key in status_by_key:
            raise RuntimeError(f"duplicate performance status row: {key}")
        status_by_key[key] = row
    observed_keys: set[tuple[str, int, int, str]] = set()
    output_hashes: dict[str, set[str]] = defaultdict(set)
    fnv_hashes: dict[str, set[str]] = defaultdict(set)

    for directory in sorted(path for path in options.performance_root.iterdir() if path.is_dir()):
        match = LABEL_RE.fullmatch(directory.name)
        if not match:
            continue
        phase, block_text, slot_text, model = match.groups()
        block, slot = int(block_text), int(slot_text)
        key = (phase, block, slot, model)
        observed_keys.add(key)
        if key not in status_by_key:
            raise RuntimeError(f"missing performance status row: {key}")
        status = status_by_key[key]
        if int(status["exit_code"]) != 0:
            raise RuntimeError(f"failed performance slot: {key}")
        samples_path = directory / "samples.tsv"
        output_path = directory / "output.bin"
        if sha256(samples_path) != status["samples_sha256"]:
            raise RuntimeError(f"samples hash mismatch: {key}")
        if sha256(output_path) != status["output_sha256"]:
            raise RuntimeError(f"output hash mismatch: {key}")
        sample_rows = read_tsv(samples_path)
        if len(sample_rows) != 100:
            raise RuntimeError(f"expected 100 measured runs in {key}, got {len(sample_rows)}")
        output_hashes[model].add(status["output_sha256"])
        fnv_hashes[model].update(row["output_fnv1a64"] for row in sample_rows)
        parsed = parse_samples(samples_path)
        for metric, values in parsed.items():
            values_by_slot[(phase, block, slot, model, metric)] = values
            slot_rows.append({
                "phase": phase,
                "block": block,
                "slot": slot,
                "model": model,
                "metric": metric,
                **stats(values),
            })
        text = (directory / "run.log").read_text(encoding="utf-8", errors="replace")
        session = SESSION_RE.search(text)
        first = FIRST_RE.search(text)
        if not session or not first:
            raise RuntimeError(f"missing session/first timing in {directory}")
        if "stage64_result status=pass" not in text:
            raise RuntimeError(f"missing pass result in {directory}")
        creation_rows.append({
            "phase": phase, "block": block, "slot": slot, "model": model,
            "inference_create_us": session.group("inference"),
            "tail_create_us": session.group("tail"), "status": "pass",
        })
        first_rows.append({
            "phase": phase, "block": block, "slot": slot, "model": model,
            "inference_us": first.group("inference"), "tail_us": first.group("tail"),
            "total_us": first.group("total"), "status": "pass",
        })
        for state in ("before", "after"):
            for line in (directory / f"state-{state}.tsv").read_text().splitlines():
                fields = line.split("\t")
                if len(fields) == 3:
                    thermal_rows.append({
                        "phase": phase, "block": block, "slot": slot, "model": model,
                        "moment": state, "kind": fields[0], "source": fields[1], "value": fields[2],
                    })

    expected_keys: set[tuple[str, int, int, str]] = set()
    for block in range(4):
        expected_keys.update(("noise", block, slot, "B2") for slot in range(4))
    for block in range(12):
        order = ("B2", "C2", "C2", "B2") if block % 2 == 0 else ("C2", "B2", "B2", "C2")
        expected_keys.update(("abba", block, slot, model) for slot, model in enumerate(order))
    if observed_keys != expected_keys or set(status_by_key) != expected_keys:
        raise RuntimeError("performance schedule/status does not match the frozen 4+12 block contract")
    if any(len(output_hashes[model]) != 1 or len(fnv_hashes[model]) != 1 for model in ("B2", "C2")):
        raise RuntimeError("output hash changed within the matched performance surface")
    if len({row["block"] for row in slot_rows if row["phase"] == "noise"}) != 4:
        raise RuntimeError("expected four B2/B2 noise blocks")
    if len({row["block"] for row in slot_rows if row["phase"] == "abba"}) != 12:
        raise RuntimeError("expected twelve B2/C2 ABBA blocks")
    write_tsv(options.tracked_root / "session_creation_timing.tsv", creation_rows)
    write_tsv(options.tracked_root / "first_run_timing.tsv", first_rows)
    write_tsv(options.tracked_root / "b2_b2_noise_floor_raw.tsv", [row for row in slot_rows if row["phase"] == "noise"])
    write_tsv(options.tracked_root / "b2_c2_abba_raw.tsv", [row for row in slot_rows if row["phase"] == "abba"])

    noise_summary: list[dict[str, object]] = []
    noise_floor: dict[tuple[str, str], float] = {}
    for metric in ("inference", "tail", "two_stage"):
        for statistic in ("median", "p95"):
            ratios = []
            for block in range(4):
                left = []
                right = []
                for slot in (0, 3):
                    values = values_by_slot[("noise", block, slot, "B2", metric)]
                    left.append(statistics.median(values) if statistic == "median" else percentile(values, 0.95))
                for slot in (1, 2):
                    values = values_by_slot[("noise", block, slot, "B2", metric)]
                    right.append(statistics.median(values) if statistic == "median" else percentile(values, 0.95))
                ratios.append(statistics.fmean(left) / statistics.fmean(right))
            mean_ratio, lower, upper = bootstrap_mean(ratios, options.seed + len(noise_summary))
            absolute_floor = max(abs(min(ratios) - 1.0), abs(max(ratios) - 1.0))
            noise_floor[(metric, statistic)] = absolute_floor
            noise_summary.append({
                "metric": metric, "statistic": statistic, "blocks": 4,
                "block_ratio_mean": mean_ratio, "block_ratio_min": min(ratios),
                "block_ratio_max": max(ratios), "bootstrap_ci_lower": lower,
                "bootstrap_ci_upper": upper, "empirical_abs_noise_floor": absolute_floor,
            })
    write_tsv(options.tracked_root / "b2_b2_noise_floor_summary.tsv", noise_summary)

    ratio_rows: list[dict[str, object]] = []
    decision_states: list[str] = []
    guardrails = {("inference", "median"): 1.02, ("two_stage", "median"): 1.02,
                  ("inference", "p95"): 1.05, ("two_stage", "p95"): 1.05}
    for metric in ("inference", "tail", "two_stage"):
        all_b2 = []
        all_c2 = []
        for block in range(12):
            for slot in range(4):
                model = "B2" if (block % 2 == 0 and slot in (0, 3)) or (block % 2 == 1 and slot in (1, 2)) else "C2"
                values = values_by_slot[("abba", block, slot, model, metric)]
                (all_b2 if model == "B2" else all_c2).extend(values)
        b2_stats, c2_stats = stats(all_b2), stats(all_c2)
        for statistic, field in (("median", "median_us"), ("p95", "p95_us")):
            block_ratios = []
            for block in range(12):
                per_model: dict[str, list[float]] = {"B2": [], "C2": []}
                for slot in range(4):
                    model = "B2" if (block % 2 == 0 and slot in (0, 3)) or (block % 2 == 1 and slot in (1, 2)) else "C2"
                    values = values_by_slot[("abba", block, slot, model, metric)]
                    per_model[model].append(statistics.median(values) if statistic == "median" else percentile(values, 0.95))
                block_ratios.append(statistics.fmean(per_model["C2"]) / statistics.fmean(per_model["B2"]))
            mean_ratio, lower, upper = bootstrap_mean(block_ratios, options.seed + 100 + len(ratio_rows))
            ratio = float(c2_stats[field]) / float(b2_stats[field])
            guardrail = guardrails.get((metric, statistic))
            floor = noise_floor[(metric, statistic)]
            if guardrail is None or ratio <= guardrail:
                state = "pass"
            elif ratio - 1.0 <= floor:
                state = "performance-equivalent-within-noise"
            else:
                state = "fail"
            if guardrail is not None:
                decision_states.append(state)
            ratio_rows.append({
                "metric": metric, "statistic": statistic,
                "b2_samples": len(all_b2), "c2_samples": len(all_c2),
                "b2_value_us": b2_stats[field], "c2_value_us": c2_stats[field],
                "c2_b2_ratio": ratio, "block_ratio_mean": mean_ratio,
                "bootstrap_ci_lower": lower, "bootstrap_ci_upper": upper,
                "guardrail": guardrail if guardrail is not None else "descriptive",
                "empirical_abs_noise_floor": floor, "decision": state,
            })
    write_tsv(options.tracked_root / "performance_ratios.tsv", ratio_rows)
    write_tsv(options.tracked_root / "b2_c2_abba_summary.tsv", ratio_rows)
    write_tsv(options.tracked_root / "thermal_frequency.tsv", thermal_rows)

    governors = {str(row["value"]) for row in thermal_rows if row["kind"] == "governor"}
    frequencies = [float(row["value"]) for row in thermal_rows if row["kind"] == "frequency_khz"]
    temperatures = [float(row["value"]) for row in thermal_rows if row["kind"] == "temperature_millic"]
    state_pass = bool(governors) and governors == {"performance"}
    state_pass = state_pass and bool(frequencies) and min(frequencies) >= 0.95 * max(frequencies)
    state_pass = state_pass and bool(temperatures) and max(temperatures) <= 85000

    pipeline_rows = []
    for raw in options.pipeline:
        surface, path = parse_pipeline(raw)
        columns: dict[str, list[float]] = defaultdict(list)
        for row in read_tsv(path):
            for key in ("preprocess_ms", "inference_ms", "tail_ms", "decode_ms", "total_ms"):
                columns[key.removesuffix("_ms")].append(float(row[key]) * 1000.0)
        for metric, values in columns.items():
            pipeline_rows.append({
                "surface": surface,
                "measurement_source": "accepted-full-val-runner",
                "metric": metric,
                **stats(values),
            })
    if options.file_components:
        component_rows = read_tsv(options.file_components)
        for column, metric in (
            ("file_read_us", "file_read"),
            ("jpeg_decode_us", "jpeg_decode"),
            ("letterbox_tensor_us", "letterbox_tensor_preparation"),
            ("preprocess_total_us", "file_preprocess_total"),
        ):
            values = [float(row[column]) for row in component_rows]
            pipeline_rows.append({
                "surface": "model-independent-board-file-components",
                "measurement_source": "Stage65D-R1-3x100-deterministic-component-probe",
                "metric": metric,
                **stats(values),
            })
    if pipeline_rows:
        write_tsv(options.tracked_root / "file_pipeline_timing.tsv", pipeline_rows)

    passed = all(state != "fail" for state in decision_states) and state_pass
    (options.tracked_root / "performance_decision.md").write_text(
        "# Stage65D-R1 performance decision\n\n"
        f"Decision: `{'pass' if passed else 'fail'}`. Four B2/B2 blocks define the empirical process/session noise floor; twelve order-balanced ABBA blocks provide the C2/B2 comparison. A fixed-ratio miss no larger than the measured noise floor is classified as `performance-equivalent-within-noise`. Governor/frequency/thermal state: `{'pass' if state_pass else 'fail'}`. File-pipeline timing is diagnostic and no camera path was run.\n",
        encoding="utf-8",
    )
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
