#!/usr/bin/env python3
"""Normalize the Stage65E matched performance and empirical noise floors."""

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

SESSION_RE = re.compile(
    r"stage64_session inference_create_us=(?P<inference>[0-9.]+) "
    r"tail_create_us=(?P<tail>[0-9.]+)"
)
FIRST_RE = re.compile(
    r"stage64_first inference_us=(?P<inference>[0-9.]+) "
    r"tail_us=(?P<tail>[0-9.]+) total_us=(?P<total>[0-9.]+)"
)
RESULT_HASH_RE = re.compile(r"stage64_result .*output_fnv1a64=(?P<hash>0x[0-9a-fA-F]+)")
LABEL_RE = re.compile(
    r"(noise_b2|noise_c2|abba|cpu)-b([0-9]+)-s([0-9]+)-(B2|C2)-(spacemit|cpu)"
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("statistics require at least one value")
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
        "p999_us": percentile(values, 0.999),
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


def expected_keys() -> set[tuple[str, int, int, str, str]]:
    result: set[tuple[str, int, int, str, str]] = set()
    for model, phase in (("B2", "noise_b2"), ("C2", "noise_c2")):
        for block in range(8):
            result.update((phase, block, slot, model, "spacemit") for slot in range(4))
    for block in range(12):
        order = ("B2", "C2", "C2", "B2") if block % 2 == 0 else ("C2", "B2", "B2", "C2")
        result.update(("abba", block, slot, model, "spacemit") for slot, model in enumerate(order))
    for block in range(4):
        order = ("B2", "C2") if block % 2 == 0 else ("C2", "B2")
        result.update(("cpu", block, slot, model, "cpu") for slot, model in enumerate(order))
    return result


def summarize_resource(path: Path) -> dict[str, int]:
    all_rows = [
        row
        for row in read_tsv(path)
        if int(row["rss_kib"]) > 0
        and int(row["threads"]) > 0
        and int(row["fds"]) > 0
    ]
    if not all_rows:
        raise RuntimeError(f"no resource samples: {path}")
    steady_candidates = [row for row in all_rows if int(row["sample"]) >= 30]
    if len(steady_candidates) < 2:
        raise RuntimeError(f"no post-initialization resource samples: {path}")
    rows = steady_candidates[:-1]
    values = {
        key: [int(row[key]) for row in rows]
        for key in ("rss_kib", "peak_rss_kib", "fds", "threads")
    }
    return {
        "resource_samples": len(all_rows),
        "steady_resource_samples": len(rows),
        "terminal_samples_excluded": 1,
        "rss_startup_first_kib": int(all_rows[0]["rss_kib"]),
        "rss_min_kib": min(values["rss_kib"]),
        "rss_max_kib": max(values["rss_kib"]),
        "hwm_max_kib": max(values["peak_rss_kib"]),
        "fds_min": min(values["fds"]),
        "fds_max": max(values["fds"]),
        "threads_min": min(values["threads"]),
        "threads_max": max(values["threads"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--performance-root", required=True, type=Path)
    parser.add_argument("--tracked-root", required=True, type=Path)
    parser.add_argument("--pipeline", action="append", default=[])
    parser.add_argument("--file-components", type=Path)
    parser.add_argument("--seed", type=int, default=65015)
    options = parser.parse_args()

    status_rows = read_tsv(options.performance_root / "status.raw.tsv")
    status_by_key: dict[tuple[str, int, int, str, str], dict[str, str]] = {}
    for row in status_rows:
        key = (row["phase"], int(row["block"]), int(row["slot"]), row["model"], row["provider"])
        if key in status_by_key:
            raise RuntimeError(f"duplicate status row: {key}")
        status_by_key[key] = row

    slot_rows: list[dict[str, object]] = []
    creation_rows: list[dict[str, object]] = []
    first_rows: list[dict[str, object]] = []
    thermal_rows: list[dict[str, object]] = []
    values_by_slot: dict[tuple[str, int, int, str, str], list[float]] = {}
    output_hashes: dict[tuple[str, str], set[str]] = defaultdict(set)
    fnv_hashes: dict[tuple[str, str], set[str]] = defaultdict(set)
    result_fnv_hashes: dict[tuple[str, str], set[str]] = defaultdict(set)
    sample_hash_modes: set[str] = set()
    observed: set[tuple[str, int, int, str, str]] = set()

    for directory in sorted(path for path in options.performance_root.iterdir() if path.is_dir()):
        match = LABEL_RE.fullmatch(directory.name)
        if not match:
            continue
        phase, block_text, slot_text, model, provider = match.groups()
        block, slot = int(block_text), int(slot_text)
        key = (phase, block, slot, model, provider)
        observed.add(key)
        status = status_by_key.get(key)
        if status is None or int(status["exit_code"]) != 0:
            raise RuntimeError(f"missing or failed status: {key}")
        for field, filename in (
            ("output_sha256", "output.bin"),
            ("samples_sha256", "samples.tsv"),
            ("resource_sha256", "resource.tsv"),
        ):
            if sha256(directory / filename) != status[field]:
                raise RuntimeError(f"{field} mismatch: {key}")
        sample_rows = read_tsv(directory / "samples.tsv")
        if len(sample_rows) != 100:
            raise RuntimeError(f"expected 100 measured runs in {key}, got {len(sample_rows)}")
        surface = (model, provider)
        output_hashes[surface].add(status["output_sha256"])
        sample_fields = set(sample_rows[0])
        required_sample_fields = {"repeat", "run", "inference_us", "tail_us", "total_us"}
        if not required_sample_fields.issubset(sample_fields):
            raise RuntimeError(f"timing sample contract drift: {key}")
        if "output_fnv1a64" in sample_fields:
            sample_hash_modes.add("per-run-fnv1a64")
            fnv_hashes[surface].update(row["output_fnv1a64"] for row in sample_rows)
        else:
            sample_hash_modes.add("not-emitted-by-bound-runner")
        resource = summarize_resource(directory / "resource.tsv")
        parsed = parse_samples(directory / "samples.tsv")
        for metric, values in parsed.items():
            values_by_slot[(phase, block, slot, model, metric)] = values
            slot_rows.append({
                "phase": phase,
                "block": block,
                "slot": slot,
                "model": model,
                "provider": provider,
                "metric": metric,
                **stats(values),
                **resource,
            })
        text = (directory / "run.log").read_text(encoding="utf-8", errors="replace")
        session = SESSION_RE.search(text)
        first = FIRST_RE.search(text)
        result_hash = RESULT_HASH_RE.search(text)
        if not session or not first or not result_hash or "stage64_result status=pass" not in text:
            raise RuntimeError(f"incomplete runner log: {directory}")
        result_fnv_hashes[surface].add(result_hash.group("hash").lower())
        creation_rows.append({
            "phase": phase,
            "block": block,
            "slot": slot,
            "model": model,
            "provider": provider,
            "inference_create_us": session.group("inference"),
            "tail_create_us": session.group("tail"),
            "status": "pass",
        })
        first_rows.append({
            "phase": phase,
            "block": block,
            "slot": slot,
            "model": model,
            "provider": provider,
            "inference_us": first.group("inference"),
            "tail_us": first.group("tail"),
            "total_us": first.group("total"),
            "status": "pass",
        })
        for moment in ("before", "after"):
            for line in (directory / f"state-{moment}.tsv").read_text().splitlines():
                fields = line.split("\t")
                if len(fields) == 3:
                    thermal_rows.append({
                        "phase": phase,
                        "block": block,
                        "slot": slot,
                        "model": model,
                        "provider": provider,
                        "moment": moment,
                        "kind": fields[0],
                        "source": fields[1],
                        "value": fields[2],
                    })

    required = expected_keys()
    if observed != required or set(status_by_key) != required:
        missing = sorted(required - observed)
        extra = sorted(observed - required)
        raise RuntimeError(f"performance schedule mismatch; missing={missing[:3]} extra={extra[:3]}")
    if any(len(values) != 1 for values in output_hashes.values()):
        raise RuntimeError("output SHA changed within an exact model/provider surface")
    if len(sample_hash_modes) != 1:
        raise RuntimeError(f"mixed per-run output-hash contracts: {sample_hash_modes}")
    per_run_hashes = sample_hash_modes == {"per-run-fnv1a64"}
    if per_run_hashes and any(len(values) != 1 for values in fnv_hashes.values()):
        raise RuntimeError("per-run output hash changed within an exact model/provider surface")
    if any(len(values) != 1 for values in result_fnv_hashes.values()):
        raise RuntimeError("result FNV changed within an exact model/provider surface")
    if sum(row["provider"] == "spacemit" for row in creation_rows if row["model"] == "B2") < 10:
        raise RuntimeError("fewer than ten clean B2 EP session creations")
    if sum(row["provider"] == "spacemit" for row in creation_rows if row["model"] == "C2") < 10:
        raise RuntimeError("fewer than ten clean C2 EP session creations")

    hash_rows = []
    for model, provider in sorted(output_hashes):
        hash_rows.append({
            "model": model,
            "provider": provider,
            "fresh_session_slots": sum(
                row["model"] == model and row["provider"] == provider
                for row in creation_rows
            ),
            "output_sha256": next(iter(output_hashes[(model, provider)])),
            "result_fnv1a64": next(iter(result_fnv_hashes[(model, provider)])),
            "per_run_sample_fingerprint": "emitted" if per_run_hashes else "not-emitted-by-bound-runner",
            "status": "pass",
        })
    write_tsv(options.tracked_root / "performance_output_hash_stability.tsv", hash_rows)

    write_tsv(options.tracked_root / "session_creation_timing.tsv", creation_rows)
    write_tsv(options.tracked_root / "first_run_timing.tsv", first_rows)
    write_tsv(options.tracked_root / "b2_b2_noise_floor_raw.tsv", [row for row in slot_rows if row["phase"] == "noise_b2"])
    write_tsv(options.tracked_root / "c2_c2_noise_floor_raw.tsv", [row for row in slot_rows if row["phase"] == "noise_c2"])
    write_tsv(options.tracked_root / "b2_c2_abba_raw.tsv", [row for row in slot_rows if row["phase"] == "abba"])

    noise_floor: dict[tuple[str, str, str], float] = {}
    noise_summary: dict[str, list[dict[str, object]]] = {"B2": [], "C2": []}
    for model, phase in (("B2", "noise_b2"), ("C2", "noise_c2")):
        for metric in ("inference", "tail", "two_stage"):
            for statistic in ("median", "p95"):
                ratios = []
                for block in range(8):
                    left = []
                    right = []
                    for slot in (0, 3):
                        values = values_by_slot[(phase, block, slot, model, metric)]
                        left.append(statistics.median(values) if statistic == "median" else percentile(values, 0.95))
                    for slot in (1, 2):
                        values = values_by_slot[(phase, block, slot, model, metric)]
                        right.append(statistics.median(values) if statistic == "median" else percentile(values, 0.95))
                    ratios.append(statistics.fmean(left) / statistics.fmean(right))
                mean_ratio, lower, upper = bootstrap_mean(ratios, options.seed + len(noise_summary[model]))
                floor = max(abs(min(ratios) - 1.0), abs(max(ratios) - 1.0))
                noise_floor[(model, metric, statistic)] = floor
                noise_summary[model].append({
                    "model": model,
                    "metric": metric,
                    "statistic": statistic,
                    "blocks": 8,
                    "block_ratio_mean": mean_ratio,
                    "block_ratio_min": min(ratios),
                    "block_ratio_max": max(ratios),
                    "bootstrap_ci_lower": lower,
                    "bootstrap_ci_upper": upper,
                    "empirical_abs_noise_floor": floor,
                })
    write_tsv(options.tracked_root / "b2_b2_noise_floor_summary.tsv", noise_summary["B2"])
    write_tsv(options.tracked_root / "c2_c2_noise_floor_summary.tsv", noise_summary["C2"])

    ratio_rows: list[dict[str, object]] = []
    decision_states: list[str] = []
    guardrails = {
        ("inference", "median"): 1.02,
        ("two_stage", "median"): 1.02,
        ("inference", "p95"): 1.05,
        ("two_stage", "p95"): 1.05,
    }
    aggregate_by_model: dict[tuple[str, str], list[float]] = defaultdict(list)
    for metric in ("inference", "tail", "two_stage"):
        for block in range(12):
            order = ("B2", "C2", "C2", "B2") if block % 2 == 0 else ("C2", "B2", "B2", "C2")
            for slot, model in enumerate(order):
                aggregate_by_model[(model, metric)].extend(values_by_slot[("abba", block, slot, model, metric)])
        b2_stats = stats(aggregate_by_model[("B2", metric)])
        c2_stats = stats(aggregate_by_model[("C2", metric)])
        for statistic, field in (("median", "median_us"), ("p95", "p95_us")):
            block_ratios = []
            for block in range(12):
                order = ("B2", "C2", "C2", "B2") if block % 2 == 0 else ("C2", "B2", "B2", "C2")
                per_model: dict[str, list[float]] = {"B2": [], "C2": []}
                for slot, model in enumerate(order):
                    values = values_by_slot[("abba", block, slot, model, metric)]
                    per_model[model].append(statistics.median(values) if statistic == "median" else percentile(values, 0.95))
                block_ratios.append(statistics.fmean(per_model["C2"]) / statistics.fmean(per_model["B2"]))
            mean_ratio, lower, upper = bootstrap_mean(block_ratios, options.seed + 100 + len(ratio_rows))
            ratio = float(c2_stats[field]) / float(b2_stats[field])
            guardrail = guardrails.get((metric, statistic))
            floor = max(noise_floor[("B2", metric, statistic)], noise_floor[("C2", metric, statistic)])
            if guardrail is None or ratio <= guardrail:
                state = "pass"
            elif ratio - 1.0 <= floor:
                state = "performance-equivalent-within-noise"
            else:
                state = "fail"
            if guardrail is not None:
                decision_states.append(state)
            ratio_rows.append({
                "metric": metric,
                "statistic": statistic,
                "b2_samples": b2_stats["samples"],
                "c2_samples": c2_stats["samples"],
                "b2_value_us": b2_stats[field],
                "c2_value_us": c2_stats[field],
                "c2_b2_ratio": ratio,
                "block_ratio_mean": mean_ratio,
                "bootstrap_ci_lower": lower,
                "bootstrap_ci_upper": upper,
                "guardrail": guardrail if guardrail is not None else "descriptive",
                "b2_empirical_abs_noise_floor": noise_floor[("B2", metric, statistic)],
                "c2_empirical_abs_noise_floor": noise_floor[("C2", metric, statistic)],
                "comparison_noise_floor": floor,
                "decision": state,
            })
    write_tsv(options.tracked_root / "performance_ratios.tsv", ratio_rows)
    write_tsv(options.tracked_root / "b2_c2_abba_summary.tsv", ratio_rows)

    cpu_rows: list[dict[str, object]] = []
    for model in ("B2", "C2"):
        for metric in ("inference", "tail", "two_stage"):
            values: list[float] = []
            for block in range(4):
                order = ("B2", "C2") if block % 2 == 0 else ("C2", "B2")
                slot = order.index(model)
                values.extend(values_by_slot[("cpu", block, slot, model, metric)])
            cpu_rows.append({"model": model, "provider": "cpu", "metric": metric, **stats(values)})
    write_tsv(options.tracked_root / "cpu_reference_timing.tsv", cpu_rows)

    tail_rows: list[dict[str, object]] = []
    for model in ("B2", "C2"):
        for metric in ("tail", "two_stage"):
            tail_rows.append({"model": model, "provider": "spacemit", "metric": metric, **stats(aggregate_by_model[(model, metric)])})
    write_tsv(options.tracked_root / "tail_two_stage_timing.tsv", tail_rows)
    write_tsv(options.tracked_root / "thermal_frequency_timing.tsv", thermal_rows)

    governors = {str(row["value"]) for row in thermal_rows if row["kind"] == "governor"}
    frequencies = [float(row["value"]) for row in thermal_rows if row["kind"] == "frequency_khz"]
    temperatures = [float(row["value"]) for row in thermal_rows if row["kind"] == "temperature_millic"]
    state_pass = governors == {"performance"}
    state_pass = state_pass and bool(frequencies) and min(frequencies) >= 0.95 * max(frequencies)
    state_pass = state_pass and bool(temperatures) and max(temperatures) <= 85000
    resource_pass = all(
        int(row["fds_max"]) - int(row["fds_min"]) <= 2
        and int(row["threads_max"]) - int(row["threads_min"]) <= 2
        and int(row["rss_max_kib"]) - int(row["rss_min_kib"]) <= 16384
        for row in slot_rows
    )

    pipeline_rows: list[dict[str, object]] = []
    for raw in options.pipeline:
        surface, path = parse_pipeline(raw)
        columns: dict[str, list[float]] = defaultdict(list)
        for row in read_tsv(path):
            for key in ("preprocess_ms", "inference_ms", "tail_ms", "decode_ms", "total_ms"):
                columns[key.removesuffix("_ms")].append(float(row[key]) * 1000.0)
        for metric, values in columns.items():
            pipeline_rows.append({"surface": surface, "measurement_source": "Stage65E-board-100-image", "metric": metric, **stats(values)})
    if options.file_components:
        component_rows = read_tsv(options.file_components)
        for column, metric in (
            ("file_read_us", "file_read"),
            ("jpeg_decode_us", "jpeg_decode"),
            ("letterbox_tensor_us", "letterbox_tensor_preparation"),
            ("preprocess_total_us", "file_preprocess_total"),
        ):
            values = [float(row[column]) for row in component_rows]
            pipeline_rows.append({"surface": "model-independent-file-components", "measurement_source": "Stage65E-3x100", "metric": metric, **stats(values)})
    if pipeline_rows:
        write_tsv(options.tracked_root / "file_pipeline_timing.tsv", pipeline_rows)

    passed = all(state != "fail" for state in decision_states) and state_pass and resource_pass
    (options.tracked_root / "performance_decision.md").write_text(
        "# Stage65E performance decision\n\n"
        f"Decision: `{'pass' if passed else 'fail'}`. Eight B2/B2 and eight C2/C2 "
        "order-balanced blocks define independent empirical process/session noise floors; "
        "twelve order-balanced B2/C2 ABBA blocks provide the matched comparison. A fixed-ratio "
        "miss no larger than the larger same-arm noise floor is classified as "
        "`performance-equivalent-within-noise`. "
        f"Governor/frequency/thermal state: `{'pass' if state_pass else 'fail'}`; per-slot "
        f"resource bounds: `{'pass' if resource_pass else 'fail'}`. CPU and file-pipeline "
        "rows are reference surfaces and no camera path was run. The identity-bound Stage65E "
        f"runner per-run sample fingerprint contract is `{'emitted-and-stable' if per_run_hashes else 'not-emitted'}`; "
        "determinism is gated by the emitted final-result FNV and exact output SHA-256 across every fresh process/session slot. "
        "All resource samples are retained; the normal process/session initialization ramp is reported "
        "separately and FD/thread/RSS bounds use samples 30 onward while excluding the final "
        "process-teardown observation, matching the stability contract.\n",
        encoding="utf-8",
    )
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
