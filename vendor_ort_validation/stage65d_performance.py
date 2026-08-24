#!/usr/bin/env python3
"""Normalize Stage65D matched-performance evidence and enforce its gates."""

from __future__ import annotations

import argparse
import csv
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


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stats(values: list[float]) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    stddev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "samples": len(values),
        "mean_us": mean,
        "stddev_us": stddev,
        "cv_pct": 0.0 if mean == 0 else 100.0 * stddev / mean,
        "median_us": statistics.median(values),
        "p90_us": percentile(values, 0.90),
        "p95_us": percentile(values, 0.95),
        "p99_us": percentile(values, 0.99),
        "min_us": min(values),
        "max_us": max(values),
    }


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_samples(path: Path) -> dict[str, list[float]]:
    result = {"inference": [], "tail": [], "two_stage": []}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            result["inference"].append(float(row["inference_us"]))
            result["tail"].append(float(row["tail_us"]))
            result["two_stage"].append(float(row["total_us"]))
    return result


def session_rows(profile_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    creation: list[dict[str, object]] = []
    first: list[dict[str, object]] = []
    for log in sorted(profile_root.glob("*-*/run.log")):
        parts = log.parent.name.split("-")
        if len(parts) < 3:
            continue
        model, provider, mode = parts[0], parts[1], "-".join(parts[2:])
        text = log.read_text(encoding="utf-8", errors="replace")
        session_match = SESSION_RE.search(text)
        first_match = FIRST_RE.search(text)
        if session_match:
            creation.append(
                {
                    "model": model,
                    "provider": provider,
                    "mode": mode,
                    "inference_create_us": session_match.group("inference"),
                    "tail_create_us": session_match.group("tail"),
                    "status": "pass",
                }
            )
        if first_match:
            first.append(
                {
                    "model": model,
                    "provider": provider,
                    "mode": mode,
                    "inference_us": first_match.group("inference"),
                    "tail_us": first_match.group("tail"),
                    "total_us": first_match.group("total"),
                    "status": "pass",
                }
            )
    return creation, first


def bootstrap_block_ratios(
    block_ratios: list[float], *, seed: int = 65009, replicates: int = 10000
) -> tuple[float, float]:
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        draw = [rng.choice(block_ratios) for _ in block_ratios]
        estimates.append(statistics.fmean(draw))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def normalize_abba(performance_root: Path, tracked_root: Path) -> tuple[bool, dict[str, float]]:
    slot_rows: list[dict[str, object]] = []
    aggregate: dict[tuple[str, str], list[float]] = defaultdict(list)
    block_values: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for samples_file in sorted(performance_root.glob("*/samples.tsv")):
        label = samples_file.parent.name
        match = re.fullmatch(r"(noise|abba)-b([0-9]+)-s([0-9]+)-(C2|B2)", label)
        if not match:
            continue
        phase, block_text, slot_text, model = match.groups()
        block, slot = int(block_text), int(slot_text)
        parsed = parse_samples(samples_file)
        for metric, values in parsed.items():
            row = {
                "phase": phase,
                "block": block,
                "slot": slot,
                "model": model,
                "metric": metric,
                **stats(values),
            }
            slot_rows.append(row)
            if phase == "abba":
                aggregate[(model, metric)].extend(values)
                block_values[(block, model, metric)].extend(values)

    fields = [
        "phase",
        "block",
        "slot",
        "model",
        "metric",
        "samples",
        "mean_us",
        "stddev_us",
        "cv_pct",
        "median_us",
        "p90_us",
        "p95_us",
        "p99_us",
        "min_us",
        "max_us",
    ]
    write_tsv(
        tracked_root / "b2_b2_noise_floor_raw.tsv",
        [row for row in slot_rows if row["phase"] == "noise"],
        fields,
    )
    write_tsv(
        tracked_root / "b2_c2_abba_raw.tsv",
        [row for row in slot_rows if row["phase"] == "abba"],
        fields,
    )
    write_tsv(tracked_root / "steady_inference_abba.tsv", [r for r in slot_rows if r["metric"] == "inference"], fields)
    write_tsv(tracked_root / "tail_timing.tsv", [r for r in slot_rows if r["metric"] == "tail"], fields)
    write_tsv(tracked_root / "two_stage_abba.tsv", [r for r in slot_rows if r["metric"] == "two_stage"], fields)

    ratios: dict[str, float] = {}
    ratio_rows: list[dict[str, object]] = []
    for metric in ("inference", "two_stage"):
        c2 = aggregate[("C2", metric)]
        b2 = aggregate[("B2", metric)]
        c2_stats, b2_stats = stats(c2), stats(b2)
        median_ratio = float(c2_stats["median_us"]) / float(b2_stats["median_us"])
        p95_ratio = float(c2_stats["p95_us"]) / float(b2_stats["p95_us"])
        block_ratios = [
            statistics.median(block_values[(block, "C2", metric)])
            / statistics.median(block_values[(block, "B2", metric)])
            for block in range(5)
        ]
        ci_lower, ci_upper = bootstrap_block_ratios(block_ratios)
        ratios[f"{metric}_median_ratio"] = median_ratio
        ratios[f"{metric}_p95_ratio"] = p95_ratio
        ratio_rows.append(
            {
                "metric": metric,
                "c2_samples": len(c2),
                "b2_samples": len(b2),
                "c2_median_us": c2_stats["median_us"],
                "b2_median_us": b2_stats["median_us"],
                "median_ratio": median_ratio,
                "c2_p95_us": c2_stats["p95_us"],
                "b2_p95_us": b2_stats["p95_us"],
                "p95_ratio": p95_ratio,
                "block_ratio_mean": statistics.fmean(block_ratios),
                "bootstrap_ci_lower": ci_lower,
                "bootstrap_ci_upper": ci_upper,
            }
        )

    write_tsv(
        tracked_root / "performance_ratios.tsv",
        ratio_rows,
        [
            "metric",
            "c2_samples",
            "b2_samples",
            "c2_median_us",
            "b2_median_us",
            "median_ratio",
            "c2_p95_us",
            "b2_p95_us",
            "p95_ratio",
            "block_ratio_mean",
            "bootstrap_ci_lower",
            "bootstrap_ci_upper",
        ],
    )
    write_tsv(
        tracked_root / "b2_c2_abba_summary.tsv",
        ratio_rows,
        [
            "metric",
            "c2_samples",
            "b2_samples",
            "c2_median_us",
            "b2_median_us",
            "median_ratio",
            "c2_p95_us",
            "b2_p95_us",
            "p95_ratio",
            "block_ratio_mean",
            "bootstrap_ci_lower",
            "bootstrap_ci_upper",
        ],
    )

    passed = (
        ratios["inference_median_ratio"] <= 1.02
        and ratios["two_stage_median_ratio"] <= 1.02
        and ratios["inference_p95_ratio"] <= 1.05
        and ratios["two_stage_p95_ratio"] <= 1.05
    )
    return passed, ratios


def pipeline_rows(coco_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(coco_root.glob("val/*/timing.tsv")):
        model, provider = path.parent.name.split("-", maxsplit=1)
        columns: dict[str, list[float]] = defaultdict(list)
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                for key in ("preprocess_ms", "inference_ms", "tail_ms", "decode_ms", "total_ms"):
                    columns[key].append(float(row[key]) * 1000.0)
        for metric, values in columns.items():
            rows.append({"model": model, "provider": provider, "metric": metric.removesuffix("_ms"), **stats(values)})
    return rows


def thermal_frequency_rows(performance_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(performance_root.glob("*/state-*.tsv")):
        phase = path.parent.name
        moment = path.stem.removeprefix("state-")
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3:
                    kind, source, value = parts
                    rows.append(
                        {
                            "slot": phase,
                            "moment": moment,
                            "kind": kind,
                            "source": source,
                            "value": value,
                        }
                    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--tracked-root", required=True, type=Path)
    args = parser.parse_args()
    args.tracked_root.mkdir(parents=True, exist_ok=True)

    creation, first = session_rows(args.raw_root / "board" / "profile")
    write_tsv(
        args.tracked_root / "session_creation_timing.tsv",
        creation,
        ["model", "provider", "mode", "inference_create_us", "tail_create_us", "status"],
    )
    write_tsv(
        args.tracked_root / "first_run_timing.tsv",
        first,
        ["model", "provider", "mode", "inference_us", "tail_us", "total_us", "status"],
    )
    combined_session = []
    for row in creation:
        combined_session.append({"surface": "session_creation", **row})
    for row in first:
        combined_session.append({
            "surface": "first_run",
            "model": row["model"],
            "provider": row["provider"],
            "mode": row["mode"],
            "inference_create_us": "",
            "tail_create_us": "",
            "inference_us": row["inference_us"],
            "tail_us": row["tail_us"],
            "total_us": row["total_us"],
            "status": row["status"],
        })
    write_tsv(
        args.tracked_root / "session_first_run_timing.tsv",
        combined_session,
        ["surface", "model", "provider", "mode", "inference_create_us", "tail_create_us", "inference_us", "tail_us", "total_us", "status"],
    )

    passed, ratios = normalize_abba(
        args.raw_root / "board" / "performance" / "abba", args.tracked_root
    )
    pipeline = pipeline_rows(args.raw_root / "board" / "coco")
    write_tsv(
        args.tracked_root / "pipeline_timing.tsv",
        pipeline,
        [
            "model",
            "provider",
            "metric",
            "samples",
            "mean_us",
            "stddev_us",
            "cv_pct",
            "median_us",
            "p90_us",
            "p95_us",
            "p99_us",
            "min_us",
            "max_us",
        ],
    )
    write_tsv(
        args.tracked_root / "file_pipeline_timing.tsv",
        pipeline,
        [
            "model",
            "provider",
            "metric",
            "samples",
            "mean_us",
            "stddev_us",
            "cv_pct",
            "median_us",
            "p90_us",
            "p95_us",
            "p99_us",
            "min_us",
            "max_us",
        ],
    )
    tail_two_stage = []
    for path in (args.tracked_root / "tail_timing.tsv", args.tracked_root / "two_stage_abba.tsv"):
        with path.open(encoding="utf-8", newline="") as stream:
            tail_two_stage.extend(csv.DictReader(stream, delimiter="\t"))
    write_tsv(
        args.tracked_root / "tail_two_stage_timing.tsv",
        tail_two_stage,
        [
            "phase", "block", "slot", "model", "metric", "samples", "mean_us",
            "stddev_us", "cv_pct", "median_us", "p90_us", "p95_us", "p99_us",
            "min_us", "max_us",
        ],
    )
    thermal = thermal_frequency_rows(args.raw_root / "board" / "performance" / "abba")
    write_tsv(
        args.tracked_root / "thermal_frequency.tsv",
        thermal,
        ["slot", "moment", "kind", "source", "value"],
    )

    governors = {row["value"] for row in thermal if row["kind"] == "governor"}
    frequency_values = [float(row["value"]) for row in thermal if row["kind"] == "frequency_khz"]
    state_pass = (not governors or governors == {"performance"}) and (
        not frequency_values or min(frequency_values) >= 0.95 * max(frequency_values)
    )
    passed = passed and state_pass
    report = args.tracked_root / "performance_decision.md"
    report.write_text(
        "# Stage65D matched performance decision\n\n"
        f"- Decision: `{'pass' if passed else 'fail'}`\n"
        f"- C2/B2 steady-inference median ratio: `{ratios['inference_median_ratio']:.9f}` (gate `<= 1.02`)\n"
        f"- C2/B2 two-stage median ratio: `{ratios['two_stage_median_ratio']:.9f}` (gate `<= 1.02`)\n"
        f"- C2/B2 steady-inference p95 ratio: `{ratios['inference_p95_ratio']:.9f}` (gate `<= 1.05`)\n"
        f"- C2/B2 two-stage p95 ratio: `{ratios['two_stage_p95_ratio']:.9f}` (gate `<= 1.05`)\n"
        f"- Governor/frequency gate: `{'pass' if state_pass else 'fail'}`\n"
        "- Protocol: two B2/B2 noise blocks followed by five process-isolated ABBA blocks, 100 measured runs per slot, CPU set 0-3.\n"
        "- File/JPEG pipeline timing is diagnostic and excluded from promotion decisions.\n",
        encoding="utf-8",
    )
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
