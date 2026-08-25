#!/usr/bin/env python3
"""Normalize symmetric short soak and segmented C2 10k soak for Stage65D-R1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stats(values: list[float]) -> dict[str, object]:
    mean = statistics.fmean(values)
    standard = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "samples": len(values), "mean_us": mean, "stddev_us": standard,
        "cv_pct": 0.0 if mean == 0 else 100.0 * standard / mean,
        "p50_us": statistics.median(values), "p95_us": percentile(values, 0.95),
        "p99_us": percentile(values, 0.99), "p999_us": percentile(values, 0.999),
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


def collect(root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, list[float]], dict[str, set[str]], dict[str, set[str]], list[dict[str, object]], list[dict[str, object]]]:
    status = read_tsv(root / "status.raw.tsv")
    windows: list[dict[str, object]] = []
    resources: list[dict[str, object]] = []
    aggregate: dict[str, list[float]] = defaultdict(list)
    output_hashes: dict[str, set[str]] = defaultdict(set)
    fnv_hashes: dict[str, set[str]] = defaultdict(set)
    thermal: list[dict[str, object]] = []
    output_diagnostics: list[dict[str, object]] = []
    seen_segments: set[tuple[str, str, str]] = set()
    for status_row in status:
        directory = root / f"segment-{status_row['segment']}-{status_row['position']}-{status_row['model']}"
        model = status_row["model"]
        key = (status_row["segment"], status_row["position"], model)
        if key in seen_segments:
            raise RuntimeError(f"duplicate soak segment: {key}")
        seen_segments.add(key)
        if int(status_row["exit_code"]) != 0:
            raise RuntimeError(f"failed soak segment: {directory}")
        for field, filename in (
            ("output_sha256", "output.bin"),
            ("samples_sha256", "samples.tsv"),
            ("resource_sha256", "resource.tsv"),
        ):
            if sha256(directory / filename) != status_row[field]:
                raise RuntimeError(f"{field} mismatch in {directory}")
        output_hashes[model].add(status_row["output_sha256"])
        values = {"inference": [], "tail": [], "two_stage": []}
        sample_rows = read_tsv(directory / "samples.tsv")
        for row in sample_rows:
            values["inference"].append(float(row["inference_us"]))
            values["tail"].append(float(row["tail_us"]))
            values["two_stage"].append(float(row["total_us"]))
            fnv_hashes[model].add(row["output_fnv1a64"])
        expected_runs = int(status_row["runs"])
        if len(sample_rows) != expected_runs:
            raise RuntimeError(f"sample count mismatch in {directory}: {len(sample_rows)} != {expected_runs}")
        run_log = (directory / "run.log").read_text(encoding="utf-8", errors="replace")
        if "stage64_result status=pass" not in run_log:
            raise RuntimeError(f"missing pass result in {directory}")
        output = np.fromfile(directory / "output.bin", dtype="<f4")
        finite = int(np.isfinite(output).sum())
        scores = output.reshape(300, 6)[:, 4] if output.size == 1800 else np.asarray([], dtype=np.float32)
        collapsed = scores.size == 0 or not np.isfinite(scores).all() or np.unique(scores).size < 2
        output_diagnostics.append({
            "root": root.name,
            "segment": status_row["segment"],
            "position": status_row["position"],
            "model": model,
            "output_elements": output.size,
            "non_finite_count": output.size - finite,
            "score_unique_count": int(np.unique(scores).size),
            "score_min": float(np.min(scores)) if scores.size else "not-available",
            "score_max": float(np.max(scores)) if scores.size else "not-available",
            "score_collapse": "yes" if collapsed else "no",
            "status": "pass" if output.size == 1800 and finite == output.size and not collapsed else "fail",
        })
        for metric, metric_values in values.items():
            aggregate[f"{model}:{metric}"].extend(metric_values)
            windows.append({
                "root": root.name, "segment": status_row["segment"],
                "position": status_row["position"], "model": model,
                "metric": metric, **stats(metric_values),
            })
        resource = read_tsv(directory / "resource.tsv")
        if not resource:
            raise RuntimeError(f"missing resource samples in {directory}")
        numeric = {
            name: [int(row[name]) for row in resource]
            for name in (
                "rss_kib",
                "peak_rss_kib",
                "fds",
                "threads",
                "voluntary_ctxt",
                "nonvoluntary_ctxt",
            )
        }
        rss_slope = (numeric["rss_kib"][-1] - numeric["rss_kib"][0]) / max(1, len(resource) - 1)
        resources.append({
            "root": root.name, "segment": status_row["segment"],
            "position": status_row["position"], "model": model,
            "samples": len(resource), "rss_first_kib": numeric["rss_kib"][0],
            "rss_last_kib": numeric["rss_kib"][-1], "rss_max_kib": max(numeric["rss_kib"]),
            "peak_rss_max_kib": max(numeric["peak_rss_kib"]), "rss_slope_kib_per_sample": rss_slope,
            "fds_min": min(numeric["fds"]), "fds_max": max(numeric["fds"]),
            "threads_min": min(numeric["threads"]), "threads_max": max(numeric["threads"]),
            "voluntary_ctxt_first": numeric["voluntary_ctxt"][0],
            "voluntary_ctxt_last": numeric["voluntary_ctxt"][-1],
            "voluntary_ctxt_delta": numeric["voluntary_ctxt"][-1] - numeric["voluntary_ctxt"][0],
            "nonvoluntary_ctxt_first": numeric["nonvoluntary_ctxt"][0],
            "nonvoluntary_ctxt_last": numeric["nonvoluntary_ctxt"][-1],
            "nonvoluntary_ctxt_delta": numeric["nonvoluntary_ctxt"][-1] - numeric["nonvoluntary_ctxt"][0],
            "exit_code": status_row["exit_code"],
        })
        for moment in ("before", "after"):
            for line in (directory / f"state-{moment}.tsv").read_text().splitlines():
                fields = line.split("\t")
                if len(fields) == 3:
                    thermal.append({
                        "root": root.name, "segment": status_row["segment"],
                        "position": status_row["position"], "model": model,
                        "moment": moment, "kind": fields[0], "source": fields[1], "value": fields[2],
                    })
    return windows, resources, aggregate, output_hashes, fnv_hashes, thermal, output_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--short-root", required=True, type=Path)
    parser.add_argument("--long-root", required=True, type=Path)
    parser.add_argument("--tracked-root", required=True, type=Path)
    options = parser.parse_args()
    short = collect(options.short_root)
    long = collect(options.long_root)

    short_rows = []
    for model in ("B2", "C2"):
        for metric in ("inference", "tail", "two_stage"):
            short_rows.append({"model": model, "metric": metric, **stats(short[2][f"{model}:{metric}"])})
    long_rows = [{"model": "C2", "metric": metric, **stats(long[2][f"C2:{metric}"])} for metric in ("inference", "tail", "two_stage")]
    write_tsv(options.tracked_root / "short_soak.tsv", short_rows)
    write_tsv(options.tracked_root / "c2_10k_soak.tsv", long_rows)

    passed = True
    resources = short[1] + long[1]
    for row in resources:
        row_pass = (
            int(row["exit_code"]) == 0
            and int(row["fds_max"]) - int(row["fds_min"]) <= 2
            and int(row["threads_max"]) - int(row["threads_min"]) <= 2
            and int(row["rss_last_kib"]) <= int(row["rss_first_kib"]) + 16384
        )
        row["status"] = "pass" if row_pass else "fail"
        passed = passed and row_pass
    write_tsv(options.tracked_root / "resource_drift.tsv", resources)

    output_diagnostics = short[6] + long[6]
    write_tsv(options.tracked_root / "soak_output_semantics.tsv", output_diagnostics)
    passed = passed and all(row["status"] == "pass" for row in output_diagnostics)

    thermal = short[5] + long[5]
    write_tsv(options.tracked_root / "thermal_frequency_log.tsv", thermal)
    governors = {str(row["value"]) for row in thermal if row["kind"] == "governor"}
    frequencies = [float(row["value"]) for row in thermal if row["kind"] == "frequency_khz"]
    temperatures = [float(row["value"]) for row in thermal if row["kind"] == "temperature_millic"]
    thermal_pass = bool(governors) and governors == {"performance"}
    thermal_pass = thermal_pass and bool(frequencies) and min(frequencies) >= 0.95 * max(frequencies)
    thermal_pass = thermal_pass and bool(temperatures) and max(temperatures) <= 85000
    passed = passed and thermal_pass

    hash_rows = []
    for model in ("B2", "C2"):
        expected_short = 2000
        actual = len(short[2][f"{model}:two_stage"])
        selected_diagnostics = [row for row in short[6] if row["model"] == model]
        model_pass = (
            len(short[3][model]) == 1
            and len(short[4][model]) == 1
            and actual == expected_short
            and all(row["status"] == "pass" for row in selected_diagnostics)
        )
        passed = passed and model_pass
        hash_rows.append({
            "surface": f"short-order-balanced-{model}", "runs": actual,
            "output_sha256_count": len(short[3][model]), "output_fnv1a64_count": len(short[4][model]),
            "non_finite_count": sum(int(row["non_finite_count"]) for row in short[6] if row["model"] == model),
            "score_collapse_segments": sum(row["score_collapse"] == "yes" for row in short[6] if row["model"] == model),
            "status": "pass" if model_pass else "fail",
        })
    windows = [row for row in long[0] if row["model"] == "C2" and row["metric"] == "two_stage"]
    first = statistics.median(float(row["p50_us"]) for row in windows[:5])
    second = statistics.median(float(row["p50_us"]) for row in windows[5:])
    drift = second / first
    long_runs = len(long[2]["C2:two_stage"])
    long_pass = (
        len(long[3]["C2"]) == 1
        and len(long[4]["C2"]) == 1
        and long_runs == 10000
        and 0.95 <= drift <= 1.05
        and all(row["status"] == "pass" for row in long[6])
    )
    passed = passed and long_pass
    hash_rows.append({
        "surface": "c2-10x1000-clean-session", "runs": long_runs,
        "output_sha256_count": len(long[3]["C2"]), "output_fnv1a64_count": len(long[4]["C2"]),
        "non_finite_count": sum(int(row["non_finite_count"]) for row in long[6]),
        "score_collapse_segments": sum(row["score_collapse"] == "yes" for row in long[6]),
        "status": "pass" if long_pass else "fail",
    })
    write_tsv(options.tracked_root / "output_hash_stability.tsv", hash_rows)
    (options.tracked_root / "stability_decision.md").write_text(
        "# Stage65D-R1 stability decision\n\n"
        f"Decision: `{'pass' if passed else 'fail'}`. The short surface contains reversed-order B2/C2 1k arms (2k runs per model total). C2 long soak uses ten clean process/session segments of 1000 runs. C2 second-half/first-half median ratio is `{drift:.9f}` (accepted range 0.95..1.05); thermal/frequency state is `{'pass' if thermal_pass else 'fail'}`.\n",
        encoding="utf-8",
    )
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
