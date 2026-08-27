#!/usr/bin/env python3
"""Normalize symmetric Stage65E short and 10k stability surfaces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


RESULT_HASH_RE = re.compile(r"stage64_result .*output_fnv1a64=(?P<hash>0x[0-9a-fA-F]+)")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stats(values: list[float]) -> dict[str, object]:
    if not values:
        raise ValueError("statistics require at least one value")
    mean = statistics.fmean(values)
    standard = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "samples": len(values),
        "mean_us": mean,
        "stddev_us": standard,
        "cv_pct": 0.0 if mean == 0 else 100.0 * standard / mean,
        "p50_us": statistics.median(values),
        "p95_us": percentile(values, 0.95),
        "p99_us": percentile(values, 0.99),
        "p999_us": percentile(values, 0.999),
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


def collect(root: Path) -> dict[str, object]:
    status = read_tsv(root / "status.raw.tsv")
    windows: list[dict[str, object]] = []
    resources: list[dict[str, object]] = []
    aggregate: dict[str, list[float]] = defaultdict(list)
    output_hashes: dict[str, set[str]] = defaultdict(set)
    fnv_hashes: dict[str, set[str]] = defaultdict(set)
    result_fnv_hashes: dict[str, set[str]] = defaultdict(set)
    sample_hash_modes: set[str] = set()
    thermal: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    identities: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()

    for status_row in status:
        model = status_row["model"]
        key = (status_row["segment"], status_row["position"], model)
        if key in seen:
            raise RuntimeError(f"duplicate soak segment: {key}")
        seen.add(key)
        directory = root / f"segment-{status_row['segment']}-{status_row['position']}-{model}"
        if int(status_row["exit_code"]) != 0:
            raise RuntimeError(f"failed soak segment: {directory}")
        for field, filename in (
            ("output_sha256", "output.bin"),
            ("samples_sha256", "samples.tsv"),
            ("resource_sha256", "resource.tsv"),
        ):
            if sha256(directory / filename) != status_row[field]:
                raise RuntimeError(f"{field} mismatch: {directory}")
        identities.append({
            "root": root.name,
            "segment": status_row["segment"],
            "position": status_row["position"],
            "model": model,
            "runs": status_row["runs"],
            "output_sha256": status_row["output_sha256"],
            "samples_sha256": status_row["samples_sha256"],
            "resource_sha256": status_row["resource_sha256"],
            "exit_code": status_row["exit_code"],
            "status": "pass",
        })
        output_hashes[model].add(status_row["output_sha256"])
        sample_rows = read_tsv(directory / "samples.tsv")
        expected_runs = int(status_row["runs"])
        if len(sample_rows) != expected_runs:
            raise RuntimeError(f"sample count mismatch: {directory}")
        values = {"inference": [], "tail": [], "two_stage": []}
        sample_fields = set(sample_rows[0])
        required_sample_fields = {"repeat", "run", "inference_us", "tail_us", "total_us"}
        if not required_sample_fields.issubset(sample_fields):
            raise RuntimeError(f"timing sample contract drift: {directory}")
        has_per_run_hash = "output_fnv1a64" in sample_fields
        sample_hash_modes.add("per-run-fnv1a64" if has_per_run_hash else "not-emitted-by-bound-runner")
        for row in sample_rows:
            values["inference"].append(float(row["inference_us"]))
            values["tail"].append(float(row["tail_us"]))
            values["two_stage"].append(float(row["total_us"]))
            if has_per_run_hash:
                fnv_hashes[model].add(row["output_fnv1a64"])
        log = (directory / "run.log").read_text(encoding="utf-8", errors="replace")
        result_hash = RESULT_HASH_RE.search(log)
        if "stage64_result status=pass" not in log or not result_hash:
            raise RuntimeError(f"missing runner pass marker: {directory}")
        result_fnv_hashes[model].add(result_hash.group("hash").lower())
        output = np.fromfile(directory / "output.bin", dtype="<f4")
        finite = int(np.isfinite(output).sum())
        scores = output.reshape(300, 6)[:, 4] if output.size == 1800 else np.asarray([], dtype=np.float32)
        collapsed = scores.size == 0 or not np.isfinite(scores).all() or np.unique(scores).size < 2
        diagnostics.append({
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
                "root": root.name,
                "segment": status_row["segment"],
                "position": status_row["position"],
                "model": model,
                "metric": metric,
                **stats(metric_values),
            })
        resource_rows = [
            row
            for row in read_tsv(directory / "resource.tsv")
            if int(row["rss_kib"]) > 0
            and int(row["threads"]) > 0
            and int(row["fds"]) > 0
        ]
        if not resource_rows:
            raise RuntimeError(f"missing resource samples: {directory}")
        steady_candidates = [
            row for row in resource_rows if int(row["sample"]) >= 30
        ]
        if len(steady_candidates) < 2:
            raise RuntimeError(f"missing post-initialization resource samples: {directory}")
        steady_resource_rows = steady_candidates[:-1]
        numeric = {
            name: [int(row[name]) for row in steady_resource_rows]
            for name in (
                "rss_kib",
                "peak_rss_kib",
                "fds",
                "threads",
                "voluntary_ctxt",
                "nonvoluntary_ctxt",
            )
        }
        resources.append({
            "root": root.name,
            "segment": status_row["segment"],
            "position": status_row["position"],
            "model": model,
            "samples_total": len(resource_rows),
            "steady_samples": len(steady_resource_rows),
            "terminal_samples_excluded": 1,
            "rss_startup_first_kib": int(resource_rows[0]["rss_kib"]),
            "rss_first_kib": numeric["rss_kib"][0],
            "rss_last_kib": numeric["rss_kib"][-1],
            "rss_max_kib": max(numeric["rss_kib"]),
            "peak_rss_max_kib": max(numeric["peak_rss_kib"]),
            "rss_slope_kib_per_sample": (
                numeric["rss_kib"][-1] - numeric["rss_kib"][0]
            ) / max(1, len(steady_resource_rows) - 1),
            "fds_min": min(numeric["fds"]),
            "fds_max": max(numeric["fds"]),
            "threads_min": min(numeric["threads"]),
            "threads_max": max(numeric["threads"]),
            "voluntary_ctxt_delta": numeric["voluntary_ctxt"][-1] - numeric["voluntary_ctxt"][0],
            "nonvoluntary_ctxt_delta": numeric["nonvoluntary_ctxt"][-1] - numeric["nonvoluntary_ctxt"][0],
            "exit_code": status_row["exit_code"],
        })
        for moment in ("before", "after"):
            for line in (directory / f"state-{moment}.tsv").read_text().splitlines():
                fields = line.split("\t")
                if len(fields) == 3:
                    thermal.append({
                        "root": root.name,
                        "segment": status_row["segment"],
                        "position": status_row["position"],
                        "model": model,
                        "moment": moment,
                        "kind": fields[0],
                        "source": fields[1],
                        "value": fields[2],
                    })
    if len(sample_hash_modes) != 1:
        raise RuntimeError(f"mixed per-run output-hash contracts in {root}: {sample_hash_modes}")
    return {
        "windows": windows,
        "resources": resources,
        "aggregate": aggregate,
        "output_hashes": output_hashes,
        "fnv_hashes": fnv_hashes,
        "result_fnv_hashes": result_fnv_hashes,
        "sample_hash_mode": next(iter(sample_hash_modes)),
        "thermal": thermal,
        "diagnostics": diagnostics,
        "identities": identities,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--short-root", required=True, type=Path)
    parser.add_argument("--b2-long-root", required=True, type=Path)
    parser.add_argument("--c2-long-root", required=True, type=Path)
    parser.add_argument("--tracked-root", required=True, type=Path)
    options = parser.parse_args()

    short = collect(options.short_root)
    b2_long = collect(options.b2_long_root)
    c2_long = collect(options.c2_long_root)
    collections = (short, b2_long, c2_long)
    sample_hash_modes = {str(collected["sample_hash_mode"]) for collected in collections}
    if len(sample_hash_modes) != 1:
        raise RuntimeError(f"mixed soak output-hash contracts: {sample_hash_modes}")
    per_run_hashes = sample_hash_modes == {"per-run-fnv1a64"}
    fixtures = read_tsv(options.tracked_root / "bounded_fixture_recheck.tsv")
    accepted_output_hash = {
        "B2": next(row["output_sha256"] for row in fixtures if row["surface"] == "B2_spacemit_fixture"),
        "C2": next(row["output_sha256"] for row in fixtures if row["surface"] == "C2_spacemit_fixture"),
    }

    short_rows: list[dict[str, object]] = []
    for model in ("B2", "C2"):
        for metric in ("inference", "tail", "two_stage"):
            short_rows.append({"model": model, "metric": metric, **stats(short["aggregate"][f"{model}:{metric}"])})
    write_tsv(options.tracked_root / "short_soak.tsv", short_rows)

    long_summaries: dict[str, list[dict[str, object]]] = {}
    for model, collected in (("B2", b2_long), ("C2", c2_long)):
        rows = [
            {"model": model, "metric": metric, **stats(collected["aggregate"][f"{model}:{metric}"])}
            for metric in ("inference", "tail", "two_stage")
        ]
        long_summaries[model] = rows
        write_tsv(options.tracked_root / f"{model.lower()}_10k_soak.tsv", rows)

    passed = True
    resources: list[dict[str, object]] = []
    for collected in collections:
        for row in collected["resources"]:
            row_pass = (
                int(row["exit_code"]) == 0
                and int(row["fds_max"]) - int(row["fds_min"]) <= 2
                and int(row["threads_max"]) - int(row["threads_min"]) <= 2
                and int(row["rss_last_kib"]) <= int(row["rss_first_kib"]) + 16384
            )
            row["status"] = "pass" if row_pass else "fail"
            passed = passed and row_pass
            resources.append(row)
    write_tsv(options.tracked_root / "resource_drift.tsv", resources)

    diagnostics = [row for collected in collections for row in collected["diagnostics"]]
    write_tsv(options.tracked_root / "soak_output_semantics.tsv", diagnostics)
    passed = passed and all(row["status"] == "pass" for row in diagnostics)

    identities = [row for collected in collections for row in collected["identities"]]
    write_tsv(options.tracked_root / "soak_segment_identity.tsv", identities)

    thermal = [row for collected in collections for row in collected["thermal"]]
    write_tsv(options.tracked_root / "thermal_frequency_soak.tsv", thermal)
    governors = {str(row["value"]) for row in thermal if row["kind"] == "governor"}
    frequencies = [float(row["value"]) for row in thermal if row["kind"] == "frequency_khz"]
    temperatures = [float(row["value"]) for row in thermal if row["kind"] == "temperature_millic"]
    thermal_pass = governors == {"performance"}
    thermal_pass = thermal_pass and bool(frequencies) and min(frequencies) >= 0.95 * max(frequencies)
    thermal_pass = thermal_pass and bool(temperatures) and max(temperatures) <= 85000
    passed = passed and thermal_pass

    hash_rows: list[dict[str, object]] = []
    drift_by_model: dict[str, float] = {}
    for model, long_collection in (("B2", b2_long), ("C2", c2_long)):
        short_runs = len(short["aggregate"][f"{model}:two_stage"])
        short_pass = (
            short_runs == 2000
            and short["output_hashes"][model] == {accepted_output_hash[model]}
            and (not per_run_hashes or len(short["fnv_hashes"][model]) == 1)
            and len(short["result_fnv_hashes"][model]) == 1
            and all(row["status"] == "pass" for row in short["diagnostics"] if row["model"] == model)
        )
        hash_rows.append({
            "surface": f"short-order-balanced-{model}",
            "runs": short_runs,
            "output_sha256_count": len(short["output_hashes"][model]),
            "accepted_fixture_sha256": accepted_output_hash[model],
            "equal_to_accepted_fixture": "yes" if short["output_hashes"][model] == {accepted_output_hash[model]} else "no",
            "output_fnv1a64_count": len(short["fnv_hashes"][model]) if per_run_hashes else "not-emitted-by-bound-runner",
            "result_fnv1a64_count": len(short["result_fnv_hashes"][model]),
            "status": "pass" if short_pass else "fail",
        })
        windows = [
            row for row in long_collection["windows"]
            if row["model"] == model and row["metric"] == "two_stage"
        ]
        first = statistics.median(float(row["p50_us"]) for row in windows[:5])
        second = statistics.median(float(row["p50_us"]) for row in windows[5:])
        drift = second / first
        drift_by_model[model] = drift
        long_runs = len(long_collection["aggregate"][f"{model}:two_stage"])
        combined_output_hashes = set(short["output_hashes"][model]) | set(long_collection["output_hashes"][model])
        combined_fnv_hashes = set(short["fnv_hashes"][model]) | set(long_collection["fnv_hashes"][model])
        combined_result_fnv_hashes = set(short["result_fnv_hashes"][model]) | set(long_collection["result_fnv_hashes"][model])
        long_pass = (
            len(windows) == 10
            and long_runs == 10000
            and combined_output_hashes == {accepted_output_hash[model]}
            and (not per_run_hashes or len(combined_fnv_hashes) == 1)
            and len(combined_result_fnv_hashes) == 1
            and 0.95 <= drift <= 1.05
            and all(row["status"] == "pass" for row in long_collection["diagnostics"])
        )
        hash_rows.append({
            "surface": f"{model.lower()}-10x1000-clean-session",
            "runs": long_runs,
            "output_sha256_count": len(combined_output_hashes),
            "accepted_fixture_sha256": accepted_output_hash[model],
            "equal_to_accepted_fixture": "yes" if combined_output_hashes == {accepted_output_hash[model]} else "no",
            "output_fnv1a64_count": len(combined_fnv_hashes) if per_run_hashes else "not-emitted-by-bound-runner",
            "result_fnv1a64_count": len(combined_result_fnv_hashes),
            "status": "pass" if long_pass else "fail",
        })
        passed = passed and short_pass and long_pass
    write_tsv(options.tracked_root / "output_hash_stability.tsv", hash_rows)

    (options.tracked_root / "stability_decision.md").write_text(
        "# Stage65E stability decision\n\n"
        f"Decision: `{'pass' if passed else 'fail'}`. The short surface contains reversed-order "
        "B2/C2 1k arms (2k runs per model). Both long surfaces contain ten clean process/session "
        "segments of 1000 runs. Second-half/first-half median ratios are "
        f"B2 `{drift_by_model['B2']:.9f}` and C2 `{drift_by_model['C2']:.9f}` "
        "(accepted range 0.95..1.05). "
        f"Thermal/frequency state: `{'pass' if thermal_pass else 'fail'}`. The first 30 valid "
        "one-second resource samples remain in raw evidence as the expected process/session "
        "initialization ramp; RSS/FD/thread drift gates use the post-initialization window and "
        "exclude the final process-teardown observation.\n"
        f"The bound runner per-run sample fingerprint contract is `{'emitted-and-stable' if per_run_hashes else 'not-emitted'}`; the emitted final-result FNV must remain stable, and exact final output SHA-256 is required across all independent short and long process/session segments and must equal the accepted bounded SpaceMIT fixture.\n",
        encoding="utf-8",
    )
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
