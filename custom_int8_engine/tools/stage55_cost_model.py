#!/usr/bin/env python3
"""Build the measured Stage55 shape lattice and Cost Model V3 evidence."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def selected_route(row: dict[str, str]) -> str:
    if row["kind"] == "conv_grouped":
        return "depthwise-v2-c8"
    n = int(row["n"])
    m = int(row["m"])
    input_c = int(row["input_c"])
    kernel = int(row["kernel"].split("x", 1)[0])
    stride = int(row["stride"].split("x", 1)[0])
    if n == 4:
        return "m12n4"
    if n == 8:
        return "m12n8"
    tail = "n16" if n % 16 == 0 else "n16-live-tail"
    if kernel == 1:
        family = "family-a-weight-stationary" if (
            input_c <= 96 and m >= 6400 and n % 16 == 0
        ) else "direct-strided"
        return f"{family}-{tail}"
    delivery = "p3-segmented" if stride == 2 else "packed-p1"
    return f"{delivery}-{tail}"


def feature_vector(row: dict[str, str], routes: list[str]) -> np.ndarray:
    m = float(row["m"])
    n = float(row["n"])
    k = float(row["k"])
    side = int(row["output_h"])
    kernel = int(row["kernel"].split("x", 1)[0])
    stride = int(row["stride"].split("x", 1)[0])
    working_set = float(row["working_set_bytes"])
    packed = float(row["packed_weight_bytes"])
    log_m, log_n, log_k = math.log(m), math.log(n), math.log(k)
    power_of_two_side = float(side > 0 and side & (side - 1) == 0)
    values = [
        1.0,
        log_m,
        log_n,
        log_k,
        math.log(m * n * k),
        math.log(working_set + 1.0),
        math.log(packed + 1.0),
        float(kernel == 1),
        float(kernel == 3),
        float(stride == 2),
        power_of_two_side,
        power_of_two_side * log_k,
        power_of_two_side * log_n,
        power_of_two_side * log_m,
        float(side % 16 == 0),
        float(side % 32 == 0),
        float(side % 64 == 0),
        float(int(n) % 16 != 0),
        float(int(m) % 12) / 12.0,
    ]
    for route in routes:
        values.append(float(row["selected_route"] == route))
    for route in routes:
        active = float(row["selected_route"] == route)
        values.extend(
            [active * log_m, active * log_n, active * log_k,
             active * power_of_two_side, active * power_of_two_side * log_k]
        )
    return np.asarray(values, dtype=np.float64)


def parse_profile(path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    operations: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("stage53_op\t"):
            fields = line.split("\t", 7)
            if len(fields) != 8:
                raise ValueError(f"malformed operation profile row: {line}")
            operations.append({
                "run": int(fields[1]),
                "operation_index": int(fields[2]),
                "resident_operation_index": int(fields[3]),
                "kind": fields[4],
                "scope": fields[5],
                "wall_us": float(fields[6]),
                "name": fields[7],
            })
        elif line.startswith("stage53_profile_run\t"):
            fields = line.split("\t")
            runs.append({
                "run": int(fields[1]),
                "outer_wall_us": float(fields[2]),
                "operation_sum_us": float(fields[3]),
                "profiled_ranges": int(fields[4]),
            })
    if len(runs) < 50:
        raise ValueError(f"expected at least 50 profiled runs, found {len(runs)}")
    measured_runs = runs[-50:]
    measured_ids = {int(row["run"]) for row in measured_runs}
    return [row for row in operations if int(row["run"]) in measured_ids], measured_runs


def summarize_profile(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["operation_index"], row["resident_operation_index"], row["kind"],
               row["scope"], row["name"])
        grouped[key].append(float(row["wall_us"]))
    output: list[dict[str, object]] = []
    for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        output.append({
            "operation_index": key[0],
            "resident_operation_index": key[1],
            "kind": key[2],
            "scope": key[3],
            "name": key[4],
            "samples": len(values),
            "mean_us": statistics.fmean(values),
            "median_us": statistics.median(values),
            "p95_us": percentile(values, 0.95),
            "implementation": "stage55-selected-explicit-or-measured",
            "confidence": "50 complete profiled observations",
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lattice", type=Path, required=True)
    parser.add_argument("--profile-log", type=Path, required=True)
    parser.add_argument("--candidate-freeze", type=Path, required=True)
    parser.add_argument("--candidate-actual", action="append", default=[])
    parser.add_argument("--headline-mean-us", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    lattice = read_tsv(args.lattice)
    for row in lattice:
        row["selected_route"] = selected_route(row)
        row["scheduler"] = "SCHED_OTHER-raw-epoch-spin"
        row["workers"] = "CPU0-3"
        row["tails"] = f"M%12={int(row['m']) % 12};N%16={int(row['n']) % 16}"
    routes = sorted({row["selected_route"] for row in lattice})

    design = [{
        "resolution": row["resolution"],
        "shape_class": row["shape_class"],
        "operation_index": row["operation_index"],
        "kind": row["kind"],
        "m": row["m"],
        "n": row["n"],
        "k": row["k"],
        "kernel": row["kernel"],
        "stride": row["stride"],
        "selected_route": row["selected_route"],
        "design_role": "bounded-graph-backed-synthetic-exact-shape",
    } for row in lattice]
    write_tsv(args.output_dir / "shape_lattice_design.tsv", design)
    write_tsv(args.output_dir / "shape_lattice_measurements.tsv", lattice)

    regular = [row for row in lattice if row["shape_class"] != "dense_1x1_n_tail_k384"]
    training_rows: list[dict[str, object]] = []
    holdout_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    errors: list[float] = []
    ridge = 0.001
    for resolution in (384, 416, 448, 480, 512, 640):
        train = [row for row in regular if int(row["resolution"]) != resolution]
        holdout = [row for row in regular if int(row["resolution"]) == resolution]
        x_train = np.asarray([feature_vector(row, routes) for row in train])
        y_train = np.log(np.asarray([float(row["mean_us"]) for row in train]))
        mean = x_train[:, 1:].mean(axis=0)
        deviation = x_train[:, 1:].std(axis=0)
        deviation[deviation < 1e-8] = 1.0
        normalized = np.column_stack(
            [np.ones(len(x_train)), (x_train[:, 1:] - mean) / deviation]
        )
        penalty = ridge * np.eye(normalized.shape[1])
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            normalized.T @ normalized + penalty, normalized.T @ y_train
        )
        for row in train:
            training_rows.append({
                "fold": f"hold-resolution-{resolution}",
                "role": "training",
                **{field: row[field] for field in (
                    "resolution", "shape_class", "operation_index", "kind", "m", "n", "k",
                    "kernel", "stride", "selected_route", "mean_us")},
            })
        for row in holdout:
            vector = feature_vector(row, routes)
            normalized_vector = np.concatenate(
                ([1.0], (vector[1:] - mean) / deviation)
            )
            prediction = float(math.exp(normalized_vector @ coefficients))
            measured = float(row["mean_us"])
            error = abs(prediction - measured) / measured * 100.0
            item = {
                "fold": f"hold-resolution-{resolution}",
                "role": "held-out-exact-shape",
                **{field: row[field] for field in (
                    "resolution", "shape_class", "operation_index", "kind", "m", "n", "k",
                    "kernel", "stride", "selected_route")},
                "measured_us": measured,
                "predicted_us": prediction,
                "absolute_percentage_error_pct": error,
                "high_uncertainty": "yes" if error > 25.0 else "no",
            }
            holdout_rows.append(item)
            validation_rows.append(item)
            errors.append(error)
    for row in lattice:
        if row["shape_class"] == "dense_1x1_n_tail_k384":
            training_rows.append({
                "fold": "channel-tail-direct-measurement",
                "role": "training-only-single-resolution-tail",
                **{field: row[field] for field in (
                    "resolution", "shape_class", "operation_index", "kind", "m", "n", "k",
                    "kernel", "stride", "selected_route", "mean_us")},
            })

    aggregate = {
        "fold": "aggregate",
        "role": "six-fold-resolution-grouped-summary",
        "resolution": "384/416/448/480/512/640",
        "shape_class": f"{len({row['shape_class'] for row in regular})} measured classes",
        "operation_index": "all",
        "kind": "dense-and-depthwise",
        "m": "varied",
        "n": "varied",
        "k": "varied",
        "kernel": "1x1/3x3",
        "stride": "1/2",
        "selected_route": "prepared-route-aware-ridge",
        "measured_us": f"{len(errors)} held-out observations",
        "predicted_us": "not-a-tail-sum",
        "absolute_percentage_error_pct": "not-a-tail-model",
        "median_absolute_percentage_error_pct": percentile(errors, 0.5),
        "mean_absolute_percentage_error_pct": statistics.fmean(errors),
        "p90_absolute_percentage_error_pct": percentile(errors, 0.9),
        "worst_absolute_percentage_error_pct": max(errors),
        "high_uncertainty": "directly-measure-power-of-two/cache-conflict-classes",
    }
    validation_rows.append(aggregate)
    write_tsv(args.output_dir / "shape_model_v3_training.tsv", training_rows)
    write_tsv(args.output_dir / "shape_model_v3_holdout.tsv", holdout_rows)
    write_tsv(args.output_dir / "shape_model_v3_validation.tsv", validation_rows)

    profile_rows, profile_runs = parse_profile(args.profile_log)
    profile_summary = summarize_profile(profile_rows)
    write_tsv(args.output_dir / "measured_latency_lut_v3.tsv", profile_summary)
    nonmac = [row for row in profile_summary
              if row["kind"] not in {"conv_dense", "conv_grouped", "matmul"}]
    write_tsv(args.output_dir / "measured_nonmac_lut_v3.tsv", nonmac)

    outer_mean = statistics.fmean(float(row["outer_wall_us"]) for row in profile_runs)
    operation_sum = statistics.fmean(float(row["operation_sum_us"]) for row in profile_runs)
    decomposition_error = (operation_sum - outer_mean) / outer_mean * 100.0
    headline_error = (outer_mean - args.headline_mean_us) / args.headline_mean_us * 100.0

    actual = {}
    for specification in args.candidate_actual:
        name, value = specification.split("=", 1)
        actual[name] = float(value)
    freeze_rows = read_tsv(args.candidate_freeze)
    comparison_rows: list[dict[str, object]] = []
    candidate_errors: list[float] = []
    for row in freeze_rows:
        name = row["candidate"]
        measured = actual.get(name)
        predicted = float(row["predicted_full_mean_us"])
        error = "unmeasured"
        status = "not-selected-or-no-complete-measurement"
        if measured is not None:
            numeric_error = (predicted - measured) / measured * 100.0
            candidate_errors.append(abs(numeric_error))
            error = numeric_error
            status = "pass" if abs(numeric_error) <= 15.0 else "fail"
        comparison_rows.append({
            **row,
            "measured_full_mean_us": measured if measured is not None else "not-selected",
            "prediction_error_pct": error,
            "hard_gate_pct": 15.0,
            "status": status,
        })
    write_tsv(args.output_dir / "candidate_prediction_freeze.tsv", freeze_rows)
    write_tsv(args.output_dir / "candidate_prediction_vs_measurement.tsv", comparison_rows)

    maximum_candidate_error = max(candidate_errors) if candidate_errors else math.inf
    model_rows = [
        {
            "surface": "current_graph_profile_decomposition",
            "predicted_mean_us": operation_sum,
            "measured_mean_us": outer_mean,
            "error_pct": decomposition_error,
            "gate_pct": 3.0,
            "status": "pass" if abs(decomposition_error) <= 3.0 else "fail",
        },
        {
            "surface": "profile_to_uninstrumented_headline",
            "predicted_mean_us": outer_mean,
            "measured_mean_us": args.headline_mean_us,
            "error_pct": headline_error,
            "gate_pct": 15.0,
            "status": "pass" if abs(headline_error) <= 15.0 else "fail",
        },
        {
            "surface": "candidate_pre_measurement_composition",
            "predicted_mean_us": "frozen-per-candidate",
            "measured_mean_us": f"{len(candidate_errors)} complete A/B surfaces",
            "error_pct": maximum_candidate_error,
            "gate_pct": 15.0,
            "status": "pass" if maximum_candidate_error <= 15.0 else "fail",
        },
        {
            "surface": "held_out_exact_shape_lattice",
            "predicted_mean_us": "not-additive-tail-prediction",
            "measured_mean_us": f"{len(errors)} held-out rows",
            "error_pct": percentile(errors, 0.5),
            "p90_error_pct": percentile(errors, 0.9),
            "worst_error_pct": max(errors),
            "gate_pct": 10.0,
            "status": "median-pass; p90-direct-measurement-required" if
                percentile(errors, 0.5) <= 10.0 else "fail",
        },
    ]
    write_tsv(args.output_dir / "full_graph_cost_model_v3.tsv", model_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
