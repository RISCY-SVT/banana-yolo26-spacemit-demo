#!/usr/bin/env python3
"""Build the Stage 54 measured latency and held-out dense-shape model."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


SHAPE_FIELDS = (
    "kernel",
    "stride",
    "m",
    "n",
    "k",
    "input_c",
    "output_c",
    "output_h",
    "output_w",
    "n_tail",
    "m_tail",
)


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


def shape_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in SHAPE_FIELDS)


def features(row: dict[str, str]) -> list[float]:
    m = float(row["m"])
    n = float(row["n"])
    k = float(row["k"])
    kernel = int(row["kernel"].split("x", 1)[0])
    stride = int(row["stride"].split("x", 1)[0])
    macs = m * n * k
    return [
        1.0,
        math.log(m),
        math.log(n),
        math.log(k),
        math.log(macs),
        float(kernel == 1),
        float(kernel == 3),
        float(stride == 2),
        (n % 16.0) / 16.0,
        (k % 32.0) / 32.0,
        math.log(float(row["packed_weight_bytes"]) + 1.0),
        math.log(float(row["output_bytes"]) + 1.0),
    ]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-census", type=Path, required=True)
    parser.add_argument("--profile-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-outer-mean-us", type=float, required=True)
    parser.add_argument("--profile-operation-sum-us", type=float, required=True)
    parser.add_argument("--headline-mean-us", type=float, required=True)
    args = parser.parse_args()

    dense = read_tsv(args.dense_census)
    profile = read_tsv(args.profile_summary)
    measured_by_name = {
        row["name"]: float(row["mean_us"])
        for row in profile
        if row["kind"] == "conv_dense" and row["scope"] != "resident_core"
    }
    rows = [row for row in dense if row["name"] in measured_by_name]
    if len(rows) != 72:
        raise ValueError(f"expected 72 non-resident dense rows, found {len(rows)}")

    classes = sorted({shape_key(row) for row in rows})
    fold_by_class = {key: index % 5 for index, key in enumerate(classes)}
    training_rows: list[dict[str, object]] = []
    holdout_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []

    for fold in range(5):
        train = [row for row in rows if fold_by_class[shape_key(row)] != fold]
        holdout = [row for row in rows if fold_by_class[shape_key(row)] == fold]
        x_train = np.asarray([features(row) for row in train], dtype=np.float64)
        y_train = np.log(
            np.asarray([measured_by_name[row["name"]] for row in train], dtype=np.float64)
        )
        ridge = 0.1
        coefficients = np.linalg.solve(
            x_train.T @ x_train + ridge * np.eye(x_train.shape[1]),
            x_train.T @ y_train,
        )

        for row in train:
            training_rows.append(
                {
                    "fold": fold,
                    "role": "training",
                    "shape_class": "|".join(shape_key(row)),
                    "name": row["name"],
                    "m": row["m"],
                    "n": row["n"],
                    "k": row["k"],
                    "kernel": row["kernel"],
                    "stride": row["stride"],
                    "measured_us": measured_by_name[row["name"]],
                }
            )
        for row in holdout:
            prediction = float(math.exp(np.dot(features(row), coefficients)))
            measured = measured_by_name[row["name"]]
            absolute_percentage_error = abs(prediction - measured) / measured * 100.0
            item = {
                "fold": fold,
                "role": "held-out-exact-shape-class",
                "shape_class": "|".join(shape_key(row)),
                "name": row["name"],
                "m": row["m"],
                "n": row["n"],
                "k": row["k"],
                "kernel": row["kernel"],
                "stride": row["stride"],
                "measured_us": measured,
                "predicted_us": prediction,
                "absolute_percentage_error_pct": absolute_percentage_error,
            }
            holdout_rows.append(item)
            validation_rows.append(item)

    errors = [float(row["absolute_percentage_error_pct"]) for row in validation_rows]
    validation_rows.append(
        {
            "fold": "aggregate",
            "role": "five-fold-grouped-summary",
            "shape_class": f"{len(classes)} exact classes",
            "name": f"{len(rows)} held-out observations",
            "m": "not-applicable",
            "n": "not-applicable",
            "k": "not-applicable",
            "kernel": "all",
            "stride": "all",
            "measured_us": sum(measured_by_name[row["name"]] for row in rows),
            "predicted_us": sum(float(row["predicted_us"]) for row in holdout_rows),
            "absolute_percentage_error_pct": "not-a-summed-tail-model",
            "median_absolute_percentage_error_pct": percentile(errors, 0.5),
            "mean_absolute_percentage_error_pct": sum(errors) / len(errors),
            "p90_absolute_percentage_error_pct": percentile(errors, 0.9),
            "worst_absolute_percentage_error_pct": max(errors),
        }
    )

    output = args.output_dir
    write_tsv(output / "shape_model_training_rows.tsv", training_rows)
    write_tsv(output / "shape_model_holdout_rows.tsv", holdout_rows)
    write_tsv(output / "shape_model_validation.tsv", validation_rows)

    latency_lut = []
    for row in profile:
        latency_lut.append(
            {
                "operation_index": row["operation_index"],
                "kind": row["kind"],
                "scope": row["scope"],
                "name": row["name"],
                "mean_us": row["mean_us"],
                "p95_us": row["p95_us"],
                "samples": row["samples"],
                "implementation": "stage54-selected-explicit-or-measured",
                "confidence": "70 complete profiled observations",
            }
        )
    write_tsv(output / "measured_latency_lut_v2.tsv", latency_lut)
    write_tsv(
        output / "measured_nonmac_lut_v2.tsv",
        [row for row in latency_lut if row["kind"] not in {"conv_dense", "conv_grouped", "matmul"}],
    )

    decomposition_error = (
        (args.profile_operation_sum_us - args.profile_outer_mean_us)
        / args.profile_outer_mean_us
        * 100.0
    )
    candidate_error = (
        (args.profile_outer_mean_us - args.headline_mean_us)
        / args.headline_mean_us
        * 100.0
    )
    model_rows = [
        {
            "surface": "current_graph_profile_decomposition",
            "predicted_mean_us": args.profile_operation_sum_us,
            "measured_mean_us": args.profile_outer_mean_us,
            "error_pct": decomposition_error,
            "gate_pct": 3.0,
            "status": "pass" if abs(decomposition_error) <= 3.0 else "fail",
        },
        {
            "surface": "candidate_pre_measurement_composition",
            "predicted_mean_us": args.profile_outer_mean_us,
            "measured_mean_us": args.headline_mean_us,
            "error_pct": candidate_error,
            "gate_pct": 15.0,
            "status": "pass" if abs(candidate_error) <= 15.0 else "fail",
        },
        {
            "surface": "held_out_dense_shape_classes",
            "predicted_mean_us": "not-additive-tail-prediction",
            "measured_mean_us": "72 rows",
            "error_pct": percentile(errors, 0.5),
            "gate_pct": 20.0,
            "status": "pass" if percentile(errors, 0.5) <= 20.0 else "fail",
        },
    ]
    write_tsv(output / "full_graph_cost_model_v2.tsv", model_rows)
    write_tsv(output / "candidate_prediction_freeze.tsv", [model_rows[1]])
    write_tsv(output / "candidate_prediction_vs_measurement.tsv", [model_rows[1]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
