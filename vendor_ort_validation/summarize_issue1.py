#!/usr/bin/env python3
"""Build corrected Stage63 issue #1 tables from raw process evidence."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--oracles", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--full-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = read_tsv(args.raw)
    oracle_rows = read_tsv(args.oracles)
    oracle_by_test = {row["test_id"]: row["oracle_sha256"] for row in oracle_rows}

    correctness: list[dict[str, object]] = []
    exits: list[dict[str, object]] = []
    for row in raw:
        expected = oracle_by_test[row["test_id"]]
        exact = row["exit_code"] == "0" and row["output_sha256"] == expected
        placement = (
            "CPUExecutionProvider"
            if row["provider"] == "cpu"
            else "SpacemiT EP appended and subgraph observed"
            if row["spacemit_appended"] == "1"
            else "not proven before process failure"
        )
        correctness.append(
            {
                "runtime": row["runtime"],
                "provider": row["provider"],
                "test_id": row["test_id"],
                "model_sha256": row["model_sha256"],
                "input_sha256": row["input_sha256"],
                "output_sha256": row["output_sha256"],
                "independent_oracle_sha256": expected,
                "exact": int(exact),
                "provider_placement": placement,
                "corrected_error_class": "none" if exact else row["error_class"],
                "raw_error_class": row["error_class"],
            }
        )
        exits.append(
            {
                "runtime": row["runtime"],
                "provider": row["provider"],
                "test_id": row["test_id"],
                "exit_code": row["exit_code"],
                "signal": row["signal"],
                "timed_out": row["timed_out"],
                "session_created": row["session_created"],
                "result_marker": row["result_marker"],
                "error_class": row["error_class"],
            }
        )

    write_tsv(
        args.output_dir / "tiny_control_correctness.tsv",
        [
            "runtime",
            "provider",
            "test_id",
            "model_sha256",
            "input_sha256",
            "output_sha256",
            "independent_oracle_sha256",
            "exact",
            "provider_placement",
            "corrected_error_class",
            "raw_error_class",
        ],
        correctness,
    )
    write_tsv(
        args.output_dir / "tiny_control_exit_signals.tsv",
        [
            "runtime",
            "provider",
            "test_id",
            "exit_code",
            "signal",
            "timed_out",
            "session_created",
            "result_marker",
            "error_class",
        ],
        exits,
    )

    by_key = {(row["runtime"], row["provider"], row["test_id"]): row for row in correctness}

    def result(runtime: str, test_id: str) -> str:
        row = by_key[(runtime, "spacemit", test_id)]
        if row["exact"] == 1:
            return "EP assigned; exact"
        error = str(row["corrected_error_class"])
        if error == "clip_minmax_compile_error":
            return "catchable clip-minmax compile error"
        if error == "abort":
            return "clip-minmax error then abort" if test_id == "A1" else "abort"
        if error == "illegal_instruction":
            return "SIGILL / exit 132"
        return error

    plugin = {row["arm"]: row for row in read_tsv(args.plugin)}
    full = read_tsv(args.full_model)
    full_int8_206 = next(
        row
        for row in full
        if row["runtime"] == "rt206"
        and row["provider"] == "spacemit"
        and row["model_surface"] == "int8"
        and row["opt_level"] == "all"
    )

    decisions = [
        {
            "issue_item": "Q/DQ Conv no kernel_shape",
            "rt204": result("rt204", "A0"),
            "rt205": result("rt205", "A0"),
            "rt206": result("rt206", "A0"),
            "rt206_classification": "unchanged",
            "basis": "positive control remains assigned and exact against independent Q/DQ Conv oracle in all three versions",
        },
        {
            "issue_item": "Q/DQ Conv kernel_shape=[3,3]",
            "rt204": result("rt204", "A1"),
            "rt205": result("rt205", "A1"),
            "rt206": result("rt206", "A1"),
            "rt206_classification": "unchanged",
            "basis": "2.0.6 emits the same clip-minmax compiler message and aborts",
        },
        {
            "issue_item": "QLinearConv",
            "rt204": result("rt204", "B"),
            "rt205": result("rt205", "B"),
            "rt206": result("rt206", "B"),
            "rt206_classification": "unchanged",
            "basis": "2.0.6 still terminates with SIGILL before output",
        },
        {
            "issue_item": "QLinearMatMul",
            "rt204": result("rt204", "C"),
            "rt205": result("rt205", "C"),
            "rt206": result("rt206", "C"),
            "rt206_classification": "unchanged",
            "basis": "2.0.6 still terminates with SIGILL before output",
        },
        {
            "issue_item": "Official plugin sample link/load",
            "rt204": "not packaged",
            "rt205": "builds; unresolved public plugin API methods",
            "rt206": "builds; ldd -r and dlopen pass; ABI query/init pass",
            "rt206_classification": "fixed",
            "basis": "provider now exports the previously unresolved public methods",
        },
        {
            "issue_item": "Independent plugin execution",
            "rt204": "not packaged",
            "rt205": "loader failure; execution not reached",
            "rt206": "assigned; exact; 1011 dispatches",
            "rt206_classification": "fixed",
            "basis": (
                "custom uint8 graph executes exact; official Track2 output remains non-exact "
                f"({plugin['official_track2_ep']['output_sha256']})"
            ),
        },
        {
            "issue_item": "Full YOLO26 INT8 EP",
            "rt204": "clip-minmax compile error / no complete EP run",
            "rt205": "clip-minmax error then abort",
            "rt206": (
                f"{full_int8_206['error_class']} / exit {full_int8_206['exit']} / "
                "no executed provider placement"
            ),
            "rt206_classification": "unchanged",
            "basis": "session creation aborts at the first quantized Conv",
        },
    ]
    fields = ["issue_item", "rt204", "rt205", "rt206", "rt206_classification", "basis"]
    write_tsv(args.output_dir / "issue1_regression_matrix.tsv", fields, decisions)
    write_tsv(args.output_dir / "issue1_final_decision_table.tsv", fields, decisions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
