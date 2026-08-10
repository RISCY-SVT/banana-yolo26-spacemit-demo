#!/usr/bin/env python3
"""Normalize XSlim Graphwise Markdown reports for Stage65B-R1."""

from __future__ import annotations

import argparse
import csv
import html
import re
from pathlib import Path
from typing import Any


TAG = re.compile(r"<[^>]+>")
HEAD = re.compile(r"one2one_cv([23])\.([012])")


def clean(value: str) -> str:
    return TAG.sub("", html.unescape(value)).strip()


def number(value: str) -> float:
    try:
        return float(clean(value))
    except ValueError:
        return float("nan")


def minmax(value: str) -> tuple[str, str]:
    fields = [item.strip() for item in clean(value).split(",")]
    return (fields[0], fields[1]) if len(fields) == 2 else ("", "")


def mapping(op: str, variable: str) -> tuple[str, str]:
    match = HEAD.search(variable) or HEAD.search(op)
    if match is None:
        return "", ""
    return {"0": "P3", "1": "P4", "2": "P5"}[match.group(2)], {
        "2": "bbox",
        "3": "confidence",
    }[match.group(1)]


def rows(lane: str, path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    table = [line for line in lines if line.lstrip().startswith("|")]
    if len(table) < 3:
        raise ValueError(f"Graphwise table not found: {path}")
    header = [clean(item) for item in table[0].strip().strip("|").split("|")]
    output: list[dict[str, Any]] = []
    for line in table[2:]:
        values = [clean(item) for item in line.strip().strip("|").split("|")]
        if len(values) != len(header):
            continue
        item = dict(zip(header, values))
        op = item.get("Op", "")
        variable = item.get("Var", "")
        pyramid, branch = mapping(op, variable)
        q_min, q_max = minmax(item.get("Q.MinMax", ""))
        f_min, f_max = minmax(item.get("F.MinMax", ""))
        snr = number(item.get("SNR", ""))
        cosine = number(item.get("Cosine", ""))
        output.append(
            {
                "lane": lane,
                "graphwise_rank": item.get("", item.get("Unnamed: 0", len(output))),
                "op": op,
                "variable": variable,
                "pyramid": pyramid,
                "branch": branch,
                "snr": snr,
                "mse": number(item.get("MSE", "")),
                "cosine": cosine,
                "q_min": q_min,
                "q_max": q_max,
                "f_min": f_min,
                "f_max": f_max,
                "f_histogram": item.get("F.Hist", ""),
                "snr_high_error": int(snr >= 0.1),
                "cosine_significant_deviation": int(cosine < 0.99),
                "source_report": path.name,
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", action="append", required=True, help="LANE=/path/to/report.md"
    )
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    all_rows: list[dict[str, Any]] = []
    for value in options.input:
        lane, separator, raw_path = value.partition("=")
        if not separator or not lane or not raw_path:
            raise ValueError(f"invalid --input value: {value}")
        all_rows.extend(rows(lane, Path(raw_path)))
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(all_rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(all_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
