#!/usr/bin/env python3
"""Run paired Stage60-control versus Stage61 N-tail full-model ABBA blocks."""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import subprocess
import tempfile
from pathlib import Path


T95_10_PAIRS = 2.262


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def read_temperature() -> float:
    values: list[float] = []
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            values.append(float(path.read_text().strip()) / 1000.0)
        except (OSError, ValueError):
            pass
    return statistics.fmean(values) if values else math.nan


def read_frequency() -> float:
    values: list[float] = []
    for cpu in range(5):
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq")
        try:
            values.append(float(path.read_text().strip()))
        except (OSError, ValueError):
            pass
    return statistics.fmean(values) if values else math.nan


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def block_mean(rows: list[dict[str, str]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resolutions", default="640,512,448,416,384,352,320,256")
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if args.cycles < 1 or args.runs < 1 or args.warmup < 0:
        raise ValueError("invalid ABBA dimensions")

    args.output_root.mkdir(parents=True, exist_ok=True)
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    raw_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    invocation = 0

    with tempfile.TemporaryDirectory(prefix="y26-stage61-abba-", dir="/dev/shm") as temporary:
        temporary_root = Path(temporary)
        for resolution in (int(value) for value in args.resolutions.split(",")):
            rows_by_arm: dict[str, list[dict[str, str]]] = {"control": [], "ntail": []}
            block_means: dict[str, list[float]] = {"control": [], "ntail": []}
            paired_differences: list[float] = []
            for cycle in range(args.cycles):
                cycle_blocks: dict[str, list[float]] = {"control": [], "ntail": []}
                for slot, arm in enumerate(("control", "ntail", "ntail", "control")):
                    invocation += 1
                    output = temporary_root / f"r{resolution}_{invocation:04d}.tsv"
                    environment = os.environ.copy()
                    environment["Y26_STAGE61_ATTENTION_NTAIL"] = "0" if arm == "control" else "1"
                    before_temperature = read_temperature()
                    before_frequency = read_frequency()
                    command = [
                        str(args.binary),
                        "--package", str(args.stage_root / "packages" / f"r{resolution}"),
                        "--input", str(args.stage_root / "fixtures" / f"r{resolution}" /
                                       f"bus_r{resolution}_nchw_f32.bin"),
                        "--output", str(output),
                        "--surface", "preprocessed",
                        "--wake", "frame-gated-spin",
                        "--warmup", str(args.warmup),
                        "--runs", str(args.runs),
                        "--repeats", "1",
                    ]
                    completed = subprocess.run(
                        command, env=environment, check=False, capture_output=True, text=True,
                    )
                    after_temperature = read_temperature()
                    after_frequency = read_frequency()
                    if completed.returncode != 0:
                        raise RuntimeError(
                            f"ABBA invocation failed r{resolution} {arm}: {completed.stderr}"
                        )
                    rows = read_rows(output)
                    if len(rows) != args.runs:
                        raise RuntimeError("benchmark sample count mismatch")
                    for row in rows:
                        raw_rows.append({
                            "resolution": resolution,
                            "cycle": cycle,
                            "slot": slot,
                            "invocation": invocation,
                            "arm": arm,
                            **row,
                            "boot_id": boot_id,
                            "temperature_before_c": before_temperature,
                            "temperature_after_c": after_temperature,
                            "frequency_before_khz": before_frequency,
                            "frequency_after_khz": after_frequency,
                        })
                    mean = block_mean(rows, "total_us")
                    rows_by_arm[arm].extend(rows)
                    block_means[arm].append(mean)
                    cycle_blocks[arm].append(mean)
                    block_rows.append({
                        "resolution": resolution,
                        "cycle": cycle,
                        "slot": slot,
                        "invocation": invocation,
                        "arm": arm,
                        "samples": len(rows),
                        "mean_us": mean,
                        "attention_mean_us": block_mean(rows, "attention_us"),
                        "p95_us": percentile([float(row["total_us"]) for row in rows], 0.95),
                        "p99_us": percentile([float(row["total_us"]) for row in rows], 0.99),
                        "output_hash": rows[0]["output_hash"],
                        "manifest_sha256": rows[0]["manifest_sha256"],
                        "boot_id": boot_id,
                        "temperature_before_c": before_temperature,
                        "temperature_after_c": after_temperature,
                        "frequency_before_khz": before_frequency,
                        "frequency_after_khz": after_frequency,
                    })
                paired_differences.extend([
                    cycle_blocks["ntail"][0] - cycle_blocks["control"][0],
                    cycle_blocks["ntail"][1] - cycle_blocks["control"][1],
                ])

            control = [float(row["total_us"]) for row in rows_by_arm["control"]]
            ntail = [float(row["total_us"]) for row in rows_by_arm["ntail"]]
            control_mean = statistics.fmean(control)
            ntail_mean = statistics.fmean(ntail)
            mean_difference = statistics.fmean(paired_differences)
            if len(paired_differences) > 1:
                margin = (
                    T95_10_PAIRS * statistics.stdev(paired_differences) /
                    math.sqrt(len(paired_differences))
                )
            else:
                margin = math.nan
            ci_low = mean_difference - margin
            ci_high = mean_difference + margin
            delta_pct = 100.0 * (ntail_mean - control_mean) / control_mean
            aligned = resolution in {640, 512, 384, 256}
            selected = (
                abs(delta_pct) <= 0.5 if aligned else
                ci_high < 0.0 and delta_pct <= -0.5 and
                percentile(ntail, 0.95) <= percentile(control, 0.95) * 1.01 and
                percentile(ntail, 0.99) <= percentile(control, 0.99) * 1.01
            )
            summary_rows.append({
                "resolution": resolution,
                "aligned_control": int(aligned),
                "samples_per_arm": len(control),
                "blocks_per_arm": len(block_means["control"]),
                "control_mean_us": control_mean,
                "ntail_mean_us": ntail_mean,
                "delta_pct": delta_pct,
                "paired_mean_difference_us": mean_difference,
                "paired_95ci_low_us": ci_low,
                "paired_95ci_high_us": ci_high,
                "control_p95_us": percentile(control, 0.95),
                "ntail_p95_us": percentile(ntail, 0.95),
                "control_p99_us": percentile(control, 0.99),
                "ntail_p99_us": percentile(ntail, 0.99),
                "control_attention_mean_us": block_mean(rows_by_arm["control"], "attention_us"),
                "ntail_attention_mean_us": block_mean(rows_by_arm["ntail"], "attention_us"),
                "output_hash": rows_by_arm["control"][0]["output_hash"],
                "hash_exact": int(
                    {row["output_hash"] for row in rows_by_arm["control"] + rows_by_arm["ntail"]}
                    == {rows_by_arm["control"][0]["output_hash"]}
                ),
                "selection_gate": "pass" if selected else "fail",
            })

    write_rows(args.output_root / "attention_ntail_abba_raw.tsv", raw_rows)
    write_rows(args.output_root / "attention_ntail_abba_blocks.tsv", block_rows)
    write_rows(args.output_root / "attention_ntail_abba_summary.tsv", summary_rows)
    for row in summary_rows:
        print(
            f"r{row['resolution']} control_us={row['control_mean_us']:.3f} "
            f"ntail_us={row['ntail_mean_us']:.3f} delta_pct={row['delta_pct']:.3f} "
            f"gate={row['selection_gate']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
