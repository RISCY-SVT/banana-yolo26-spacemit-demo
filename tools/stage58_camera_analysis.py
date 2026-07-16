#!/usr/bin/env python3
"""Summarize buffered Stage58 camera/demo timing without mixing benchmark arms."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from pathlib import Path


SUMMARY_RE = re.compile(r"^SUMMARY source=(?:camera|video) (.*)$")


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def parse_summary(log: Path) -> dict[str, str]:
    found = ""
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = SUMMARY_RE.match(line)
        if match:
            found = match.group(1)
    if not found:
        raise ValueError(f"no SUMMARY row in {log}")
    result: dict[str, str] = {}
    for match in re.finditer(r'(\w+)=("[^"]*"|\S+)', found):
        result[match.group(1)] = match.group(2).strip('"')
    return result


def run_name(path: Path) -> str:
    return path.stem


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metric_files = sorted(args.camera_dir.glob("final-*.tsv"))
    metric_files = [path for path in metric_files if "detections" not in path.name]
    if not metric_files:
        raise ValueError("no final camera timing files")

    phase_names = [
        "capture_ms", "resize_letterbox_ms", "bgr_to_rgb_ms", "preprocess_ms",
        "executor_ms", "postprocess_ms", "render_ms", "display_ms", "record_ms",
        "total_ms", "read_to_display_ms",
    ]
    summaries: list[dict[str, str]] = []
    raw_path = args.output_dir / "camera_timing_raw.tsv"
    with raw_path.open("w", encoding="utf-8", newline="") as raw_stream:
        raw_writer = None
        for metric_path in metric_files:
            name = run_name(metric_path)
            log_path = args.log_dir / f"{name}.log"
            if not log_path.is_file():
                continue
            summary = parse_summary(log_path)
            rows: list[dict[str, str]] = []
            with metric_path.open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream, delimiter="\t"):
                    if row["measured"] == "1":
                        rows.append(row)
            if not rows:
                raise ValueError(f"no measured rows in {metric_path}")
            if raw_writer is None:
                raw_writer = csv.DictWriter(
                    raw_stream, fieldnames=["run"] + list(rows[0]), delimiter="\t",
                    lineterminator="\n")
                raw_writer.writeheader()
            for row in rows:
                raw_writer.writerow({"run": name, **row})

            result = {
                "run": name,
                "profile": summary["profile"],
                "flow": summary["flow"],
                "effective_format": summary["effective_format"],
                "measured_frames": str(len(rows)),
                "captured_frames": summary["captured_frames"],
                "dropped_frames": summary["dropped_frames"],
                "drop_pct": summary["drop_pct"],
                "capture_fps": summary["capture_fps"],
                "processed_fps": summary["processed_fps"],
                "displayed_fps": summary["displayed_fps"],
                "recording_fps": summary["recording_fps"],
                "recording_mode": summary["recording_mode"],
            }
            for phase in phase_names:
                values = [float(row[phase]) for row in rows]
                result[f"{phase}_mean"] = f"{statistics.fmean(values):.6f}"
                result[f"{phase}_p95"] = f"{percentile(values, 0.95):.6f}"
            summaries.append(result)

    summary_path = args.output_dir / "camera_timing_summary.tsv"
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]), delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)

    drop_path = args.output_dir / "camera_drop_summary.tsv"
    drop_fields = [
        "run", "flow", "effective_format", "captured_frames", "measured_frames",
        "dropped_frames", "drop_pct", "capture_fps", "processed_fps",
        "capture_backend_drop_status",
    ]
    with drop_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=drop_fields, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        for row in summaries:
            writer.writerow({
                key: row.get(key, "unknown-not-exposed-by-v4l2-opencv")
                for key in drop_fields
            })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
