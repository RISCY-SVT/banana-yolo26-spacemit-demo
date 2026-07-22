#!/usr/bin/env python3
"""Summarize Stage60 camera schema-v2 samples without claiming sensor FPS."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path
from typing import Any


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def read_metrics(path: Path) -> tuple[str, list[dict[str, str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "# metrics_schema_version=2":
        raise ValueError(f"camera metric schema mismatch: {path}")
    rows = list(csv.DictReader(lines[1:], delimiter="\t"))
    measured = [row for row in rows if row["measured"] == "1"]
    if not measured:
        raise ValueError(f"no measured camera rows: {path}")
    return "2", measured


def read_system(path: Path) -> tuple[float, float, float]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    temperatures = [float(row["mean_thermal_c"]) for row in rows if row["mean_thermal_c"]]
    frequencies = [float(row["mean_cpu0_4_khz"]) for row in rows if row["mean_cpu0_4_khz"]]
    return (
        statistics.fmean(temperatures) if temperatures else 0.0,
        max(temperatures) if temperatures else 0.0,
        min(frequencies) if frequencies else 0.0,
    )


def read_application(path: Path) -> dict[str, str]:
    values = {
        "camera_requested": "",
        "camera_effective_format": "",
        "backend_reported_fps": "",
        "capture_backend": "",
        "opencv_threads": "",
        "profile": "",
        "flow": "",
    }
    log = path.read_text(encoding="utf-8", errors="replace")
    requested = re.search(r"camera_requested=(\S+) effective=(.+)$", log, re.MULTILINE)
    if requested is not None:
        values["camera_requested"] = requested.group(1)
        values["camera_effective_format"] = requested.group(2).strip()
    backend = re.search(r"capture_backend=(\S+)", log)
    if backend is not None:
        values["capture_backend"] = backend.group(1)
    profile = re.search(r"profile=(\S+) flow=(\S+) opencv_threads=(\d+)", log)
    if profile is not None:
        values["profile"] = profile.group(1)
        values["flow"] = profile.group(2)
        values["opencv_threads"] = profile.group(3)

    stdout = path.with_name(path.name.replace(".application.log", ".stdout.log")).read_text(
        encoding="utf-8", errors="replace"
    )
    effective = re.search(r'effective_format="([^"]+)"', stdout)
    if effective is not None:
        values["camera_effective_format"] = effective.group(1)
    reported = re.search(r"backend_reported_fps=([0-9.]+)", stdout)
    if reported is not None:
        values["backend_reported_fps"] = reported.group(1)
    return values


def summarize(path: Path, raw_v4l2_fps: float) -> dict[str, Any]:
    match = re.fullmatch(r"r(\d+)-(run\d+|soak)\.metrics\.tsv", path.name)
    if match is None:
        raise ValueError(f"unexpected Stage60 camera filename: {path.name}")
    resolution = int(match.group(1))
    arm = match.group(2)
    schema, rows = read_metrics(path)
    start_ns = int(rows[0]["measured_window_start_ns"])
    end_ns = int(rows[-1]["frame_done_ns"])
    duration = (end_ns - start_ns) / 1.0e9
    if duration <= 0.0:
        raise ValueError(f"invalid measured window: {path}")
    captured = int(rows[-1]["captured_measured"])
    replacements = int(rows[-1]["application_slot_replacements_measured"])
    consumer = [float(row["consumer_loop_ms"]) for row in rows]
    read_display = [float(row["decoded_read_return_to_display_call_ms"]) for row in rows]
    inference = [float(row["executor_ms"]) for row in rows]
    preprocess = [float(row["preprocess_ms"]) for row in rows]
    render = [float(row["render_ms"]) for row in rows]
    display = [float(row["display_ms"]) for row in rows]
    temperature, max_temperature, min_frequency = read_system(
        path.with_name(path.name.replace(".metrics.tsv", ".system.tsv"))
    )
    application = read_application(
        path.with_name(path.name.replace(".metrics.tsv", ".application.log"))
    )
    return {
        "resolution": resolution,
        "arm": arm,
        "metric_schema_version": schema,
        "camera_requested": application["camera_requested"],
        "camera_effective_format": application["camera_effective_format"],
        "backend_reported_fps": application["backend_reported_fps"],
        "capture_backend": application["capture_backend"],
        "opencv_threads": application["opencv_threads"],
        "profile": application["profile"],
        "flow": application["flow"],
        "raw_v4l2_dequeued_compressed_buffer_fps_reference": f"{raw_v4l2_fps:.9f}",
        "measured_seconds": f"{duration:.6f}",
        "processed_frames": len(rows),
        "captured_decoded_frames": captured,
        "opencv_decoded_frame_fps": f"{captured / duration:.9f}",
        "processed_displayed_fps": f"{len(rows) / duration:.9f}",
        "application_slot_replacements": replacements,
        "application_slot_replacement_pct": f"{100.0 * replacements / captured:.9f}",
        "consumer_loop_mean_ms": f"{statistics.fmean(consumer):.6f}",
        "consumer_loop_p95_ms": f"{percentile(consumer, 0.95):.6f}",
        "consumer_loop_p99_ms": f"{percentile(consumer, 0.99):.6f}",
        "read_return_to_display_call_mean_ms": f"{statistics.fmean(read_display):.6f}",
        "read_return_to_display_call_p95_ms": f"{percentile(read_display, 0.95):.6f}",
        "inference_mean_ms": f"{statistics.fmean(inference):.6f}",
        "inference_p95_ms": f"{percentile(inference, 0.95):.6f}",
        "preprocess_mean_ms": f"{statistics.fmean(preprocess):.6f}",
        "render_mean_ms": f"{statistics.fmean(render):.6f}",
        "display_mean_ms": f"{statistics.fmean(display):.6f}",
        "mean_thermal_c": f"{temperature:.6f}",
        "max_thermal_c": f"{max_temperature:.6f}",
        "min_cpu0_4_khz": f"{min_frequency:.0f}",
        "surface_definition": "decoded-read-return-to-GUI-display-call; not sensor-to-screen",
        "source_file": path.name,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-v4l2-fps", type=float, default=30.0016)
    args = parser.parse_args()
    rows = [summarize(path, args.raw_v4l2_fps)
            for path in sorted(args.input.glob("r*.metrics.tsv"))]
    if not rows:
        raise ValueError(f"no Stage60 camera metrics under {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"camera_arms={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
