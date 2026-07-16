#!/usr/bin/env python3
"""Summarize buffered Stage58 camera/demo timing without mixing benchmark arms."""

from __future__ import annotations

import argparse
import collections
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


def parse_summary(log: Path) -> dict[str, str] | None:
    found = ""
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = SUMMARY_RE.match(line)
        if match:
            found = match.group(1)
    if not found:
        return None
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

    metric_files = []
    for path in sorted(args.camera_dir.glob("*.tsv")):
        if "detections" in path.name or not (args.log_dir / f"{path.stem}.log").is_file():
            continue
        with path.open(encoding="utf-8", errors="replace") as stream:
            if "measured" not in stream.readline().split("\t"):
                continue
        metric_files.append(path)
    if not metric_files:
        raise ValueError("no final camera timing files")

    phase_names = [
        "capture_ms", "resize_letterbox_ms", "bgr_to_rgb_ms", "preprocess_ms",
        "executor_ms", "postprocess_ms", "render_ms", "display_ms", "record_ms",
        "total_ms", "read_to_display_ms",
    ]
    summaries: list[dict[str, str]] = []
    aggregate_rows: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    aggregate_summaries: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    raw_path = args.output_dir / "camera_timing_raw.tsv"
    with raw_path.open("w", encoding="utf-8", newline="") as raw_stream:
        raw_writer = None
        for metric_path in metric_files:
            name = run_name(metric_path)
            log_path = args.log_dir / f"{name}.log"
            if not log_path.is_file():
                continue
            summary = parse_summary(log_path)
            if summary is None:
                continue
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

            arm = re.sub(r"-r[0-9]+$", "", name)
            aggregate_rows[arm].extend(rows)

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
                "recorded_frames": summary["recorded_frames"],
                "recording_fps": summary["recording_fps"],
                "recording_mode": summary["recording_mode"],
            }
            for phase in phase_names:
                values = [float(row[phase]) for row in rows]
                result[f"{phase}_mean"] = f"{statistics.fmean(values):.6f}"
                result[f"{phase}_p95"] = f"{percentile(values, 0.95):.6f}"
            summaries.append(result)
            aggregate_summaries[arm].append(result)

    for arm in sorted(aggregate_rows):
        runs = aggregate_summaries[arm]
        if len(runs) < 2:
            continue
        rows = aggregate_rows[arm]
        measured = sum(int(run["measured_frames"]) for run in runs)
        captured = sum(int(run["captured_frames"]) for run in runs)
        dropped = sum(int(run["dropped_frames"]) for run in runs)
        elapsed = sum(int(run["measured_frames"]) / float(run["processed_fps"])
                      for run in runs)
        capture_elapsed = sum(int(run["captured_frames"]) / float(run["capture_fps"])
                              for run in runs)
        recorded = sum(int(run["recorded_frames"]) for run in runs)
        recording_elapsed = sum(
            int(run["recorded_frames"]) / float(run["recording_fps"])
            for run in runs if float(run["recording_fps"]) > 0.0)
        aggregate = {
            "run": f"aggregate-{arm}",
            "profile": runs[0]["profile"],
            "flow": runs[0]["flow"],
            "effective_format": runs[0]["effective_format"],
            "measured_frames": str(measured),
            "captured_frames": str(captured),
            "dropped_frames": str(dropped),
            "drop_pct": f"{100.0 * dropped / captured:.6f}" if captured else "0.000000",
            "capture_fps": f"{captured / capture_elapsed:.6f}",
            "processed_fps": f"{measured / elapsed:.6f}",
            "displayed_fps": (f"{measured / elapsed:.6f}"
                              if all(float(run["displayed_fps"]) > 0.0 for run in runs)
                              else "0.000000"),
            "recorded_frames": str(recorded),
            "recording_fps": (f"{recorded / recording_elapsed:.6f}"
                              if recording_elapsed > 0.0 else "0.000000"),
            "recording_mode": runs[0]["recording_mode"],
        }
        for phase in phase_names:
            values = [float(row[phase]) for row in rows]
            aggregate[f"{phase}_mean"] = f"{statistics.fmean(values):.6f}"
            aggregate[f"{phase}_p95"] = f"{percentile(values, 0.95):.6f}"
        summaries.append(aggregate)

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

    by_run = {row["run"]: row for row in summaries}
    selected = by_run.get("aggregate-release-c5", by_run.get("aggregate-final-c5"))
    if selected is not None:
        recording = by_run.get("release-camera-demo-30s",
                               by_run.get("camera-demo-30s", {}))
        soak = by_run.get("release-c5-soak", by_run.get("final-c5-soak", {}))
        report_en = f"""# Full Camera FPS

The selected complete camera surface is `{selected['effective_format']}`, V4L2,
`{selected['profile']}`, latest-frame flow, GUI display, boxes, and timing overlay,
without O2. Three independent 180-second runs processed and displayed
{selected['processed_fps']} FPS across {selected['measured_frames']} measured frames.
Capture arrival was {selected['capture_fps']} FPS; the queue-depth-one application
replaced {selected['drop_pct']}% of captured frames because capture was faster than
the complete pipeline.

Mean/p95 software time from `VideoCapture::read` return through display/event
handling was {selected['read_to_display_ms_mean']} / {selected['read_to_display_ms_p95']}
ms. Mean total software-loop time was {selected['total_ms_mean']} ms. These values
include capture, exact letterbox and RGB conversion, executor, deletterboxing,
boxes, overlay, `imshow`, and event handling. They are not pure-model FPS and are
not sensor-to-screen latency because sensor timestamps were not correlated.

The separate MJPG AVI recording arm processed {recording.get('recording_fps', 'not-measured')}
FPS. The 30-minute selected-profile camera soak processed
{soak.get('measured_frames', 'not-measured')} frames at
{soak.get('processed_fps', 'not-measured')} FPS and completed without an application
failure. OpenCV/V4L2 did not expose an independent driver-drop count, so the report
only claims application replacements.
"""
        report_ru = f"""# Полная частота кадров с камеры

Выбранный полный камерный режим: `{selected['effective_format']}`, V4L2, профиль
`{selected['profile']}`, очередь последнего кадра, графическое окно, рамки и панель
измерений, без O2. В трех независимых прогонах по 180 секунд обработано и показано
{selected['measured_frames']} кадров со средней частотой {selected['processed_fps']}
кадра/с. Кадры поступали с частотой {selected['capture_fps']} кадра/с; очередь
глубиной один заменила {selected['drop_pct']}% кадров более свежими, поскольку
полный тракт работает медленнее захвата.

Среднее и p95 программного времени от возврата `VideoCapture::read` до показа и
обработки события составили {selected['read_to_display_ms_mean']} и
{selected['read_to_display_ms_p95']} мс. Среднее время полного программного цикла —
{selected['total_ms_mean']} мс. В него входят захват, точное преобразование с полями,
RGB, исполнитель, пересчет координат, рамки, панель, `imshow` и обработка событий.
Это не частота одной модели и не задержка от сенсора до экрана: метки времени
сенсора не сопоставлялись.

Отдельный режим записи MJPG AVI показал
{recording.get('recording_fps', 'не измерено')} кадра/с. Тридцатиминутный прогон
выбранного режима обработал {soak.get('measured_frames', 'не измерено')} кадров со
скоростью {soak.get('processed_fps', 'не измерено')} кадра/с и завершился без сбоя
приложения. OpenCV/V4L2 не выдал отдельный счетчик потерь драйвера, поэтому указаны
только замены кадров внутри приложения.
"""
        (args.output_dir / "camera_full_fps_report_en.md").write_text(
            report_en, encoding="utf-8")
        (args.output_dir / "camera_full_fps_report_ru.md").write_text(
            report_ru, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
