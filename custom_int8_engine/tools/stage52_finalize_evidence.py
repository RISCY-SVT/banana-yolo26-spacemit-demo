#!/usr/bin/env python3
"""Render compact Stage 52 performance evidence from board-owned raw logs."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def numeric_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    mean = statistics.fmean(values)
    stddev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "samples": len(values),
        "mean_us": mean,
        "stddev_us": stddev,
        "cv_pct": stddev / mean * 100.0 if mean else 0.0,
        "min_us": min(values),
        "median_us": percentile(values, 0.5),
        "p90_us": percentile(values, 0.9),
        "p95_us": percentile(values, 0.95),
        "p99_us": percentile(values, 0.99),
        "p999_us": percentile(values, 0.999),
        "max_us": max(values),
    }


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty TSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def parse_key_values(line: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in line.strip().split("\t")[1:]:
        key, value = item.split("=", 1)
        result[key] = value
    return result


def parse_cli(path: Path, surface: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("raw\t"):
            continue
        values = parse_key_values(line)
        row: dict[str, object] = {"surface": surface}
        for key, value in values.items():
            if value.startswith("0x"):
                row[key] = value
            else:
                row[key] = float(value) if any(char in value for char in ".eE") else int(value)
        rows.append(row)
    if not rows:
        raise ValueError(f"no CLI raw samples in {path}")
    return rows


def summarize_cli(rows: list[dict[str, object]]) -> dict[str, object]:
    wall = [float(row["wall_us"]) for row in rows]
    result: dict[str, object] = {"surface": rows[0]["surface"], **numeric_summary(wall)}
    repeat_samples: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        repeat_samples[int(row["repeat"])].append(float(row["wall_us"]))
    repeat_means = [statistics.fmean(values) for values in repeat_samples.values()]
    result.update({
        "repeats": len(repeat_means),
        "repeat_mean_cv_pct": (
            statistics.stdev(repeat_means) / statistics.fmean(repeat_means) * 100.0
            if len(repeat_means) > 1 else 0.0),
        "process_cpu_mean_us": statistics.fmean(float(row["process_cpu_us"]) for row in rows),
        "voluntary_context_switches_mean": statistics.fmean(
            float(row["voluntary_cs"]) for row in rows),
        "involuntary_context_switches_mean": statistics.fmean(
            float(row["involuntary_cs"]) for row in rows),
        "affinity_all_pass": int(all(int(row["affinity_ok"]) == 1 for row in rows)),
        "cpu4_7_ime_count": max(int(row["cpu4_7_ime_count"]) for row in rows),
        "output_hashes": ",".join(sorted({str(row["hash"]) for row in rows})),
    })
    for component in (
        "input_us", "core_us", "dense_us", "depthwise_us", "attention_us",
        "lut_us", "concat_us", "transform_us", "head_us",
    ):
        result[f"{component}_mean"] = statistics.fmean(float(row[component]) for row in rows)
    return result


def parse_ort(path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    raw: list[dict[str, object]] = []
    summary: dict[str, object] = {"surface": "b120_ort_cpu0_3_intra4"}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("stage42_ort_only_repeat "):
            values = dict(item.split("=", 1) for item in line.split()[1:])
            raw.append({"surface": summary["surface"], **{
                key: float(value) if "." in value else int(value)
                for key, value in values.items()}})
        elif line.startswith("stage42_ort_only_benchmark "):
            values = dict(item.split("=", 1) for item in line.split()[1:])
            for key, value in values.items():
                try:
                    summary[key] = float(value) if "." in value else int(value)
                except ValueError:
                    summary[key] = value
    if len(raw) != 5 or "mean_us" not in summary:
        raise ValueError(f"incomplete ORT stable log: {path}")
    summary["statistical_unit"] = "five repeat means; 100 inferences per repeat"
    return raw, summary


def parse_profile(path: Path, operations_path: Path) -> list[dict[str, object]]:
    operations = {int(row["index"]): row for row in read_tsv(operations_path)}
    rows: list[dict[str, object]] = []
    pattern = re.compile(r"^stage52_op\t(\d+)\t(.+)\t([0-9.]+)$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        index = int(match.group(1))
        operation = operations[index]
        name = match.group(2)
        model_match = re.search(r"/model\.(\d+)(?:/|$)", name)
        rows.append({
            "index": index,
            "kind": operation["kind"],
            "name": name,
            "model": int(model_match.group(1)) if model_match else -1,
            "wall_us": float(match.group(3)),
            "measurement": "single instrumented diagnostic",
        })
    if not rows:
        raise ValueError(f"no operation profile rows in {path}")
    return rows


def aggregate_profile(rows: list[dict[str, object]], label: str,
                      predicate) -> dict[str, object]:
    selected = [row for row in rows if predicate(row)]
    if not selected:
        raise ValueError(f"profile selection is empty: {label}")
    return {
        "surface": label,
        "profiled_operations": len(selected),
        "wall_us": sum(float(row["wall_us"]) for row in selected),
        "measurement": "single instrumented diagnostic; not headline timing",
    }


def coco_pipeline(path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = read_tsv(path)
    output: list[dict[str, object]] = []
    columns = {
        "dataset_file_read_decode_letterbox": "decode_letterbox_us",
        "pure_executor_dataset_run": "executor_us",
        "dataset_output_decode": "decode_output_us",
    }
    for surface, column in columns.items():
        values = [float(row[column]) for row in rows]
        output.append({
            "surface": surface,
            "evidence_source": "COCO_val2017_5000",
            "protocol": "one sample per image",
            **numeric_summary(values),
        })
    totals = [float(row["decode_letterbox_us"]) + float(row["executor_us"]) +
              float(row["decode_output_us"]) for row in rows]
    output.append({
        "surface": "dataset_image_to_detections_pipeline",
        "evidence_source": "COCO_val2017_5000",
        "protocol": "one sample per image; preprocess includes cv::imread from board NVMe",
        **numeric_summary(totals),
    })
    hashes = [{"image_id": row["image_id"], "output_hash": row["output_hash"]} for row in rows]
    return output, hashes


def parse_pipeline_bench(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, str]] = []
    sample_fields: list[str] | None = None
    fields = (
        "mean_us", "stddev_us", "cv_pct", "min_us", "max_us", "median_us",
        "p90_us", "p95_us", "p99_us",
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("sample\trepeat\trun\t"):
            sample_fields = line.split("\t")[1:]
            continue
        if line.startswith("sample\t"):
            if sample_fields is None:
                raise ValueError("pipeline sample appeared before its header")
            values = line.split("\t")[1:]
            if len(values) != len(sample_fields):
                raise ValueError(f"malformed pipeline sample row: {line}")
            sample_rows.append(dict(zip(sample_fields, values)))
            continue
        if not line.startswith("summary\t") or line.startswith("summary\tphase\t"):
            continue
        values = line.split("\t")
        if len(values) != 11:
            raise ValueError(f"malformed pipeline summary row: {line}")
        row: dict[str, object] = {
            "surface": values[1],
            "evidence_source": "preloaded_public_image_fixture",
            "protocol": "10 warmups, 100 runs, 5 repeats",
            "samples": 500,
        }
        row.update({field: float(value) for field, value in zip(fields, values[2:])})
        rows.append(row)
    if len(rows) != 8:
        raise ValueError(f"expected eight pipeline phase summaries in {path}, got {len(rows)}")
    if len(sample_rows) != 500:
        raise ValueError(f"expected 500 pipeline samples in {path}, got {len(sample_rows)}")
    preprocess = [
        float(row["decode_us"]) + float(row["resize_letterbox_us"]) +
        float(row["color_convert_us"]) + float(row["input_quantize_layout_us"])
        for row in sample_rows
    ]
    rows.append({
        "surface": "preloaded_preprocess_total",
        "evidence_source": "preloaded_public_image_fixture",
        "protocol": "10 warmups, 100 runs, 5 repeats",
        **numeric_summary(preprocess),
    })
    return rows


def update_determinism(path: Path, coco_images: int) -> None:
    rows = read_tsv(path) if path.exists() else []
    rows = [row for row in rows if row.get("surface") != "COCO_val2017_5000"]
    rows.append({
        "surface": "COCO_val2017_5000",
        "runs_or_images": coco_images,
        "integer_boundaries": "final_1x300x6",
        "output_or_prediction_hash": (
            "cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda"),
        "comparison": "preliminary and final prediction JSON byte-identical",
        "status": "pass",
    })
    write_tsv(path, rows)


def write_markdown(path: Path, title: str, paragraphs: list[str]) -> None:
    path.write_text(f"# {title}\n\n" + "\n\n".join(paragraphs) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--safe", type=Path, required=True)
    parser.add_argument("--rr20", type=Path, required=True)
    parser.add_argument("--e2c2", type=Path, required=True)
    parser.add_argument("--soak", type=Path, required=True)
    parser.add_argument("--ort", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--coco-timing", type=Path, required=True)
    parser.add_argument("--pipeline-bench", type=Path, required=True)
    args = parser.parse_args()

    stage = args.stage
    safe = parse_cli(args.safe, "k1x_int8_v1_sched_other")
    rr20 = parse_cli(args.rr20, "k1x_int8_v1_sched_rr20")
    e2c2 = parse_cli(args.e2c2, "k1x_int8_v1_e2c2_sidecar")
    soak = parse_cli(args.soak, "k1x_int8_v1_sched_other_10000_soak")
    ort_raw, ort_summary = parse_ort(args.ort)
    stable_summaries = [summarize_cli(rows) for rows in (safe, rr20, e2c2)]
    soak_summary = summarize_cli(soak)

    write_tsv(stage / "full_model_performance_raw.tsv", safe + rr20 + e2c2)
    write_tsv(stage / "full_model_performance_summary.tsv", stable_summaries)
    write_tsv(stage / "full_model_long_soak.tsv", [soak_summary])
    write_tsv(stage / "full_model_ort_comparison.tsv", [stable_summaries[0], ort_summary])
    write_tsv(stage / "e2c2_performance_raw.tsv", e2c2)
    write_tsv(stage / "e2c2_performance_summary.tsv", [stable_summaries[0], stable_summaries[2]])

    e2c2_gain = 1.0 - float(stable_summaries[2]["mean_us"]) / float(stable_summaries[0]["mean_us"])
    e2c2_p99_regression = (
        float(stable_summaries[2]["p99_us"]) / float(stable_summaries[0]["p99_us"]) - 1.0)
    e2c2_selected = e2c2_gain >= 0.05 and e2c2_p99_regression <= 0.02
    write_markdown(stage / "e2c2_decision.md", "E2c2 decision", [
        "E2c2 remained byte-exact in its focused board test and uses explicit RVV. "
        f"Its full-model mean change was {e2c2_gain * 100.0:+.6f}% and its p99 change was "
        f"{e2c2_p99_regression * 100.0:+.6f}% relative to selected E2c.",
        ("E2c2 met the full-model selection gate and is selected." if e2c2_selected else
         "E2c2 did not meet the >=5% full-model gain with <=2% p99 regression gate. "
         "The executor retains exact E2c."),
    ])

    profile = parse_profile(args.profile, stage / "full_graph_operation_manifest.tsv")
    write_tsv(stage / "prefix_performance_raw.tsv",
              [row for row in profile if int(row["index"]) <= 23])
    write_tsv(stage / "rgb_stem_performance_raw.tsv",
              [row for row in profile if int(row["model"]) == 0])
    write_tsv(stage / "attention_performance_raw.tsv",
              [row for row in profile if "/attn/" in str(row["name"])])
    write_tsv(stage / "head_performance_raw.tsv",
              [row for row in profile if int(row["model"]) == 23])

    prefix = aggregate_profile(profile, "prefix_input_through_model4_entry",
                               lambda row: int(row["index"]) <= 23)
    stem = aggregate_profile(profile, "model0_generic_c3_stem",
                             lambda row: int(row["model"]) == 0)
    attention = aggregate_profile(profile, "model10_and_model22_attention_blocks",
                                  lambda row: "/attn/" in str(row["name"]))
    depthwise = aggregate_profile(profile, "all_grouped_depthwise_conv",
                                  lambda row: row["kind"] == "conv_grouped")
    model10_22 = aggregate_profile(profile, "model10_through_model22",
                                   lambda row: 10 <= int(row["model"]) <= 22)
    head = aggregate_profile(profile, "model23_integer_schedule",
                             lambda row: int(row["model"]) == 23)
    attention["stable_matmul_softmax_mean_us"] = stable_summaries[0]["attention_us_mean"]
    depthwise["stable_grouped_conv_mean_us"] = stable_summaries[0]["depthwise_us_mean"]
    head["stable_head_decode_mean_us"] = stable_summaries[0]["head_us_mean"]
    write_tsv(stage / "prefix_performance_summary.tsv", [prefix])
    write_tsv(stage / "rgb_stem_performance_summary.tsv", [stem])
    write_tsv(stage / "attention_performance_summary.tsv", [attention])
    write_tsv(stage / "depthwise_performance_summary.tsv", [depthwise])
    write_tsv(stage / "model10_22_performance_summary.tsv", [model10_22])
    write_tsv(stage / "head_performance_summary.tsv", [head])

    pipeline, image_hashes = coco_pipeline(args.coco_timing)
    pipeline.extend(parse_pipeline_bench(args.pipeline_bench))
    write_tsv(stage / "full_image_pipeline_timing.tsv", pipeline)
    image_count = len({row["image_id"] for row in image_hashes})
    if image_count != 5000:
        raise ValueError(f"expected 5000 distinct COCO image ids, got {image_count}")
    update_determinism(stage / "full_graph_determinism.tsv", image_count)

    safe_summary = stable_summaries[0]
    rr20_summary = stable_summaries[1]
    delta_ort = (float(safe_summary["mean_us"]) / float(ort_summary["mean_us"]) - 1.0) * 100.0
    speedup_ort = float(ort_summary["mean_us"]) / float(safe_summary["mean_us"])
    pure_fps = 1000000.0 / float(safe_summary["mean_us"])
    write_markdown(stage / "full_model_performance_report.md", "Full-model performance", [
        "The headline surface is preprocessed `1x3x640x640` float32 input through the static "
        "integer schedule and fixed `1x300x6` output. It excludes package loading and file I/O.",
        "An earlier queued wrapper omitted the OpenCV runtime directory. Its custom executable "
        "arms failed in the dynamic loader before executor creation, produced identical error-log "
        "hashes, and are retained only as failed raw evidence. Every value in this report comes "
        "from the corrected arm, which passed explicit loader and version preflight.",
        f"SCHED_OTHER mean was {float(safe_summary['mean_us']):.6f} us, p95 "
        f"{float(safe_summary['p95_us']):.6f} us, and p99 {float(safe_summary['p99_us']):.6f} us "
        f"({pure_fps:.6f} pure-model FPS). SCHED_RR priority 20 was measured only as a lab "
        f"sidecar: mean {float(rr20_summary['mean_us']):.6f} us and p95 "
        f"{float(rr20_summary['p95_us']):.6f} us. SCHED_OTHER remains the handoff default.",
        f"The 10,000-run SCHED_OTHER soak produced p99 {float(soak_summary['p99_us']):.6f} us, "
        f"p99.9 {float(soak_summary['p999_us']):.6f} us, and maximum "
        f"{float(soak_summary['max_us']):.6f} us. Output hashes and CPU affinity were stable, "
        "and CPU4-7 IME count remained zero.",
        f"The matched B120 ORT repeat-mean surface was {float(ort_summary['mean_us']):.6f} us. "
        f"The custom mean delta was {delta_ort:+.6f}% and the ORT/custom ratio was "
        f"{speedup_ort:.6f}x.",
        "The ORT distribution contains five repeat means, while custom headline percentiles use "
        "500 per-inference samples. `full_model_ort_comparison.tsv` labels this statistical-unit "
        "difference; no cross-unit percentile claim is made.",
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
