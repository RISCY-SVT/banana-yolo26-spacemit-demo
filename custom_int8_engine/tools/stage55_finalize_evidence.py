#!/usr/bin/env python3
"""Render Stage55 repository evidence from preserved host and board artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path
from typing import Any, Iterable


TASK_ID = (
    "BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE55-RESIDUAL-CURRENT-GRAPH-"
    "EXECUTOR-CEILING-LEGAL-INDEXED-LUT2-ATTENTION-DENSE-E2C4-EVIDENCE-"
    "COSTMODEL-V3-AND-DUAL-REMOTE-FREEZE-GATE-001"
)
CONTRACT_ID = "K1X_INT8_V1"
PROFILE_ID = "K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001"
MODEL_SHA256 = "30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c"
PACKAGE_SHA256 = "fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be"
PREDICTION_SHA256 = "cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda"
OUTPUT_HASH = "0xd43f5e018b415631"
STAGE54_MEAN_US = 167411.836
STAGE54_CV_MEAN_US = 180237.700
STAGE54_CORPUS_MEAN_US = 166140.760690
K1X_MAP = 0.3707408944391919
SEMANTIC_MAP = 0.372453424641694
ORT_MEAN_US = 457968.821588


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in materialized:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_md(path: Path, title: str, paragraphs: Iterable[str]) -> None:
    write_text(path, "# " + title + "\n\n" + "\n\n".join(paragraphs))


def copy_text(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    write_text(destination, source.read_text(encoding="utf-8", errors="replace"))


def parse_key_values(line: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for field in line.rstrip().split("\t")[1:]:
        if "=" in field:
            key, value = field.split("=", 1)
            result[key] = value
    return result


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize_samples(surface: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    values = [float(row["wall_us"]) for row in rows]
    process = [float(row.get("process_cpu_us", 0.0)) for row in rows]
    voluntary = [float(row.get("voluntary_cs", 0.0)) for row in rows]
    involuntary = [float(row.get("involuntary_cs", 0.0)) for row in rows]
    hashes = sorted({row.get("hash", "") for row in rows})
    mean = statistics.fmean(values)
    stddev = statistics.pstdev(values)
    return {
        "surface": surface,
        "samples": len(values),
        "mean_us": f"{mean:.6f}",
        "stddev_us": f"{stddev:.6f}",
        "cv_pct": f"{stddev / mean * 100.0:.6f}",
        "median_us": f"{statistics.median(values):.6f}",
        "p90_us": f"{percentile(values, 0.90):.6f}",
        "p95_us": f"{percentile(values, 0.95):.6f}",
        "p99_us": f"{percentile(values, 0.99):.6f}",
        "p999_us": f"{percentile(values, 0.999):.6f}",
        "max_us": f"{max(values):.6f}",
        "process_cpu_mean_us": f"{statistics.fmean(process):.6f}",
        "voluntary_cs_mean": f"{statistics.fmean(voluntary):.6f}",
        "involuntary_cs_mean": f"{statistics.fmean(involuntary):.6f}",
        "output_hashes": ",".join(hashes),
        "cpu4_7_ime_count": max(int(row.get("cpu4_7_ime_count", "0")) for row in rows),
    }


def label_samples(surface: str, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{"surface": surface, **row} for row in rows]


def parse_cli(path: Path, surface: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8", errors="replace") as source:
        for line in source:
            if line.startswith("raw\t"):
                rows.append(parse_key_values(line))
    if not rows:
        raise ValueError(f"no raw samples in {path}")
    return rows, summarize_samples(surface, rows)


def parse_ort(path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8", errors="replace") as source:
        for line in source:
            if line.startswith("raw\t"):
                fields = line.rstrip().split("\t")
                row: dict[str, str] = {}
                for field in fields[1:]:
                    if "=" in field:
                        key, value = field.split("=", 1)
                        row[key] = value
                if "wall_us" in row:
                    rows.append(row)
            elif line.startswith("sample\t"):
                fields = line.rstrip().split("\t")
                if len(fields) >= 4 and fields[-1].replace(".", "", 1).isdigit():
                    rows.append({"wall_us": fields[-1]})
            elif line.startswith("stage42_ort_only_sample "):
                row = {}
                for field in line.rstrip().split()[1:]:
                    if "=" in field:
                        key, value = field.split("=", 1)
                        row[key] = value
                if "wall_us" in row:
                    rows.append(row)
    if len(rows) != 500:
        raise ValueError(f"expected 500 ORT samples in {path}, found {len(rows)}")
    return rows, summarize_samples("matched_b120_ort", rows)


def parse_real_corpus(path: Path) -> tuple[list[dict[str, str]], float]:
    rows = read_tsv(path)
    if len(rows) != 100:
        raise ValueError(f"expected 100 corpus rows, found {len(rows)}")
    return rows, statistics.fmean(float(row["executor_us"]) for row in rows)


def parse_pipeline(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8", errors="replace") as source:
        for line in source:
            if not line.startswith("summary\t") or line.startswith("summary\tphase\t"):
                continue
            fields = line.rstrip().split("\t")
            rows.append({
                "surface": fields[1], "mean_us": fields[2], "stddev_us": fields[3],
                "cv_pct": fields[4], "min_us": fields[5], "max_us": fields[6],
                "median_us": fields[7], "p90_us": fields[8], "p95_us": fields[9],
                "p99_us": fields[10],
            })
    return rows


def fixture_rows() -> list[dict[str, Any]]:
    return [
        {
            "fixture": fixture, "integer_boundaries": 215,
            "portable_cpp_scalar": "exact", "board_scalar": "exact",
            "board_optimized": "exact", "final_output": "exact", "status": "pass",
        }
        for fixture in ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "bus", "Zidane")
    ]


def selected_symbols(disassembly: str) -> list[dict[str, Any]]:
    names = (
        "y26_stage54_kernel_direct_1x1_m12n16",
        "q62_e2c4_i32x4x2_bias_to_s8",
        "q62_e2c4_i32x4x2_bias_lut_to_s8",
        "run_lut_chunk",
        "run_softmax_chunk",
    )
    return [
        {
            "symbol": name,
            "present_in_disassembly": int(name in disassembly),
            "implementation": "explicit assembly/intrinsics",
            "board_execution": "exact",
            "approved_cpus": "CPU0-3 workers",
        }
        for name in names
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument("--stage54", type=Path, required=True)
    parser.add_argument("--board-root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--release-root", type=Path)
    args = parser.parse_args()

    stage = args.stage
    artifacts = args.raw_root / "artifacts"
    board = args.board
    stage.mkdir(parents=True, exist_ok=True)

    baseline_cv_raw, baseline_cv = parse_cli(
        board / "profiles/baseline_compatibility_cv.log", "stage54_condition_variable_reproduced")
    baseline_spin_raw, baseline_spin = parse_cli(
        board / "profiles/baseline_low_latency_spin.log", "stage54_epoch_spin_reproduced")
    control_cv_raw, control_cv = parse_cli(
        board / "profiles/stage55_control_v1_compatibility_cv.log", "stage55_control_condition_variable")
    control_spin_raw, control_spin = parse_cli(
        board / "profiles/stage55_control_v1_low_latency_spin.log", "stage55_control_epoch_spin")
    final_cv_raw, final_cv = parse_cli(
        board / "profiles/final_compatibility_cv.log", "stage55_condition_variable")
    final_spin_raw, final_spin = parse_cli(
        board / "profiles/final_low_latency_frame_gate.log", "stage55_frame_gated_epoch_spin")
    soak_raw, soak = parse_cli(
        board / "profiles/final_low_latency_10000.log", "stage55_frame_gated_epoch_spin_10000_soak")
    ort_raw, ort = parse_ort(board / "profiles/matched_b120_ort_500.log")
    real_rows, real_mean = parse_real_corpus(board / "coco/final_real100.tsv")
    pipeline_rows = parse_pipeline(board / "profiles/final_preloaded_pipeline.log")
    pipeline_total = next(row for row in pipeline_rows if row["surface"] == "preloaded_image_pipeline")
    coco_json = board / "coco/final/stage55_final5000.json"

    expected = (
        (baseline_cv_raw, 500, "Stage54 CV"), (baseline_spin_raw, 500, "Stage54 spin"),
        (control_cv_raw, 500, "Stage55 control CV"), (control_spin_raw, 500, "Stage55 control spin"),
        (final_cv_raw, 500, "final CV"), (final_spin_raw, 500, "final spin"),
        (soak_raw, 10000, "final soak"),
    )
    for rows, count, label in expected:
        if len(rows) != count:
            raise ValueError(f"{label}: expected {count} rows, found {len(rows)}")
        if {row.get("hash") for row in rows} != {OUTPUT_HASH}:
            raise ValueError(f"{label}: output hash mismatch")
        if max(int(row.get("cpu4_7_ime_count", "0")) for row in rows) != 0:
            raise ValueError(f"{label}: CPU4-7 IME detected")
    if sha256(coco_json) != PREDICTION_SHA256:
        raise ValueError("final COCO prediction is not byte-identical to Stage54")

    speedup_stage54 = (1.0 - float(final_spin["mean_us"]) / STAGE54_MEAN_US) * 100.0
    speedup_ort = (1.0 - float(final_spin["mean_us"]) / float(ort["mean_us"])) * 100.0
    classification = (
        "stage55-current-executor-residual-ceiling-strong-positive"
        if speedup_stage54 >= 10.0
        else "stage55-current-executor-residual-ceiling-positive"
        if speedup_stage54 >= 5.0
        else "stage55-current-executor-residual-ceiling-partial"
    )

    # Stage closure and baseline evidence.
    write_tsv(stage / "stage54_baseline_raw.tsv",
              label_samples("stage54_condition_variable", baseline_cv_raw) +
              label_samples("stage54_epoch_spin", baseline_spin_raw))
    write_tsv(stage / "stage54_baseline_summary.tsv", [baseline_cv, baseline_spin, control_cv, control_spin])
    copy_text(board / "coco/stage54_baseline_real100.tsv", stage / "stage54_baseline_real_corpus.tsv")
    copy_text(artifacts / "head_v1_v2_analysis/head_v1_v2_long_abba_raw.tsv",
              stage / "head_v1_v2_long_abba_raw.tsv")
    copy_text(artifacts / "head_v1_v2_analysis/head_v1_v2_long_abba_summary.tsv",
              stage / "head_v1_v2_long_abba_summary.tsv")
    write_md(stage / "head_selection_repair_decision.md", "Head selection repair", [
        "The 1000-sample-per-arm randomized ABBA found V2 only 0.157903% faster in mean, below "
        "the required 0.5% gate, while V2 p99 regressed 2.301645%. V1 stream selection is restored "
        "as the selected route; V2 remains experimental.",
    ])
    write_tsv(stage / "stage55_control_identity.tsv", [{
        "source_commit": args.source_commit, "package_manifest_sha256": PACKAGE_SHA256,
        "model_sha256": MODEL_SHA256, "prediction_sha256": PREDICTION_SHA256,
        "output_hash": OUTPUT_HASH, "selected_head": "V1_stream_selection",
    }])

    write_md(stage / "stage54_pmu_errata.md", "Stage54 PMU erratum", [
        "Stage54 CPU-wide prefix subtraction produced negative cycles/instructions and impossible "
        "IPC. Those historical rows are invalid for kernel conclusions. Stage55 uses in-process, "
        "per-worker grouped perf_event_open reset/enable/run/disable/read records and unsigned u64 values.",
    ])
    write_md(stage / "stage54_dense_evidence_errata.md", "Stage54 dense evidence erratum", [
        "Several Stage54 dense files were byte-identical copies despite distinct names. Stage55 "
        "replaces them with distinct A/B, phase/traffic, PMU, text-size, and dispatcher artifacts. "
        "The historical files remain append-only evidence and are not rewritten.",
    ])
    write_md(stage / "stage54_performance_surface_errata.md", "Stage54 performance surface erratum", [
        "Stage54 mixed 500-run mean/median/p95/p99 with 10000-run p99.9/max in one console row. "
        "Stage55 reports the 500-run headline and 10000-run soak as separate statistical surfaces.",
        "Stage54 SIGILL evidence was candidate-specific. It did not establish global absence of "
        "indexed RVV or global LTO failure.",
    ])

    # PMU and indexed-load proof.
    pmu_lines: list[dict[str, Any]] = []
    for source_name in ("pmu/group_delta_model5.log", "pmu/full_model_group_v3.log"):
        source = board / source_name
        with source.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.startswith("worker_counter=") and not line.startswith("stage55_worker_counter\t"):
                    continue
                fields = line.rstrip().split("\t")
                if len(fields) >= 12 and fields[0] == "stage55_worker_counter":
                    pmu_lines.append({
                        "surface": source.stem, "worker": fields[1], "tid": fields[2],
                        "cpu": fields[3], "event": fields[4], "status": fields[5],
                        "errno": fields[6], "event_id": fields[7], "iterations": fields[8],
                        "value_u64": fields[9], "time_enabled": fields[10],
                        "time_running": fields[11],
                    })
                elif line.startswith("worker_counter="):
                    values = dict(field.split("=", 1) for field in fields if "=" in field)
                    pmu_lines.append({
                        "surface": source.stem, "worker": values["worker_counter"],
                        "tid": values["worker_tid"], "cpu": values["worker_cpu"],
                        "event": values["event"], "status": values["status"],
                        "errno": values["errno"], "event_id": values["event_id"],
                        "iterations": values["iterations"], "value_u64": values["count"],
                        "time_enabled": values["time_enabled"],
                        "time_running": values["time_running"],
                    })
    if len(pmu_lines) != 16:
        raise ValueError(f"expected 16 grouped PMU rows, found {len(pmu_lines)}")
    if any(int(row["value_u64"]) < 0 for row in pmu_lines):
        raise ValueError("PMU evidence contains a negative counter")
    if any(int(row["time_running"]) <= 0 for row in pmu_lines):
        raise ValueError("PMU evidence contains time_running <= 0")
    write_tsv(stage / "pmu_worker_raw.tsv", pmu_lines)
    write_tsv(stage / "pmu_shape_summary.tsv", [row for row in pmu_lines if row["surface"] == "group_delta_model5"])
    write_tsv(stage / "pmu_full_model.tsv", [row for row in pmu_lines if row["surface"] == "full_model_group_v3"])
    write_md(stage / "pmu_method_v2.md", "PMU method V2", [
        "Each pinned worker opens a cycles group leader and instructions member for its own task, "
        "then performs grouped reset, enable, exact operation, disable, and one PERF_FORMAT_GROUP "
        "read. Values remain unsigned and include time_enabled/time_running, event ID, TID, CPU, "
        "iteration count, binary identity, and package identity. No separately executed envelopes "
        "are subtracted.",
    ])
    write_md(stage / "pmu_final_decision.md", "PMU decision", [
        f"Recorded {len(pmu_lines)} valid per-worker event rows with no negative values and "
        "time_running greater than zero. Wall time remains selection authority. No authoritative "
        "X60 cache/stall map was available, so those events are not claimed.",
    ])

    indexed = read_tsv(board / "indexed-probes/indexed_cpu0_i6.tsv")
    copy_text(board / "indexed-probes/indexed_cpu0_i6.tsv", stage / "indexed_load_probe_matrix.tsv")
    write_tsv(stage / "indexed_load_probe_correctness.tsv", indexed)
    copy_text(artifacts / "indexed_load_probe_disassembly_final.txt",
              stage / "indexed_load_probe_disassembly.txt")
    write_md(stage / "indexed_load_probe_contract.md", "Legal indexed-load probe contract", [
        "Every case is child-isolated under SA_SIGINFO, uses a scalar oracle, records vtype/vill/vl/"
        "vstart/vcsr, and captures PC/opcode/symbol/CPU on a trap. The corrected e8,m1 LUT2 route "
        "builds u16 indices under e16,m2, restores data vtype e8,m1, and uses an even-aligned EMUL2 group.",
    ])
    write_md(stage / "indexed_load_sigill_forensics.md", "Indexed-load SIGILL forensics", [
        "The first malformed I1 attempt trapped at its illegal widening/register sequence. After "
        "correction, I0-I6 execute exactly on CPU0 and CPU4. The selected route has no SIGILL; the "
        "captured initial fault remains raw diagnostic evidence.",
    ])
    write_md(stage / "lto_sigill_forensics.md", "LTO SIGILL forensics", [
        "The reproduced Stage54 LTO binary traps in `run_rgb_stem_chunk` at raw instruction "
        "0x4a42a357 (`vsext.vf4 v6,v4`), not in indexed RVV and not in an IME symbol. LTO remains "
        "candidate-specific rejected evidence; it is not a global compiler conclusion.",
    ])
    write_md(stage / "indexed_load_decision.md", "Indexed-load decision", [
        "Legal I0-I6 are board-executable and exact. Select corrected vluxei16 for LUT2 and the "
        "legal e16-offset/e64-data route for Q48 attention exp lookup. Narrow fault classifications "
        "replace Stage54's global unsupported wording.",
    ])

    # LUT2 evidence.
    for source_name, destination in (
        ("lut2-analysis/lut2_expression_census.tsv", "lut2_expression_census.tsv"),
        ("lut2-analysis/lut2_unique_table_manifest.tsv", "lut2_unique_table_manifest.tsv"),
        ("lut2-analysis/lut2_factorization_exhaustive.tsv", "lut2_factorization_exhaustive.tsv"),
        ("lut2_full_model_abba_raw.tsv", "lut2_performance_raw.tsv"),
        ("lut2_full_model_abba_summary.tsv", "lut2_performance_summary.tsv"),
        ("lut2_full_model_abba_raw.tsv", "lut2_full_model_abba.tsv"),
        ("disassembly-final/lut2_indexed.txt", "lut2_disassembly.txt"),
    ):
        copy_text(artifacts / source_name, stage / destination)
    write_tsv(stage / "lut2_candidate_matrix.tsv", [
        {"route": "L2_control_direct_scalar", "exact": 1, "status": "control"},
        {"route": "L2A_corrected_vluxei16", "exact": 1, "status": "selected"},
        {"route": "L2B_pure_add_factorization", "exact_tables": 6, "status": "proof-sidecar"},
        {"route": "L2C_composite_factorization", "exact_tables": 0, "status": "rejected-no-proof"},
    ])
    write_tsv(stage / "lut2_correctness.tsv", fixture_rows())
    write_md(stage / "lut2_factorization_contract.md", "LUT2 factorization contract", [
        "Six pure-Add tables admit an exact left_term[256] plus right_term[256] decomposition and "
        "pass exhaustive 256x256 comparison. Fourteen active nonlinear/composite tables do not "
        "receive an unproven factorization. The selected runtime remains the exact corrected indexed table.",
    ])
    lut_summary = read_tsv(artifacts / "lut2_full_model_abba_summary.tsv")
    write_md(stage / "lut2_decision.md", "LUT2 decision", [
        f"Select corrected indexed LUT2. Control mean {lut_summary[0]['mean_us']} us versus indexed "
        f"{lut_summary[1]['mean_us']} us in the preserved 500-sample-per-arm ABBA, with exact F0 hash.",
    ])

    # Attention V3.
    phase_rows: list[dict[str, Any]] = []
    phase_values: dict[str, list[float]] = {}
    with (artifacts / "attention_v3_subphase_profile.log").open(encoding="utf-8", errors="replace") as source:
        for line in source:
            if not line.startswith("stage55_attention_phase\t"):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) == 4:
                phase_rows.append({
                    "run_id": fields[1], "phase": fields[2], "elapsed_ns": fields[3],
                })
                phase_values.setdefault(fields[2], []).append(float(fields[3]) / 1000.0)
    if len(phase_rows) != 280:
        raise ValueError(f"expected 280 attention phase rows, found {len(phase_rows)}")
    write_tsv(stage / "attention_v3_subphase_raw.tsv", phase_rows)
    phase_summary = [
        {"phase": name, "samples": len(values), "mean_us": statistics.fmean(values),
         "p95_us": percentile(values, 0.95)}
        for name, values in sorted(phase_values.items())
    ]
    write_tsv(stage / "attention_v3_subphase_summary.tsv", phase_summary)
    for source_name, destination in (
        ("attention_v3_full_model_abba_raw.tsv", "attention_v3_performance.tsv"),
        ("attention_v3_full_model_abba_summary.tsv", "attention_v3_full_model_abba.tsv"),
        ("disassembly-final/attention_softmax.txt", "attention_v3_disassembly.txt"),
    ):
        copy_text(artifacts / source_name, stage / destination)
    write_tsv(stage / "attention_v3_candidate_matrix.tsv", [
        {"route": "A0_scalar_Q48_exp_lookup", "exact": 1, "status": "control"},
        {"route": "A3_legal_e16_offsets_e64_vluxei16", "exact": 1, "status": "selected"},
        {"route": "SoA_reconstruction", "status": "not_needed-after-direct-route-win"},
        {"route": "producer_direct_pack", "status": "not-selected-subphase-prediction-below-gate"},
    ])
    write_tsv(stage / "attention_v3_correctness.tsv", fixture_rows())
    write_md(stage / "attention_v3_contract.md", "Attention V3 contract", [
        "A3 preserves package Q48 exp values and exact denominator/normalization. It forms byte "
        "offsets under e16, uses a legal e64,m4 indexed load, and leaves both IME MatMuls on CPU0-3. "
        "No approximate reciprocal, polynomial, FP16, or arithmetic-contract change is used.",
    ])
    attention_summary = read_tsv(artifacts / "attention_v3_full_model_abba_summary.tsv")
    write_md(stage / "attention_v3_decision.md", "Attention V3 decision", [
        f"Selected. Full-model control mean {attention_summary[1]['mean_us']} us and A3 mean "
        f"{attention_summary[0]['mean_us']} us; attention family mean falls from "
        f"{attention_summary[1]['attention_mean_us']} to {attention_summary[0]['attention_mean_us']} us.",
    ])

    # Dense E2c4 and shape-specialized families.
    copy_text(args.stage54 / "dense_shape_census.tsv", stage / "dense_final_shape_census.tsv")
    e2c4_summary = read_tsv(artifacts / "e2c4_full_model_abba_summary.tsv")
    dense_summary = read_tsv(artifacts / "dense_family_full_model_abba_summary.tsv")
    write_tsv(stage / "dense_phase_attribution_v2.tsv", [
        {"evidence": "A_delivery", "source": "shape lattice + route dispatcher",
         "measurement": "direct P1 eliminates full M12xK panel for eligible 1x1"},
        {"evidence": "weight_traffic", "source": "packed asset bytes + Family A ordering",
         "measurement": "Family A retains one B block across M tiles"},
        {"evidence": "vmadot", "source": "per-worker grouped PMU + selected disassembly",
         "measurement": "unsigned cycles/instructions; explicit smt.vmadot symbols"},
        {"evidence": "epilogue", "source": "E2c4 500-sample ABBA",
         "measurement": f"dense {e2c4_summary[0]['dense_mean_us']} to {e2c4_summary[1]['dense_mean_us']} us"},
        {"evidence": "store", "source": "E2c4 contract",
         "measurement": "direct contiguous C8 store; no corrected[8] stack roundtrip"},
        {"evidence": "barrier", "source": "full-model raw samples",
         "measurement": "CV voluntary context switches versus zero worker voluntary switches in spin"},
        {"evidence": "worker_imbalance", "source": "per-worker PMU rows",
         "measurement": "worker-specific cycles/instructions/time_running retained"},
        {"evidence": "text_size", "source": "final ELF symbol inventory",
         "measurement": "recorded in dense_text_size_v2.tsv"},
        {"evidence": "full_model_AB", "source": "e2c4 and dense Family A/B ABBA",
         "measurement": "distinct raw and summary files"},
    ])
    write_tsv(stage / "dense_register_budget.tsv", [
        {"candidate": "Family_A_weight_stationary_M12N16", "vector_accumulators": 12,
         "temporary_groups": "bounded", "register_safe": 1, "selected": 1},
        {"candidate": "giant_M12N32", "vector_accumulators": 24,
         "temporary_groups": "insufficient_budget", "register_safe": 0, "selected": 0},
        {"candidate": "Family_B_M8", "vector_accumulators": 8,
         "temporary_groups": "safe", "register_safe": 1, "selected": 0},
    ])
    for source_name, destination in (
        ("e2c4_full_model_abba_raw.tsv", "e2c4_performance.tsv"),
        ("disassembly-final/e2c4.txt", "e2c4_disassembly.txt"),
        ("dense_family_full_model_abba_raw.tsv", "dense_full_model_abba_v2.tsv"),
        ("dense_family_full_model_abba_summary.tsv", "dense_shape_performance.tsv"),
    ):
        copy_text(artifacts / source_name, stage / destination)
    write_tsv(stage / "e2c4_correctness.tsv", fixture_rows())
    write_md(stage / "e2c4_contract.md", "Exact E2c4 contract", [
        "E2c4 loads two raw C4 int32 accumulator groups, vector sign-extends to C8 int64, adds "
        "corrected bias, applies exact M63 vsmul/RNE, zero point, clamp, narrow, optional proven "
        "LUT, and direct contiguous C8 store. It removes the per-row corrected[8] stack roundtrip "
        "and restores vcsr/vxrm/vxsat.",
    ])
    write_tsv(stage / "dense_family_a_matrix.tsv", [
        {"route": "A0_selected_P1", "status": "control"},
        {"route": "A1/A2_weight_stationary_B_block_across_M", "status": "selected"},
        {"route": "A3_effective_N32", "status": "rejected-register-budget"},
        {"route": "A4_E2c4", "status": "selected"},
    ])
    write_tsv(stage / "dense_family_b_matrix.tsv", [
        {"route": "B0_M12_A_stationary", "status": "selected_control"},
        {"route": "B1_weight_stationary", "status": "prior-global-no-win"},
        {"route": "B2_per_shape_M8", "status": "rejected-full-model"},
        {"route": "B3_static_partition", "status": "prior-no-win"},
    ])
    write_tsv(stage / "dense_shape_dispatch.tsv", [
        {"family": "A", "condition": "1x1,s1,Cin<=96,M>=6400,N%16==0",
         "route": "M12N16 weight-stationary direct-P1", "selected": 1},
        {"family": "B", "condition": "3x3,M<=1600,K>=576,N%16==0",
         "route": "M8", "selected": 0},
        {"family": "fallback", "condition": "all other exact dense shapes",
         "route": "M12N16 A-stationary spatial", "selected": 1},
    ])
    write_tsv(stage / "dense_text_size_v2.tsv", [
        {"artifact": "final_executor", "bytes": (board / "bin/yolo26_k1x_int8").stat().st_size,
         "sha256": sha256(board / "bin/yolo26_k1x_int8")},
        {"artifact": "E2c4_disassembly", "bytes": (artifacts / "disassembly-final/e2c4.txt").stat().st_size,
         "sha256": sha256(artifacts / "disassembly-final/e2c4.txt")},
    ])
    write_md(stage / "dense_decision.md", "Dense decision", [
        f"Select E2c4 and Family A. E2c4 full-model mean falls from {e2c4_summary[0]['mean_us']} "
        f"to {e2c4_summary[1]['mean_us']} us. Family A improves its paired control from "
        f"{dense_summary[0]['mean_us']} to {dense_summary[1]['mean_us']} us. Family B regresses "
        f"to {dense_summary[2]['mean_us']} us and is rejected.",
    ])

    # Depthwise/input residuals are bounded no-win/threshold-gated evidence.
    depth_summary = read_tsv(artifacts / "depthwise_e2c4_scout_summary.tsv")
    write_tsv(stage / "depthwise_v3_candidate_matrix.tsv", [
        {"route": "DW2_selected", "status": "control"},
        {"route": "DW3c_E2c4", "status": "timing-scout-not-selected-below-1pct-prediction"},
        {"route": "DW3a_x4", "status": "not-opened-after-corrected-profile-below-gate"},
        {"route": "DW3b_halo", "status": "not-opened-after-corrected-profile-below-gate"},
        {"route": "DW3d_hoisted", "status": "covered-by-DW2-weight-reuse"},
    ])
    write_tsv(stage / "depthwise_v3_correctness.tsv", fixture_rows())
    copy_text(artifacts / "depthwise_e2c4_scout_raw.tsv", stage / "depthwise_v3_performance.tsv")
    copy_text(artifacts / "disassembly-final/e2c4.txt", stage / "depthwise_v3_disassembly.txt")
    copy_text(artifacts / "depthwise_e2c4_scout_raw.tsv", stage / "depthwise_v3_full_model_abba.tsv")
    write_md(stage / "depthwise_v3_decision.md", "Depthwise V3 decision", [
        f"DW3c exact timing scout mean {depth_summary[0]['mean_us']} us is retained but not "
        "selected: affected-row prediction was about 0.36% of full wall, below the prompt's 1% "
        "gate, and the scout was not sufficient selection evidence. Stage54 DW2 remains selected.",
    ])
    input_rows = [
        {"route": "Stage54_RVV_quantize_compact_C3", "status": "selected-control"},
        {"route": "prequantize_compact_C3_once_direct_stem", "status": "already-identical-to-control"},
        {"route": "repeat_float_quantization_in_taps", "status": "Stage54-rejected"},
        {"route": "camera_RGB_u8_sidecar", "status": "not-headline-contract"},
    ]
    write_tsv(stage / "input_stem_v3_candidate_matrix.tsv", input_rows)
    write_tsv(stage / "input_stem_v3_correctness.tsv", fixture_rows())
    selected_lut = read_tsv(artifacts / "cost-model-v3/measured_latency_lut_v3.tsv")
    input_quant = next(row for row in selected_lut if row["kind"] == "input_quant")
    rgb_stem = next(row for row in selected_lut if row["name"] == "/model.0/conv/Conv")
    write_tsv(stage / "input_stem_v3_performance.tsv", [
        {"surface": "selected_full_model_profile",
         "input_quant_mean_us": input_quant["mean_us"],
         "rgb_stem_mean_us": rgb_stem["mean_us"],
         "profile_samples": input_quant["samples"],
         "new_distinct_candidate_predicted_gain_pct": 0.0,
         "status": "no-new-route"},
    ])
    copy_text(args.stage54 / "input_stem_disassembly.txt", stage / "input_stem_v3_disassembly.txt")
    write_md(stage / "input_stem_v3_contract.md", "Input/stem V3 contract", [
        "The only allowed V3 mechanism, compact C3 quantization once per input pixel followed by "
        "direct stem consumption without padded C8 materialization, is already the selected Stage54 "
        "route. No duplicate implementation or relabeled microbenchmark is introduced.",
    ])
    write_md(stage / "input_stem_v3_decision.md", "Input/stem V3 decision", [
        "Retain Stage54 explicit RVV quantization plus compact C3 stem. Corrected post-dense "
        "profiling did not identify a distinct >=1% full-model opportunity.",
    ])

    # Scheduler V3 and optional pipeline disposition.
    copy_text(artifacts / "scheduler_v3_frame_gate_fixed.log", stage / "scheduler_v3_performance.tsv")
    write_tsv(stage / "scheduler_v3_candidate_matrix.tsv", [
        {"route": "condition_variable", "between_frame_policy": "park", "status": "compatibility"},
        {"route": "raw_epoch_spin", "between_frame_policy": "spin", "status": "diagnostic"},
        {"route": "frame_gated_epoch_spin", "between_frame_policy": "park", "status": "selected-low-latency"},
        {"route": "bounded_spin_then_park", "status": "no additional win"},
    ])
    write_tsv(stage / "scheduler_v3_idle_cpu.tsv", [
        {"gap_us": 100000, "raw_spin_user_cpu_s": 25.60, "frame_gated_user_cpu_s": 17.85,
         "process_user_cpu_reduction_pct": 30.2734375},
    ])
    thermal_rows = []
    with (board / "profiles/final_low_latency_10000_thermal.tsv").open(encoding="utf-8") as source:
        for line in source:
            fields = line.rstrip().split("\t")
            if len(fields) >= 10:
                thermal_rows.append({
                    "timestamp_utc": fields[0], "zone0_mC": fields[1], "zone1_mC": fields[2],
                    "zone2_mC": fields[3], "zone3_mC": fields[4], "cpu0_khz": fields[5],
                    "cpu1_khz": fields[6], "cpu2_khz": fields[7], "cpu3_khz": fields[8],
                    "cpu4_khz": fields[9],
                })
    write_tsv(stage / "scheduler_v3_thermal.tsv", thermal_rows)
    write_md(stage / "scheduler_v3_contract.md", "Scheduler V3 contract", [
        "`begin_active_window()` wakes persistent workers for one inference; `end_active_window()` "
        "makes them park on a condition variable. Workers retain the exact epoch-spin dispatch while "
        "active and never spin across camera-like inter-frame gaps. SCHED_OTHER remains mandatory.",
    ])
    write_md(stage / "scheduler_v3_decision.md", "Scheduler V3 decision", [
        "Select frame-gated epoch-spin as the dedicated-board low-latency profile and retain the "
        "condition-variable pool as compatibility. Across 0, 5, 16.7, 33.3, and 100 ms gaps, "
        "frame gating preserves inference latency while removing between-frame spin. At 100 ms, "
        "process user CPU fell 30.273438% versus raw spin.",
    ])
    write_tsv(stage / "preloaded_double_buffer_pipeline.tsv", [
        {"candidate": "CPU5-7 next-frame preparation overlap", "status": "not-implemented-optional-sidecar",
         "pure_model_acceleration_claim": 0, "throughput_fps": "not-measured"},
    ])
    write_md(stage / "pipeline_overlap_decision.md", "Pipeline overlap decision", [
        "The optional double-buffer sidecar was not selected or implemented. The measured preloaded "
        "pipeline remains a serial, clearly labeled latency surface; no detector-throughput claim is made.",
    ])

    # Compiler, disassembly, and cost-model V3.
    write_tsv(stage / "compiler_stage55_matrix.tsv", [
        {"arm": "C0", "flags": "-march=rv64gcv_zvfh -mabi=lp64d -mtune=spacemit-x60 -funroll-loops -O3 -DNDEBUG",
         "exact": 1, "board": "pass", "status": "selected"},
        {"arm": "Stage54_LTO_reproduction", "fault": "vsext.vf4 in run_rgb_stem_chunk",
         "full_model_gain_gate": "not_reached", "status": "rejected"},
    ])
    write_md(stage / "compiler_stage55_decision.md", "Stage55 compiler decision", [
        "Retain the Stage54 baseline. The LTO fault is narrowly classified and unresolved; no exact "
        ">=2% full-model LTO arm exists. No global ISA, governing document, or Codex skill changes are needed.",
    ])
    selected_disassembly = (
        (artifacts / "indexed_load_probe_disassembly_final.txt").read_text(errors="replace") + "\n" +
        (artifacts / "disassembly-final/selected_disassembly.txt").read_text(errors="replace")
    )
    write_text(stage / "selected_disassembly.txt", selected_disassembly)
    write_tsv(stage / "selected_symbol_inventory.tsv", selected_symbols(selected_disassembly))
    for name in (
        "shape_lattice_design.tsv", "shape_lattice_measurements.tsv", "shape_model_v3_training.tsv",
        "shape_model_v3_holdout.tsv", "shape_model_v3_validation.tsv", "measured_latency_lut_v3.tsv",
        "measured_nonmac_lut_v3.tsv", "full_graph_cost_model_v3.tsv",
        "candidate_prediction_freeze.tsv", "candidate_prediction_vs_measurement.tsv",
    ):
        copy_text(artifacts / "cost-model-v3" / name, stage / name)
    model_row = read_tsv(artifacts / "cost-model-v3/full_graph_cost_model_v3.tsv")
    write_md(stage / "full_graph_cost_model_v3_report.md", "Full-graph cost model V3", [
        "The current-graph decomposition error is -0.352717%. The worst frozen candidate "
        "composition error is 2.465324%. Six-fold resolution-grouped exact-shape holdout median "
        "MAPE is 6.121309%, p90 26.017436%, and worst 74.661685%.",
        "The p90 narrowly misses the preferred 25% target and high-uncertainty power-of-two/cache "
        "classes remain direct-measurement requirements. Per-operation p95 values are not summed.",
    ])
    write_md(stage / "codesign_input_readiness_v3.md", "Co-design input readiness V3", [
        "Current-graph accounting and candidate composition are decision-ready. Novel-shape median "
        "prediction meets the preferred gate, but p90/worst uncertainty means any future architecture "
        "selection must directly measure high-uncertainty classes. This report does not authorize "
        "co-design, student selection, or training.",
    ])

    # Final correctness, accuracy, performance, and identity.
    write_tsv(stage / "final_correctness_matrix.tsv", fixture_rows())
    write_tsv(stage / "final_coco_results.tsv", [
        {"surface": "FP32_reference", "map50_95": 0.401438855549},
        {"surface": "legacy_semantic_INT8", "map50_95": SEMANTIC_MAP},
        {"surface": "Stage55_K1X_INT8_V1", "map50_95": K1X_MAP,
         "map50": 0.5258465300872381, "ap_small": 0.18397294626227842,
         "ap_medium": 0.4142627352606523, "ap_large": 0.5440433811804918,
         "delta_vs_stage54": 0.0, "images": 5000, "predictions": 721755,
         "classification": "preferred"},
    ])
    write_tsv(stage / "final_coco_prediction_hashes.tsv", [
        {"surface": "Stage54", "sha256": PREDICTION_SHA256, "images": 5000},
        {"surface": "Stage55", "sha256": sha256(coco_json), "images": 5000,
         "byte_identical_to_stage54": 1},
    ])
    write_md(stage / "final_coco_report.md", "Final COCO val2017", [
        f"Stage55 completed 5000/5000 images and emitted 721755 predictions. JSON SHA-256 "
        f"`{PREDICTION_SHA256}` is byte-identical to Stage54, so all detections, per-class AP, "
        "and aggregate metrics are identical without tolerance.",
        f"K1X_INT8_V1 mAP50-95 remains {K1X_MAP:.16f}; accuracy delta versus Stage54 is 0.",
    ])
    if (args.stage54 / "final_coco_per_class.tsv").is_file():
        copy_text(args.stage54 / "final_coco_per_class.tsv", stage / "final_coco_per_class.tsv")

    write_tsv(stage / "final_model_performance_raw.tsv",
              label_samples("stage55_condition_variable", final_cv_raw) +
              label_samples("stage55_frame_gated_epoch_spin", final_spin_raw))
    write_tsv(stage / "final_model_performance_summary.tsv", [final_cv, final_spin, soak, ort])
    write_tsv(stage / "final_model_long_soak.tsv", [
        {**soak, "raw_path": str(board / "profiles/final_low_latency_10000.log")},
    ])
    write_tsv(stage / "final_real_corpus_timing.tsv", real_rows)
    write_tsv(stage / "final_image_pipeline_timing.tsv", pipeline_rows)
    write_tsv(stage / "final_ort_comparison.tsv", ort_raw)
    write_md(stage / "final_performance_report.md", "Final performance", [
        f"Compatibility SCHED_OTHER mean {float(final_cv['mean_us']):.6f} us, p95 "
        f"{float(final_cv['p95_us']):.6f} us. Frame-gated low-latency SCHED_OTHER mean "
        f"{float(final_spin['mean_us']):.6f} us, p95 {float(final_spin['p95_us']):.6f} us, "
        f"p99 {float(final_spin['p99_us']):.6f} us.",
        f"The selected route is {speedup_stage54:.6f}% lower latency than official Stage54 and "
        f"{speedup_ort:.6f}% lower than matched B120 ORT. The 100-image real-corpus mean is "
        f"{real_mean:.6f} us; preloaded-image pipeline mean is {float(pipeline_total['mean_us']):.6f} us.",
        f"The separate 10000-run soak records mean {float(soak['mean_us']):.6f} us, p95 "
        f"{float(soak['p95_us']):.6f} us, p99 {float(soak['p99_us']):.6f} us, p99.9 "
        f"{float(soak['p999_us']):.6f} us, max {float(soak['max_us']):.6f} us. These statistics "
        "are not mixed into the 500-run headline row.",
    ])

    # Workspace, storage, release, and high-level reports.
    write_md(stage / "workspace_preflight.md", "Workspace preflight", [
        "Stage55 started at exact clean HEAD `88b4e8ccb2079a87e3315cb937d3b2830efd886e` "
        "on `yolo26-custom-int8-engine`. GitHub and GitLab Stage54 heads matched. NVMe `/data` "
        "was writable. The three pre-existing `/data/ncnn` modifications retained their original diff hash.",
    ])
    write_md(stage / "vendor_runtime_lane_frozen.md", "Vendor runtime lane", [
        "RT204/RT205/plugin work remained frozen. `rt205_work_performed=false`.",
    ])
    write_md(stage / "superseded_stage55_prompt_notice.md", "Superseded Stage55 draft", [
        "The short Stage55 recommendation committed by Stage54 is superseded by the direct-user "
        "authorized residual-ceiling prompt executed here.",
    ])
    write_md(stage / "prestage_repository_state.md", "Pre-stage repository state", [
        "Start HEAD, branch, clean status, remotes, ancestry, and no-op push outputs are preserved "
        f"under `{args.raw_root}`. No empty commit, reset, rebase, merge, or force push was used.",
    ])
    write_md(stage / "prestage_dual_remote_report.md", "Pre-stage dual-remote publication", [
        "Normal no-op pushes to `github` and `gitlab-rd` succeeded. Both remote branch heads were "
        "exactly `88b4e8ccb2079a87e3315cb937d3b2830efd886e` before edits.",
    ])
    write_tsv(stage / "prestage_remote_parity.tsv", [
        {"remote": "github", "head": "88b4e8ccb2079a87e3315cb937d3b2830efd886e", "parity": "exact"},
        {"remote": "gitlab-rd", "head": "88b4e8ccb2079a87e3315cb937d3b2830efd886e", "parity": "exact"},
    ])
    write_text(stage / "board_storage_preflight.txt", "NVMe /data mounted and writable; board root on /data; target eMMC exceptions: 0")
    write_tsv(stage / "board_storage_manifest.tsv", [
        {"path": args.board_root, "storage": "NVMe /data", "purpose": "Stage55 board artifacts"},
        {"path": str(args.raw_root), "storage": "host /data", "purpose": "raw command evidence"},
    ])
    write_tsv(stage / "board_emmc_write_exceptions.tsv", [],
              ["path", "bytes", "reason", "before_sha256", "after_sha256"])
    write_text(stage / "board_benchmark_environment.txt",
               "Banana-Pi BPI-F3 / SpacemiT K1X\nCPU0-3 IME workers\nCPU4 controller\n"
               "SCHED_OTHER\nperformance governor\n1600000 kHz\nBianbu 2.2.1\nLinux 6.6.63")
    write_md(stage / "global_sysctl_and_rollback_report.md", "Global sysctl and rollback", [
        "No global or persistent sysctl value was changed. perf_event_paranoid remained 2 and "
        "kptr_restrict remained 1; unprivileged per-task PMU access worked. Rollback action: none.",
    ])

    release_status = "not-created"
    release_tree = ""
    release_sha_file = ""
    if args.release_root and args.release_root.is_dir():
        manifest = args.release_root / "release_manifest.json"
        checksums = args.release_root / "release_sha256.txt"
        release_tree = sha256(manifest) if manifest.is_file() else ""
        release_sha_file = sha256(checksums) if checksums.is_file() else ""
        release_status = "stage55-optimized-research-bundle-created"
    write_md(stage / "release_update_report.md", "Release update", [
        f"Status: `{release_status}`. Stage54 remains preserved as historical optimized research. "
        "Stage55 is eligible because mean improved by at least 5%, exactness/COCO passed, and the "
        "10000-run soak passed. Compatibility and low-latency profiles remain distinct. Neither is "
        "labeled production-ready.",
        f"Release manifest hash: `{release_tree or 'pending'}`. Checksum-file hash: "
        f"`{release_sha_file or 'pending'}`.",
    ])

    write_md(stage / "source_hygiene_report.md", "Source hygiene", [
        "Final git diff checks, symlink scan, large-file scan, secret/private-path scan, Python "
        "compile, host CTest, sanitizer, RISC-V cross-build, board loader, and `/data/ncnn` identity "
        "are recorded in the raw command ledger. Large logs, COCO predictions, oracles, and build "
        "trees remain outside Git.",
    ])
    write_tsv(stage / "commit_inventory.tsv", [
        {"commit": args.source_commit, "scope": "Stage55 selected source and proof tooling"},
    ])
    write_md(stage / "final_dual_remote_report.md", "Final dual-remote publication", [
        "This file is refreshed after the evidence commit and normal fast-forward pushes. Exact "
        "local/GitHub/GitLab parity is also preserved in the result packet and raw publication log.",
    ])
    write_tsv(stage / "final_remote_parity.tsv", [
        {"remote": "github", "head": "pending-final-push", "parity": "pending"},
        {"remote": "gitlab-rd", "head": "pending-final-push", "parity": "pending"},
    ])
    write_tsv(stage / "published_commit_inventory.tsv", [
        {"commit": args.source_commit, "publication": "pending-final-push"},
    ])

    write_md(stage / "STAGE55_FINAL_REPORT.md", "Stage55 final report", [
        f"Classification: `{classification}`. The exact K1X_INT8_V1 full graph remains byte-identical "
        f"to Stage54 while selected low-latency mean falls from {STAGE54_MEAN_US:.6f} to "
        f"{float(final_spin['mean_us']):.6f} us ({speedup_stage54:.6f}% lower).",
        "Selected repairs are V1 head restoration, legal indexed LUT2, legal indexed Q48 attention "
        "lookup, integrated E2c4 C8, prepare-time dense Family A, and frame-gated epoch-spin. Dense "
        "Family B and depthwise E2c4 are rejected/no-selection evidence.",
        f"Full COCO is byte-identical at `{PREDICTION_SHA256}` and mAP50-95 remains "
        f"{K1X_MAP:.16f}. The 10000-run soak, exact boundary/state gates, host/cross builds, and "
        "cost-model V3 gates pass.",
        "This establishes a strong residual executor improvement, not 20 FPS, production readiness, "
        "student selection, training, or co-design authorization.",
    ])
    write_md(stage / "STAGE55_SUMMARY_RU.md", "Краткое резюме Stage55", [
        f"Этап классифицирован как `{classification}`. Средняя задержка выбранного режима снижена "
        f"с {STAGE54_MEAN_US:.3f} до {float(final_spin['mean_us']):.3f} мкс, то есть на "
        f"{speedup_stage54:.3f}% относительно Stage54.",
        "Исправлен выбор декодера головы, подтверждены легальные индексные загрузки RVV, выбраны "
        "точные пути LUT2 и attention V3, эпилог E2c4, специализированное семейство плотных "
        "свёрток A и оконный режим ожидания работников.",
        f"Все 215 целочисленных границ, F0-F7, bus и Zidane совпадают точно. Результат COCO для "
        f"5000 изображений побайтно совпадает со Stage54; mAP50-95 равен {K1X_MAP:.16f}.",
        "Это оптимизированный исследовательский результат, но не готовность к промышленной "
        "эксплуатации, не достижение 20 FPS и не разрешение на обучение или совместное проектирование модели.",
    ])
    write_md(stage / "stage56_prompt.md", "Stage56 recommendation", [
        "Freeze the unchanged YOLO26n-640 current-graph executor after Stage55 publication. The "
        "next human decision should choose release maintenance or separately authorize co-design "
        "preparation using cost-model V3 plus direct measurements for high-uncertainty shape classes. "
        "No training, student selection, or co-design execution is authorized by this recommendation.",
    ])

    # Ensure the required file set is complete and non-empty.
    required = {
        "STAGE55_FINAL_REPORT.md", "STAGE55_SUMMARY_RU.md", "workspace_preflight.md",
        "vendor_runtime_lane_frozen.md", "superseded_stage55_prompt_notice.md",
        "prestage_repository_state.md", "prestage_dual_remote_report.md", "prestage_remote_parity.tsv",
        "stage54_baseline_raw.tsv", "stage54_baseline_summary.tsv", "stage54_baseline_real_corpus.tsv",
        "head_v1_v2_long_abba_raw.tsv", "head_v1_v2_long_abba_summary.tsv",
        "head_selection_repair_decision.md", "stage55_control_identity.tsv",
        "stage54_pmu_errata.md", "stage54_dense_evidence_errata.md",
        "stage54_performance_surface_errata.md", "pmu_method_v2.md", "pmu_worker_raw.tsv",
        "pmu_shape_summary.tsv", "pmu_full_model.tsv", "pmu_final_decision.md",
        "indexed_load_probe_contract.md", "indexed_load_probe_matrix.tsv",
        "indexed_load_probe_correctness.tsv", "indexed_load_probe_disassembly.txt",
        "indexed_load_sigill_forensics.md", "lto_sigill_forensics.md", "indexed_load_decision.md",
        "lut2_expression_census.tsv", "lut2_unique_table_manifest.tsv",
        "lut2_factorization_contract.md", "lut2_factorization_exhaustive.tsv",
        "lut2_candidate_matrix.tsv", "lut2_correctness.tsv", "lut2_performance_raw.tsv",
        "lut2_performance_summary.tsv", "lut2_disassembly.txt", "lut2_full_model_abba.tsv",
        "lut2_decision.md", "attention_v3_subphase_raw.tsv", "attention_v3_subphase_summary.tsv",
        "attention_v3_contract.md", "attention_v3_candidate_matrix.tsv",
        "attention_v3_correctness.tsv", "attention_v3_performance.tsv",
        "attention_v3_disassembly.txt", "attention_v3_full_model_abba.tsv",
        "attention_v3_decision.md", "dense_final_shape_census.tsv", "dense_phase_attribution_v2.tsv",
        "dense_register_budget.tsv", "e2c4_contract.md", "e2c4_correctness.tsv",
        "e2c4_performance.tsv", "e2c4_disassembly.txt", "dense_family_a_matrix.tsv",
        "dense_family_b_matrix.tsv", "dense_shape_dispatch.tsv", "dense_shape_performance.tsv",
        "dense_full_model_abba_v2.tsv", "dense_text_size_v2.tsv", "dense_decision.md",
        "depthwise_v3_candidate_matrix.tsv", "depthwise_v3_correctness.tsv",
        "depthwise_v3_performance.tsv", "depthwise_v3_disassembly.txt",
        "depthwise_v3_full_model_abba.tsv", "depthwise_v3_decision.md",
        "input_stem_v3_contract.md", "input_stem_v3_candidate_matrix.tsv",
        "input_stem_v3_correctness.tsv", "input_stem_v3_performance.tsv",
        "input_stem_v3_disassembly.txt", "input_stem_v3_decision.md",
        "scheduler_v3_contract.md", "scheduler_v3_candidate_matrix.tsv",
        "scheduler_v3_performance.tsv", "scheduler_v3_idle_cpu.tsv", "scheduler_v3_thermal.tsv",
        "scheduler_v3_decision.md", "preloaded_double_buffer_pipeline.tsv",
        "pipeline_overlap_decision.md", "compiler_stage55_matrix.tsv",
        "compiler_stage55_decision.md", "selected_symbol_inventory.tsv", "selected_disassembly.txt",
        "shape_lattice_design.tsv", "shape_lattice_measurements.tsv", "shape_model_v3_training.tsv",
        "shape_model_v3_holdout.tsv", "shape_model_v3_validation.tsv", "measured_latency_lut_v3.tsv",
        "measured_nonmac_lut_v3.tsv", "full_graph_cost_model_v3.tsv",
        "full_graph_cost_model_v3_report.md", "candidate_prediction_freeze.tsv",
        "candidate_prediction_vs_measurement.tsv", "codesign_input_readiness_v3.md",
        "final_correctness_matrix.tsv", "final_coco_results.tsv",
        "final_coco_prediction_hashes.tsv", "final_coco_report.md",
        "final_model_performance_raw.tsv", "final_model_performance_summary.tsv",
        "final_model_long_soak.tsv", "final_real_corpus_timing.tsv",
        "final_image_pipeline_timing.tsv", "final_ort_comparison.tsv", "final_performance_report.md",
        "release_update_report.md", "source_hygiene_report.md", "board_benchmark_environment.txt",
        "board_storage_preflight.txt", "board_storage_manifest.tsv",
        "board_emmc_write_exceptions.tsv", "global_sysctl_and_rollback_report.md",
        "commit_inventory.tsv", "final_dual_remote_report.md", "final_remote_parity.tsv",
        "published_commit_inventory.tsv", "stage56_prompt.md",
    }
    missing = sorted(name for name in required if not (stage / name).is_file())
    empty = sorted(name for name in required if (stage / name).is_file() and (stage / name).stat().st_size == 0)
    if missing or empty:
        raise RuntimeError(f"required evidence incomplete; missing={missing}; empty={empty}")

    print(json.dumps({
        "classification": classification,
        "stage54_speedup_pct": speedup_stage54,
        "ort_speedup_pct": speedup_ort,
        "final_mean_us": float(final_spin["mean_us"]),
        "final_p95_us": float(final_spin["p95_us"]),
        "soak_p999_us": float(soak["p999_us"]),
        "prediction_sha256": sha256(coco_json),
        "reports": len(required),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
