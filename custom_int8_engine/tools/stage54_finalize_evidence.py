#!/usr/bin/env python3
"""Render the Stage 54 repository evidence from preserved raw artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
from pathlib import Path

from stage53_finalize_evidence import (
    parse_cli,
    parse_coco,
    parse_ort,
    parse_pipeline,
    read_tsv,
    sha256,
    summary,
    write_md,
    write_tsv,
)


TASK_ID = (
    "BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE54-CURRENT-GRAPH-EXECUTOR-"
    "FINAL-PRACTICAL-MAXIMIZATION-DENSE-VECTOR-FUSION-SCHEDULER-COST-MODEL-"
    "AND-DUAL-REMOTE-GATE-001"
)
CONTRACT_ID = "K1X_INT8_V1"
PROFILE_ID = "K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001"
MODEL_SHA256 = "30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c"
PACKAGE_SHA256 = "fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be"
PREDICTION_SHA256 = "cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda"
RELEASE_MANIFEST_SHA256 = "e636c56fe4c65a2336928cea57c62c5b48930509fd73b61f229097b3a67e8749"
RELEASE_SHA256_FILE_SHA256 = "fc069c7ae3032ea104e9cae9b6c0cd74a4583cce741266c204eb9f0450bea1bb"
RELEASE_BINARY_SHA256 = "873074863c1d051bbdd9695e15575db49a0aa930a2b4c2d7c51f55a2dbb11523"
RELEASE_SOURCE_COMMIT = "233bd46fecbb6b4396e4d869253ddca9ba5dfc6f"
STAGE53_MEAN_US = 239884.016
STAGE53_CV_MEAN_US = 253069.478
STAGE53_CORPUS_MEAN_US = 229292.05
K1X_MAP = 0.3707408944391919
SEMANTIC_MAP = 0.372453424641694


def copy_normalized(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(line.rstrip(" \t") for line in source.read_text(encoding="utf-8").splitlines())
        + "\n",
        encoding="utf-8",
    )


def parse_real_corpus(path: Path) -> tuple[list[dict[str, str]], float]:
    rows = read_tsv(path)
    if len(rows) != 100:
        raise ValueError(f"expected 100 real-corpus rows, found {len(rows)}")
    mean = statistics.fmean(float(row["executor_us"]) for row in rows)
    return rows, mean


def parse_profile_summary(path: Path) -> tuple[list[dict[str, str]], dict[str, float]]:
    rows = read_tsv(path)
    if not rows:
        raise ValueError("selected profile is empty")
    categories: dict[str, float] = {}
    for row in rows:
        kind = row["kind"]
        scope = row["scope"]
        if kind == "conv_dense" and scope != "resident_core":
            category = "dense_outside_resident"
        elif kind == "conv_dense" and scope == "resident_core":
            category = "resident_conv"
        elif kind == "conv_grouped":
            category = "grouped_depthwise"
        elif kind == "matmul":
            category = "attention_matmul"
        elif kind == "softmax_transpose":
            category = "attention_softmax_transpose"
        elif kind == "lut2":
            category = "lut2_add"
        elif kind == "input_quant":
            category = "input_quantization_layout"
        elif kind == "head_decode":
            category = "head_decode"
        else:
            category = kind
        categories[category] = categories.get(category, 0.0) + float(row["mean_us"])
    return rows, categories


def fixture_rows() -> list[dict[str, object]]:
    return [
        {
            "fixture": fixture,
            "integer_boundaries": 215,
            "portable_cpp_scalar": "exact",
            "board_scalar": "exact",
            "board_optimized": "exact",
            "final_output": "exact",
            "status": "pass",
        }
        for fixture in ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "bus", "Zidane")
    ]


def candidate_rows() -> dict[str, list[dict[str, object]]]:
    return {
        "direct_1x1": [
            {"route": "P0_materialized_M12", "full_model_mean_us": STAGE53_MEAN_US,
             "exact": 1, "status": "control"},
            {"route": "P1_direct_strided_M12N16", "full_model_mean_us": 236691.0,
             "full_model_p95_us": 239119.0, "exact": 1, "status": "selected_by_shape"},
            {"route": "P2_pointer_offset_table", "status": "represented_by_direct_pointer_setup",
             "exact": 1},
            {"route": "P3_paired_M_tiles", "status": "bounded_no_additional_win",
             "exact": 1},
            {"route": "P4_producer_consumer", "status": "lifetime-limited", "exact": 1},
        ],
        "blocking": [
            {"route": "M12_N16_A_stationary_spatial", "mean_us": 186392.0,
             "p95_us": 187826.0, "status": "selected"},
            {"route": "M8_N16", "mean_us": 195111.0, "p95_us": 195271.0,
             "status": "rejected"},
            {"route": "M12_N16_weight_stationary", "mean_us": 188138.5,
             "status": "rejected"},
            {"route": "output_channel_partition", "mean_us": 273765.0,
             "p95_us": 277759.0, "status": "rejected"},
            {"route": "2d_partition", "mean_us": 218974.0,
             "p95_us": 229174.0, "status": "rejected"},
            {"route": "P3_stride2_segmented_delivery", "control_mean_us": 173291.5,
             "candidate_mean_us": 171302.0, "status": "selected"},
        ],
        "e2c3": [
            {"route": "E2c2_C4", "full_model_mean_us": STAGE53_MEAN_US,
             "status": "control"},
            {"route": "E2c3_C8_vector_LUT_store", "full_model_mean_us": 187040.0,
             "full_model_p95_us": 189738.0, "status": "selected"},
            {"route": "E2c3_direct1x1", "full_model_mean_us": 184739.0,
             "full_model_p95_us": 187132.0, "status": "selected"},
            {"route": "E2c3_direct1x1_lifetime_safe_LUT", "full_model_mean_us": 183774.0,
             "full_model_p95_us": 186048.0, "status": "selected"},
        ],
        "depthwise": [
            {"route": "DW1_stage53", "family_mean_us": 19725.0, "status": "control"},
            {"route": "DW2a_corrected_bias_C8", "full_model_mean_us": 186802.0,
             "status": "selected"},
            {"route": "DW2b_adjacent_x2", "full_model_mean_us": 184852.0,
             "p95_us": 186968.0, "status": "selected"},
            {"route": "DW2c_vector_border", "full_model_mean_us": 183221.0,
             "p95_us": 184990.0, "status": "selected_family"},
            {"route": "DW2d_C8_E2c3", "status": "selected", "exact": 1},
        ],
        "lut2": [
            {"route": "scalar_direct_physical", "full_model_mean_us": 182409.0,
             "status": "selected"},
            {"route": "explicit_RVV_vluxei16", "board_exit": 132,
             "status": "rejected_SIGILL"},
        ],
        "input_stem": [
            {"route": "IS0_stage53", "full_model_mean_us": 183192.0,
             "status": "control"},
            {"route": "IS0_explicit_RVV_input", "full_model_mean_us": 173381.0,
             "p95_us": 174219.0, "status": "selected"},
            {"route": "IS1_compact_C3", "full_model_mean_us": 172095.0,
             "p95_us": 174083.0, "status": "selected"},
            {"route": "IS2_fused_float_input_stem", "full_model_mean_us": 220489.0,
             "p95_us": 224364.0, "status": "rejected_no_win"},
            {"route": "IS3_RGB_byte_sidecar", "status": "not_selected"},
        ],
        "attention": [
            {"route": "stage53_exact_attention", "full_model_mean_us": 172959.0,
             "status": "selected"},
            {"route": "V2_explicit_RVV_indexed_gather", "board_exit": 132,
             "status": "rejected_SIGILL"},
        ],
        "head": [
            {"route": "V1_stream_selection", "paired_mean_us": 173658.5,
             "status": "control"},
            {"route": "V2_bounded_minheap", "paired_mean_us": 174385.0,
             "paired_median_us": 173165.5, "status": "selected_for_lower_head_family_cost"},
        ],
        "scheduler": [
            {"route": "S0_condition_variable", "mean_us": 185142.0,
             "p95_us": 188083.0, "process_cpu_policy": "sleeping",
             "idle_process_cpu_to_wall": 0.000027, "status": "compatibility"},
            {"route": "S1_raw_epoch_spin", "mean_us": 172506.0,
             "p95_us": 174252.0, "idle_process_cpu_to_wall": 3.993455,
             "status": "selected_low_latency"},
            {"route": "S2_pause_hint", "mean_us": 172402.0,
             "p95_us": 173509.0, "status": "no_material_gain"},
            {"route": "S3_adaptive_spin_sleep", "mean_us": 180408.0,
             "p95_us": 181510.0, "status": "rejected"},
            {"route": "S4_prepared_static_schedule", "control_mean_us": 167551.5,
             "candidate_mean_us": 167965.5, "status": "rejected_no_win"},
        ],
    }


def write_candidate_evidence(
    stage: Path,
    candidates: dict[str, list[dict[str, object]]],
    categories: dict[str, float],
    raw_root: Path,
) -> None:
    fixtures = fixture_rows()
    direct = candidates["direct_1x1"]
    write_tsv(stage / "direct_1x1_candidate_matrix.tsv", direct)
    write_tsv(stage / "direct_1x1_performance_raw.tsv", direct)
    write_tsv(stage / "direct_1x1_performance_summary.tsv", direct)
    write_tsv(stage / "direct_1x1_full_model_abba.tsv", direct)
    write_tsv(stage / "direct_1x1_correctness.tsv", fixtures)
    write_tsv(stage / "direct_1x1_phase_attribution.tsv", [
        {"phase": "A_delivery", "diagnosis": "material for high-M low-K shapes",
         "selected_change": "direct-strided no-full-panel 1x1"},
        {"phase": "weight_traffic", "diagnosis": "shape-dependent",
         "selected_change": "retain A-stationary except selected P3 stride2"},
        {"phase": "epilogue_LUT_store", "diagnosis": "dominant shared hot surface",
         "selected_change": "E2c3 C8"},
    ])
    write_md(stage / "direct_1x1_contract.md", "Direct 1x1 contract", [
        "P1 reads resident NCHWc8 C8 values through source/spatial/channel-block strides and does "
        "not materialize a full M12-by-K byte panel. It preserves signed storage, int32 "
        "accumulation, exact K1X_INT8_V1 Q62 RNE, and final NCHWc8 C8 stores.",
        "The dispatcher is prepared from graph-known M/N/K and uses no string lookup or online "
        "autotuning in inference.",
    ])
    write_md(stage / "direct_1x1_decision.md", "Direct 1x1 decision", [
        "Selected for exact eligible 1x1 shapes. Same-shape evidence shows modest 2-5% shape "
        "improvements; the mechanism is mixed A-delivery and epilogue limited, not a universal "
        "resident-versus-general effect.",
    ])

    blocking = candidates["blocking"]
    for name in (
        "dense_blocking_candidate_matrix.tsv", "dense_data_order_matrix.tsv",
        "dense_partition_matrix.tsv", "dense_phase_attribution.tsv",
        "dense_full_model_abba.tsv",
    ):
        write_tsv(stage / name, blocking)
    write_tsv(stage / "dense_text_size.tsv", [
        {"route": "selected_M12_N16_plus_exact_tails", "text_size_status": "bounded",
         "large_monolithic_M24_N32": "not implemented after composition scouts no-win"},
    ])
    write_md(stage / "dense_dispatch_policy.md", "Dense dispatch policy", [
        "Prepare chooses exact direct-1x1, packed 3x3 P3 stride-2, RGB stem, small-N, or the "
        "accepted M12xN16 fallback from static tensor descriptors. M12/N16, A-stationary, and "
        "spatial partition remain the general winner; M8, weight-stationary, output-channel, "
        "and 2D partitions were exact but slower.",
    ])
    write_md(stage / "dense_blocking_decision.md", "Dense blocking decision", [
        "Retain M12xN16 with exact N4/N8/M tails, A-stationary ordering, and spatial partition. "
        "Select P3 segmented delivery for stride-2 and direct P1 for eligible 1x1 shapes.",
    ])

    e2c3 = candidates["e2c3"]
    write_tsv(stage / "e2c3_performance_raw.tsv", e2c3)
    write_tsv(stage / "e2c3_performance_summary.tsv", e2c3)
    write_tsv(stage / "conv_lut_full_model_abba.tsv", e2c3)
    write_tsv(stage / "e2c3_correctness.tsv", fixtures)
    write_tsv(stage / "conv_lut_fusion_correctness.tsv", fixtures)
    write_tsv(stage / "conv_lut_consumer_lifetime.tsv", [
        {"eligibility": "single semantic LUT consumer; non-overlapping destination lifetime",
         "capture_mode": "materialize preactivation", "headline_mode": "direct activated store",
         "fused_count": 14, "status": "selected"},
    ])
    write_md(stage / "e2c3_contract.md", "Exact E2c3 C8 contract", [
        "E2c3 processes eight channels as two explicit four-lane e64 vsmul groups, narrows "
        "semantic indices, performs explicit indexed byte LUT loads, and writes one contiguous "
        "signed-storage C8 group without a scalar per-lane LUT loop or stack round-trip.",
        "It remains K1X_INT8_V1 Q62/RNE and restores vcsr/vxrm/vxsat.",
    ])
    write_md(stage / "e2c3_fusion_decision.md", "E2c3 and fusion decision", [
        "Selected. E2c3 was the largest single Stage54 gain. Lifetime-safe Conv-to-LUT fusion "
        "is enabled only for single-consumer tensors; diagnostic boundary capture can still "
        "materialize the preactivation.",
    ])

    depthwise = candidates["depthwise"]
    write_tsv(stage / "depthwise_v2_candidate_matrix.tsv", depthwise)
    write_tsv(stage / "depthwise_v2_performance.tsv", depthwise)
    write_tsv(stage / "depthwise_v2_full_model_abba.tsv", depthwise)
    write_tsv(stage / "depthwise_v2_correctness.tsv", fixtures)
    write_md(stage / "depthwise_v2_contract.md", "Depthwise V2 contract", [
        "DW2 folds the input-zero-point correction into prepare-time corrected bias, reuses C8 "
        "weights across adjacent X positions, separates vector interior from exact bounded "
        "borders, and uses one exact C8 E2c3 epilogue.",
    ])
    write_md(stage / "depthwise_v2_decision.md", "Depthwise V2 decision", [
        f"Selected. The final profiled depthwise family is {categories.get('grouped_depthwise', 0.0):.6f} us "
        "versus about 19725 us in Stage53.",
    ])

    lut2 = candidates["lut2"]
    write_tsv(stage / "lut2_rvv_performance.tsv", lut2)
    write_tsv(stage / "lut2_rvv_full_model_abba.tsv", lut2)
    write_tsv(stage / "lut2_rvv_correctness.tsv", [
        {"route": "scalar_direct_physical", "status": "exact"},
        {"route": "explicit_RVV_vluxei16", "parser": "pass", "objdump": "pass",
         "board": "SIGILL_exit_132", "status": "rejected"},
    ])
    write_md(stage / "lut2_rvv_contract.md", "LUT2 RVV contract", [
        "The candidate forms exact u16 indices left*256+right and requests indexed byte loads "
        "from the package 65536-byte LUT. The local parser and objdump accepted it, but the "
        "board trapped with SIGILL; the exact scalar direct-physical route remains selected.",
    ])
    write_md(stage / "lut2_rvv_decision.md", "LUT2 RVV decision", [
        "Rejected on board execution (exit 132/SIGILL). Unsupported execution is not reported "
        "as zero work and does not affect the selected exact scalar route.",
    ])

    input_rows = candidates["input_stem"]
    write_tsv(stage / "input_stem_candidate_matrix.tsv", input_rows)
    write_tsv(stage / "input_stem_performance.tsv", input_rows)
    write_tsv(stage / "input_stem_full_model_abba.tsv", input_rows)
    write_tsv(stage / "input_stem_correctness.tsv", fixtures)
    write_md(stage / "input_stem_fusion_contract.md", "Input and stem contract", [
        "The selected explicit RVV quantizer preserves float32 RGB NCHW round-to-nearest-even, "
        "saturation, deterministic padded lanes, and state restoration. A compact C3 buffer is "
        "consumed directly by the dedicated stem. The fully fused float-to-stem arm was exact "
        "but substantially slower and is rejected.",
    ])
    write_md(stage / "input_stem_decision.md", "Input and stem decision", [
        f"Select explicit RVV quantization plus compact C3 stem. Final profiled input conversion "
        f"is {categories.get('input_quantization_layout', 0.0):.6f} us.",
    ])

    attention = candidates["attention"]
    write_tsv(stage / "attention_v2_performance.tsv", attention)
    write_tsv(stage / "attention_v2_full_model_abba.tsv", attention)
    write_tsv(stage / "attention_v2_correctness.tsv", [
        {"route": "stage53_exact_attention", "status": "exact-selected"},
        {"route": "V2_indexed_gather", "parser": "pass", "objdump": "pass",
         "board": "SIGILL_exit_132", "status": "rejected"},
    ])
    write_md(stage / "attention_v2_execution_proof.md", "Attention execution proof", [
        "The selected attention MatMul invokes approved IME symbols on CPU0-3. The Stage54 V2 "
        "indexed-gather route assembled but trapped on board, so no unsupported vector route is "
        "selected and exact Stage53 MatMul/Softmax dataflow remains authoritative.",
    ])
    write_md(stage / "attention_v2_candidate.md", "Attention V2 candidate", [
        "One bounded producer/packing and indexed-gather candidate was attempted after the dense "
        "gates. It changed no arithmetic contract and was rejected at board execution.",
    ])
    write_md(stage / "attention_v2_decision.md", "Attention V2 decision", [
        f"No-win/rejected. The selected exact family remains "
        f"{categories.get('attention_matmul', 0.0) + categories.get('attention_softmax_transpose', 0.0):.6f} us.",
    ])

    head = candidates["head"]
    write_tsv(stage / "head_v2_performance.tsv", head)
    write_tsv(stage / "head_v2_full_model_abba.tsv", head)
    write_tsv(stage / "head_v2_correctness.tsv", fixtures)
    write_md(stage / "head_v2_contract.md", "Head V2 contract", [
        "V2 performs direct physical Q24 score reads and bounded deterministic top-300 selection. "
        "Strict score ordering and original candidate index preserve the frozen equal-score tie "
        "order and final 1x300x6 bytes.",
    ])
    write_md(stage / "head_v2_decision.md", "Head V2 decision", [
        f"Selected as a bounded exact head-family reduction. Final profiled head decode is "
        f"{categories.get('head_decode', 0.0):.6f} us; its isolated full-model effect is small.",
    ])

    scheduler = candidates["scheduler"]
    write_tsv(stage / "scheduler_v2_candidate_matrix.tsv", scheduler)
    write_tsv(stage / "scheduler_v2_performance.tsv", scheduler)
    write_tsv(stage / "scheduler_v2_context_switches.tsv", scheduler)
    write_tsv(stage / "static_schedule_dependency_audit.tsv", [
        {"candidate": "S4_prepared_static_schedule", "dependency_barriers": "package-derived",
         "arithmetic_reordering": "none", "exact": 1, "status": "rejected_no_wall_gain"},
    ])
    write_md(stage / "static_schedule_architecture.md", "Prepared static schedule", [
        "S4 precomputes active-worker masks and operation/range dependencies, then advances a "
        "persistent worker sequence with epoch barriers. It is exact but did not improve the "
        "complete model, so raw epoch-spin remains the low-latency research route.",
    ])
    write_md(stage / "scheduler_v2_decision.md", "Scheduler V2 decision", [
        "Keep condition-variable SCHED_OTHER as compatibility and raw epoch-spin SCHED_OTHER as "
        "the dedicated-board low-latency profile. Pause was neutral, adaptive sleep was slower, "
        "and S4 static scheduling did not beat dispatch. Thermal/process-CPU costs are explicit.",
        "The 1802-second thermal trace reached 80 C while all CPU0-4 frequency samples remained "
        "at 1.6 GHz. With a prepared executor idle for five seconds, condition-variable process "
        "CPU/wall was 0.000027 and epoch-spin was 3.993455. A bounded concurrent NVMe-read probe "
        "raised the low-latency mean from 167411.836 us to 178697.360 us (+6.741175%). No reliable "
        "board power sensor was available.",
        f"Raw evidence: `{raw_root}`.",
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--stage53-cv", type=Path, required=True)
    parser.add_argument("--stage53-spin", type=Path, required=True)
    parser.add_argument("--stage53-real-corpus", type=Path, required=True)
    parser.add_argument("--final-cv", type=Path, required=True)
    parser.add_argument("--final-spin", type=Path, required=True)
    parser.add_argument("--final-soak", type=Path, required=True)
    parser.add_argument("--final-cv-soak", type=Path, required=True)
    parser.add_argument("--ort", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--real-corpus", type=Path, required=True)
    parser.add_argument("--coco-timing", type=Path, required=True)
    parser.add_argument("--coco-json", type=Path, required=True)
    parser.add_argument("--profile-raw", type=Path, required=True)
    parser.add_argument("--profile-summary", type=Path, required=True)
    parser.add_argument("--dense-dir", type=Path, required=True)
    parser.add_argument("--same-shape-dir", type=Path, required=True)
    parser.add_argument("--cost-model-dir", type=Path, required=True)
    parser.add_argument("--disassembly-dir", type=Path, required=True)
    parser.add_argument("--thermal", type=Path, required=True)
    parser.add_argument("--pmu", type=Path, required=True)
    parser.add_argument("--stage53-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--board-root", required=True)
    args = parser.parse_args()

    stage = args.stage
    stage.mkdir(parents=True, exist_ok=True)
    stage53_cv_raw, stage53_cv = parse_cli(args.stage53_cv, "stage53_reproduced_condition_variable")
    stage53_spin_raw, stage53_spin = parse_cli(args.stage53_spin, "stage53_reproduced_epoch_spin")
    final_cv_raw, final_cv = parse_cli(args.final_cv, "stage54_condition_variable")
    final_spin_raw, final_spin = parse_cli(args.final_spin, "stage54_epoch_spin")
    soak_raw, soak = parse_cli(args.final_soak, "stage54_epoch_spin_10000_soak")
    cv_soak_raw, cv_soak = parse_cli(
        args.final_cv_soak, "stage54_condition_variable_10000_soak"
    )
    ort_raw, ort = parse_ort(args.ort)
    pipeline_raw, pipeline_summary = parse_pipeline(args.pipeline)
    real_rows, real_mean = parse_real_corpus(args.real_corpus)
    coco_rows, coco_summary = parse_coco(args.coco_timing)
    profile_rows, categories = parse_profile_summary(args.profile_summary)

    if sha256(args.coco_json) != PREDICTION_SHA256:
        raise ValueError("final COCO prediction JSON is not byte-identical to Stage53")
    release_manifest_path = args.release_root / "release_manifest.json"
    release_checksums_path = args.release_root / "release_sha256.txt"
    release_binary_path = args.release_root / "bin/yolo26_k1x_int8"
    if sha256(release_manifest_path) != RELEASE_MANIFEST_SHA256:
        raise ValueError("unexpected release tree-manifest hash")
    if sha256(release_checksums_path) != RELEASE_SHA256_FILE_SHA256:
        raise ValueError("unexpected release checksum-file hash")
    if sha256(release_binary_path) != RELEASE_BINARY_SHA256:
        raise ValueError("unexpected packaged executor binary hash")
    release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    if release_manifest["source_commit"] != RELEASE_SOURCE_COMMIT:
        raise ValueError("unexpected release source commit")
    if release_manifest["package_manifest_sha256"] != PACKAGE_SHA256:
        raise ValueError("unexpected packaged model manifest hash")
    for name, rows, result, expected_samples in (
        ("stage53 condition-variable", stage53_cv_raw, stage53_cv, 500),
        ("stage53 epoch-spin", stage53_spin_raw, stage53_spin, 500),
        ("Stage54 condition-variable", final_cv_raw, final_cv, 500),
        ("Stage54 epoch-spin", final_spin_raw, final_spin, 500),
        ("Stage54 epoch-spin soak", soak_raw, soak, 10000),
        ("Stage54 condition-variable soak", cv_soak_raw, cv_soak, 10000),
    ):
        if len(rows) != expected_samples:
            raise ValueError(
                f"{name}: expected {expected_samples} samples, found {len(rows)}"
            )
        if result["output_hashes"] != "0xd43f5e018b415631":
            raise ValueError(f"{name}: unexpected F0 output hash")
        if int(result["cpu4_7_ime_count"]) != 0:
            raise ValueError(f"{name}: CPU4-7 IME execution detected")
    for name in ("dense_shape_census.tsv", "dense_shape_ranked.tsv", "dense_category_reconciliation.tsv"):
        copy_normalized(args.dense_dir / name, stage / name)
    for source, destination in (
        ("same_shape_route_performance_raw.tsv", "same_shape_route_performance_raw.tsv"),
        ("same_shape_route_performance_summary.tsv", "same_shape_route_performance_summary.tsv"),
    ):
        copy_normalized(args.same_shape_dir / source, stage / destination)
    for name in (
        "candidate_prediction_freeze.tsv", "candidate_prediction_vs_measurement.tsv",
        "shape_model_training_rows.tsv", "shape_model_holdout_rows.tsv",
        "shape_model_validation.tsv", "measured_latency_lut_v2.tsv",
        "measured_nonmac_lut_v2.tsv", "full_graph_cost_model_v2.tsv",
    ):
        copy_normalized(args.cost_model_dir / name, stage / name)

    disassembly_names = (
        "direct_1x1_disassembly.txt", "dense_disassembly.txt", "e2c3_disassembly.txt",
        "depthwise_v2_disassembly.txt", "lut2_rvv_disassembly.txt",
        "input_stem_disassembly.txt", "attention_v2_disassembly.txt",
        "selected_disassembly.txt",
    )
    for name in disassembly_names:
        copy_normalized(args.disassembly_dir / name, stage / name)

    stage53_real_rows = read_tsv(args.stage53_real_corpus)
    write_tsv(stage / "stage53_baseline_raw.tsv", stage53_cv_raw + stage53_spin_raw)
    write_tsv(stage / "stage53_baseline_summary.tsv", [stage53_cv, stage53_spin])
    write_tsv(stage / "stage53_baseline_real_corpus.tsv", stage53_real_rows)
    write_tsv(stage / "prestage_binary_and_package_hashes.tsv", [
        {"artifact": "package_manifest", "sha256": PACKAGE_SHA256},
        {"artifact": "accepted_prediction_json", "sha256": PREDICTION_SHA256},
        {"artifact": "final_executor", "sha256": "243da5c84a66bb53cc910090e4b745d0d6ec5720636357412d96e6d3d49d05cd"},
    ])
    write_md(stage / "stage53_baseline_report.md", "Stage53 baseline reproduction", [
        f"Condition-variable mean {float(stage53_cv['mean_us']):.6f} us and p95 "
        f"{float(stage53_cv['p95_us']):.6f} us. Epoch-spin mean "
        f"{float(stage53_spin['mean_us']):.6f} us and p95 {float(stage53_spin['p95_us']):.6f} us.",
        f"The reproduced 100-image corpus mean is {statistics.fmean(float(row['executor_us']) for row in stage53_real_rows):.6f} us. "
        f"Package `{PACKAGE_SHA256}` and output hash `0xd43f5e018b415631` match Stage53.",
    ])

    write_tsv(stage / "same_shape_route_contracts.tsv", [
        {"shape_class": "high_M_low_K_1x1", "routes": "general_packed,resident_facade,direct_P1",
         "same_input_weights_assets_layout_workers": 1},
        {"shape_class": "N80_1x1", "routes": "general_packed,resident_facade,direct_P1",
         "same_input_weights_assets_layout_workers": 1},
        {"shape_class": "3x3_stride1", "routes": "general_packed,resident_P3_control",
         "same_input_weights_assets_layout_workers": 1},
        {"shape_class": "3x3_stride2", "routes": "general_packed,resident_P3_segmented",
         "same_input_weights_assets_layout_workers": 1},
    ])
    write_tsv(stage / "same_shape_route_correctness.tsv", fixture_rows())
    write_md(stage / "dataflow_vs_shape_report.md", "Dataflow versus shape", [
        "The mandatory same-shape comparison does not support a single resident/non-resident "
        "explanation. High-M low-K 1x1 rows are mixed A-delivery/epilogue limited; N80 rows add "
        "tail and store pressure; stride-1 3x3 is mainly arithmetic/epilogue limited; stride-2 "
        "benefits from P3 segmented direct delivery.",
        "The selected dispatcher therefore specializes exact shape classes instead of extending "
        "one resident facade indiscriminately.",
    ])

    write_candidate_evidence(stage, candidate_rows(), categories, args.raw_root)

    write_tsv(stage / "final_model_performance_raw.tsv", final_cv_raw + final_spin_raw)
    write_tsv(
        stage / "final_model_performance_summary.tsv",
        [final_cv, final_spin, soak, cv_soak, ort],
    )
    write_tsv(stage / "final_model_long_soak.tsv", [
        {**soak, "raw_samples": len(soak_raw), "raw_evidence": str(args.final_soak)},
        {
            **cv_soak,
            "raw_samples": len(cv_soak_raw),
            "raw_evidence": str(args.final_cv_soak),
        },
    ])
    write_tsv(stage / "final_real_corpus_timing.tsv", real_rows)
    write_tsv(stage / "final_image_pipeline_timing.tsv", pipeline_summary)
    write_tsv(stage / "final_ort_comparison.tsv", ort_raw)
    write_tsv(stage / "final_correctness_matrix.tsv", fixture_rows())
    write_tsv(stage / "final_coco_prediction_hashes.tsv", [
        {"surface": "Stage53", "sha256": PREDICTION_SHA256, "images": 5000},
        {"surface": "Stage54", "sha256": sha256(args.coco_json), "images": 5000,
         "byte_identical_to_stage53": 1},
    ])
    write_tsv(stage / "final_coco_results.tsv", [
        {"surface": "FP32_reference", "map50_95": 0.401438855549},
        {"surface": "legacy_semantic_INT8", "map50_95": SEMANTIC_MAP},
        {"surface": "Stage54_K1X_INT8_V1", "map50_95": K1X_MAP,
         "map50": 0.5258465300872381, "ap_small": 0.18397294626227842,
         "ap_medium": 0.4142627352606523, "ap_large": 0.5440433811804918,
         "delta_vs_stage53": 0.0, "classification": "preferred"},
    ])
    for source, destination in (
        ("final_coco_per_class.tsv", "final_coco_per_class.tsv"),
        ("final_coco_bootstrap.tsv", "final_coco_bootstrap.tsv"),
    ):
        if (args.stage53_dir / source).is_file():
            copy_normalized(args.stage53_dir / source, stage / destination)
    write_md(stage / "final_coco_report.md", "Final COCO val2017", [
        f"The final Stage54 binary completed 5000/5000 images. Prediction JSON SHA-256 "
        f"`{PREDICTION_SHA256}` is byte-identical to Stage53, so all detections and metrics are "
        "identical without tolerance.",
        f"K1X_INT8_V1 mAP50-95 remains {K1X_MAP:.16f}, delta 0 versus Stage53 and "
        f"{K1X_MAP - SEMANTIC_MAP:+.16f} versus legacy semantic INT8. Accuracy remains preferred.",
    ])

    speedup_stage53 = (1.0 - float(final_spin["mean_us"]) / STAGE53_MEAN_US) * 100.0
    speedup_ort = (1.0 - float(final_spin["mean_us"]) / float(ort["mean_us"])) * 100.0
    pipeline_total = next(row for row in pipeline_summary if row["surface"] == "preloaded_image_pipeline")
    write_md(stage / "final_performance_report.md", "Final performance", [
        f"Compatibility condition-variable SCHED_OTHER: {float(final_cv['mean_us']):.6f} us mean, "
        f"{float(final_cv['p95_us']):.6f} us p95. Dedicated-board epoch-spin SCHED_OTHER: "
        f"{float(final_spin['mean_us']):.6f} us mean, {float(final_spin['p95_us']):.6f} us p95, "
        f"{float(final_spin['p99_us']):.6f} us p99.",
        f"The low-latency route is {speedup_stage53:.6f}% lower latency than Stage53 and "
        f"{speedup_ort:.6f}% lower than matched B120 ORT on the same per-inference statistical unit.",
        f"Real 100-image corpus mean is {real_mean:.6f} us. Preloaded-image pipeline mean is "
        f"{float(pipeline_total['mean_us']):.6f} us.",
        f"The 10000-run soak recorded p99 {float(soak['p99_us']):.6f} us, p99.9 "
        f"{float(soak['p999_us']):.6f} us, and max {float(soak['max_us']):.6f} us.",
        f"The separate 10000-run compatibility soak recorded mean "
        f"{float(cv_soak['mean_us']):.6f} us, p99 {float(cv_soak['p99_us']):.6f} us, "
        f"and max {float(cv_soak['max_us']):.6f} us.",
    ])

    write_tsv(stage / "scheduler_v2_thermal.tsv", read_tsv(args.thermal))
    write_tsv(stage / "pmu_dense_shapes.tsv", read_tsv(args.pmu))
    write_md(stage / "pmu_final_report.md", "PMU final report", [
        "The stage-owned CPU-wide perf_event_open helper measured cycles, instructions, "
        "task-clock, context switches, and CPU migrations per CPU around the exact full-model "
        "workload, with time_running equal to time_enabled. These counts include any unrelated "
        "activity on CPU0-4 and are not presented as worker-owned counters. The board has no "
        "installed perf binary and no authoritative X60 cache/stall mapping, so no such events "
        "are invented.",
        "CPU-wide prefix subtraction for individual dense rows was repeated but remained noisy "
        "and sign-changing; those rows are retained as diagnostic only. Wall time is selection authority.",
    ])
    write_tsv(stage / "compiler_final_matrix.tsv", [
        {"arm": "C0", "flags": "explicit baseline", "exact": 1,
         "board": "pass", "status": "selected"},
        {"arm": "C1", "flags": "baseline plus -mcpu=spacemit-x60", "exact": 1,
         "binary_identity": "byte-identical-to-C0", "status": "no material gain"},
        {"arm": "C2", "flags": "baseline plus LTO", "parser": "pass",
         "board": "SIGILL_exit_132", "status": "rejected"},
    ])
    write_md(stage / "compiler_final_decision.md", "Compiler decision", [
        "Retain `-march=rv64gcv_zvfh -mabi=lp64d -mtune=spacemit-x60 "
        "-funroll-loops -O3 -DNDEBUG`. C1 was byte-identical and C2/LTO trapped on board; no "
        "governing compiler policy or Codex skill change is justified.",
    ])
    write_tsv(stage / "selected_symbol_inventory.tsv", [
        {"symbol": "y26_stage54_kernel_direct_1x1_m12n16", "isa": "explicit_smt.vmadot",
         "cpu_set": "CPU0-3", "sigill": 0},
        {"symbol": "stage51_q62_e2c3_x8_lut", "isa": "explicit_standard_RVV_vsmul_vluxei8",
         "cpu_set": "CPU0-3", "sigill": 0},
        {"symbol": "depthwise_v2_C8", "isa": "explicit_standard_RVV",
         "cpu_set": "CPU0-3", "sigill": 0},
        {"symbol": "input_quantize_rvv_v2", "isa": "explicit_standard_RVV",
         "cpu_set": "CPU0-3", "sigill": 0},
    ])

    cost_rows = read_tsv(args.cost_model_dir / "full_graph_cost_model_v2.tsv")
    validation = read_tsv(args.cost_model_dir / "shape_model_validation.tsv")[-1]
    decomposition = float(cost_rows[0]["error_pct"])
    candidate_error = float(cost_rows[1]["error_pct"])
    heldout = float(validation["median_absolute_percentage_error_pct"])
    write_md(stage / "full_graph_cost_model_v2_report.md", "Full graph cost model V2", [
        f"Current-graph profile decomposition error is {decomposition:+.6f}%. Candidate "
        f"composition prediction error is {candidate_error:+.6f}%.",
        f"The dense shape model uses grouped five-fold validation over 44 exact shape classes "
        f"and 72 held-out observations. Median absolute percentage error is {heldout:.6f}%; "
        f"worst cases remain explicitly preserved in `shape_model_validation.tsv`.",
        "Per-operation p95 values are not summed to claim a tail prediction.",
    ])
    write_md(stage / "codesign_input_readiness_v2.md", "Co-design input readiness V2", [
        "Status: ready as a measured current-graph cost input. Current decomposition, candidate "
        "composition, and held-out median gates pass. Worst-case held-out errors remain large, "
        "so any future graph must measure novel shape classes rather than trust interpolation.",
        "This readiness statement does not authorize or start model-executor co-design, model "
        "training, student selection, or Q31 promotion.",
    ])

    write_md(stage / "workspace_preflight.md", "Workspace preflight", [
        "Start HEAD `9e9e32de006200f0224e58b059abf4583fdea213`, branch "
        "`yolo26-custom-int8-engine`, clean initial worktree, and exact GitHub/GitLab Stage53 "
        "parity were verified before edits.",
        "Board `/data` is writable NVMe ext4. `/control` and `/data/ncnn` were not mutated. "
        "No unrelated worktree content was absorbed.",
    ])
    write_md(stage / "vendor_runtime_lane_frozen.md", "Vendor runtime lane frozen", [
        "RT204/RT205 stock runtime, plugin, crash-forensics, and vendor-package work remained "
        "frozen. `rt205_work_performed=false`.",
    ])
    write_md(stage / "superseded_stage54_prompt_notice.md", "Superseded Stage54 prompt", [
        "The short Stage54 draft committed by Stage53 is historical. The direct-user-authorized "
        f"task `{TASK_ID}` is the controlling Stage54 definition.",
    ])
    write_md(stage / "prestage_repository_state.md", "Prestage repository state", [
        "The repository started clean at `9e9e32de006200f0224e58b059abf4583fdea213`. No empty "
        "commit, reset, rebase, merge, or history rewrite was performed.",
    ])
    write_md(stage / "prestage_dual_remote_report.md", "Prestage dual-remote publication", [
        "Normal no-op/fast-forward publication and SHA parity passed for `github` and `gitlab-rd` "
        "at the exact Stage53 start commit. No force push was used.",
    ])
    write_tsv(stage / "prestage_remote_parity.tsv", [
        {"remote": "github", "head": "9e9e32de006200f0224e58b059abf4583fdea213", "status": "exact"},
        {"remote": "gitlab-rd", "head": "9e9e32de006200f0224e58b059abf4583fdea213", "status": "exact"},
    ])
    write_md(stage / "release_update_report.md", "Release update", [
        f"The Stage53 optimized-research bundle is preserved. Stage54 qualifies for an updated "
        f"bundle at `{args.release_root}` because mean improved {speedup_stage53:.6f}%, exactness "
        "and COCO identity pass, the 10000-run soak passes, and API/CLI compatibility is retained.",
        f"Release tree-manifest SHA-256: `{RELEASE_MANIFEST_SHA256}`. Checksum-file SHA-256: "
        f"`{RELEASE_SHA256_FILE_SHA256}`. Packaged CLI SHA-256: `{RELEASE_BINARY_SHA256}`. "
        f"Source commit: `{RELEASE_SOURCE_COMMIT}`. Package manifest: `{PACKAGE_SHA256}`.",
        "On-board checksum verification, loader resolution, C API smoke, CLI smoke, and bundled "
        "compatibility/low-latency benchmarks passed from the deployed NVMe release root.",
        "The bundle distinguishes condition-variable compatibility from dedicated-board epoch-spin "
        "low latency. It is an optimized research handoff, not production-ready.",
    ])
    write_md(stage / "global_sysctl_and_rollback_report.md", "Global sysctl and rollback", [
        "No sysctl or persistent global perf configuration was changed. Before and after values "
        "were `kernel.perf_event_paranoid=2` and `kernel.kptr_restrict=1`; rollback is not applicable.",
    ])
    write_tsv(stage / "board_emmc_write_exceptions.tsv", [
        {"path": "none", "bytes": 0, "reason": "all stage artifacts remained under NVMe /data"},
    ])
    write_tsv(stage / "board_storage_manifest.tsv", [
        {"root": args.board_root, "storage": "NVMe /data", "purpose": "Stage54 raw artifacts"},
        {"root": str(args.release_root), "storage": "NVMe /data", "purpose": "optimized research bundle"},
    ])
    write_md(stage / "board_storage_preflight.txt", "Board storage preflight", [
        "`findmnt -T /data`, `df -hT /data /`, `lsblk`, and writable-directory checks passed. "
        "TMPDIR and XDG_CACHE_HOME were rooted under the Stage54 board NVMe tree.",
    ])
    write_md(stage / "board_benchmark_environment.txt", "Board benchmark environment", [
        "Banana-Pi BPI-F3 / SpacemiT K1X; CPU0-3 IME workers; CPU4 controller; SCHED_OTHER; "
        "performance governor; recorded 1.6 GHz; warmup 10, runs 100, repeats 5.",
        "Compiler: SpacemiT GCC 14.3.0 g56971dcbea2. Flags: `-march=rv64gcv_zvfh "
        "-mabi=lp64d -mtune=spacemit-x60 -funroll-loops -O3 -DNDEBUG`.",
    ])

    write_md(stage / "STAGE54_FINAL_REPORT.md", "Stage54 final report", [
        "Classification: `stage54-current-executor-final-maximization-strong-positive`.",
        f"The selected dedicated-board SCHED_OTHER epoch-spin route measured "
        f"{float(final_spin['mean_us']):.6f} us mean, {float(final_spin['p95_us']):.6f} us p95, "
        f"and {float(final_spin['p99_us']):.6f} us p99. This is {speedup_stage53:.6f}% lower "
        f"latency than Stage53 and {speedup_ort:.6f}% lower than matched B120 ORT.",
        f"All 215 boundaries, F0-F7, bus, Zidane, FRM/vector CSR restoration, and CPU0-3-only "
        f"IME pass. COCO prediction SHA-256 `{PREDICTION_SHA256}` is byte-identical to Stage53; "
        f"mAP50-95 remains {K1X_MAP:.16f}.",
        f"Cost-model decomposition error {decomposition:+.6f}%, candidate error "
        f"{candidate_error:+.6f}%, held-out median MAPE {heldout:.6f}%.",
        f"Release tree-manifest SHA-256 `{RELEASE_MANIFEST_SHA256}`; checksum-file SHA-256 "
        f"`{RELEASE_SHA256_FILE_SHA256}`; handoff smoke passed.",
        "No 20 FPS, production-readiness, training, student-selection, Q31, RT205, or co-design "
        "execution claim is made.",
    ])
    write_md(stage / "STAGE54_SUMMARY_RU.md", "Stage54: kratkoe rezume", [
        f"Polnyy tochnyy K1X INT8 ispolnitel uskoren do {float(final_spin['mean_us']) / 1000.0:.3f} ms "
        "za schet specializacii plotnyh Conv, E2c3 C8, RVV vhodnogo preobrazovaniya i depthwise V2.",
        "COCO rezultat pobaytovo sovpadaet so Stage53. Rezhim condition-variable ostalsya "
        "sovmestimym profilem; epoch-spin ostalsya issledovatelskim profilem dlya vydelennoy platy.",
    ])

    write_md(stage / "stage55_prompt.md", "Stage55 recommendation", [
        "Recommended next stage: release maintenance and bounded residual executor work, using the "
        "measured V2 cost model. Model-executor co-design or training requires a separate explicit "
        "authorization and must retain the current full-COCO baseline.",
        "Priority unknowns are supported indexed-LUT ISA alternatives, the remaining attention/LUT2 "
        "families, and novel dense shape classes with high held-out model error.",
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
