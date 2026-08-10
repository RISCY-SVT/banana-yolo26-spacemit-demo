#!/usr/bin/env python3
"""Derive compact Stage65B-R1 reports from completed raw host evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


LANES = ("B1", "B2", "B3", "B4", "B5", "B6")
FULL_HYBRID_ARMS = ("H0", "H1", "H3", "H5", "H6", "H8")
TSV_FIELD_LIMIT = 16 * 1024 * 1024
STAGE64_METRICS = {
    "surface": "B0",
    "images": 5000,
    "images_with_predictions": 4997,
    "failures": 0,
    "non_finite_predictions": 0,
    "prediction_count": 782544,
    "prediction_sha256": (
        "6162fc26a654f19e21a7ba65f064ab1c3f651a318453944e25026f2e75ae3a00"
    ),
    "map50_95": 0.35876850879267863,
    "map50": 0.5141763070517243,
    "map75": 0.3888792314680792,
    "ap_small": 0.179113696685979,
    "ap_medium": 0.4176396163742976,
    "ap_large": 0.5165509362490022,
    "ar_1": 0.3163094526658381,
    "ar_10": 0.5356407594171744,
    "ar_100": 0.6024801627718729,
    "ar_small": 0.37385174173614794,
    "ar_medium": 0.649859296442491,
    "ar_large": 0.8020013088757183,
    "evidence_class": "imported-stage64-exact-artifact",
}
FP32_METRICS = {
    "surface": "FP32",
    "images": 5000,
    "prediction_count": 548128,
    "prediction_sha256": (
        "e8c97ebf44727670cdc44c3fa5ce50df6748e849dd4211b9367db92b5da96c1a"
    ),
    "map50_95": 0.40473065112282053,
    "map50": 0.5712619071892974,
    "map75": 0.43463577176335666,
    "ap_small": 0.19778857258539873,
    "ap_medium": 0.4414523593136039,
    "ap_large": 0.586958794539891,
    "ar_1": 0.3294452013730641,
    "ar_10": 0.5427769653531678,
    "ar_100": 0.5898703789064664,
    "evidence_class": "imported-stage64-same-source-fp32",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--stage-dir", required=True, type=Path)
    parser.add_argument("--stage64-derived", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    # Boundary histogram summaries legitimately exceed Python's 128 KiB default.
    csv.field_size_limit(TSV_FIELD_LIMIT)
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def fields(rows: Iterable[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                result.append(field)
    return result


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields(rows)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            values = {name: row.get(name, "") for name in fieldnames}
            if values[fieldnames[-1]] in (None, ""):
                values[fieldnames[-1]] = "NA"
            writer.writerow(values)


def candidate_models(root: Path, lane: str) -> tuple[Path, Path]:
    base = root / "postprocess" / lane / "candidate-gate" / lane / "models"
    prefix = f"stage65b_r1_{lane.lower()}"
    return (
        base / f"{prefix}.inference.onnx",
        base / f"{prefix}.postprocess.onnx",
    )


def config_values(path: Path) -> tuple[int, int, int, str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    calibration = config["calibration_parameters"]
    quantization = config["quantization_parameters"]
    return (
        int(calibration["calibration_step"]),
        int(quantization["precision_level"]),
        int(quantization["finetune_level"]),
        str(calibration["input_parameters"][0]["data_list_path"]),
    )


def first_summary(root: Path, lane: str) -> Path:
    suffix = "run1-long" if lane == "B5" else "run1"
    return root / "quantization" / f"{lane}-{suffix}.tsv"


def normalized_report_hash(path: Path, run_root: Path) -> str:
    text = path.read_text(encoding="utf-8")
    text = text.replace(str(run_root), "<RUN_ROOT>")
    text = re.sub(r"20\d\d-\d\d-\d\d[ T]\d\d:\d\d:\d\d(?:\.\d+)?", "<TIMESTAMP>", text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def quantization_reports(options: argparse.Namespace) -> None:
    raw = options.raw_root
    stage = options.stage_dir
    baseline: list[dict[str, Any]] = [
        {
            "lane": "B0",
            "corpus": "Stage64-val-derived",
            "calibration_count": 50,
            "precision_level": 0,
            "finetune_level": 1,
            "model_sha256": (
                "29e08be834afb8925ca02af69d9a25df05449e9367ef3d8dd8ca4d57cf59a4fb"
            ),
            "reproducibility": "imported-exact-identity",
            "status": "imported-historical-control",
        }
    ]
    runtime_rows: list[dict[str, Any]] = []
    reproducibility: list[dict[str, Any]] = []
    tree_rows: list[dict[str, Any]] = []
    corpus = {
        "B1": "COCO-train-natural-C50",
        "B2": "COCO-train-natural-C50",
        "B3": "COCO-train-natural-C200",
        "B4": "COCO-train-natural-C500",
        "B5": "COCO-train-natural-C1000",
        "B6": "COCO-train-size-balanced-C500",
    }
    for lane in LANES:
        config = raw / "configs" / "effective_configs" / f"{lane}.json"
        count, precision, finetune, image_list = config_values(config)
        first_path = first_summary(raw, lane)
        second_path = raw / "quantization" / f"{lane}-run2.tsv"
        first = read_tsv(first_path)[0]
        second = read_tsv(second_path)[0]
        same_model = first["output_sha256"] == second["output_sha256"]
        first_run_root = Path(first["output_model"]).parent.parent
        second_run_root = Path(second["output_model"]).parent.parent
        first_report = next((first_run_root / "output").glob("*_report.md"))
        second_report = next((second_run_root / "output").glob("*_report.md"))
        normalized_equal = normalized_report_hash(
            first_report, first_run_root
        ) == normalized_report_hash(second_report, second_run_root)
        if not same_model or not normalized_equal:
            raise RuntimeError(
                f"reproducibility gate failed for {lane}: "
                f"model_equal={same_model} report_equal={normalized_equal}"
            )
        baseline.append(
            {
                "lane": lane,
                "corpus": corpus[lane],
                "calibration_count": count,
                "precision_level": precision,
                "finetune_level": finetune,
                "effective_config_sha256": sha256(config),
                "effective_image_list": image_list,
                "model_sha256": first["output_sha256"],
                "reproducibility": "byte-identical" if same_model else "mismatch",
                "status": "pass" if same_model else "fail",
            }
        )
        for generation, summary_path, row in (
            ("run1", first_path, first),
            ("run2", second_path, second),
        ):
            runtime_rows.append(
                {
                    "lane": lane,
                    "generation": generation,
                    "summary": str(summary_path),
                    "returncode": row["returncode"],
                    "elapsed_seconds": row["elapsed_seconds"],
                    "output_sha256": row["output_sha256"],
                    "node_count": row["node_count"],
                    "qdq_count": row["qdq_count"],
                    "qlinear_count": row["qlinear_count"],
                    "tree_file_count": row["tree_file_count"],
                    "tree_byte_count": row["tree_byte_count"],
                    "maximum_rss_kbytes": time_value(Path(row["time_v"]), "Maximum resident set size"),
                }
            )
            manifest = Path(row["output_model"]).parent.parent / "generated-output-tree-manifest.tsv"
            for item in read_tsv(manifest):
                tree_rows.append({"lane": lane, "generation": generation, **item})
        reproducibility.append(
            {
                "lane": lane,
                "run1_model_sha256": first["output_sha256"],
                "run2_model_sha256": second["output_sha256"],
                "deployable_onnx_byte_equal": int(same_model),
                "run1_report_sha256": sha256(first_report),
                "run2_report_sha256": sha256(second_report),
                "normalized_analysis_report_equal": int(normalized_equal),
                "random_seed": first["random_seed"],
                "launcher": Path(first["launcher"]).name,
                "status": "pass" if same_model and normalized_equal else "fail",
            }
        )
    baseline.append(
        {
            "lane": "O1",
            "corpus": "Open-Images-V7-validation",
            "calibration_count": 0,
            "precision_level": 1,
            "finetune_level": 2,
            "status": "not-run-nonblocking-allowlisted-image-surface-unavailable",
        }
    )
    write_tsv(stage / "baseline_quantization_matrix.tsv", baseline)
    write_tsv(stage / "quantization_runtime.tsv", runtime_rows)
    write_tsv(stage / "quantization_reproducibility.tsv", reproducibility)
    write_tsv(stage / "generated_tree_manifest.tsv", tree_rows)


def time_value(path: Path, label: str) -> str:
    if not path.is_file():
        return ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip().startswith(label):
            return line.rsplit(":", 1)[-1].strip()
    return ""


def conformance_reports(options: argparse.Namespace) -> None:
    raw = options.raw_root
    stage = options.stage_dir
    conformance: list[dict[str, Any]] = []
    qdq: list[dict[str, Any]] = []
    precision: list[dict[str, Any]] = []
    conv: list[dict[str, Any]] = []
    semantic: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    finals: list[dict[str, Any]] = []
    fixed_finals: list[dict[str, Any]] = []
    fixed_identity: list[dict[str, Any]] = []
    for lane in LANES:
        base = raw / "postprocess" / lane
        gate_root = base / "candidate-gate" / lane
        conformance.extend(read_tsv(gate_root / "candidate-gate.tsv"))
        for row in read_tsv(gate_root / "audit" / "qdq_schema_census.tsv"):
            qdq.append({"lane": lane, **row})
        for row in read_tsv(gate_root / "audit" / "conv_kernel_shape_audit.tsv"):
            conv.append({"lane": lane, **row})
        precision.extend(read_tsv(base / "precision-surface.tsv"))
        semantic_root = gate_root / "semantic-100"
        semantic.extend(
            {"fixture_scope": "H500-first-100", **row}
            for row in read_tsv(semantic_root / "host_cpu_semantic_matrix.tsv")
        )
        scores.extend(
            {"fixture_scope": "H500-first-100", **row}
            for row in read_tsv(semantic_root / "score_collapse_gate.tsv")
        )
        finals.extend(read_tsv(semantic_root / "host_final_output_comparison.tsv"))
        fixed_root = base / "fixed-fixtures"
        semantic.extend(
            {"fixture_scope": "fixed-F0-bus-Zidane-canonical", **row}
            for row in read_tsv(fixed_root / "host_cpu_semantic_matrix.tsv")
        )
        scores.extend(
            {"fixture_scope": "fixed-F0-bus-Zidane-canonical", **row}
            for row in read_tsv(fixed_root / "score_collapse_gate.tsv")
        )
        fixed_finals.extend(
            {"fixture_scope": "fixed-F0-bus-Zidane-canonical", **row}
            for row in read_tsv(fixed_root / "host_final_output_comparison.tsv")
        )
        fixed_identity.extend(
            {"lane": lane, **row}
            for row in read_tsv(base / "fixed-fixture-identity.tsv")
        )
    write_tsv(stage / "model_conformance.tsv", conformance)
    write_tsv(stage / "qdq_schema_census.tsv", qdq)
    write_tsv(stage / "precision_surface.tsv", precision)
    write_tsv(stage / "conv_kernel_shape_audit.tsv", conv)
    write_tsv(stage / "host_semantic_matrix.tsv", semantic)
    write_tsv(stage / "score_collapse_gate.tsv", scores)
    write_tsv(stage / "host_100_image_results.tsv", finals)
    write_tsv(stage / "fixed_fixture_results.tsv", fixed_finals)
    write_tsv(stage / "fixed_fixture_identity.tsv", fixed_identity)

    full_root = raw / "full-matrix"
    selected = (full_root / "selected-global-candidate.txt").read_text(
        encoding="utf-8"
    ).strip()
    h500 = full_root / "host-h500" / selected
    h500_rows = read_tsv(h500 / "host_cpu_semantic_matrix.tsv")
    write_tsv(stage / "host_H500_results.tsv", h500_rows)


def scout_and_graphwise_reports(options: argparse.Namespace) -> None:
    raw = options.raw_root
    stage = options.stage_dir
    scout: list[dict[str, Any]] = []
    graphwise: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []
    for lane in LANES:
        base = raw / "postprocess" / lane
        scout.extend(read_tsv(base / "scout500" / "metrics" / "results.tsv"))
        graphwise.extend(read_tsv(base / "graphwise-normalized.tsv"))
        for row in read_tsv(base / "boundary-saturation.tsv"):
            pyramid, branch = boundary_map(row["quant_tensor"])
            boundaries.append({"lane": lane, "pyramid": pyramid, "branch": branch, **row})
    write_tsv(stage / "scout500_results.tsv", scout)
    write_tsv(stage / "graphwise_normalized.tsv", graphwise)
    qparam_fields = (
        "lane", "pyramid", "branch", "float_tensor", "quant_tensor",
        "raw_quantized_tensor", "source_op", "dtype", "axis", "granularity",
        "scale", "zero_point", "representable_min", "representable_max",
    )
    write_tsv(
        stage / "boundary_qparams.tsv",
        [{name: row.get(name, "") for name in qparam_fields} for row in boundaries],
    )
    write_tsv(stage / "boundary_saturation.tsv", boundaries)
    pyramid_fields = (
        "lane", "pyramid", "branch", "sample_count", "fp32_min", "fp32_max",
        "below_range_count", "below_range_fraction", "above_range_count",
        "above_range_fraction", "quantized_min_rail_hits",
        "quantized_max_rail_hits", "rail_hit_fraction", "mean_bias", "mae",
        "normalized_mae", "cosine",
    )
    write_tsv(
        stage / "pyramid_branch_error.tsv",
        [{name: row.get(name, "") for name in pyramid_fields} for row in boundaries],
    )
    hotspots = sorted(
        graphwise,
        key=lambda row: (
            -int(row["snr_high_error"]),
            int(row["cosine_significant_deviation"]) * -1,
            -float(row["snr"]) if finite(row["snr"]) else math.inf,
        ),
    )[:30]
    lines = [
        "# Graphwise hotspots",
        "",
        "Thresholds are diagnostic: SNR >= 0.1 and cosine < 0.99. They do not establish causality.",
        "",
        "| lane | pyramid | branch | op | tensor | SNR | cosine |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for row in hotspots:
        lines.append(
            f"| {row['lane']} | {row['pyramid']} | {row['branch']} | "
            f"{row['op']} | {row['variable']} | {row['snr']} | {row['cosine']} |"
        )
    (stage / "graphwise_hotspots.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def finite(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except ValueError:
        return False


def boundary_map(name: str) -> tuple[str, str]:
    match = re.search(r"one2one_cv([23])\.([012])", name)
    if match is None:
        return "", ""
    return (
        {"0": "P3", "1": "P4", "2": "P5"}[match.group(2)],
        {"2": "bbox", "3": "confidence"}[match.group(1)],
    )


def metrics(path: Path) -> dict[str, str]:
    rows = read_tsv(path)
    if len(rows) != 1:
        raise ValueError(f"expected one metrics row: {path}")
    return rows[0]


def full_coco_reports(options: argparse.Namespace) -> tuple[str, str, str]:
    raw = options.raw_root
    stage = options.stage_dir
    matrix = raw / "full-matrix"
    results: list[dict[str, Any]] = [FP32_METRICS, STAGE64_METRICS]
    size_rows: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = [
        {
            "surface": row["surface"],
            "images": row["images"],
            "prediction_count": row["prediction_count"],
            "prediction_sha256": row["prediction_sha256"],
            "evidence_class": row["evidence_class"],
        }
        for row in (FP32_METRICS, STAGE64_METRICS)
    ]
    b0_metrics_root = raw / "tmp" / "b0-metrics-tool-test"
    size_rows.extend(read_tsv(b0_metrics_root / "size_bins.tsv"))
    per_class.extend(read_tsv(b0_metrics_root / "per_class.tsv"))
    import_fp32_detail(options, size_rows, per_class)

    for lane in LANES:
        root = matrix / "full-coco" / lane
        row = metrics(root / "metrics" / "results.tsv")
        row["evidence_class"] = "stage65b-r1-host-cpu"
        results.append(row)
        size_rows.extend(read_tsv(root / "metrics" / "size_bins.tsv"))
        per_class.extend(read_tsv(root / "metrics" / "per_class.tsv"))
        prediction_rows.append(
            {
                "surface": lane,
                "images": row["images"],
                "prediction_count": row["prediction_count"],
                "prediction_sha256": row["prediction_sha256"],
                "evidence_class": row["evidence_class"],
            }
        )
    hybrid_metrics: dict[str, dict[str, str]] = {}
    for arm in FULL_HYBRID_ARMS:
        root = matrix / "hybrid-full-coco" / arm
        row = metrics(root / "metrics" / "results.tsv")
        row["evidence_class"] = "stage65b-r1-hybrid-host-cpu"
        results.append(row)
        hybrid_metrics[arm] = row
        size_rows.extend(read_tsv(root / "metrics" / "size_bins.tsv"))
        per_class.extend(read_tsv(root / "metrics" / "per_class.tsv"))
        prediction_rows.append(
            {
                "surface": arm,
                "images": row["images"],
                "prediction_count": row["prediction_count"],
                "prediction_sha256": row["prediction_sha256"],
                "evidence_class": row["evidence_class"],
            }
        )
    results.append(
        {
            "surface": "O1",
            "images": 0,
            "evidence_class": "not-run-nonblocking",
            "status": "allowlisted-image-surface-unavailable",
        }
    )
    write_tsv(stage / "full_coco_results.tsv", results)
    write_tsv(stage / "full_coco_size_bins.tsv", size_rows)
    write_tsv(stage / "full_coco_per_class.tsv", per_class)
    write_tsv(stage / "prediction_hashes.tsv", prediction_rows)

    hybrid_scout: list[dict[str, Any]] = []
    for arm in ("H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"):
        hybrid_scout.append(
            metrics(matrix / "hybrid-scout500" / arm / "metrics" / "results.tsv")
        )
    write_tsv(stage / "hybrid_boundary_scout.tsv", hybrid_scout)
    write_tsv(stage / "hybrid_boundary_coco.tsv", list(hybrid_metrics.values()))
    write_hybrid_h500(stage, matrix)
    verdict = causal_verdict(hybrid_metrics)
    causal_candidate = (matrix / "selected-global-candidate.txt").read_text(
        encoding="utf-8"
    ).strip()
    write_causal_report(stage, verdict, causal_candidate, hybrid_metrics)
    write_comparison(stage, results)
    return verdict, best_global(results), causal_candidate


def import_fp32_detail(
    options: argparse.Namespace,
    size_rows: list[dict[str, Any]],
    per_class: list[dict[str, Any]],
) -> None:
    size_source = options.stage64_derived / "full_coco_size_bins.tsv"
    for row in read_tsv(size_source):
        if row["surface"] == "IMPORTED_CANONICAL_FP32/cpu-full5000":
            size_rows.append(
                {
                    "surface": "FP32",
                    "size_bin": row["area"],
                    "ap50_95": row["ap50_95"],
                    "ar_100": "",
                }
            )
    class_source = options.stage64_derived / "full_coco_per_class.tsv"
    for row in read_tsv(class_source):
        if row["surface"] == "IMPORTED_CANONICAL_FP32/cpu-full5000":
            per_class.append(
                {
                    "surface": "FP32",
                    "category_id": row["category_id"],
                    "class_name": row["name"],
                    "instances": "",
                    "ap50_95": row["ap50_95"],
                    "ap50": "",
                    "ar_100": "",
                }
            )


def write_hybrid_h500(stage: Path, matrix: Path) -> None:
    rows = read_tsv(matrix / "hybrid-h500-all" / "score.tsv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["surface"]].append(row)
    output: list[dict[str, Any]] = []
    for arm in sorted(grouped):
        arm_rows = grouped[arm]
        prediction = matrix / "hybrid-h500-all" / arm / "predictions.json"
        output.append(
            {
                "arm": arm,
                "images": len(arm_rows),
                "collapsed_outputs": sum(int(row["collapsed"]) for row in arm_rows),
                "non_finite_values": sum(int(row["non_finite_count"]) for row in arm_rows),
                "mean_score_max": sum(float(row["score_max"]) for row in arm_rows) / len(arm_rows),
                "mean_positive_scores": sum(int(row["positive_scores"]) for row in arm_rows) / len(arm_rows),
                "prediction_sha256": sha256(prediction),
                "replacement": hybrid_replacement(arm),
            }
        )
    write_tsv(stage / "hybrid_boundary_ablation.tsv", output)


def hybrid_replacement(arm: str) -> str:
    return {
        "H0": "none-all-candidate",
        "H1": "FP32-P5-confidence",
        "H2": "FP32-P5-bbox",
        "H3": "FP32-P4-confidence",
        "H4": "FP32-P3-confidence",
        "H5": "FP32-P5-bbox-and-confidence",
        "H6": "FP32-all-confidence",
        "H7": "FP32-all-bbox",
        "H8": "FP32-all-six-control",
    }[arm]


def recovery(rows: dict[str, dict[str, str]], arm: str, metric_name: str) -> float:
    h0 = float(rows["H0"][metric_name])
    h8 = float(rows["H8"][metric_name])
    gap = h8 - h0
    if abs(gap) < 0.001:
        return float("nan")
    return (float(rows[arm][metric_name]) - h0) / gap


def causal_verdict(rows: dict[str, dict[str, str]]) -> str:
    values = {
        (arm, metric_name): recovery(rows, arm, metric_name)
        for arm in ("H1", "H3", "H5", "H6")
        for metric_name in ("map50_95", "ap_large")
    }
    if not all(math.isfinite(value) for value in values.values()):
        return "earlier-subgraph-or-tail-interaction"
    h1_map = values[("H1", "map50_95")]
    h1_large = values[("H1", "ap_large")]
    h5_large = values[("H5", "ap_large")]
    h6_map = values[("H6", "map50_95")]
    h6_large = values[("H6", "ap_large")]
    if (
        h1_map >= 0.5
        and h1_large >= 0.5
        and h6_large - h1_large < 0.2
        and h5_large - h1_large < 0.2
    ):
        return "p5-confidence-causality-supported"
    if h6_map >= 0.5 and h6_large >= 0.5 and h6_large - h1_large >= 0.2:
        return "multi-scale-confidence-causality-supported"
    if h6_map < 0.25 and h6_large < 0.25:
        return "pyramid-boundary-hypothesis-not-supported"
    return "earlier-subgraph-or-tail-interaction"


def write_causal_report(
    stage: Path,
    verdict: str,
    causal_candidate: str,
    rows: dict[str, dict[str, str]],
) -> None:
    lines = [
        "# Causal decision",
        "",
        f"Classification: `{verdict}`.",
        f"Hybrid decision candidate selected by the frozen scout gate: `{causal_candidate}`.",
        "",
        "The decision applies the rules frozen in `causal_predeclared_rules.md`.",
        "Graphwise and clipping results remain supporting diagnostics only.",
        "",
        "| arm | mAP50-95 | AP-large | mAP recovery | AP-large recovery |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in FULL_HYBRID_ARMS:
        map_recovery = recovery(rows, arm, "map50_95") if arm not in {"H0", "H8"} else (0.0 if arm == "H0" else 1.0)
        large_recovery = recovery(rows, arm, "ap_large") if arm not in {"H0", "H8"} else (0.0 if arm == "H0" else 1.0)
        lines.append(
            f"| {arm} | {rows[arm]['map50_95']} | {rows[arm]['ap_large']} | "
            f"{map_recovery:.6f} | {large_recovery:.6f} |"
        )
    (stage / "causal_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def best_global(rows: list[dict[str, Any]]) -> str:
    candidates = [row for row in rows if row.get("surface") in LANES]
    return str(max(candidates, key=lambda row: (float(row["map50_95"]), float(row["ap_large"])))["surface"])


def write_comparison(stage: Path, rows: list[dict[str, Any]]) -> None:
    selected = [row for row in rows if row.get("surface") in {"FP32", "B0", *LANES}]
    lines = [
        "# Stage64 to Stage65B-R1 comparison",
        "",
        "COCO-train calibration is evaluation-disjoint from val2017 but not training-independent: model metadata proves COCO training lineage.",
        "",
        "| surface | factor | mAP50-95 | AP-small | AP-medium | AP-large | prediction SHA-256 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    factors = {
        "FP32": "same-source floating-point reference",
        "B0": "Stage64 val-derived C50 P0/F1",
        "B1": "train C50 P0/F1 corpus effect",
        "B2": "train C50 P1/F2 settings effect",
        "B3": "train C200 P1/F2 count",
        "B4": "train C500 P1/F2 count",
        "B5": "train C1000 P1/F2 count",
        "B6": "train balanced C500 P1/F2 size coverage",
    }
    for row in selected:
        lines.append(
            f"| {row['surface']} | {factors[row['surface']]} | {row.get('map50_95', '')} | "
            f"{row.get('ap_small', '')} | {row.get('ap_medium', '')} | {row.get('ap_large', '')} | "
            f"{row.get('prediction_sha256', '')} |"
        )
    (stage / "stage64_stage65b_r1_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def stage64_reconciliation(options: argparse.Namespace) -> None:
    stage = options.stage_dir
    boundaries = read_tsv(stage / "boundary_saturation.tsv")
    p5 = [row for row in boundaries if row["pyramid"] == "P5" and row["branch"] == "confidence"]
    lines = [
        "# Stage64 boundary reconciliation",
        "",
        "Stage64 historical P5-confidence observations were cosine about `0.987823`, normalized MAE about `0.555`, FP32 above ceiling on 79/100 images, S8 at ceiling on 80/100, and mean-logit bias about `+1.546`.",
        "",
        "Stage65B-R1 evaluates aggregate H500 tensor elements with the released boundary-audit tool. Per-image hashes are preserved in the raw TSV; element counts are not represented as image counts.",
        "",
        "| lane | FP32 min | FP32 max | representable max | above fraction | rail fraction | bias | normalized MAE | cosine |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in p5:
        lines.append(
            f"| {row['lane']} | {row['fp32_min']} | {row['fp32_max']} | {row['representable_max']} | "
            f"{row['above_range_fraction']} | {row['rail_hit_fraction']} | {row['mean_bias']} | "
            f"{row['normalized_mae']} | {row['cosine']} |"
        )
    (stage / "stage64_boundary_reconciliation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def recommendations(
    options: argparse.Namespace,
    verdict: str,
    best: str,
    causal_candidate: str,
) -> None:
    stage = options.stage_dir
    raw = options.raw_root
    model, _ = candidate_models(raw, best)
    metrics_row = metrics(raw / "full-matrix" / "full-coco" / best / "metrics" / "results.tsv")
    boundary_rows = [
        row
        for row in read_tsv(stage / "boundary_saturation.tsv")
        if row["lane"] == causal_candidate
    ]
    confidence_rows = {
        row["pyramid"]: row for row in boundary_rows if row["branch"] == "confidence"
    }
    if verdict == "p5-confidence-causality-supported":
        targeted_rows = [confidence_rows["P5"]]
        targeted_policy = (
            "Retain the terminal P5-confidence Conv/QDQ boundary in floating-point "
            f"precision while preserving the `{best}` calibration corpus and all other "
            "quantization settings."
        )
        expected_direction = "recover the dominant mAP and AP-large gap with one bounded exclusion"
    elif verdict == "multi-scale-confidence-causality-supported":
        targeted_rows = [confidence_rows[level] for level in ("P3", "P4", "P5")]
        targeted_policy = (
            "Retain the terminal P3/P4/P5 confidence Conv/QDQ boundaries in floating-point "
            f"precision while preserving the `{best}` calibration corpus and all other "
            "quantization settings."
        )
        expected_direction = "recover the multi-scale confidence loss across AP size bins"
    else:
        targeted_rows = []
        targeted_policy = (
            "No boundary-specific policy is selected; first localize the earlier inference "
            "subgraph or split/tail interaction with additional host evidence."
        )
        expected_direction = "unknown; a pyramid-boundary exclusion is not justified"
    target_lines = "\n".join(
        f"  - `{row['quant_tensor']}`; source op `{row['source_op']}`; "
        f"scale `{row['scale']}`; zero point `{row['zero_point']}`"
        for row in targeted_rows
    ) or "  - none selected by the causal gate"
    text = f"""# Later candidate recommendations

At most two policies are recommended; neither was generated or tested on K1X in this stage.

## 1. Global policy

- Lane: `{best}`
- Inference model SHA-256: `{sha256(model)}`
- Host mAP50-95: `{metrics_row['map50_95']}`
- Host AP-small/medium/large: `{metrics_row['ap_small']}` / `{metrics_row['ap_medium']}` / `{metrics_row['ap_large']}`
- Requirement: repeat signed-QDQ conformance and then separately authorize ORT 2.0.6 board placement, correctness, timing, and soak gates.

## 2. Targeted proposal

- Causal classification: `{verdict}`
- Hybrid decision lane: `{causal_candidate}` (selected before full COCO by the frozen scout rule).
- Exact implicated tensor/source-op sites:
{target_lines}
- Proposed precision/exclusion: {targeted_policy}
- Expected accuracy direction: {expected_direction}.
- Risk: `precision_level=1` can retain higher-precision regions and may change SpacemiT provider compatibility or fallback. Host graph inventory is not placement evidence.
- Required next gates: deterministic generation, host semantic/COCO repeat, signed-QDQ audit, then separately authorized K1X provider-placement and correctness validation.
"""
    (stage / "later_candidate_recommendations.md").write_text(text, encoding="utf-8")
    readiness = f"""# Later-stage readiness

Host baseline matrix: complete.

Hybrid causal classification: `{verdict}`.

Global full-COCO candidate: `{best}`.

Hybrid decision candidate: `{causal_candidate}`.

Readiness is `host-evidence-complete-board-not-authorized`. This stage provides no SpacemiT EP placement, latency, stability, or promotion evidence. A later stage requires explicit user authorization and must consume exact model/config hashes from this report without regenerating them silently.
"""
    (stage / "stage65b_r2_or_stage65c_readiness.md").write_text(
        readiness, encoding="utf-8"
    )


def final_reports(
    options: argparse.Namespace,
    verdict: str,
    best: str,
    causal_candidate: str,
) -> None:
    stage = options.stage_dir
    result_rows = read_tsv(stage / "full_coco_results.tsv")
    by_surface = {row["surface"]: row for row in result_rows}
    classification = {
        "p5-confidence-causality-supported": "stage65b-r1-evaluation-disjoint-calibration-p5-confidence-causality-supported-host-accuracy-proof-complete",
        "multi-scale-confidence-causality-supported": "stage65b-r1-evaluation-disjoint-calibration-multi-scale-confidence-causality-supported-host-pareto-complete",
        "pyramid-boundary-hypothesis-not-supported": "stage65b-r1-evaluation-disjoint-calibration-pyramid-hypothesis-not-supported-host-root-cause-remains-open",
        "earlier-subgraph-or-tail-interaction": "stage65b-r1-evaluation-disjoint-calibration-pyramid-hypothesis-not-supported-host-root-cause-remains-open",
    }[verdict]
    report = f"""# Stage65B-R1 final report

## Classification

`{classification}`

Publication classification: `host-research-evidence-only-no-board-or-runtime-promotion`.

## Corpus

The official COCO train2017 archive was acquired and selectively extracted. Exact JPEG and canonical decoded-pixel overlap with val2017 is zero for all selected calibration and H500 images. Model metadata proves COCO training lineage, so this corpus is evaluation-disjoint but not training-independent.

Open Images O1 is `not-run-nonblocking`: official metadata was captured, while the official image object host was outside the launch allowlist and an allowlisted equivalent returned HTTP 403.

## PTQ and host gates

All B1-B6 deployable ONNX models are byte-identical across two clean seeded generations. Each passed ONNX checking, signed-QDQ/QLinear/UINT8/kernel-shape/six-boundary/tail-identity checks, the accepted fixed F0/bus, Zidane, and canonical-image semantics, and 100-image host score-collapse gates. F0 and bus intentionally share the accepted `real_bus_preprocessed` tensor; the JPEG and input-tensor identities are recorded in `fixed_fixture_identity.tsv`. The best full-COCO global host candidate is `{best}`. Hybrid causality was run on `{causal_candidate}`, selected earlier by the frozen scout rule; both identities are recorded in `later_candidate_recommendations.md`.

## Accuracy

| surface | mAP50-95 | AP-small | AP-medium | AP-large |
|---|---:|---:|---:|---:|
| FP32 | {by_surface['FP32']['map50_95']} | {by_surface['FP32']['ap_small']} | {by_surface['FP32']['ap_medium']} | {by_surface['FP32']['ap_large']} |
| B0 | {by_surface['B0']['map50_95']} | {by_surface['B0']['ap_small']} | {by_surface['B0']['ap_medium']} | {by_surface['B0']['ap_large']} |
| {best} | {by_surface[best]['map50_95']} | {by_surface[best]['ap_small']} | {by_surface[best]['ap_medium']} | {by_surface[best]['ap_large']} |

Full B1-B6, H0/H1/H3/H5/H6/H8, per-class, size-bin, prediction-hash, Graphwise, and boundary evidence is in the adjacent reports.

## Causality

`{verdict}` for candidate `{causal_candidate}` under the predeclared recovery thresholds. The result comes from full COCO boundary replacement, not correlation metrics alone.

## Scope

No board command, ORT 2.0.6 placement claim, performance/soak run, XSlim source/release mutation, targeted model generation, QAT, training, issue update, or runtime promotion occurred.

Git closure, protected-project invariance, and result-packet identity are appended during publication closure.
"""
    (stage / "STAGE65B_R1_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    ru = f"""# Stage65B-R1: краткое резюме

Классификация: `{classification}`.

Официальный COCO train2017 получен и использован только выборочно. Для выбранных калибровочных наборов и H500 доказано нулевое совпадение с val2017 по SHA-256 JPEG и каноническим RGB-пикселям. Модель обучалась на COCO, поэтому набор независим от оценки, но не от обучения.

Все B1-B6 воспроизводимы байт-в-байт и прошли host-проверки signed-QDQ, графа, хвоста, фиксированных F0/bus, Zidane и canonical fixtures, а также score-collapse. F0 и bus используют один принятый входной тензор `real_bus_preprocessed`. Лучший глобальный host-кандидат по полному COCO: `{best}`. Гибридная причинная проверка выполнена для `{causal_candidate}`, выбранного заранее по scout-метрике.

Причинный вывод по полному COCO для `{causal_candidate}`: `{verdict}`. Board/K1X, размещение SpacemiT EP, скорость и soak в этом этапе не запускались и не заявляются.
"""
    (stage / "STAGE65B_R1_SUMMARY_RU.md").write_text(ru, encoding="utf-8")
    decisions = f"""# Human decision options

1. Review the host-selected global policy `{best}` for a separately authorized board-validation stage.
2. Review the targeted proposal derived from `{verdict}`; generation remains unauthorized here.
3. Require an external-domain corpus with an allowlisted official image surface if training-independence, rather than evaluation-disjointness, is required.

No option is an automatic authorization for source work, Stage65B-R2, Stage65C, board execution, or runtime promotion.
"""
    (stage / "human_decision_options.md").write_text(decisions, encoding="utf-8")


def main() -> int:
    options = parse_args()
    summary = options.raw_root / "full-matrix" / "full-matrix-summary.tsv"
    if not summary.is_file() or read_tsv(summary)[0]["status"] != "pass":
        raise RuntimeError("full host matrix is not complete")
    quantization_reports(options)
    conformance_reports(options)
    scout_and_graphwise_reports(options)
    verdict, best, causal_candidate = full_coco_reports(options)
    stage64_reconciliation(options)
    recommendations(options, verdict, best, causal_candidate)
    final_reports(options, verdict, best, causal_candidate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
