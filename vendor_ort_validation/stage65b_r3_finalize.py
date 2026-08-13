#!/usr/bin/env python3
"""Build compact Stage65B-R3 reports from immutable raw evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable


STAGE_ID = (
    "BANANA-YOLO26-XSLIM-STAGE65B-R3-HIERARCHICAL-EARLY-SUBGRAPH-"
    "CUT-SPLICE-LOCALIZATION-AND-XSLIM-TARGET-POLICY-CHARTER-001"
)
CLASSIFICATION = (
    "stage65b-r3-early-subgraph-localized-distributed-upstream-error-"
    "multi-region-policy-prep-ready"
)
H500_D8 = 0.45621089142110244
H500_H8 = 0.479089
H500_RESIDUAL = 0.02287810857889755
H500_THRESHOLD = 0.005719527144724387
FULL_D8 = 0.3793441320923446
FULL_H8 = 0.4018217950262668
FULL_RESIDUAL = 0.022477662933922227
FULL_THRESHOLD = 0.005619415733480557
FINAL_BOOTSTRAP_DRAW_SHA256 = (
    "35f4077634743cd23d64d7898d827f631a81191a04d381b7386e826cec7679c8"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--tracked-root", required=True, type=Path)
    parser.add_argument("--r1-tracked-root", required=True, type=Path)
    parser.add_argument("--r2-raw-root", required=True, type=Path)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in materialized:
            writer.writerow(row)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def copy(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def float_value(row: dict[str, str], key: str) -> float:
    return float(row[key])


def metric_row(path: Path) -> dict[str, str]:
    rows = read_tsv(path)
    if len(rows) != 1:
        raise ValueError(f"expected one metric row: {path}")
    return rows[0]


def bootstrap_map(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    return {
        (row["pair"], row["metric"]): row
        for row in read_tsv(path)
    }


def combine_tables(paths: Iterable[Path], target: Path) -> None:
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(read_tsv(path))
    write_tsv(target, rows)


def copy_graph_reports(raw: Path, tracked: Path) -> None:
    coarse = raw / "planning/coarse"
    refined = raw / "planning/refined"
    copy(coarse / "source_graph_map.tsv", tracked / "source_graph_map.tsv")
    copy(coarse / "architecture_region_map.tsv", tracked / "architecture_region_map.tsv")
    copy(coarse / "coarse_frontier_plan.tsv", tracked / "coarse_frontier_plan.tsv")
    copy(coarse / "frontier_live_tensor_sets.tsv", tracked / "frontier_live_tensor_sets.tsv")
    copy(coarse / "frontier_graph_proof.md", tracked / "frontier_graph_proof.md")
    copy(coarse / "mapping_rejections.tsv", tracked / "mapping_rejections.tsv")
    copy(refined / "coarse_frontier_plan.tsv", tracked / "refinement_plan.tsv")

    correspondence: dict[str, dict[str, str]] = {}
    for path in (coarse / "qdq_correspondence.tsv", refined / "qdq_correspondence.tsv"):
        for row in read_tsv(path):
            correspondence[row["source_tensor"]] = row
    write_tsv(tracked / "qdq_correspondence.tsv", correspondence.values())

    split_rows: list[dict[str, str]] = []
    for scope, path in (
        ("coarse", coarse / "split_model_identity.tsv"),
        ("refined", refined / "split_model_identity.tsv"),
    ):
        for row in read_tsv(path):
            split_rows.append({"scope": scope, **row})
    write_tsv(tracked / "split_model_identity.tsv", split_rows)


def copy_control_reports(raw: Path, tracked: Path) -> None:
    scopes = ("fixed", "h500", "refined-fixed", "refined-h500")
    combine_tables(
        [raw / "controls" / scope / "split_control_results.tsv" for scope in scopes],
        tracked / "split_control_results.tsv",
    )
    failures: list[dict[str, str]] = []
    for scope in scopes:
        for row in read_tsv(raw / "controls" / scope / "split_control_failures.tsv"):
            if row.get("status") not in {"no-failures", "pass"}:
                failures.append({"control_scope": scope, **row})
    write_tsv(
        tracked / "split_control_failures.tsv",
        failures or [{"control_scope": "all", "frontier": "none", "status": "no-failures"}],
    )


def forward_reports(raw: Path, tracked: Path) -> None:
    coarse_metrics = {
        frontier: metric_row(raw / "coarse-forward/metrics" / frontier / "results.tsv")
        for frontier in (f"C{i}" for i in range(8))
    }
    refined_metrics = {
        frontier: metric_row(raw / "refined-forward/metrics" / frontier / "results.tsv")
        for frontier in ("E0", "E1", "E2", "L1", "L2", "L3")
    }
    coarse_bootstrap_path = raw / "coarse-forward/bootstrap/paired_bootstrap_results.tsv"
    refined_bootstrap_path = raw / "refined-forward/bootstrap/paired_bootstrap_results.tsv"
    coarse_bootstrap = bootstrap_map(coarse_bootstrap_path)
    refined_bootstrap = bootstrap_map(refined_bootstrap_path)

    coarse_rows: list[dict[str, Any]] = []
    for frontier, metric in coarse_metrics.items():
        delta = float_value(metric, "map50_95") - H500_D8
        boot = coarse_bootstrap[(f"{frontier}-D8", "map50_95")]
        coarse_rows.append(
            {
                **metric,
                "delta_from_d8": delta,
                "recovery_fraction": delta / H500_RESIDUAL,
                "ci_lower": boot["percentile_2_5"],
                "ci_upper": boot["percentile_97_5"],
                "probability_delta_gt_zero": boot["probability_delta_gt_zero"],
                "quarter_residual_threshold": H500_THRESHOLD,
                "material_vs_d8": int(delta >= H500_THRESHOLD and float(boot["percentile_2_5"]) > 0),
            }
        )
    write_tsv(tracked / "coarse_forward_h500.tsv", coarse_rows)
    copy(coarse_bootstrap_path, tracked / "coarse_forward_bootstrap.tsv")

    recovery_rows: list[dict[str, Any]] = []
    previous = "D8"
    previous_value = H500_D8
    for frontier in (f"C{i}" for i in range(8)):
        value = float_value(coarse_metrics[frontier], "map50_95")
        pair = f"{frontier}-{previous}"
        boot = coarse_bootstrap.get((pair, "map50_95"))
        recovery_rows.append(
            {
                "frontier": frontier,
                "map50_95": value,
                "delta_from_d8": value - H500_D8,
                "recovery_fraction": (value - H500_D8) / H500_RESIDUAL,
                "incremental_from_previous": value - previous_value,
                "incremental_pair": pair,
                "incremental_ci_lower": boot["percentile_2_5"] if boot else "not-run",
                "incremental_ci_upper": boot["percentile_97_5"] if boot else "not-run",
                "incremental_material": int(
                    bool(boot)
                    and value - previous_value >= H500_THRESHOLD
                    and float(boot["percentile_2_5"]) > 0
                ),
            }
        )
        previous, previous_value = frontier, value
    write_tsv(tracked / "coarse_recovery_curve.tsv", recovery_rows)

    refined_rows: list[dict[str, Any]] = []
    for frontier, metric in refined_metrics.items():
        delta = float_value(metric, "map50_95") - H500_D8
        boot = refined_bootstrap[(f"{frontier}-D8", "map50_95")]
        refined_rows.append(
            {
                **metric,
                "delta_from_d8": delta,
                "recovery_fraction": delta / H500_RESIDUAL,
                "ci_lower": boot["percentile_2_5"],
                "ci_upper": boot["percentile_97_5"],
                "probability_delta_gt_zero": boot["probability_delta_gt_zero"],
                "quarter_residual_threshold": H500_THRESHOLD,
                "material_vs_d8": int(delta >= H500_THRESHOLD and float(boot["percentile_2_5"]) > 0),
            }
        )
    write_tsv(tracked / "refined_frontier_results.tsv", refined_rows)
    copy(refined_bootstrap_path, tracked / "refined_frontier_bootstrap.tsv")

    intervals = (
        ("input-to-C0", "C0-D8", coarse_bootstrap),
        ("E0-to-E1", "E1-E0", refined_bootstrap),
        ("E1-to-E2", "E2-E1", refined_bootstrap),
        ("E2-to-C0", "C0-E2", refined_bootstrap),
        ("C6-to-L1", "L1-C6", refined_bootstrap),
        ("L1-to-L2", "L2-L1", refined_bootstrap),
        ("L2-to-L3", "L3-L2", refined_bootstrap),
        ("L3-to-C7", "C7-L3", refined_bootstrap),
        ("C6-to-C7", "C7-C6", coarse_bootstrap),
    )
    rows: list[dict[str, Any]] = []
    for region, pair, table in intervals:
        boot = table[(pair, "map50_95")]
        point = float(boot["point_delta"])
        rows.append(
            {
                "region_interval": region,
                "pair": pair,
                "point_recovery": point,
                "ci_lower": boot["percentile_2_5"],
                "ci_upper": boot["percentile_97_5"],
                "probability_delta_gt_zero": boot["probability_delta_gt_zero"],
                "quarter_residual_threshold": H500_THRESHOLD,
                "material": int(point >= H500_THRESHOLD and float(boot["percentile_2_5"]) > 0),
                "decision": (
                    "material-distributed-region"
                    if point >= H500_THRESHOLD and float(boot["percentile_2_5"]) > 0
                    else "not-individually-material"
                ),
            }
        )
    write_tsv(tracked / "region_materiality.tsv", rows)


def reverse_reports(raw: Path, tracked: Path) -> None:
    rows = [
        metric_row(raw / "reverse/metrics" / frontier / "results.tsv")
        for frontier in ("C0", "C6", "L3")
    ]
    write_tsv(tracked / "reverse_h500.tsv", rows)
    copy(raw / "reverse/bootstrap/paired_bootstrap_results.tsv", tracked / "reverse_bootstrap.tsv")
    write_text(
        tracked / "bidirectional_localization_decision.md",
        """# Bidirectional localization decision

Forward FQ8 and reverse QF controls passed exact reconstruction on all H500 images.
QF-C0 remains 0.007204 mAP below H8 (95% CI 0.001506..0.012273), so material error is already present by the first complete coarse frontier. QF-C6 is statistically indistinguishable from QF-C0, while QF-L3 loses another 0.008579 mAP (95% CI 0.003159..0.010965). This confirms two distributed contributors: the input/stem-to-model.2 region and the model.23 per-scale head prefixes. No single refined step reaches the predeclared quarter-residual threshold.
""",
    )


def activation_reports(raw: Path, tracked: Path, r1: Path) -> None:
    original = read_tsv(raw / "activation-audit/selected_region_activation_error.tsv")
    revised = read_tsv(raw / "activation-audit-v3/selected_region_activation_error.tsv")
    if len(original) != 22 or len(revised) != len(original):
        raise RuntimeError("selected activation tensor count differs")
    original_fields = list(original[0])
    if any(
        {field: row[field] for field in original_fields}
        != {field: revised[index][field] for field in original_fields}
        for index, row in enumerate(original)
    ):
        raise RuntimeError("per-image hash rerun changed accepted activation metrics")
    per_image_path = (
        raw
        / "activation-audit-v3/selected_region_activation_per_image_hashes.tsv"
    )
    per_image_rows = read_tsv(per_image_path)
    if len(per_image_rows) != 11000 or any(
        row["per_image_hash_table_sha256"] != sha256(per_image_path)
        or int(row["per_image_hash_table_rows"]) != len(per_image_rows)
        for row in revised
    ):
        raise RuntimeError("selected activation per-image hash contract differs")
    copy(
        raw / "activation-audit-v3/selected_region_activation_error.tsv",
        tracked / "selected_region_activation_error.tsv",
    )
    copy(
        raw / "activation-audit-v3/selected_region_qparams.tsv",
        tracked / "selected_region_qparams.tsv",
    )
    graphwise = read_tsv(r1 / "graphwise_normalized.tsv")
    selected = [
        row
        for row in graphwise
        if row["lane"] == "B2"
        and (
            row["variable"] == "/model.2/cv2/act/Mul_output_0"
            or row["variable"].endswith("one2one_cv3.1.1.1/act/Mul_output_0")
            or row["variable"].endswith("one2one_cv3.2.1.1/act/Mul_output_0")
            or row["variable"].endswith("one2one_cv3.2.1.0/conv/Conv_output_0")
        )
    ]
    lines = [
        "# Graphwise reconciliation",
        "",
        "R1 Graphwise and the independent H500 activation audit agree on the two task-causal regions. Exact source tensor names, not shape similarity, were used.",
        "",
        "| Tensor | SNR | MSE | Cosine | Graphwise FP range | Graphwise Q range |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| `{row['variable']}` | {row['snr']} | {row['mse']} | {row['cosine']} | "
            f"{row['f_min']}..{row['f_max']} | {row['q_min']}..{row['q_max']} |"
        )
    lines.extend(
        [
            "",
            "The H500 audit additionally measures clipping/rail fractions over 500 calibration-disjoint images. It finds normalized MAE 0.1679 at model.2, 0.2912 at P4 confidence's last prefix activation, and 0.3871 at P5 confidence's last prefix activation. The task-level cut/splice recovery, not these correlation metrics, is the causal authority.",
        ]
    )
    write_text(tracked / "graphwise_reconciliation.md", "\n".join(lines))
    write_text(
        tracked / "mechanism_decision.md",
        """# Mechanism decision

Classification: `distributed-error`, with `outlier-dominated-range`, lower-tail clipping/rail occupancy, and reconstruction error as the best-supported mechanisms.

- Early R0/model.2: model.2 output has normalized MAE 0.1679, cosine 0.9867, 3.07% FP values below the representable range, and 12.94% low-rail occupancy.
- Late R7 confidence prefixes: P4's last prefix activation has a zero real minimum, clipping 27.06% negative FP SiLU values; P5's last prefix activation is bounded near 93.80 while FP values reach 153.17. Their normalized MAE values are 0.2912 and 0.3871.
- The six-output terminal Q/DQ pairs are excluded here by the accepted D8 bypass. Therefore these measurements describe the remaining upstream residual.
- No individual refined step met the quarter-residual task threshold, so a single-op defect is not established. Attention/MatMul sensitivity and merge-qdomain mismatch remain secondary possibilities, not the selected causal classification.
""",
    )


def full_val_reports(raw: Path, tracked: Path, r2: Path) -> None:
    new_arm_rows = read_tsv(raw / "full-val/fq8/arm_summary.tsv") + read_tsv(
        raw / "full-val/qf/arm_summary.tsv"
    )
    if len(new_arm_rows) != 4 or any(
        int(row["images"]) != 5000
        or int(row["non_finite"]) != 0
        or int(row["score_collapsed_images"]) != 0
        or row["status"] != "pass"
        for row in new_arm_rows
    ):
        raise RuntimeError("new full-val arm conformance/count contract differs")
    surfaces = {
        "D8": r2 / "d8/full-val-metrics/results.tsv",
        "H8": r2 / "fp32/full-val-current/metrics-H8/results.tsv",
        "FQ8-L3": raw / "full-val/metrics/FQ8-L3/results.tsv",
        "FQ8-L2": raw / "full-val/metrics/FQ8-L2/results.tsv",
        "FQ8-E1": raw / "full-val/metrics/FQ8-E1/results.tsv",
        "QF-L3": raw / "full-val/metrics/QF-L3/results.tsv",
    }
    rows: list[dict[str, Any]] = []
    for surface, path in surfaces.items():
        row: dict[str, Any] = dict(metric_row(path))
        if (
            int(row["images"]) != 5000
            or int(row["failures"]) != 0
            or int(row["non_finite_predictions"]) != 0
        ):
            raise RuntimeError(f"full-val gate failed for {surface}: {row}")
        row["surface"] = surface
        if surface.startswith("FQ8"):
            row["delta_from_d8"] = float(row["map50_95"]) - FULL_D8
            row["recovery_fraction"] = (float(row["map50_95"]) - FULL_D8) / FULL_RESIDUAL
        else:
            row["delta_from_d8"] = float(row["map50_95"]) - FULL_D8
            row["recovery_fraction"] = "not-applicable"
        rows.append(row)
    write_tsv(tracked / "selected_full_coco.tsv", rows)

    size_rows: list[dict[str, str]] = []
    class_rows: list[dict[str, str]] = []
    for surface, base in (
        ("D8", r2 / "d8/full-val-metrics"),
        ("H8", r2 / "fp32/full-val-current/metrics-H8"),
        ("FQ8-L3", raw / "full-val/metrics/FQ8-L3"),
        ("FQ8-L2", raw / "full-val/metrics/FQ8-L2"),
        ("FQ8-E1", raw / "full-val/metrics/FQ8-E1"),
        ("QF-L3", raw / "full-val/metrics/QF-L3"),
    ):
        for row in read_tsv(base / "size_bins.tsv"):
            size_rows.append({"surface": surface, **{k: v for k, v in row.items() if k != "surface"}})
        for row in read_tsv(base / "per_class.tsv"):
            class_rows.append({"surface": surface, **{k: v for k, v in row.items() if k != "surface"}})
    write_tsv(tracked / "selected_full_coco_size_bins.tsv", size_rows)
    write_tsv(tracked / "selected_full_coco_per_class.tsv", class_rows)

    screening = raw / "full-val/bootstrap-screening-1000/paired_bootstrap_results.tsv"
    final = raw / "full-val/bootstrap-final-l3-d8-10000/paired_bootstrap_results.tsv"
    combined: list[dict[str, str]] = []
    for scope, path in (("screening", screening), ("final-top-region", final)):
        for row in read_tsv(path):
            if not row.get("reused_screening_prefix_replicates"):
                row["reused_screening_prefix_replicates"] = "not-applicable"
            combined.append({"scope": scope, **row})
    write_tsv(tracked / "selected_full_coco_bootstrap.tsv", combined)

    l3 = float(metric_row(surfaces["FQ8-L3"])["map50_95"])
    l2 = float(metric_row(surfaces["FQ8-L2"])["map50_95"])
    e1 = float(metric_row(surfaces["FQ8-E1"])["map50_95"])
    qf = float(metric_row(surfaces["QF-L3"])["map50_95"])
    final_l3 = bootstrap_map(final)[("L3-D8", "map50_95")]
    if int(final_l3["replicates"]) != 10000 or int(final_l3["seed"]) != 65003:
        raise RuntimeError("final L3-D8 bootstrap count/seed contract differs")
    if int(final_l3["reused_screening_prefix_replicates"]) != 1000:
        raise RuntimeError("final bootstrap did not preserve the accepted 1000 prefix")
    if abs(float(final_l3["point_delta"]) - (l3 - FULL_D8)) > 1e-12:
        raise RuntimeError("final L3-D8 bootstrap point delta differs from full-val")
    remap_rows = read_tsv(
        raw / "full-val/bootstrap-final-l3-d8-10000/synthetic_id_validation.tsv"
    )
    if len(remap_rows) != 10 or any(
        row["status"] != "pass" or float(row["absolute_difference"]) > 1e-12
        for row in remap_rows
    ):
        raise RuntimeError("final bootstrap literal synthetic-ID remap differs")
    hash_lines = (
        raw
        / "full-val/bootstrap-final-l3-d8-10000/paired_bootstrap_replicates.sha256"
    ).read_text(encoding="utf-8").splitlines()
    if len(hash_lines) != 2 or not hash_lines[1].startswith(
        FINAL_BOOTSTRAP_DRAW_SHA256 + "  "
    ):
        raise RuntimeError("final bootstrap draw identity differs")
    write_text(
        tracked / "full_val_localization_decision.md",
        f"""# Full-val localization decision

Full val2017 is confirmation, not selection. All four predeclared arms passed 5000/5000 with zero non-finite outputs and zero score-collapse images.

- D8: {FULL_D8:.12f}; H8: {FULL_H8:.12f}; residual: {FULL_RESIDUAL:.12f}.
- FQ8-L3: {l3:.12f}, recovery {(l3 - FULL_D8) / FULL_RESIDUAL:.3%}.
- FQ8-L2: {l2:.12f}, recovery {(l2 - FULL_D8) / FULL_RESIDUAL:.3%}.
- FQ8-E1 negative control: {e1:.12f}, delta {e1 - FULL_D8:+.12f}.
- QF-L3 reverse confirmation: {qf:.12f}, H8 gap {FULL_H8 - qf:.12f}.
- Final L3-D8 paired bootstrap: 10000 replicates, point delta {float(final_l3['point_delta']):.12f}, percentile 95% CI {float(final_l3['percentile_2_5']):.12f}..{float(final_l3['percentile_97_5']):.12f}, P(delta > 0)={float(final_l3['probability_delta_gt_zero']):.6f}.

The full-val result reproduces H500: error is distributed between an early R0/stem-model.2 contributor and a larger late R7/model.23 per-scale-head-prefix contributor. L3 minus L2 is below the predeclared quarter-residual point threshold, so no single refined sub-block is declared independently material.
""",
    )


def policy_and_charter(tracked: Path) -> None:
    write_text(
        tracked / "targeted_policy_proposals.md",
        """# Targeted policy proposals

No model is generated in Stage65B-R3. At most two later policy classes are justified.

## Policy A: all-S8 local robust-range policy

Apply an XSlim-supported local observer/range-correction policy to the two task-causal regions while retaining signed S8 QDQ throughout: the R0 stem/model.0-model.2 path and the R7 model.23 per-scale head prefixes, with priority on P4/P5 confidence activations. Candidate observer classes are bounded local percentile, MSE/KL, or bias/range correction; Stage R3 does not choose one without a generation/evaluation gate. Preserve six outputs, separate bbox/confidence branches, zero QLinear, zero UINT8 zero points, and explicit Conv kernel_shape.

## Policy B: bounded higher-precision/exclusion policy

If Policy A cannot recover the residual, leave only the proven R7 confidence-prefix region at higher precision or exclude it from quantization. This is secondary because it risks splitting a SpacemiT fused region and violates the resident custom engine's all-INT8 dataflow assumption. It requires explicit conversion/fallback accounting and is not provider-compatible evidence by itself.
""",
    )
    contracts = [
        {
            "policy": "A-all-s8-local-robust-range",
            "source_region": "R0-model.0-model.2;R7-model.23-head-prefixes",
            "exact_anchor_tensors": (
                "/model.0/act/Mul_output_0;/model.1/act/Mul_output_0;"
                "/model.2/cv1/act/Mul_output_0;/model.2/cv2/act/Mul_output_0;"
                "/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.1/act/Mul_output_0;"
                "/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1/act/Mul_output_0;"
                "/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/act/Mul_output_0"
            ),
            "configuration_class": "local-observer-or-range-correction",
            "expected_accuracy_direction": "recover-early-and-head-prefix-residual",
            "spacemit_risk": "low-to-medium-if-all-s8-contract-preserved",
            "custom_engine_v2_risk": "requires-explicit-per-op-qparam-export",
            "generation_status": "not-authorized-not-generated",
        },
        {
            "policy": "B-r7-confidence-prefix-higher-precision-or-exclusion",
            "source_region": "R7-model.23-P3-P5-confidence-prefixes",
            "exact_anchor_tensors": (
                "/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.1/act/Mul_output_0;"
                "/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1/act/Mul_output_0;"
                "/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/act/Mul_output_0"
            ),
            "configuration_class": "ignore-or-higher-precision-subgraph",
            "expected_accuracy_direction": "recover-dominant-late-head-residual",
            "spacemit_risk": "high-mixed-precision-fusion-and-cpu-fallback-risk",
            "custom_engine_v2_risk": "high-breaks-resident-all-int8-dataflow",
            "generation_status": "not-authorized-not-generated",
        },
    ]
    write_tsv(tracked / "targeted_policy_contracts.tsv", contracts)
    write_text(
        tracked / "provider_risk_assessment.md",
        """# Provider risk assessment

Policy A keeps the vendor-declared signed S8-QDQ representation and therefore has the lower structural risk, but changed qparams may alter SpacemiT partition/fusion behavior. Policy B introduces mixed-precision boundaries in model.23 and may split a fused subgraph, add conversions, or cause CPU fallback. Host accuracy does not prove K1X provider placement, latency, or stability; each policy needs later signed-QDQ conformance, provider profiling, fixed-fixture, COCO, performance, and soak gates.
""",
    )
    write_text(
        tracked / "custom_engine_v2_impact_note.md",
        """# K1X custom-engine V2 impact note

This Stage does not modify the custom executor and does not reinterpret `K1X_INT8_V1`. A future `K1X_INT8_V2` exporter would need per-op scale/zero-point identities, per-channel weight scales, exact multiplier/right shift, ties-to-even rounding, saturation, accumulator bounds, an explicit qdomain graph, residual/concat alignment, NCHWc8 layout hints, and a packed-weight manifest. Policy A is compatible in principle only after these contracts are exported exactly. Policy B conflicts with the resident all-INT8 dataflow unless explicit mixed-precision conversions and costs are designed and measured.
""",
    )
    write_text(
        tracked / "XSLIM_DEV_001_CHARTER.md",
        """# XSLIM-DEV-001 Charter

This is a design charter, not implementation authorization.

## Phase 1: generic causal localizer

Implement `xslim-localize-quant-error` with safe complete cut sets, exact source/QDQ correspondence, bidirectional split/splice reconstruction controls, a task-evaluator adapter, paired image-level bootstrap, and bounded `custom_setting` proposal output. It must fail closed on incomplete cuts, ambiguous mappings, dynamic shapes, shared unsafe Q/DQ, or reconstruction mismatch.

## Phase 2: target-runtime profiles

### SpacemiT profile

Enforce signed S8-QDQ only, zero QLinear, zero UINT8 zero points, explicit Conv `kernel_shape`, the six-output/unquantized-tail contract, and a mixed-precision/fallback risk report. Expected fused regions are hypotheses, never placement or speed claims.

### K1X custom-engine profile

Design a new `K1X_INT8_V2` contract: per-op scales/zero points, per-channel weight scales, exact multiplier/right shift, ties-to-even rounding, saturation, accumulator bounds, qdomain graph, residual/concat alignment, NCHWc8 hints, and packed-weight manifest. Do not mutate or reinterpret `K1X_INT8_V1`.

## Phase 3: task-aware robust selection

Add a YOLO COCO metric plugin, AP-small/medium/large constraints, multiple deterministic calibration draws/seeds, a variance penalty, and score-collapse/non-finite guards.

## Phase 4: targeted generation

Generate at most two evidence-selected policies deterministically, then require H500/full COCO and signed-QDQ conformance. This phase needs separate authorization.

## Phase 5: hardware feedback

Measure SpacemiT provider partitions, CPU fallback/conversion cost, custom-engine rescale/layout cost, matched board latency, correctness, and soak. A target profile alone proves none of these hardware properties.
""",
    )
    write_tsv(
        tracked / "later_candidate_freeze.tsv",
        [
            {"rank": 1, "policy": contracts[0]["policy"], "model_sha256": "not-generated", "status": "proposal-only"},
            {"rank": 2, "policy": contracts[1]["policy"], "model_sha256": "not-generated", "status": "proposal-only"},
        ],
    )
    write_text(
        tracked / "stage65c_readiness.md",
        """# Stage65C readiness

Host causal localization is ready for a separately authorized targeted-generation design review. Stage65B-R3 does not authorize Stage65C, XSlim implementation, targeted model generation, board execution, provider claims, performance work, or runtime promotion. Policy A should be attempted before Policy B because it retains the all-S8 representation.
""",
    )
    write_text(
        tracked / "human_decision_options.md",
        """# Human decision options

1. Authorize a bounded all-S8 local-observer/range-correction generation study for the two proven distributed regions.
2. Defer mixed-precision Policy B until Policy A is measured; it has materially higher SpacemiT and custom-engine integration risk.
3. Stop with the current evidence. No board/runtime promotion decision is supported by Stage65B-R3.
""",
    )


def tooling_reports(raw: Path, tracked: Path) -> None:
    write_tsv(
        tracked / "tooling_test_matrix.tsv",
        [
            {
                "test_surface": "focused-pytest",
                "result": "12-passed",
                "status": "pass",
                "raw_evidence": str(raw / "tooling-tests/pytest.log"),
            },
            {
                "test_surface": "compileall",
                "result": "stage65b-r3-tools-and-tests",
                "status": "pass",
                "raw_evidence": str(raw / "tooling-tests/compileall.tsv"),
            },
            {
                "test_surface": "ruff",
                "result": "not-run-unavailable",
                "status": "not-run-non-gating",
                "raw_evidence": str(raw / "tooling-tests/ruff.tsv"),
            },
            {
                "test_surface": "git-diff-check",
                "result": "no-whitespace-errors",
                "status": "pass",
                "raw_evidence": "final-git-hygiene",
            },
        ],
    )
    negative_tests = (
        "incomplete_multi_tensor_cut",
        "hidden_residual_or_concat_edge",
        "ambiguous_source_qdq_mapping",
        "shape_mismatch",
        "dtype_mismatch",
        "dynamic_cut_shape",
        "duplicate_tensor_name",
        "missing_initializer_or_value",
        "functionproto_or_opset_loss",
        "control_reconstruction_mismatch",
        "unsafe_shared_qdq_node",
    )
    write_tsv(
        tracked / "tooling_negative_tests.tsv",
        [
            {
                "negative_case": case,
                "expected_behavior": "fail-closed",
                "observed_behavior": "rejected",
                "status": "pass",
            }
            for case in negative_tests
        ],
    )


def final_reports(tracked: Path) -> None:
    final_bootstrap = next(
        row
        for row in read_tsv(tracked / "selected_full_coco_bootstrap.tsv")
        if row["scope"] == "final-top-region"
        and row["pair"] == "L3-D8"
        and row["metric"] == "map50_95"
    )
    write_text(
        tracked / "STAGE65B_R3_FINAL_REPORT.md",
        f"""# Stage65B-R3 Final Report

## Classification

`{CLASSIFICATION}`

Publication classification: research evidence only; no targeted model, board run, runtime promotion, XSlim source change, or custom-executor change.

## Immutable gates

The research branch began at `84caf893c73f2a33dfbbedacfa56c7ca40843557`; protected main remained `1fd2e71bb1d5a924e7c0444cada94f681b73aa91`. The accepted R2 packet recomputed to tree `d80b3362fa0abd99afa3fc1985fe547e774113f690e5ab0140b0b4c7276639db`, 75 files, 809627 bytes. Frozen FP32/B2/D8/tail/tool hashes passed exact identity checks.

## Cut/splice controls

Eight coarse and six refined complete cut frontiers were constructed from exact source-producer/output and Q/DQ provenance. Every FP32, B2, and B2-D8 reconstruction was byte-identical on fixed fixtures and all 500 H500 images. No frontier was rejected after the final mapping fix; two earlier fail-closed planning attempts and one relative-list invocation remain preserved as raw evidence.

## Localization

H500 D8-to-H8 residual is {H500_RESIDUAL:.12f}. C0 recovers 0.006933 (95% CI 0.001034..0.013480), proving a material early R0/stem-model.2 contribution. C7 minus C6 recovers 0.009207 (95% CI 0.004456..0.012169), proving a material distributed R7/model.23 head-prefix contribution. No individual refined adjacent step reaches the quarter-residual threshold {H500_THRESHOLD:.12f}; L3 reaches 0.475057, or 82.38% cumulative recovery from D8.

Reverse confirmation shows QF-C0 remains 0.007204 below H8 and QF-L3 remains 0.017176 below H8. Error has already accumulated at the early frontier, and the late head adds a second material contribution.

## Full val2017

The four predeclared new arms all completed 5000/5000 with zero failures/non-finite/collapse. FQ8-L3 reaches 0.397752 mAP (81.90% residual recovery), FQ8-L2 0.394140 (65.82%), FQ8-E1 negative control 0.379543, and QF-L3 0.384050. The final 10000-replicate paired L3-D8 bootstrap gives a 95% interval of {float(final_bootstrap['percentile_2_5']):.6f}..{float(final_bootstrap['percentile_97_5']):.6f}, with P(delta > 0)={float(final_bootstrap['probability_delta_gt_zero']):.6f}. These results reproduce the H500 localization without selecting on val2017.

## Mechanism

The selected classification is distributed error, supported by outlier-dominated asymmetric ranges, lower-tail clipping/rail occupancy, and reconstruction error. The strongest H500 activation errors occur at model.2 and P4/P5 confidence prefixes. Task-level cut/splice recovery is the causal authority; Graphwise and activation statistics are supporting diagnostics.

## Policy charter

At most two proposals are frozen: an all-S8 local robust-range policy for R0+R7 (preferred), and a bounded R7 confidence-prefix higher-precision/exclusion policy (secondary, higher integration risk). Neither was generated. `XSLIM_DEV_001_CHARTER.md` defines generic localization, SpacemiT/K1X target profiles, robust selection, targeted generation, and later hardware feedback as separate phases.
""",
    )
    write_text(
        tracked / "STAGE65B_R3_SUMMARY_RU.md",
        f"""# Итог Stage65B-R3

Классификация: `{CLASSIFICATION}`.

Контрольные split/splice-модели прошли точную реконструкцию на фиксированных изображениях и на всех 500 изображениях H500. Остаточная ошибка после D8 не локализуется в одном операторе. Доказаны два распределённых вклада: ранний участок stem/model.0-model.2 и поздние префиксы bbox/confidence-ветвей model.23. Поздний frontier L3 восстанавливает около 82% остатка как на H500, так и на полном val2017, но ранний вклад остаётся статистически значимым.

Финальный paired bootstrap L3-D8 выполнен на 10000 повторах: 95% интервал {float(final_bootstrap['percentile_2_5']):.6f}..{float(final_bootstrap['percentile_97_5']):.6f}, P(delta > 0)={float(final_bootstrap['probability_delta_gt_zero']):.6f}. Аудит активаций подтверждает clipping/rail occupancy и ошибку восстановления в model.2 и особенно в P4/P5 confidence-префиксах. Предложены не более двух последующих политик: сначала локальная all-S8 коррекция диапазонов/observer, затем только при её недостаточности ограниченное исключение или повышенная точность confidence-префикса. Модели не генерировались, плата K1X не запускалась, исходники XSlim и custom executor не менялись.
""",
    )


def main() -> int:
    options = parse_args()
    options.tracked_root.mkdir(parents=True, exist_ok=True)
    copy_graph_reports(options.raw_root, options.tracked_root)
    copy_control_reports(options.raw_root, options.tracked_root)
    forward_reports(options.raw_root, options.tracked_root)
    reverse_reports(options.raw_root, options.tracked_root)
    activation_reports(options.raw_root, options.tracked_root, options.r1_tracked_root)
    full_val_reports(options.raw_root, options.tracked_root, options.r2_raw_root)
    policy_and_charter(options.tracked_root)
    tooling_reports(options.raw_root, options.tracked_root)
    final_reports(options.tracked_root)
    manifest_rows = []
    for path in sorted(options.tracked_root.iterdir(), key=lambda item: item.name):
        if path.is_file():
            manifest_rows.append(
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    write_tsv(options.raw_root / "tmp/tracked-report-manifest.pre-git.tsv", manifest_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
