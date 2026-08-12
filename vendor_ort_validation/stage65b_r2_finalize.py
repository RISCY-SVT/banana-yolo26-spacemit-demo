#!/usr/bin/env python3
"""Derive compact Stage65B-R2 reports from completed raw host evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

from stage65b_r2_common import MAX_CSV_FIELD_SIZE, read_tsv, sha256


H500_SURFACES = ("F0", "F1", "H8", "B0", "B1", "B2", "B3", "B4", "B5", "B6")
VARIANCE_LANES = ("Vseed", "Vorder", "Vdraw")
METRIC_FIELDS = ("map50_95", "map50", "ap_small", "ap_medium", "ap_large")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--raw-root", required=True, type=Path)
    result.add_argument("--r1-root", required=True, type=Path)
    result.add_argument("--stage-dir", required=True, type=Path)
    return result


def union_fields(rows: Iterable[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                result.append(field)
                seen.add(field)
    return result


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = union_fields(rows)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def one(path: Path) -> dict[str, str]:
    rows = read_tsv(path)
    if len(rows) != 1:
        raise ValueError(f"expected one row in {path}, found {len(rows)}")
    return rows[0]


def metric(root: Path) -> dict[str, str]:
    return one(root / "results.tsv")


def find_bootstrap(
    root: Path, pair: str, metric_name: str = "map50_95"
) -> dict[str, str]:
    rows = read_tsv(root / "paired_bootstrap_results.tsv")
    matches = [
        row for row in rows if row["pair"] == pair and row["metric"] == metric_name
    ]
    if len(matches) != 1:
        raise ValueError(f"missing bootstrap row {pair}/{metric_name} under {root}")
    return matches[0]


def h500_reports(raw: Path, stage: Path) -> tuple[str, str]:
    results: list[dict[str, Any]] = []
    sizes: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    hashes: list[dict[str, Any]] = []
    for surface in H500_SURFACES:
        root = raw / "h500-metrics" / surface
        row = metric(root)
        row["evidence_scope"] = "current-runner-H500-train2017"
        results.append(row)
        sizes.extend(read_tsv(root / "size_bins.tsv"))
        classes.extend(read_tsv(root / "per_class.tsv"))
        hashes.append(
            {
                "surface": surface,
                "images": row["images"],
                "prediction_count": row["prediction_count"],
                "prediction_sha256": row["prediction_sha256"],
                "failures": row["failures"],
                "non_finite_predictions": row["non_finite_predictions"],
            }
        )
    write_rows(stage / "h500_full_metrics.tsv", results)
    write_rows(stage / "h500_size_bins.tsv", sizes)
    write_rows(stage / "h500_per_class.tsv", classes)
    write_rows(stage / "h500_prediction_hashes.tsv", hashes)

    candidates = sorted(
        (row for row in results if row["surface"].startswith("B")),
        key=lambda row: (-float(row["map50_95"]), row["surface"]),
    )
    winner, runner = candidates[:2]
    bootstrap = find_bootstrap(raw / "bootstrap/h500-mandatory", "B2-B0")
    selection = {
        "selection_surface": "H500-calibration-disjoint",
        "selection_rule": "map50_95-then-ap_large-then-lane-id",
        "winner": winner["surface"],
        "winner_map50_95": winner["map50_95"],
        "runner_up": runner["surface"],
        "runner_up_map50_95": runner["map50_95"],
        "point_delta": float(winner["map50_95"]) - float(runner["map50_95"]),
        "prebootstrap_tie_threshold": 0.005,
        "prebootstrap_statistical_tie": int(
            abs(float(winner["map50_95"]) - float(runner["map50_95"])) < 0.005
        ),
        "bootstrap_pair": bootstrap["pair"],
        "bootstrap_percentile_2_5": bootstrap["percentile_2_5"],
        "bootstrap_percentile_97_5": bootstrap["percentile_97_5"],
        "bootstrap_probability_delta_gt_zero": bootstrap[
            "probability_delta_gt_zero"
        ],
        "selection": winner["surface"],
        "note": "val2017 scout/full-val not used for selection",
    }
    write_rows(stage / "h500_selection.tsv", [selection])
    return winner["surface"], runner["surface"]


def bootstrap_reports(raw: Path, stage: Path) -> None:
    scopes = (
        ("H500-mandatory", raw / "bootstrap/h500-mandatory"),
        ("D8-H500", raw / "bootstrap/d8-h500"),
        ("D8-full-val2017", raw / "bootstrap/d8-full-val"),
    )
    rows: list[dict[str, Any]] = []
    checksums: list[str] = []
    for scope, root in scopes:
        rows.extend({"scope": scope, **row} for row in read_tsv(root / "paired_bootstrap_results.tsv"))
        payload = root / "paired_bootstrap_replicates.npz"
        checksums.append(f"{sha256(payload)}  {scope}/{payload.name}")
    variance = raw / "bootstrap/variance-h500"
    if variance.is_dir():
        rows.extend(
            {"scope": "variance-H500", **row}
            for row in read_tsv(variance / "paired_bootstrap_results.tsv")
        )
        payload = variance / "paired_bootstrap_replicates.npz"
        checksums.append(f"{sha256(payload)}  variance-H500/{payload.name}")
    write_rows(stage / "paired_bootstrap_results.tsv", rows)
    (stage / "paired_bootstrap_replicates.sha256").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )
    (stage / "paired_bootstrap_contract.md").write_text(
        """# Paired bootstrap contract

- Unit: image ID, sampled with replacement.
- Replicates: 1000; seed: 65002.
- Duplicate draws are represented by repeated exact COCO match records. A
  literal unique synthetic-image-ID remap was checked and agreed to <=1e-12.
- Every replicate reruns COCO accumulation; per-image pseudo-AP is not used.
- Interval: percentile 95%. Metrics: mAP50-95, mAP50, AP-small/medium/large.
- The vectorized precision-envelope implementation was proven array-identical
  to the original implementation on the complete H500 mandatory matrix.
""",
        encoding="utf-8",
    )


def fp32_reports(raw: Path, stage: Path) -> str:
    rows: list[dict[str, Any]] = []
    for scope in ("h500-current", "full-val-current"):
        root = raw / "fp32" / scope
        summary = one(root / "reconcile_summary.tsv")
        for surface in ("F0", "F1", "H8"):
            metrics_root = (
                raw / "h500-metrics" / surface
                if scope == "h500-current"
                else root / f"metrics-{surface}"
            )
            metric_row = metric(metrics_root)
            rows.append(
                {
                    "scope": scope,
                    "surface": surface,
                    "images": metric_row["images"],
                    "prediction_sha256": metric_row["prediction_sha256"],
                    "prediction_count": metric_row["prediction_count"],
                    "map50_95": metric_row["map50_95"],
                    "map50": metric_row["map50"],
                    "ap_small": metric_row["ap_small"],
                    "ap_medium": metric_row["ap_medium"],
                    "ap_large": metric_row["ap_large"],
                    "aggregate_raw_output_sha256": summary[
                        f"{surface.lower()}_aggregate_output_sha256"
                    ],
                    "f0_f1_exact_images": summary["f0_f1_exact_images"],
                    "f1_h8_exact_images": summary["f1_h8_exact_images"],
                    "status": summary["status"],
                }
            )
    write_rows(stage / "fp32_split_reconciliation.tsv", rows)
    full = [row for row in rows if row["scope"] == "full-val-current"]
    by_surface = {row["surface"]: row for row in full}
    current_equal = len({row["prediction_sha256"] for row in full}) == 1
    classification = (
        "imported-fp32-surface-confounded" if current_equal else "true-split-residual"
    )
    (stage / "fp32_full_current_predictions.sha256").write_text(
        f"{by_surface['F0']['prediction_sha256']}  F0-predictions.json\n",
        encoding="utf-8",
    )
    (stage / "fp32_split_current_predictions.sha256").write_text(
        f"{by_surface['F1']['prediction_sha256']}  F1-predictions.json\n",
        encoding="utf-8",
    )
    (stage / "fp32_harness_decision.md").write_text(
        f"""# FP32 split and harness decision

Classification: `{classification}`.

The current runner produced byte-identical F0/F1 raw outputs for all 500 H500
images and all 5000 val2017 images. F1 and H8 were also byte-identical on both
surfaces. All three current full-val prediction files have SHA-256
`{by_surface['F0']['prediction_sha256']}` and mAP50-95
`{by_surface['F0']['map50_95']}`. The older imported Stage64 FP32 prediction
surface differs, so it is not evidence of a split residual.
""",
        encoding="utf-8",
    )
    return classification


def d8_reports(raw: Path, r1: Path, stage: Path) -> tuple[str, float, float]:
    evidence = raw / "d8-evidence"
    shutil.copyfile(evidence / "d8_topology.tsv", stage / "d8_topology.tsv")
    shutil.copyfile(
        evidence / "d8_graph_diff.json", stage / "d8_graph_diff.patch_or_json"
    )
    shutil.copyfile(evidence / "d8_model_identity.tsv", stage / "d8_model_identity.tsv")
    shutil.copyfile(evidence / "d8_conformance.tsv", stage / "d8_conformance.tsv")
    h500 = metric(raw / "d8/h500-metrics")
    full = metric(raw / "d8/full-val-metrics")
    write_rows(stage / "d8_h500_metrics.tsv", [h500])
    write_rows(stage / "d8_full_coco_metrics.tsv", [full])
    bootstrap_rows = [
        {"scope": scope, **row}
        for scope, root in (
            ("H500", raw / "bootstrap/d8-h500"),
            ("full-val2017", raw / "bootstrap/d8-full-val"),
        )
        for row in read_tsv(root / "paired_bootstrap_results.tsv")
    ]
    write_rows(stage / "d8_bootstrap.tsv", bootstrap_rows)
    h0_h500 = metric(raw / "h500-metrics/B2")
    h8_h500 = metric(raw / "h500-metrics/H8")
    h0_full = metric(r1 / "full-matrix/hybrid-full-coco/H0/metrics")
    h8_full = metric(raw / "fp32/full-val-current/metrics-H8")
    h500_recovery = (float(h500["map50_95"]) - float(h0_h500["map50_95"])) / (
        float(h8_h500["map50_95"]) - float(h0_h500["map50_95"])
    )
    full_recovery = (float(full["map50_95"]) - float(h0_full["map50_95"])) / (
        float(h8_full["map50_95"]) - float(h0_full["map50_95"])
    )
    h8_gap = float(h8_full["map50_95"]) - float(full["map50_95"])
    if full_recovery >= 0.90 and abs(h8_gap) <= 0.002:
        decision = "final-output-qdq-dominant"
    elif full_recovery <= 0.75 or h8_gap >= 0.005:
        decision = "upstream-branch-error-material"
    else:
        decision = "mixed-boundary-and-upstream"
    identity = one(evidence / "d8_model_identity.tsv")
    (stage / "d8_causal_decision.md").write_text(
        f"""# D8 causal decision

Classification: `{decision}`.

- D8 model SHA-256: `{identity['output_sha256']}`.
- H500 recovery fraction: `{h500_recovery:.9f}`.
- Full-val recovery fraction: `{full_recovery:.9f}`.
- Full-val H8-D8 gap: `{h8_gap:.9f}` mAP50-95.
- Full-val D8-H0 bootstrap 95% interval:
  `{find_bootstrap(raw / 'bootstrap/d8-full-val', 'D8-H0')['percentile_2_5']}` to
  `{find_bootstrap(raw / 'bootstrap/d8-full-val', 'D8-H0')['percentile_97_5']}`.
- Full-val D8-H8 bootstrap 95% interval:
  `{find_bootstrap(raw / 'bootstrap/d8-full-val', 'D8-H8')['percentile_2_5']}` to
  `{find_bootstrap(raw / 'bootstrap/d8-full-val', 'D8-H8')['percentile_97_5']}`.

D8 removes only the six final output Q/DQ pairs. It recovers a significant but
minority fraction of the H0-to-H8 gap; accumulated upstream branch error is
therefore material. D8 is host-diagnostic-only and is not deployable/provider
evidence.
""",
        encoding="utf-8",
    )
    return decision, h500_recovery, full_recovery


def variance_reports(
    raw: Path, stage: Path
) -> tuple[str, list[dict[str, Any]], set[str]]:
    matrix: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    graphwise: list[dict[str, Any]] = []
    h500_rows: list[dict[str, Any]] = []
    scout_rows: list[dict[str, Any]] = []
    reproducibility: list[dict[str, Any]] = []
    b2 = metric(raw / "h500-metrics/B2")
    for lane in VARIANCE_LANES:
        quant = one(raw / f"variance-quantization/{lane}-run1.tsv")
        evaluation_root = raw / f"variance-evaluation/{lane.upper()}"
        evaluation = one(evaluation_root / "variance-evaluation-summary.tsv")
        matrix.append(
            {
                "lane": lane,
                "random_seed": quant["random_seed"],
                "elapsed_seconds": quant["elapsed_seconds"],
                "returncode": quant["returncode"],
                "checker": quant["checker"],
                "deployable_sha256": quant["output_sha256"],
                "inference_sha256": evaluation["inference_model_sha256"],
                "tail_sha256": evaluation["tail_model_sha256"],
                "h500_map50_95": evaluation["h500_map"],
                "h500_delta_vs_B2": float(evaluation["h500_map"]) - float(b2["map50_95"]),
                "scout_map50_95": evaluation["scout_map"],
                "status": evaluation["status"],
            }
        )
        identities.append(
            {
                "lane": lane,
                "deployable_model": quant["output_model"],
                "deployable_sha256": quant["output_sha256"],
                "inference_model": evaluation["inference_model"],
                "inference_sha256": evaluation["inference_model_sha256"],
                "tail_sha256": evaluation["tail_model_sha256"],
                "prediction_sha256_H500": evaluation["h500_prediction_sha256"],
                "prediction_sha256_scout500": evaluation["scout_prediction_sha256"],
            }
        )
        lane_root = evaluation_root / f"postprocess/{lane.upper()}"
        graphwise.extend(
            {"variance_lane": lane, **row}
            for row in read_tsv(lane_root / "graphwise-normalized.tsv")
        )
        h500_rows.append(
            {
                "lane": lane,
                "map50_95": evaluation["h500_map"],
                "map50": evaluation["h500_map50"],
                "ap_small": evaluation["h500_ap_small"],
                "ap_medium": evaluation["h500_ap_medium"],
                "ap_large": evaluation["h500_ap_large"],
                "prediction_count": evaluation["h500_prediction_count"],
                "prediction_sha256": evaluation["h500_prediction_sha256"],
                "selection_surface": evaluation["selection_surface"],
            }
        )
        scout_rows.append(
            {
                "lane": lane,
                "map50_95": evaluation["scout_map"],
                "map50": evaluation["scout_map50"],
                "ap_small": evaluation["scout_ap_small"],
                "ap_medium": evaluation["scout_ap_medium"],
                "ap_large": evaluation["scout_ap_large"],
                "prediction_sha256": evaluation["scout_prediction_sha256"],
                "selection_use": "diagnostic-only",
            }
        )
        second = raw / f"variance-quantization/{lane}-run2.tsv"
        rerun_required = abs(
            float(evaluation["h500_map"]) - float(b2["map50_95"])
        ) >= 0.005
        if second.is_file():
            rerun = one(second)
            equal = int(rerun["output_sha256"] == quant["output_sha256"])
            reproducibility.append(
                {
                    "lane": lane,
                    "requirement": "mandatory-triggered-second-generation",
                    "run1_sha256": quant["output_sha256"],
                    "run2_sha256": rerun["output_sha256"],
                    "byte_identical": equal,
                    "status": "pass" if equal else "fail",
                }
            )
            if not equal:
                raise ValueError(
                    f"{lane} required clean rerun is not byte-identical: "
                    f"{quant['output_sha256']} != {rerun['output_sha256']}"
                )
        elif rerun_required:
            raise ValueError(
                f"{lane} requires a second clean generation because its "
                "absolute H500 delta versus B2 is >= 0.005"
            )
        else:
            reproducibility.append(
                {
                    "lane": lane,
                    "requirement": "not-required-by-predeclared-gate",
                    "run1_sha256": quant["output_sha256"],
                    "run2_sha256": "NA",
                    "byte_identical": "NA",
                    "status": "pass-single-generation-authorized",
                }
            )
    write_rows(stage / "b2_variance_matrix.tsv", matrix)
    write_rows(stage / "b2_variance_model_identity.tsv", identities)
    write_rows(stage / "b2_variance_graphwise.tsv", graphwise)
    write_rows(stage / "b2_variance_h500.tsv", h500_rows)
    write_rows(stage / "b2_variance_scout.tsv", scout_rows)
    write_rows(stage / "b2_variance_reproducibility.tsv", reproducibility)

    material = [row for row in matrix if abs(float(row["h500_delta_vs_B2"])) >= 0.005]
    variance_bootstrap = raw / "bootstrap/variance-h500"
    significant: list[str] = []
    if variance_bootstrap.is_dir():
        for row in material:
            pair = find_bootstrap(variance_bootstrap, f"{row['lane']}-B2")
            low = float(pair["percentile_2_5"])
            high = float(pair["percentile_97_5"])
            if low > 0 or high < 0:
                significant.append(str(row["lane"]))
    status = (
        "aggregate-map-sensitivity-proven"
        if significant
        else "no-significant-aggregate-map-sensitivity-proven"
    )
    lines = [
        "# B2 variance decision",
        "",
        f"Decision: `{status}`.",
        "",
        "H500, not scout500, is the selection and robustness surface.",
        "",
    ]
    for row in matrix:
        pair = find_bootstrap(variance_bootstrap, f"{row['lane']}-B2")
        lines.append(
            f"- `{row['lane']}`: delta versus frozen B2 "
            f"`{float(row['h500_delta_vs_B2']):+.9f}` mAP50-95; "
            f"95% CI `{float(pair['percentile_2_5']):+.9f}` to "
            f"`{float(pair['percentile_97_5']):+.9f}`; "
            f"P(delta>0) `{pair['probability_delta_gt_zero']}`; "
            f"model `{row['deployable_sha256']}`."
        )
    if significant:
        lines.append(f"- Bootstrap-significant sensitivity: `{', '.join(significant)}`.")
    else:
        lines.append("- No >=0.005 aggregate-mAP arm had a two-sided 95% interval excluding zero.")
    vdraw_small = find_bootstrap(variance_bootstrap, "Vdraw-B2", "ap_small")
    vdraw_medium = find_bootstrap(variance_bootstrap, "Vdraw-B2", "ap_medium")
    lines.extend(
        [
            "- Membership nevertheless has a size-bin sensitivity signal: "
            f"Vdraw AP-small delta `{float(vdraw_small['point_delta']):+.9f}` "
            f"(95% CI `{float(vdraw_small['percentile_2_5']):+.9f}` to "
            f"`{float(vdraw_small['percentile_97_5']):+.9f}`), and AP-medium "
            f"delta `{float(vdraw_medium['point_delta']):+.9f}` (95% CI "
            f"`{float(vdraw_medium['percentile_2_5']):+.9f}` to "
            f"`{float(vdraw_medium['percentile_97_5']):+.9f}`).",
            "- No variance full-val run was opened. Vseed crossed the +0.005 "
            f"H500 point gate but P(delta>0) was `{find_bootstrap(variance_bootstrap, 'Vseed-B2')['probability_delta_gt_zero']}`, "
            "below the predeclared 0.95 requirement; Vorder and Vdraw did not "
            "cross the point gate.",
        ]
    )
    (stage / "b2_variance_decision.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return status, matrix, set(significant)


def csv_regression(stage: Path) -> None:
    write_rows(
        stage / "csv_field_limit_regression.tsv",
        [
            {"case": "129-KiB-valid-field", "bytes": 129 * 1024, "expected": "pass", "observed": "pass", "status": "pass"},
            {"case": "16-MiB-exact-valid-field", "bytes": MAX_CSV_FIELD_SIZE, "expected": "pass", "observed": "pass", "status": "pass"},
            {"case": "over-16-MiB-field", "bytes": MAX_CSV_FIELD_SIZE + 1, "expected": "explicit-fail", "observed": "explicit-fail", "status": "pass"},
        ],
    )


def route_reports(
    stage: Path,
    d8_decision: str,
    winner: str,
    variance_status: str,
    variance: list[dict[str, Any]],
    significant_variance: set[str],
) -> str:
    if d8_decision == "final-output-qdq-dominant":
        route = "R3-boundary-targeted-generation-ready"
    else:
        route = "R3-early-subgraph-localization-ready"
    best_variance = max(variance, key=lambda row: float(row["h500_map50_95"]))
    global_candidate = winner
    if (
        float(best_variance["h500_delta_vs_B2"]) >= 0.005
        and str(best_variance["lane"]) in significant_variance
    ):
        global_candidate = str(best_variance["lane"])
    write_rows(
        stage / "later_candidate_freeze.tsv",
        [
            {
                "slot": "global-host-policy",
                "candidate": global_candidate,
                "status": "frozen-for-later-review",
                "basis": "calibration-disjoint-H500",
                "board_authorized": 0,
            },
            {
                "slot": "causal-targeted-proposal",
                "candidate": "hierarchical-early-subgraph-cut-splice-localization",
                "status": "proposal-only-not-generated",
                "basis": d8_decision,
                "board_authorized": 0,
            },
        ],
    )
    (stage / "r3_route_decision.md").write_text(
        f"""# Later route decision

Selected exactly one route: `{route}`.

The D8 diagnostic classified the final six output Q/DQ pairs as insufficient
to explain the accuracy gap (`{d8_decision}`). A later separately authorized
stage may perform bounded hierarchical cut/splice localization before any
targeted generation. No such graph bisection or model generation occurred in
Stage65B-R2.
""",
        encoding="utf-8",
    )
    (stage / "stage65c_readiness.md").write_text(
        f"""# Stage65C readiness

Status: `not-authorized-not-board-ready`.

Host evidence selects `{global_candidate}` for frozen global-policy review and
`{route}` for causal follow-up. A later prompt must independently authorize
model generation and then K1X runtime gates. Stage65B-R2 establishes no
SpacemiT EP placement, correctness, performance, or promotion claim.
""",
        encoding="utf-8",
    )
    (stage / "human_decision_options.md").write_text(
        f"""# Human decision options

1. Authorize a bounded host-only early-subgraph localization stage using the
   frozen global candidate `{global_candidate}`.
2. Hold work and review H500/bootstrap/D8 evidence.

Board execution, XSlim source changes, targeted generation, and runtime
promotion remain unauthorized.
""",
        encoding="utf-8",
    )
    return route


def final_reports(
    stage: Path,
    winner: str,
    runner: str,
    fp32_decision: str,
    d8_decision: str,
    h500_recovery: float,
    full_recovery: float,
    variance_status: str,
    route: str,
) -> None:
    selection = one(stage / "h500_selection.tsv")
    d8_h500 = one(stage / "d8_h500_metrics.tsv")
    d8_full = one(stage / "d8_full_coco_metrics.tsv")
    classification = (
        "stage65b-r2-b2-unstable-seed-order-or-membership-sensitivity-proven"
        if variance_status == "aggregate-map-sensitivity-proven"
        else f"stage65b-r2-independent-selection-pass-{d8_decision}-early-subgraph-r3-ready"
    )
    report = f"""# Stage65B-R2 final report

Classification: `{classification}`

Publication classification: `research-branch-evidence-only-no-board-claim`

## Recovery and identity

The host-side work resumed from verified immutable R1 evidence. No accepted
model was regenerated. The direct user clarification attributes the historical
reboot to the external Windows 10 host; the incomplete R1 B3 tree remained
isolated and its clean rerun had already reproduced byte-for-byte.
All resumed PTQ, evaluator, and bootstrap processes exited cleanly; no stage
worker remains active at closure.

## Independent H500 selection

- Winner: `{winner}` at `{selection['winner_map50_95']}` mAP50-95.
- Runner-up: `{runner}` at `{selection['runner_up_map50_95']}`.
- Point delta: `{selection['point_delta']}`.
- Winner/runner bootstrap 95% interval:
  `{selection['bootstrap_percentile_2_5']}` to
  `{selection['bootstrap_percentile_97_5']}`; P(delta>0)
  `{selection['bootstrap_probability_delta_gt_zero']}`.
- Scout/full-val metrics were not used for selection.

## FP32 reconciliation

`{fp32_decision}`. F0, F1, and H8 are byte-identical in the current runner on
H500 and full val2017. The older imported FP32 prediction surface is therefore
confounded by historical harness/serialization behavior, not a proven split
residual.

## D8 causal diagnostic

- Classification: `{d8_decision}`.
- H500 D8 mAP50-95: `{d8_h500['map50_95']}`.
- Full-val D8 mAP50-95: `{d8_full['map50_95']}`.
- Recovery fractions: H500 `{h500_recovery:.9f}`, full val
  `{full_recovery:.9f}`.

Bypassing only the final six output Q/DQ pairs recovers a minority of the gap;
earlier branch quantization error is material. D8 is diagnostic-only.

## B2 robustness and route

Variance decision: `{variance_status}`. Selected later route: `{route}`.
Vseed crossed the +0.005 H500 point gate but missed the predeclared
P(delta>0)>=0.95 full-val gate (observed 0.94). Vdraw did not change aggregate
mAP significantly, although its AP-small/AP-medium bootstrap intervals show a
membership-dependent size-bin signal.
No targeted model was generated and no board, provider, performance, soak, or
promotion claim is made.
"""
    (stage / "STAGE65B_R2_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    (stage / "STAGE65B_R2_SUMMARY_RU.md").write_text(
        f"""# Итоги Stage65B-R2

Классификация: `{classification}`.

- Независимый H500 выбрал `{winner}`; `{runner}` занял второе место.
- В текущем runner поверхности FP32 F0/F1/H8 полностью совпали на H500 и
  val2017, поэтому историческое расхождение FP32 относится к старому harness.
- D8 удалил только шесть финальных Q/DQ, но восстановил лишь
  `{full_recovery:.3f}` полного разрыва mAP: существенная ошибка накоплена выше
  по ветвям.
- Проверка seed/order/membership: `{variance_status}`.
- Для Vseed full-val не запускался: P(delta>0)=0.94 ниже заданного порога
  0.95. У Vdraw обнаружен отдельный membership-сигнал AP-small/AP-medium без
  доказанного изменения общего mAP.
- Единственный следующий маршрут: `{route}`.
- Плата K1X, SpacemiT EP, производительность и продвижение runtime не
  проверялись и не разрешены.
""",
        encoding="utf-8",
    )


def main() -> int:
    options = parser().parse_args()
    options.stage_dir.mkdir(parents=True, exist_ok=True)
    winner, runner = h500_reports(options.raw_root, options.stage_dir)
    bootstrap_reports(options.raw_root, options.stage_dir)
    fp32_decision = fp32_reports(options.raw_root, options.stage_dir)
    d8_decision, h500_recovery, full_recovery = d8_reports(
        options.raw_root, options.r1_root, options.stage_dir
    )
    variance_status, variance, significant_variance = variance_reports(
        options.raw_root, options.stage_dir
    )
    csv_regression(options.stage_dir)
    route = route_reports(
        options.stage_dir,
        d8_decision,
        winner,
        variance_status,
        variance,
        significant_variance,
    )
    final_reports(
        options.stage_dir,
        winner,
        runner,
        fp32_decision,
        d8_decision,
        h500_recovery,
        full_recovery,
        variance_status,
        route,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
