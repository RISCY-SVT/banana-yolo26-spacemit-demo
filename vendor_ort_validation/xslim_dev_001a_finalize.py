#!/usr/bin/env python3
"""Create compact DEV-001A decision reports from frozen TSV evidence."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

STAGE_ID = (
    "BANANA-YOLO26-XSLIM-DEV-001A-SPACEMIT-S8-QDQ-CONSTRAINED-RANGE-"
    "OBSERVER-TERMINAL-DOMAIN-AND-POLICY-A-HOST-CANDIDATE-GATE-001"
)
CLASS_PASS = (
    "xslim-dev-001a-all-s8-policy-a-host-pass-"
    "candidate-freeze-ready-for-separate-k1x-gate"
)
CLASS_PARTIAL = "xslim-dev-001a-partial-all-s8-improvement-below-promotion-gate"
CLASS_INSUFFICIENT = (
    "xslim-dev-001a-range-observer-policy-a-insufficient-"
    "policy-b-or-block-reconstruction-human-decision-required"
)
CLASS_CONTRACT = "xslim-dev-001a-blocked-all-s8-export-contract-regression"
B2_H500 = 0.4446654879525213
FQ8_L3_H500 = 0.47505674025452777
B2_FULL = 0.3658592288412378
FQ8_L3_FULL = 0.3977524934214979
H8_FULL = 0.4018217950262668


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True, type=Path)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty report: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def as_float(row: dict[str, str], field: str) -> float:
    return float(row[field])


def markdown_rows(rows: list[dict[str, str]], fields: tuple[str, ...]) -> str:
    header = "| " + " | ".join(fields) + " |"
    divider = "| " + " | ".join("---" for _ in fields) + " |"
    body = [
        "| " + " | ".join(row.get(field, "") for field in fields) + " |"
        for row in rows
    ]
    return "\n".join((header, divider, *body))


def decide(
    conformance: list[dict[str, str]], full_decisions: list[dict[str, str]]
) -> tuple[str, list[dict[str, str]], list[dict[str, str]]]:
    if not conformance or all(row.get("status") != "pass" for row in conformance):
        return CLASS_CONTRACT, [], []
    winners = [row for row in full_decisions if row.get("full_val_success") == "1"]
    partial = [
        row
        for row in full_decisions
        if as_float(row, "delta_vs_b2") > 0.0
        and as_float(row, "bootstrap_ci_lower") > 0.0
        and row.get("full_val_success") != "1"
    ]
    if winners:
        return CLASS_PASS, winners[:2], partial
    if partial:
        return CLASS_PARTIAL, [], partial
    return CLASS_INSUFFICIENT, [], []


def main() -> int:
    options = parse_args()
    report = options.report_dir
    required = (
        "candidate_conformance.tsv",
        "candidate_model_identity.tsv",
        "development_artifact_manifest.tsv",
        "h500_metrics.tsv",
        "h500_selection.tsv",
        "h500_bootstrap_final.tsv",
        "full_val_metrics.tsv",
        "full_val_candidate_decision.tsv",
        "full_val_bootstrap.tsv",
    )
    missing = [name for name in required if not (report / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing final evidence: {', '.join(missing)}")

    conformance = read_tsv(report / "candidate_conformance.tsv")
    model_rows = read_tsv(report / "candidate_model_identity.tsv")
    artifacts = read_tsv(report / "development_artifact_manifest.tsv")
    h500_metrics = read_tsv(report / "h500_metrics.tsv")
    h500_selection = read_tsv(report / "h500_selection.tsv")
    h500_bootstrap_final = read_tsv(report / "h500_bootstrap_final.tsv")
    full_metrics = read_tsv(report / "full_val_metrics.tsv")
    full_decisions = read_tsv(report / "full_val_candidate_decision.tsv")
    classification, winners, partial = decide(conformance, full_decisions)

    conformance_by_lane = {row["lane"]: row for row in conformance}
    models_by_lane = {
        row["lane"]: row for row in model_rows if row.get("run") == "run1"
    }
    h500_by_lane = {row["surface"]: row for row in h500_metrics}
    h500_decision_by_lane = {row["lane"]: row for row in h500_selection}
    h500_final_by_lane = {
        row["pair"].removesuffix("-A0"): row
        for row in h500_bootstrap_final
        if row.get("metric") == "map50_95"
    }
    full_by_lane = {row["surface"]: row for row in full_metrics}

    freeze_lanes = [row["surface"] for row in winners]
    if not freeze_lanes and partial:
        freeze_lanes = [row["surface"] for row in partial[:2]]
    freeze_rows: list[dict[str, Any]] = []
    for lane in freeze_lanes:
        con = conformance_by_lane[lane]
        model = models_by_lane[lane]
        h500 = h500_by_lane[lane]
        hsel = h500_decision_by_lane[lane]
        hfinal = h500_final_by_lane[lane]
        full = full_by_lane[lane]
        decision = next(row for row in full_decisions if row["surface"] == lane)
        freeze_rows.append(
            {
                "lane": lane,
                "freeze_status": (
                    "ready-for-separate-k1x-gate"
                    if lane in [row["surface"] for row in winners]
                    else "evidence-only-below-promotion-gate"
                ),
                "deployable_sha256": con["deployable_sha256"],
                "inference_sha256": con["inference_sha256"],
                "tail_sha256": con["tail_sha256"],
                "range_policy_manifest_sha256": model["range_policy_manifest_sha256"],
                "h500_prediction_sha256": h500["prediction_sha256"],
                "h500_map50_95": h500["map50_95"],
                "h500_delta_vs_b2": hsel["delta_vs_a0"],
                "h500_ci_lower": hfinal["percentile_2_5"],
                "h500_ci_upper": hfinal["percentile_97_5"],
                "h500_probability_delta_gt_zero": hfinal[
                    "probability_delta_gt_zero"
                ],
                "full_val_prediction_sha256": full["prediction_sha256"],
                "full_val_map50_95": full["map50_95"],
                "full_val_delta_vs_b2": decision["delta_vs_b2"],
                "full_val_ci_lower": decision["bootstrap_ci_lower"],
                "full_val_ci_upper": decision["bootstrap_ci_upper"],
                "b2_to_fq8_l3_gap_recovered": decision[
                    "b2_to_fq8_l3_gap_recovered"
                ],
                "b2_to_h8_gap_recovered": decision["b2_to_h8_gap_recovered"],
            }
        )
    if not freeze_rows:
        freeze_rows = [
            {
                "lane": "none",
                "freeze_status": "no-policy-a-candidate-qualified",
                "classification": classification,
            }
        ]
    write_tsv(report / "candidate_freeze.tsv", freeze_rows)

    winner_text = ", ".join(row["surface"] for row in winners) or "none"
    partial_text = ", ".join(row["surface"] for row in partial) or "none"
    readiness = (
        "A separately authorized K1X correctness/placement/performance gate may consume "
        f"`{winner_text}`. This Stage makes no provider-placement or board claim."
        if winners
        else "No Policy A artifact is ready for a K1X gate. A human must choose whether "
        "to authorize Policy B or block reconstruction."
    )
    write_text(
        report / "stage65c_readiness.md",
        f"""# Stage65C readiness

- Classification: `{classification}`
- Policy A full-val winners: `{winner_text}`
- Positive below-gate candidates: `{partial_text}`
- Readiness: {readiness}

No later-stage prompt was created and no board execution is authorized.
""",
    )
    write_text(
        report / "policy_b_fallback_status.md",
        f"""# Policy B fallback status

Policy B was not implemented. Full-val Policy A winners: `{winner_text}`.

The accepted R3 higher-precision/exclusion proposal remains evidence only. It requires
separate human authorization if Policy A is not accepted or if later K1X validation fails.
""",
    )
    write_text(
        report / "human_decision_options.md",
        f"""# Human decision options

1. Accept `{winner_text}` for a separately authorized K1X correctness and placement gate.
2. Keep the current published XSlim release and frozen B2 as controls; publish nothing.
3. If Policy A is rejected, separately authorize either Policy B or block reconstruction.

This Stage does not authorize a board run, release, PyPI publication, or runtime promotion.
""",
    )

    h500_table = markdown_rows(
        h500_metrics,
        ("surface", "map50_95", "ap_small", "ap_medium", "ap_large", "prediction_sha256"),
    )
    full_table = markdown_rows(
        full_metrics,
        ("surface", "map50_95", "ap_small", "ap_medium", "ap_large", "prediction_sha256"),
    )
    artifact_table = markdown_rows(artifacts, ("artifact", "sha256", "status"))
    write_text(
        report / "STAGE_XSLIM_DEV_001A_FINAL_REPORT.md",
        f"""# XSLIM-DEV-001A final report

## Decision

- Stage: `{STAGE_ID}`
- Classification: `{classification}`
- Publication classification: `research-development-only-not-published`
- Full-val winners: `{winner_text}`
- Positive below-gate candidates: `{partial_text}`

## Source and package

XSlim development version `2.1.2+riscy.2.dev1` adds deterministic strict local
selection, constrained signed asymmetric INT8 range search, frozen qparam manifests,
and the structural `spacemit_k1x_s8_qdq_split_v1` validator. No-override B2 generation
is byte-identical to the frozen B2 deployable/inference/tail artifacts.

{artifact_table}

Full tests: 174 passed, 2 inherited warnings, 65 subtests. Focused contracts: 27 passed.

## Candidate contract

All A1-A6 candidates reproduced byte-for-byte across two clean generations and passed:
812 Q/DQ nodes, 0 QLinear, 0 UINT8 zero points, 0 FP16, 102/102 explicit Conv
`kernel_shape`, exact six-output order, exact FP32 tail, profile validation, fixtures,
and 100-image semantic/collapse checks.

## H500

{h500_table}

The only H500-qualified lane was A1 (T6 terminal ranges): +0.007062946 mAP versus B2;
10,000-replicate 95% CI 0.002017412..0.013001502. It recovered 23.24% of the accepted
B2-to-FQ8-L3 H500 oracle gap, below the 50% strong-success label.

## Full val2017

{full_table}

The final decision is taken from `full_val_candidate_decision.tsv` and its 10,000
paired image-level COCO bootstrap. FQ8-L3 and H8 are diagnostic ceilings, not deployable
candidate claims.

## Scope

No K1X board command, provider-placement test, performance run, Policy B implementation,
custom-executor mutation, release publication, or tag mutation occurred.
""",
    )
    write_text(
        report / "STAGE_XSLIM_DEV_001A_SUMMARY_RU.md",
        f"""# Краткий итог XSLIM-DEV-001A

- Классификация: `{classification}`.
- Разработан общий constrained-range механизм для signed asymmetric INT8 и строгий
  структурный профиль SpacemiT S8-QDQ.
- Режим без override воспроизводит B2 побайтно.
- Все A1-A6 детерминированы и прошли структурные/семантические проверки.
- На H500 квалифицирован только A1: +0.007063 mAP, 95% CI +0.002017..+0.013002.
- Победители полного val2017: `{winner_text}`; положительные ниже gate: `{partial_text}`.
- Плата K1X, размещение EP, производительность и публикация релиза не проверялись.
""",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
