#!/usr/bin/env python3
"""Create compact Stage65E closure, passport, and human-decision reports."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
from pathlib import Path


STAGE_ID = (
    "BANANA-YOLO26-XSLIM-STAGE65E-FP32-OPERATING-POINT-LEDGER-B2-C2-"
    "UNCONDITIONAL-PERFORMANCE-STABILITY-AND-FUSION-FEASIBILITY-CLOSURE-001"
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty TSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def report_decision(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return "pass" if "Decision: `pass`" in text else "fail"


def markdown_status(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Status: `") and line.endswith("`."):
            return line.removeprefix("Status: `").removesuffix("`.")
    raise RuntimeError(f"missing status in {path}")


def format_metric(value: str | float) -> str:
    return f"{float(value):.9f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracked-root", required=True, type=Path)
    parser.add_argument("--stage-raw-root", required=True, type=Path)
    parser.add_argument("--start-head", required=True)
    parser.add_argument("--shared-log", required=True, type=Path)
    options = parser.parse_args()
    root = options.tracked_root
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    accuracy = read_tsv(root / "accuracy_absolute.tsv")
    by_surface = {row["surface"]: row for row in accuracy}
    delta_b2 = {row["surface"]: row for row in read_tsv(root / "accuracy_delta_to_b2.tsv")}
    gap = {row["surface"]: row for row in read_tsv(root / "accuracy_gap_recovery.tsv")}
    ratios = read_tsv(root / "performance_ratios.tsv")
    ratio_by_key = {(row["metric"], row["statistic"]): row for row in ratios}
    performance = report_decision(root / "performance_decision.md")
    stability = report_decision(root / "stability_decision.md")
    fusion_text = (root / "fusion_feasibility_decision.md").read_text(encoding="utf-8")
    fusion_justified = "Status: `stage65f-justified`" in fusion_text

    failed_guardrails = [row for row in ratios if row["decision"] == "fail"]
    if stability != "pass":
        classification = "stage65e-vendor-passport-partial-runtime-performance-or-stability-fail-closed"
    elif performance != "pass" or failed_guardrails:
        classification = (
            "stage65e-vendor-passport-complete-b2-c2-performance-or-stability-"
            "difference-material-human-decision-required"
        )
    else:
        classification = (
            "stage65e-vendor-passport-complete-performance-stability-and-fusion-"
            "feasibility-closed-human-profile-decision-required"
        )

    frozen = read_tsv(root / "frozen_model_identity.tsv")
    model_hash = {
        (row["surface"], row["kind"]): row["sha256"]
        for row in frozen
    }
    passport_rows: list[dict[str, object]] = []
    for model, host, board in (
        ("B2", "B2_HOST", "B2_BOARD_EP"),
        ("C2", "C2_HOST", "C2_BOARD_EP"),
        ("A1", "A1_HOST", "A1_BOARD_EP"),
    ):
        row = by_surface[host]
        board_row = by_surface[board]
        passport_rows.append({
            "artifact": model,
            "deployable_sha256": model_hash[(model, "deployable")],
            "inference_sha256": model_hash[(model, "inference")],
            "tail_sha256": model_hash[("common", "tail")],
            "host_map50_95": row["map50_95"],
            "board_ep_map50_95": board_row["map50_95"],
            "board_ep_ap_small": board_row["ap_small"],
            "board_ep_ap_medium": board_row["ap_medium"],
            "board_ep_ap_large": board_row["ap_large"],
            "board_ep_ar_small": board_row["ar_small"],
            "board_ep_ar_medium": board_row["ar_medium"],
            "board_ep_ar_large": board_row["ar_large"],
            "placement": "one-925-node-SpacemiT-partition" if model in {"B2", "C2"} else "historical-accepted",
            "performance": performance if model in {"B2", "C2"} else "not-rerun-historical-artifact",
            "stability": stability if model in {"B2", "C2"} else "not-rerun-historical-artifact",
            "disposition": {
                "B2": "universal-vendor-control",
                "C2": "frozen-higher-AP-waiver-only-not-promoted",
                "A1": "historical-frozen-artifact",
            }[model],
        })
    write_tsv(root / "vendor_lane_final_passport.tsv", passport_rows)

    operating = read_tsv(root / "operating_point_census.tsv")
    op_rows = {
        row["surface"]: row
        for row in operating
        if row["score_threshold"] == "0.25"
        and row["iou_threshold"] == "0.5"
        and row["max_dets"] == "100"
        and row["area"] == "all"
    }
    report_surfaces = (
        "FP32_HOST",
        "B2_HOST",
        "A1_HOST",
        "C2_HOST",
        "B2_BOARD_EP",
        "C2_BOARD_EP",
    )
    accuracy_table = [
        f"| {surface} | {format_metric(by_surface[surface]['map50_95'])} | "
        f"{format_metric(by_surface[surface]['ap_large'])} | "
        f"{format_metric(by_surface[surface]['ar_small'])} | "
        f"{format_metric(by_surface[surface]['ar_large'])} |"
        for surface in report_surfaces
    ]
    op_table = [
        f"| {surface} | {op_rows[surface]['tp']} | {op_rows[surface]['fp']} | "
        f"{op_rows[surface]['fn']} | {float(op_rows[surface]['precision']):.6f} | "
        f"{float(op_rows[surface]['recall']):.6f} | {float(op_rows[surface]['f1']):.6f} |"
        for surface in report_surfaces
    ]
    inference_median = ratio_by_key[("inference", "median")]
    inference_p95 = ratio_by_key[("inference", "p95")]
    two_stage_median = ratio_by_key[("two_stage", "median")]
    two_stage_p95 = ratio_by_key[("two_stage", "p95")]
    b2_soak = next(row for row in read_tsv(root / "b2_10k_soak.tsv") if row["metric"] == "two_stage")
    c2_soak = next(row for row in read_tsv(root / "c2_10k_soak.tsv") if row["metric"] == "two_stage")
    stable_hashes = read_tsv(root / "output_hash_stability.tsv")
    stable = all(row["status"] == "pass" for row in stable_hashes)
    application_performance = {
        (row["surface"], row["role"]): row
        for row in read_tsv(root / "cross_surface_application_performance_table.tsv")
    }
    custom_pure = application_performance[("accepted-custom-engine", "custom-pure-executor")]
    custom_total = application_performance[
        ("accepted-custom-engine", "custom-total-model-to-1x300x6")
    ]
    optimization = read_tsv(root / "ort_optimization_matrix.tsv")
    accepted_opt_levels = [
        f"{row['model']}:{row['opt_level']}"
        for row in optimization
        if row["decision"] == "pass"
    ]
    offline_status = markdown_status(root / "offline_optimization_capability.md")
    iobinding_status = markdown_status(root / "iobinding_capability.md")
    ep_context_status = markdown_status(root / "ep_context_capability.md")

    main_report = f"""# Stage65E final report

Classification: `{classification}`

Publication classification: `not-authorized-not-attempted`

Stage: `{STAGE_ID}`

## Scope and immutable inputs

Stage65E did not retry or relax the historical Stage65D-R1 accuracy gate. It used the exact frozen FP32, B2, A1, C2 and common-tail bytes, kept XSlim and the custom executor read-only, used no camera, and ran the unconditional B2/C2 performance and stability passport after runtime/placement controls passed.

## Accuracy ledger

The frozen FP32 prediction re-accumulated to exact mAP50-95 `{by_surface['FP32_HOST']['map50_95']}` with the accepted prediction hash. All ten frozen host/board surfaces matched their accepted aggregate metrics.

| Surface | mAP50-95 | AP-L | AR-S | AR-L |
|---|---:|---:|---:|---:|
{chr(10).join(accuracy_table)}

C2 host improves over B2 by `{format_metric(delta_b2['C2_HOST']['map50_95'])}` mAP and recovers `{float(gap['C2_HOST']['fraction_host_gap_recovered']):.2%}` of the B2-to-FP32 mAP gap. On board EP the mAP gain is `{format_metric(delta_b2['C2_BOARD_EP']['map50_95'])}`. The historical universal gate remains failed because the predeclared AR-small and AR-large point/interval requirements were not all met; C2 is therefore a higher-AP profile candidate, not a universal replacement.

At score 0.25, IoU 0.50, maxDets 100, area all:

| Surface | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(op_table)}

The complete census covers five score thresholds, two IoUs, maxDets 100/300, four area bins, and per-class rows using COCOeval `evalImgs` match/ignore arrays.

## Runtime, performance and stability

B2 and C2 each re-attested as one equal 925-source-node SpaceMIT partition with zero unexpected CPU inference events; the common 34-node FP32 tail remained intentional CPU work.

Matched C2/B2 ratios are inference median `{float(inference_median['c2_b2_ratio']):.6f}`, inference p95 `{float(inference_p95['c2_b2_ratio']):.6f}`, two-stage median `{float(two_stage_median['c2_b2_ratio']):.6f}`, and two-stage p95 `{float(two_stage_p95['c2_b2_ratio']):.6f}`. Performance decision: `{performance}`. The decision uses eight B2/B2 and eight C2/C2 process/session noise-floor blocks plus twelve order-balanced ABBA blocks.

Two earlier incomplete harness roots are explicitly excluded: one exposed an orphaned watchdog-sleep design defect and one exposed a normal `/proc` process-exit sampling race. The clean accepted root was fresh, complete and independent; neither partial root contributes a timing row. No reboot is inferred from either root.

The first read-only custom-context root is also excluded as a tooling-output collision: omitting `--output-json` caused the executor's `/dev/stdout` JSON writer to truncate the redirected benchmark stream. A clean v2 run separated JSON and produced the complete deterministic 5x100 timing grid; no model, package or executable bytes changed.

Both models completed reversed-order short soaks and ten clean-session 1000-run segments. B2 10k two-stage median is `{float(b2_soak['p50_us']):.3f}` us; C2 is `{float(c2_soak['p50_us']):.3f}` us. Stability decision: `{stability}`; stable output/resource contract: `{'pass' if stable else 'fail'}`.

## Read-only application context

The accepted custom package remains `0.10.0-internal-rd.1`, contract `K1X_INT8_V1`, with model SHA `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`. Its fixed-input median is `{float(custom_pure['median_us']):.3f}` us for the custom pure executor and `{float(custom_total['median_us']):.3f}` us to `1x300x6`. Same-boot timing is application-level context only because model lineage, export, quantization, runtime, affinity and output implementation differ. No direct speedup ratio is claimed, and no rebuild or source mutation occurred.

## Fusion feasibility

Stage65F opening condition: `{'met-but-not-authorized' if fusion_justified else 'not-met'}`. Exact accepted ORT-level rows are `{', '.join(accepted_opt_levels)}`; BASIC/EXTENDED/ALL changed six-boundary bytes and were rejected even where a point timing looked lower. Offline optimization is `{offline_status}`, I/O Binding is `{iobinding_status}`, and EPContext is `{ep_context_status}`. Shipped plugin APIs and XSlim YoloDecode were capability-audited without changing accepted artifacts. The measured tail share is an upper bound only; no exact-tail implementation gain was projected. See `fusion_opportunity_ledger.tsv`.

## Disposition

- B2 remains the universal vendor control.
- C2 remains the frozen, best accepted same-source INT8 AP/mAP artifact and may be considered only through an explicit application waiver after reviewing TP/FP/FN and recall costs.
- A1 remains historical.
- Vendor PTQ search and provider-numerics localization remain closed.
- No runtime, model, XSlim release, custom executor, camera path or fusion implementation was promoted.

Raw evidence: `{options.stage_raw_root}`. Shared log: `{options.shared_log}`. Result-packet identity is recorded by the packet's own manifest and post-push attestation to avoid a self-referential tracked hash.

Timestamp: `{timestamp}`.
"""
    (root / "STAGE65E_FINAL_REPORT.md").write_text(main_report, encoding="utf-8")

    ru_report = f"""# Краткий отчет Stage65E

Классификация: `{classification}`.

Stage не менял модели, qparams, XSlim или custom executor и не пересматривал исторический отказ Stage65D-R1. FP32 повторно посчитан тем же COCO evaluator и точно совпал: mAP50-95 `{by_surface['FP32_HOST']['map50_95']}`.

C2 остается лучшим зафиксированным INT8-вариантом по mAP: на host он выше B2 на `{format_metric(delta_b2['C2_HOST']['map50_95'])}`, на SpaceMIT EP платы — на `{format_metric(delta_b2['C2_BOARD_EP']['map50_95'])}`. Но универсальным профилем C2 не стал: ранее заданные ограничения recall, прежде всего AR-small и граничный AR-large, полностью не выполнены. Поэтому B2 остается универсальным контролем, а C2 допускается только как отдельный профиль с осознанным человеческим waiver.

Размещение B2 и C2 одинаковое: один SpaceMIT-раздел на 925 исходных узлов, неожиданного CPU fallback нет. Сопоставимый ABBA-тест дал отношение медиан C2/B2 `{float(inference_median['c2_b2_ratio']):.6f}` для inference и `{float(two_stage_median['c2_b2_ratio']):.6f}` для inference+tail. Решение по производительности: `{performance}`. Оба варианта прошли короткие взаимно переставленные прогоны и по 10 000 запусков в десяти чистых сессиях; решение по стабильности: `{stability}`.

Сравнение с custom engine является только сравнением готовых приложений: модели, квантизация, affinity и backend различаются. Его медиана составила `{float(custom_pure['median_us']):.3f}` мкс для pure executor и `{float(custom_total['median_us']):.3f}` мкс до `1x300x6`; прямой коэффициент ускорения не заявляется. Первый custom-context root исключен: без `--output-json` JSON и benchmark конфликтовали в stdout. Чистый v2 дал полную детерминированную сетку 5x100 без изменения package или executable.

Камера не использовалась. Из уровней ORT точные boundary-выходы сохранил только DISABLE_ALL; BASIC/EXTENDED/ALL отклонены из-за численного drift. Offline optimization: `{offline_status}`, I/O Binding: `{iobinding_status}`, EPContext: `{ep_context_status}`. Реализация и продвижение не выполнялись. Stage65F: `{'обоснован, но не разрешен' if fusion_justified else 'не обоснован текущими порогами'}`.

Весь подробный паспорт TP/FP/FN, latency, ресурсов и capability находится в TSV/MD этого Stage. Время: `{timestamp}`.
"""
    (root / "STAGE65E_SUMMARY_RU.md").write_text(ru_report, encoding="utf-8")

    passport_md = f"""# Vendor lane final passport

## Accepted roles

- **B2:** universal vendor control; exact model hashes are in `vendor_lane_final_passport.tsv`.
- **C2:** frozen higher-AP candidate. Host mAP gain versus B2 is `{format_metric(delta_b2['C2_HOST']['map50_95'])}` and board-EP gain is `{format_metric(delta_b2['C2_BOARD_EP']['map50_95'])}`; universal recall non-inferiority did not pass.
- **A1:** historical frozen research artifact.

## Execution passport

Placement: pass, one equal 925-node partition. Performance: `{performance}`. Stability: `{stability}`. Camera: not used. Runtime promotion: not authorized. Vendor PTQ candidate search and provider-numerics lane: closed.

The accepted custom engine is retained as a different-surface application reference, never as an engine-only or quantizer-only comparison.
"""
    (root / "VENDOR_LANE_FINAL_PASSPORT.md").write_text(passport_md, encoding="utf-8")

    waiver = f"""# C2 high-AP profile human waiver template

This template does not grant a waiver or promote C2.

## Frozen identity

- C2 deployable: `{model_hash[('C2', 'deployable')]}`
- C2 inference: `{model_hash[('C2', 'inference')]}`
- Common tail: `{model_hash[('common', 'tail')]}`
- B2 universal control inference: `{model_hash[('B2', 'inference')]}`

## Required application decision

- Application and version:
- Critical classes:
- Selected score/IoU/maxDets operating point:
- False-negative cost and permitted FN increase:
- False-positive cost and permitted FP increase:
- Minimum recall by object-size/application bin:
- Accepted C2-versus-B2 AP/AR deltas:
- TP/FP/FN rows reviewed from `operating_point_census.tsv`:
- Performance and 10k stability evidence reviewed:
- Rollback trigger and B2 rollback procedure:
- Reviewer, timestamp and explicit approval:

The waiver must be application-specific, must not describe C2 as the universal vendor baseline, and cannot authorize runtime publication, model mutation, or camera claims.
"""
    (root / "C2_HIGH_AP_PROFILE_HUMAN_WAIVER_TEMPLATE.md").write_text(waiver, encoding="utf-8")

    options_text = """# Human decision options

1. Retain B2 as the only universal vendor profile. This requires no waiver.
2. Review C2 as a separate higher-AP/slight-recall-trade-off application profile using the provided waiver template and an application-specific operating point.
3. Keep C2 research-only. This preserves the complete passport without changing deployed defaults.
4. Consider a separately authorized runtime/fusion Stage only when `STAGE65F_CHARTER_DRAFT.md` says the quantitative opening rule was met.

No option is automatic. Head-only QAT, co-design, a new PTQ search, camera validation and same-source K1X_INT8_V2 remain separately authorized work.
"""
    (root / "human_decision_options.md").write_text(options_text, encoding="utf-8")

    readiness = f"""# Stage readiness or blocker

Stage65E classification: `{classification}`.

The evidence passport is complete. B2 remains universal. C2 is waiver-ready only as a separate higher-AP application profile because the historical universal recall gate remains immutable. Stage65F feasibility opening: `{'justified-but-not-authorized' if fusion_justified else 'not-justified-not-opened'}`. No later Stage is executed or authorized here.
"""
    (root / "stage_readiness_or_blocker.md").write_text(readiness, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
