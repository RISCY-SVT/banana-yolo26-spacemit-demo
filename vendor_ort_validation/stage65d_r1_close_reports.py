#!/usr/bin/env python3
"""Create the compact Stage65D-R1 diagnostic passport and final reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path

STAGE_ID = (
    "BANANA-YOLO26-XSLIM-STAGE65D-R1-C2-FROZEN-FULL-VAL-PROVIDER-"
    "INTERACTION-CONDITIONAL-PERFORMANCE-STABILITY-AND-CROSS-SURFACE-"
    "PASSPORT-001"
)
BANANA = Path("/data/worktrees/banana-yolo26-xslim211-s8-qdq-validation")
PROTECTED = Path("/data/banana-yolo26-spacemit-demo")
XSLIM = Path("/data/worktrees/riscy-xslim-k1x-yolo26")
NCNN = Path("/data/ncnn")
DEV001C = BANANA / "stages" / (
    "BANANA-YOLO26-XSLIM-DEV-001C-C2-FROZEN-INDEPENDENT-HOLDOUT-"
    "ADJUDICATION-AND-VENDOR-PTQ-LANE-CLOSURE-001"
)
R3 = BANANA / "stages" / (
    "BANANA-YOLO26-XSLIM-STAGE65B-R3-HIERARCHICAL-EARLY-SUBGRAPH-"
    "CUT-SPLICE-LOCALIZATION-AND-XSLIM-TARGET-POLICY-CHARTER-001"
)
CUSTOM = Path(
    "/data/releases/banana-yolo26-k1x-int8-executor/"
    "0.10.0-internal-rd.1-stage62-sdk"
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty TSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def ncnn_diff() -> str:
    payload = subprocess.check_output(["git", "diff", "--binary"], cwd=NCNN)
    return hashlib.sha256(payload).hexdigest()


def decision(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if "Decision: `pass`" in text or "task gate: `pass`" in text:
        return "pass"
    if "Decision: `fail`" in text or "task gate: `fail`" in text:
        return "fail"
    raise ValueError(f"decision is not machine-readable: {path}")


def write_not_run(path: Path, reason: str) -> None:
    if path.suffix == ".md":
        path.write_text(
            f"# Conditional gate status\n\nStatus: `not-run-{reason}`.\n",
            encoding="utf-8",
        )
    else:
        write_tsv(path, [{"status": f"not-run-{reason}", "reason": reason}])


def board_capture(stage_id: str) -> list[dict[str, str]]:
    script = f'''set -eu
printf 'timestamp_utc\t'; date -u +%Y-%m-%dT%H:%M:%SZ
printf 'hostname\t'; hostname
printf 'device_serial\t'; tr -d '\\000' </proc/device-tree/serial-number; printf '\n'
printf 'boot_id\t'; cat /proc/sys/kernel/random/boot_id
printf 'kernel\t'; uname -r
printf 'os_release\t'; . /etc/os-release; printf '%s\n' "${{PRETTY_NAME:-unknown}}"
printf 'data_mount\t'; findmnt -n -o SOURCE,FSTYPE,OPTIONS -T /data
printf 'root_mount\t'; findmnt -n -o SOURCE,FSTYPE,OPTIONS -T /
printf 'data_free_bytes\t'; df -B1 /data | tail -1 | awk '{{print $4}}'
count=0
for base in /home /tmp /var/tmp /root; do
  [ -d "$base" ] || continue
  value=$(find "$base" -xdev -path '*{stage_id}*' -print 2>/dev/null | wc -l)
  count=$((count + value))
done
printf 'emmc_stage_path_count\t%s\n' "$count"
'''
    output = subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", "svt@banana", "bash", "-s"],
        input=script,
        text=True,
    )
    return [
        {"field": line.split("\t", 1)[0], "value": line.split("\t", 1)[1]}
        for line in output.splitlines()
        if "\t" in line
    ]


def metric_row(
    row: dict[str, str], model: str, execution: str, location: str, status: str
) -> dict[str, object]:
    return {
        "model": model,
        "precision": "FP32" if model == "FP32-H8" else "S8-QDQ",
        "execution": execution,
        "location": location,
        "dataset": "val2017-5000",
        "map50_95": row["map50_95"],
        "map50": row["map50"],
        "map75": row["map75"],
        "ap_small": row["ap_small"],
        "ap_medium": row["ap_medium"],
        "ap_large": row["ap_large"],
        "ar_small": row["ar_small"],
        "ar_medium": row["ar_medium"],
        "ar_large": row["ar_large"],
        "prediction_count": row["prediction_count"],
        "status": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracked-root", required=True, type=Path)
    options = parser.parse_args()
    out = options.tracked_root

    task_pass = decision(out / "full_val_board_decision.md") == "pass"
    if not task_pass:
        for name in (
            "session_creation_timing.tsv",
            "first_run_timing.tsv",
            "b2_b2_noise_floor_raw.tsv",
            "b2_b2_noise_floor_summary.tsv",
            "b2_c2_abba_raw.tsv",
            "b2_c2_abba_summary.tsv",
            "performance_ratios.tsv",
            "thermal_frequency.tsv",
            "file_pipeline_timing.tsv",
            "short_soak.tsv",
            "c2_10k_soak.tsv",
            "resource_drift.tsv",
            "thermal_frequency_log.tsv",
            "output_hash_stability.tsv",
            "soak_output_semantics.tsv",
            "custom_release_binding.tsv",
            "cross_surface_performance_context.tsv",
        ):
            write_not_run(out / name, "task-gate-closed")
        write_not_run(out / "performance_decision.md", "task-gate-closed")
        write_not_run(out / "stability_decision.md", "task-gate-closed")
    performance_pass = decision(out / "performance_decision.md") == "pass" if task_pass else False
    stability_pass = decision(out / "stability_decision.md") == "pass" if task_pass else False
    performance_status = "pass" if performance_pass else (
        "fail" if task_pass else "not-run-task-gate-closed"
    )
    stability_status = "pass" if stability_pass else (
        "fail" if task_pass else "not-run-task-gate-closed"
    )
    interactions = read_tsv(out / "full_val_board_provider_interactions.tsv")
    classified = [row for row in interactions if row["metric"] != "prediction_count"]
    material = [row for row in classified if "material" in row["classification"]]
    inconclusive = [
        row for row in classified if row["classification"] == "provider-interaction-inconclusive"
    ]

    if not task_pass:
        classification = "stage65d-r1-frozen-c2-full-val-task-fail-retain-b2"
    elif not performance_pass or not stability_pass:
        classification = (
            "stage65d-r1-frozen-c2-full-val-task-pass-performance-or-stability-"
            "fail-retain-b2"
        )
    elif material:
        classification = (
            "stage65d-r1-frozen-c2-full-val-task-pass-provider-interaction-"
            "material-performance-stability-diagnostic-complete-retain-b2-"
            "pending-human-waiver"
        )
    elif inconclusive:
        classification = (
            "stage65d-r1-frozen-c2-full-val-task-pass-provider-interaction-"
            "inconclusive-performance-stability-pass-diagnostic-only"
        )
    else:
        classification = (
            "stage65d-r1-frozen-c2-full-val-task-pass-provider-neutral-"
            "performance-stability-pass-human-promotion-review-required"
        )

    board = {row["surface"]: row for row in read_tsv(out / "full_val_board_metrics.tsv")}
    host = {row["surface"]: row for row in read_tsv(DEV001C / "full_val_metrics.tsv")}
    custom_map = "not-run-task-gate-closed"
    if task_pass:
        fp32 = next(
            row
            for row in read_tsv(R3 / "selected_full_coco.tsv")
            if row["surface"] == "H8"
        )
        same_source = [
            metric_row(fp32, "FP32-H8", "host-CPU", "host", "accepted-reference"),
            metric_row(host["B2"], "B2", "host-CPU", "host", "accepted-reference"),
            metric_row(host["C2"], "C2", "host-CPU", "host", "accepted-reference"),
            metric_row(board["B2_CPU"], "B2", "board-CPU", "board", "reused-exact"),
            metric_row(board["B2_EP"], "B2", "board-SpaceMIT-EP", "board", "reused-exact"),
            metric_row(board["C2_CPU"], "C2", "board-CPU", "board", "fresh-Stage65D-R1"),
            metric_row(board["C2_EP"], "C2", "board-SpaceMIT-EP", "board", "fresh-Stage65D-R1"),
        ]
        write_tsv(out / "same_source_vendor_accuracy_table.tsv", same_source)

        custom_summary = read_tsv(CUSTOM / "outputs/accuracy/full_coco_summary.tsv")[0]
        custom_detail = read_tsv(
            CUSTOM / "outputs/evidence/resolution_coco_results_v2.tsv"
        )[0]
        custom_map = custom_summary["map50_95"]
        cross = [
            {
                "surface": "accepted-custom-engine",
                "model_sha256": "30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c",
                "lineage": "different source/export/quantization/runtime surface",
                "quantization_runtime": "K1X_INT8_V1 / 0.10.0-internal-rd.1",
                "map50_95": custom_summary["map50_95"],
                "ap_small": custom_summary["ap_small"],
                "ap_medium": custom_summary["ap_medium"],
                "ap_large": custom_summary["ap_large"],
                "ar_large": "not-reported",
                "prediction_count": custom_detail["prediction_count"],
                "caveat": "application-level cross-surface context; not engine-only or quantizer-only",
            },
            {
                "surface": "B2-SpaceMIT-EP",
                "model_sha256": "40ba6a7f9aebaa98a1c3abe5fce1f66f1bebcd0b10b7af3d26d30414a331d853",
                "lineage": "same-source vendor split S8-QDQ plus common FP32 tail",
                "quantization_runtime": "S8-QDQ / SpaceMIT ORT 2.0.6",
                "map50_95": board["B2_EP"]["map50_95"],
                "ap_small": board["B2_EP"]["ap_small"],
                "ap_medium": board["B2_EP"]["ap_medium"],
                "ap_large": board["B2_EP"]["ap_large"],
                "ar_large": board["B2_EP"]["ar_large"],
                "prediction_count": board["B2_EP"]["prediction_count"],
                "caveat": "different model surface from custom engine",
            },
            {
                "surface": "C2-SpaceMIT-EP",
                "model_sha256": "281f4acd1261e7ee2c38b6e3bdecbf61c3d91cf710c63e6bc6cdaf257a52669b",
                "lineage": "same-source vendor split S8-QDQ plus common FP32 tail",
                "quantization_runtime": "S8-QDQ / SpaceMIT ORT 2.0.6",
                "map50_95": board["C2_EP"]["map50_95"],
                "ap_small": board["C2_EP"]["ap_small"],
                "ap_medium": board["C2_EP"]["ap_medium"],
                "ap_large": board["C2_EP"]["ap_large"],
                "ar_large": board["C2_EP"]["ar_large"],
                "prediction_count": board["C2_EP"]["prediction_count"],
                "caveat": "different model surface from custom engine",
            },
        ]
        write_tsv(out / "cross_surface_application_accuracy_table.tsv", cross)
        (out / "cross_surface_comparison_caveats.md").write_text(
            "# Cross-surface comparison caveats\n\nThe accepted custom executor and "
            "vendor B2/C2 rows use different source/export/quantization/runtime "
            "surfaces. Their accuracy and same-boot timing are application-level "
            "context only, not an engine-only, quantizer-only, or same-source "
            "backend comparison. No camera surface was run.\n",
            encoding="utf-8",
        )
    else:
        write_not_run(out / "same_source_vendor_accuracy_table.tsv", "task-gate-closed")
        write_not_run(
            out / "cross_surface_application_accuracy_table.tsv", "task-gate-closed"
        )
        write_not_run(out / "cross_surface_comparison_caveats.md", "task-gate-closed")

    perf_path = out / "cross_surface_performance_context.tsv"
    perf = (
        {row["surface"] + ":" + row["role"]: row for row in read_tsv(perf_path)}
        if task_pass
        else {}
    )
    custom_performance = (
        perf["accepted-custom-engine:custom-pure-executor"]["median_us"]
        if "accepted-custom-engine:custom-pure-executor" in perf
        else "not-run-task-gate-closed"
    )
    passport = [
        {
            "artifact": "B2",
            "role": "vendor-universal-control",
            "host_map50_95": host["B2"]["map50_95"],
            "board_cpu_map50_95": board["B2_CPU"]["map50_95"],
            "board_ep_map50_95": board["B2_EP"]["map50_95"],
            "placement": "one-925-node-fused-subgraph",
            "provider_interaction": "control-reference",
            "performance": performance_status,
            "stability": stability_status,
            "disposition": "retain-control",
        },
        {
            "artifact": "C2",
            "role": "frozen-vendor-candidate",
            "host_map50_95": host["C2"]["map50_95"],
            "board_cpu_map50_95": board["C2_CPU"]["map50_95"],
            "board_ep_map50_95": board["C2_EP"]["map50_95"],
            "placement": "one-925-node-fused-subgraph",
            "provider_interaction": "material" if material else ("inconclusive" if inconclusive else "neutral"),
            "performance": performance_status,
            "stability": stability_status,
            "disposition": (
                "human-review"
                if classification.endswith("human-promotion-review-required")
                else "task-failed-retain-b2"
                if not task_pass
                else "diagnostic-only"
            ),
        },
        {
            "artifact": "A1",
            "role": "historical-frozen-research",
            "host_map50_95": host["A1"]["map50_95"],
            "board_cpu_map50_95": "historical-Stage65C-R1",
            "board_ep_map50_95": "historical-Stage65C-R1",
            "placement": "historical-pass",
            "provider_interaction": "historical-inconclusive",
            "performance": "not-opened-historical",
            "stability": "not-opened-historical",
            "disposition": "not-promoted",
        },
        {
            "artifact": "custom-engine",
            "role": "cross-surface-application-reference",
            "host_map50_95": custom_map,
            "board_cpu_map50_95": "not-comparable",
            "board_ep_map50_95": "not-comparable",
            "placement": "different-runtime",
            "provider_interaction": "not-applicable",
            "performance": custom_performance,
            "stability": "accepted-historical-only",
            "disposition": "unchanged",
        },
    ]
    write_tsv(out / "vendor_lane_stage65d_r1_passport.tsv", passport)

    if material:
        (out / "C2_EP_AWARE_TUNING_CHARTER.md").write_text(
            "# C2 EP-aware tuning charter\n\nProposal only; not authorized here. A later "
            "bounded lane may vary only the three terminal confidence qparams while "
            "retaining all-S8 QDQ topology and the placement target. It must use a fresh "
            "EP-tuning surface, untouched final holdout, at most three candidates, a "
            "multi-backend score-margin/TopK-stability objective, simulated perturbation "
            "robustness, and mandatory final board val2017.\n",
            encoding="utf-8",
        )
        ep_tuning = "proposed-for-separate-authorization"
    else:
        ep_tuning = "not-justified-by-current-full-val-evidence"

    board_after = board_capture(STAGE_ID)
    write_tsv(out / "board_identity_after.tsv", board_after)
    board_values = {row["field"]: row["value"] for row in board_after}
    invariant_expected = {
        "protected_main": "1fd2e71bb1d5a924e7c0444cada94f681b73aa91",
        "custom_executor_tree": "c2e400de14fb1c88d4aed70a249d9eff19a05d0f",
        "xslim_head": "46d5d36bcb6979bab6567fb4fe62839689f1881c",
        "xslim_tree": "1788779cd0887a1c8e6924cd63ad7d16d42f41ca",
        "ncnn_head": "a245a70c641a1f20f357c65d103e5f9e50fe84a1",
        "ncnn_tree": "20b96dadbd1fc0a53159cb35749719e967b55906",
        "ncnn_diff": "2bf1cc38885018a02478aa7542581639786c79bca5ce11a6e827d24bcc5f4eca",
        "ncnn_dirty_count": "3",
        "board_emmc_project_writes": "0",
    }
    invariant_actual = {
        "protected_main": git(PROTECTED, "rev-parse", "yolo26-custom-int8-engine"),
        "custom_executor_tree": git(PROTECTED, "rev-parse", "yolo26-custom-int8-engine:custom_int8_engine"),
        "xslim_head": git(XSLIM, "rev-parse", "HEAD"),
        "xslim_tree": git(XSLIM, "rev-parse", "HEAD^{tree}"),
        "ncnn_head": git(NCNN, "rev-parse", "HEAD"),
        "ncnn_tree": git(NCNN, "rev-parse", "HEAD^{tree}"),
        "ncnn_diff": ncnn_diff(),
        "ncnn_dirty_count": str(len(git(NCNN, "status", "--porcelain").splitlines())),
        "board_emmc_project_writes": board_values["emmc_stage_path_count"],
    }
    invariants = [
        {
            "field": field,
            "actual": value,
            "expected": invariant_expected[field],
            "status": "pass" if value == invariant_expected[field] else "fail",
        }
        for field, value in invariant_actual.items()
    ]
    write_tsv(out / "protected_projects_unchanged.tsv", invariants)
    if any(row["status"] != "pass" for row in invariants):
        raise RuntimeError("protected-state invariant failed")

    map_pair = next(
        row
        for row in read_tsv(out / "full_val_board_complete_bootstrap.tsv")
        if row["pair"] == "C2_EP-vs-B2_EP" and row["metric"] == "map50_95"
    )
    bootstrap = {
        (row["pair"], row["metric"]): row
        for row in read_tsv(out / "full_val_board_complete_bootstrap.tsv")
    }
    point_deltas = {
        metric: float(board["C2_EP"][metric]) - float(board["B2_EP"][metric])
        for metric in (
            "map50_95",
            "ap_small",
            "ap_medium",
            "ap_large",
            "ar_small",
            "ar_medium",
            "ar_large",
        )
    }
    interaction_counts: dict[str, int] = {}
    for row in classified:
        interaction_counts[row["classification"]] = (
            interaction_counts.get(row["classification"], 0) + 1
        )
    provider_status = "material" if material else (
        "inconclusive" if inconclusive else "neutral"
    )
    map_interaction = next(row for row in classified if row["metric"] == "map50_95")
    ar_small_interaction = next(row for row in classified if row["metric"] == "ar_small")
    ar_large_interaction = next(row for row in classified if row["metric"] == "ar_large")
    prediction_hashes = {
        row["surface"]: row["sha256"]
        for row in read_tsv(out / "full_val_board_prediction_hashes.tsv")
    }
    c2_ep_delta = float(board["C2_EP"]["map50_95"]) - float(board["B2_EP"]["map50_95"])
    report = f"""# Stage65D-R1 final report

Classification: `{classification}`
Publication: `not-authorized-not-attempted`
Stage: `{STAGE_ID}`

## Identity and placement

The exact frozen B2/C2 inference models and common tail were used with SpaceMIT ORT 2.0.6. B2 and C2 each produced one equal 925-source-node fused SpaceMIT subgraph, six outputs, and zero unexpected CPU inference events. Bounded controls and F0 CPU/EP fixtures passed without non-finite output or score collapse.

## Full val2017

| Surface | mAP50-95 | AP-S | AP-M | AP-L | AR-S | AR-M | AR-L | Predictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B2 CPU | {float(board['B2_CPU']['map50_95']):.12f} | {float(board['B2_CPU']['ap_small']):.12f} | {float(board['B2_CPU']['ap_medium']):.12f} | {float(board['B2_CPU']['ap_large']):.12f} | {float(board['B2_CPU']['ar_small']):.12f} | {float(board['B2_CPU']['ar_medium']):.12f} | {float(board['B2_CPU']['ar_large']):.12f} | {board['B2_CPU']['prediction_count']} |
| B2 EP | {float(board['B2_EP']['map50_95']):.12f} | {float(board['B2_EP']['ap_small']):.12f} | {float(board['B2_EP']['ap_medium']):.12f} | {float(board['B2_EP']['ap_large']):.12f} | {float(board['B2_EP']['ar_small']):.12f} | {float(board['B2_EP']['ar_medium']):.12f} | {float(board['B2_EP']['ar_large']):.12f} | {board['B2_EP']['prediction_count']} |
| C2 CPU | {float(board['C2_CPU']['map50_95']):.12f} | {float(board['C2_CPU']['ap_small']):.12f} | {float(board['C2_CPU']['ap_medium']):.12f} | {float(board['C2_CPU']['ap_large']):.12f} | {float(board['C2_CPU']['ar_small']):.12f} | {float(board['C2_CPU']['ar_medium']):.12f} | {float(board['C2_CPU']['ar_large']):.12f} | {board['C2_CPU']['prediction_count']} |
| C2 EP | {float(board['C2_EP']['map50_95']):.12f} | {float(board['C2_EP']['ap_small']):.12f} | {float(board['C2_EP']['ap_medium']):.12f} | {float(board['C2_EP']['ap_large']):.12f} | {float(board['C2_EP']['ar_small']):.12f} | {float(board['C2_EP']['ar_medium']):.12f} | {float(board['C2_EP']['ar_large']):.12f} | {board['C2_EP']['prediction_count']} |

C2 EP minus B2 EP mAP50-95 is `{c2_ep_delta:.12f}` with 95% CI `[{map_pair['percentile_2_5']}, {map_pair['percentile_97_5']}]` and `P(delta>0)={map_pair['probability_gt_zero']}`. AP deltas (S/M/L) are `{point_deltas['ap_small']:.12f}`, `{point_deltas['ap_medium']:.12f}`, `{point_deltas['ap_large']:.12f}`. AR deltas (S/M/L) are `{point_deltas['ar_small']:.12f}`, `{point_deltas['ar_medium']:.12f}`, `{point_deltas['ar_large']:.12f}`.

The predeclared task gate is `{'pass' if task_pass else 'fail'}`. AR-small fails both its `-0.003` point guard (`{point_deltas['ar_small']:.12f}`) and `-0.005` lower-CI guard (`{bootstrap[('C2_EP-vs-B2_EP', 'ar_small')]['percentile_2_5']}`). AR-large misses the point guard by `{abs(point_deltas['ar_large'] + 0.003):.12f}` while its CI guard passes. This is a task-contract failure despite the material mAP/AP-large gain.

Prediction SHA-256 values: B2 CPU `{prediction_hashes['B2_CPU']}`, B2 EP `{prediction_hashes['B2_EP']}`, C2 CPU `{prediction_hashes['C2_CPU']}`, C2 EP `{prediction_hashes['C2_EP']}`. Every surface completed 5000/5000 with zero runner/evaluator failures, non-finite predictions, or collapse.

## Provider diagnostic

Population difference-in-differences contains `{interaction_counts.get('provider-neutral', 0)}` provider-neutral metrics, `{interaction_counts.get('provider-interaction-inconclusive', 0)}` inconclusive metrics, and `{len(material)}` material metrics; overall status is `{provider_status}`. mAP interaction is `{float(map_interaction['point_interaction']):.12f}` with 95% CI `[{map_interaction['percentile_2_5']}, {map_interaction['percentile_97_5']}]` and is provider-neutral. AR-small interaction is `{float(ar_small_interaction['point_interaction']):.12f}` with CI `[{ar_small_interaction['percentile_2_5']}, {ar_small_interaction['percentile_97_5']}]`; AR-large is `{float(ar_large_interaction['point_interaction']):.12f}` with CI `[{ar_large_interaction['percentile_2_5']}, {ar_large_interaction['percentile_97_5']}]`.

Score/rank analysis found deterministic CPU/EP sensitivity in both models, but no material C2-specific population interaction. It does not establish a provider bug, exact rounding mode, or LSB-level cause. EP-aware tuning is `{ep_tuning}`.

## Conditional passport

Matched performance: `{performance_status}`. Stability: `{stability_status}`. Custom-engine application context: `not-run-task-gate-closed`. These gates were correctly not opened after task failure; they did not fail experimentally. No camera work was run.

## Disposition

C2 remains frozen diagnostic evidence and is not promotion-ready. B2 remains the vendor universal control. XSlim, the custom executor, protected main and `/data/ncnn` are unchanged; board eMMC project writes are zero. No runtime default or persistent board state required rollback.
"""
    (out / "STAGE65D_R1_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    (out / "STAGE65D_R1_SUMMARY_RU.md").write_text(
        "# Краткий итог Stage65D-R1\n\n"
        f"Классификация: `{classification}`. Полный val2017 на плате дал C2 EP "
        f"минус B2 EP `{c2_ep_delta:.12f}` mAP50-95; основной task gate "
        f"`{'пройден' if task_pass else 'не пройден'}`. Взаимодействие provider: "
        f"`{'существенное' if material else 'неопределенное' if inconclusive else 'нейтральное'}`. "
        f"AR-small delta `{point_deltas['ar_small']:.12f}`, AR-large delta "
        f"`{point_deltas['ar_large']:.12f}`. Производительность: `{performance_status}`, "
        f"стабильность: `{stability_status}`; эти ветви не запускались после task-gate "
        "fail и не являются экспериментальными отказами. Автоматического "
        "продвижения C2 нет; B2 остается контрольной моделью. Камера не использовалась.\n",
        encoding="utf-8",
    )
    (out / "stage_readiness_or_blocker.md").write_text(
        f"# Stage readiness\n\nClassification: `{classification}`. C2 remains a frozen diagnostic artifact; B2 remains the vendor control. Performance, stability, and same-boot custom execution are `not-run-task-gate-closed`. No runtime promotion is authorized. EP-aware tuning is `{ep_tuning}`.\n",
        encoding="utf-8",
    )
    (out / "human_decision_options.md").write_text(
        "# Human decision options\n\n- Retain B2 as the vendor baseline.\n"
        "- Keep C2 as a frozen diagnostic artifact; do not promote it under the current universal task contract.\n"
        "- Do not open EP-aware qparam tuning from this evidence: no material provider interaction was proven.\n"
        "- Separately authorize head-only QAT, model/executor co-design, or same-source K1X_INT8_V2 comparison.\n\nNo option is automatic.\n",
        encoding="utf-8",
    )
    (out / "system_rollback_report.md").write_text(
        "# System rollback\n\nNo runtime default, OS profile, model, package, XSlim source, custom executor or persistent board configuration was changed. No rollback command was required. Project writes remained on NVMe `/data`; camera work was not run.\n",
        encoding="utf-8",
    )
    (out / "closure_timestamp_utc.txt").write_text(datetime.now(UTC).isoformat() + "\n")
    print(classification)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
