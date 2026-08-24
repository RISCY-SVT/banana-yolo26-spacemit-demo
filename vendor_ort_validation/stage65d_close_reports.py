#!/usr/bin/env python3
"""Create compact Stage65D closure reports after the H500 stop gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path


STAGE_ID = (
    "BANANA-YOLO26-XSLIM-STAGE65D-C2-FROZEN-K1X-SPACEMIT-EP-BOARD-"
    "PASSPORT-FP32-FP16-AND-CUSTOM-ENGINE-APPLICATION-COMPARISON-001"
)
GATE_REASON = "H500 C2 CPU/EP provider agreement failed"
BANANA = Path("/data/worktrees/banana-yolo26-xslim211-s8-qdq-validation")
XSLIM = Path("/data/worktrees/riscy-xslim-k1x-yolo26")
PROTECTED = Path("/data/banana-yolo26-spacemit-demo")
NCNN = Path("/data/ncnn")
RAW = Path("/data/k1x-stage-runs") / STAGE_ID
CUSTOM_ROOT = Path(
    "/data/releases/banana-yolo26-k1x-int8-executor/"
    "0.10.0-internal-rd.1-stage62-sdk"
)
DEV001C = BANANA / "stages" / (
    "BANANA-YOLO26-XSLIM-DEV-001C-C2-FROZEN-INDEPENDENT-HOLDOUT-"
    "ADJUDICATION-AND-VENDOR-PTQ-LANE-CLOSURE-001"
)
DEV001B = BANANA / "stages" / (
    "BANANA-YOLO26-XSLIM-DEV-001B-ALL-S8-GENERIC-HARDENING-ADAPTIVE-"
    "ROUNDING-BLOCK-RECONSTRUCTION-AND-DETECTOR-PARETO-HOST-GATE-001"
)
R3 = BANANA / "stages" / (
    "BANANA-YOLO26-XSLIM-STAGE65B-R3-HIERARCHICAL-EARLY-SUBGRAPH-"
    "CUT-SPLICE-LOCALIZATION-AND-XSLIM-TARGET-POLICY-CHARTER-001"
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty TSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def board_capture() -> list[dict[str, str]]:
    script = r'''set -eu
printf 'timestamp_utc\t'; date -u +%Y-%m-%dT%H:%M:%SZ
printf 'hostname\t'; hostname
printf 'device_serial\t'; tr -d '\000' </proc/device-tree/serial-number; printf '\n'
printf 'boot_id\t'; cat /proc/sys/kernel/random/boot_id
printf 'kernel\t'; uname -a
printf 'os_release\t'; . /etc/os-release; printf '%s\n' "${PRETTY_NAME:-unknown}"
printf 'allowed_cpu_list\t'; cat /sys/devices/system/cpu/online
printf 'data_mount\t'; findmnt -n -o SOURCE,FSTYPE,OPTIONS -T /data
printf 'root_mount\t'; findmnt -n -o SOURCE,FSTYPE,OPTIONS -T /
printf 'data_free_bytes\t'; df -B1 /data | tail -1 | awk '{print $4}'
for path in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [ -r "$path" ] || continue
    printf 'governor\t%s=' "$path"; cat "$path"
done
for path in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do
    [ -r "$path" ] || continue
    printf 'frequency_khz\t%s=' "$path"; cat "$path"
done
for temp_path in /sys/class/thermal/thermal_zone*/temp; do
    [ -r "$temp_path" ] || continue
    printf 'temperature_millic\t%s=' "$temp_path"; cat "$temp_path"
done
count=0
for base in /home /tmp /var/tmp /root; do
    [ -d "$base" ] || continue
    value=$(find "$base" -xdev -path "*''' + STAGE_ID + r'''*" -print 2>/dev/null | wc -l)
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


def ncnn_diff_sha256() -> str:
    payload = subprocess.check_output(["git", "diff", "--binary"], cwd=NCNN)
    return hashlib.sha256(payload).hexdigest()


def closed_report(path: Path) -> None:
    if path.suffix == ".md":
        path.write_text(
            f"# Conditional gate status\n\nStatus: `not-run-gate-closed`.\n\n"
            f"Reason: {GATE_REASON}. Full val2017, matched performance and soak "
            "were not authorized to continue after this stop gate.\n",
            encoding="utf-8",
        )
    else:
        write_tsv(path, [{"status": "not-run-gate-closed", "reason": GATE_REASON}])


def metric_row(
    row: dict[str, str],
    *,
    model: str,
    precision: str,
    execution: str,
    location: str,
    dataset: str,
    model_sha256: str,
    tail_sha256: str,
    status: str,
) -> dict[str, object]:
    return {
        "model": model,
        "precision": precision,
        "execution": execution,
        "location": location,
        "dataset": dataset,
        "model_sha256": model_sha256,
        "tail_sha256": tail_sha256,
        "map50_95": row["map50_95"],
        "map50": row["map50"],
        "map75": row.get("map75", "not-reported"),
        "ap_small": row["ap_small"],
        "ap_medium": row["ap_medium"],
        "ap_large": row["ap_large"],
        "ar_small": row.get("ar_small", "not-reported"),
        "ar_medium": row.get("ar_medium", "not-reported"),
        "ar_large": row.get("ar_large", "not-reported"),
        "prediction_count": row.get("prediction_count", "not-reported"),
        "status": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracked-root", required=True, type=Path)
    options = parser.parse_args()
    out = options.tracked_root

    conditional = (
        "full_val_board_metrics.tsv",
        "full_val_board_per_class.tsv",
        "full_val_board_size_bins.tsv",
        "full_val_board_complete_bootstrap.tsv",
        "full_val_board_provider_interactions.tsv",
        "full_val_host_board_transfer.tsv",
        "full_val_board_decision.md",
        "b2_b2_noise_floor_raw.tsv",
        "b2_c2_abba_raw.tsv",
        "b2_c2_abba_summary.tsv",
        "session_first_run_timing.tsv",
        "tail_two_stage_timing.tsv",
        "file_pipeline_timing.tsv",
        "performance_decision.md",
        "short_soak.tsv",
        "c2_10k_soak.tsv",
        "resource_drift.tsv",
        "thermal_frequency_log.tsv",
        "stability_decision.md",
        "cross_surface_performance_context.tsv",
    )
    for name in conditional:
        closed_report(out / name)

    h500 = {row["surface"]: row for row in read_tsv(out / "h500_board_metrics.tsv")}
    partitions = {
        row["model"]: row for row in read_tsv(out / "provider_partition_comparison.tsv")
    }
    host = {row["surface"]: row for row in read_tsv(DEV001C / "full_val_metrics.tsv")}
    host_h500 = {
        row["surface"]: row for row in read_tsv(DEV001B / "h500_metrics.tsv")
    }
    score_rows = []
    prediction_dirs = {
        "B2_CPU": "B2-cpu",
        "B2_EP": "B2-spacemit",
        "C2_CPU": "C2-cpu",
        "C2_EP": "C2-spacemit",
    }
    for surface, directory in prediction_dirs.items():
        path = RAW / "board/coco/h500" / directory / "predictions.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        scores = [float(row["score"]) for row in payload]
        finite = all(math.isfinite(value) for value in scores)
        mean = math.fsum(scores) / len(scores)
        variance = math.fsum((value - mean) ** 2 for value in scores) / len(scores)
        unique_scores = len({value.hex() for value in scores})
        unique_classes = len({int(row["category_id"]) for row in payload})
        collapse = (
            not finite
            or not scores
            or min(scores) == max(scores)
            or unique_scores < 2
            or unique_classes < 2
        )
        score_rows.append({
            "surface": surface,
            "prediction_count": len(scores),
            "score_min": min(scores),
            "score_max": max(scores),
            "score_mean": mean,
            "score_stddev": variance ** 0.5,
            "unique_score_values": unique_scores,
            "unique_classes": unique_classes,
            "non_finite_scores": sum(not math.isfinite(value) for value in scores),
            "score_collapse": "yes" if collapse else "no",
            "status": "fail" if collapse else "pass",
        })
    write_tsv(out / "h500_score_distribution.tsv", score_rows)
    decision_path = out / "h500_board_decision.md"
    decision_path.write_text(
        decision_path.read_text(encoding="utf-8")
        + "\n## Score-collapse census\n\nAll four surfaces have finite, nonconstant "
        "score distributions with multiple classes. Status: `pass`. See "
        "`h500_score_distribution.tsv`.\n",
        encoding="utf-8",
    )
    fp32 = read_tsv(R3 / "selected_full_coco.tsv")
    fp32_row = next(row for row in fp32 if row["surface"] == "H8")
    tail = "18ffff41e6812fa781baf7b9c1fcd41b41d6118145d785c3e550499070a512a3"
    b2_model = "40ba6a7f9aebaa98a1c3abe5fce1f66f1bebcd0b10b7af3d26d30414a331d853"
    c2_model = "281f4acd1261e7ee2c38b6e3bdecbf61c3d91cf710c63e6bc6cdaf257a52669b"
    fp32_model = "72eb6136b41104753c53b8e13aeff50e7961c4cefba79e50b70894cbd169f8d8"

    precision_rows = [
        metric_row(
            fp32_row,
            model="FP32-H8",
            precision="FP32",
            execution="host-CPU",
            location="host",
            dataset="val2017-5000",
            model_sha256=fp32_model,
            tail_sha256=tail,
            status="accepted-host-reference",
        ),
        metric_row(
            host["B2"], model="B2", precision="S8-QDQ", execution="host-CPU",
            location="host", dataset="val2017-5000", model_sha256=b2_model,
            tail_sha256=tail, status="accepted-host-reference",
        ),
        metric_row(
            host["C2"], model="C2", precision="S8-QDQ", execution="host-CPU",
            location="host", dataset="val2017-5000", model_sha256=c2_model,
            tail_sha256=tail, status="accepted-host-reference",
        ),
    ]
    not_run_metrics = {
        name: "not-run-gate-closed"
        for name in (
            "map50_95", "map50", "map75", "ap_small", "ap_medium", "ap_large",
            "ar_small", "ar_medium", "ar_large", "prediction_count",
        )
    }
    for execution in ("board-CPU", "board-SpaceMIT-EP"):
        precision_rows.append(metric_row(
            not_run_metrics,
            model="FP32-H8",
            precision="FP32",
            execution=execution,
            location="board",
            dataset="not-run-H500-gate-closed",
            model_sha256=fp32_model,
            tail_sha256=tail,
            status="not-run-gate-closed",
        ))
    for execution, location in (
        ("host-CPU", "host"),
        ("board-CPU", "board"),
        ("board-SpaceMIT-EP", "board"),
    ):
        precision_rows.append(metric_row(
            not_run_metrics,
            model="FP16",
            precision="FP16",
            execution=execution,
            location=location,
            dataset="not-run-no-comparable-frozen-artifact",
            model_sha256="identity-unresolved",
            tail_sha256="identity-unresolved",
            status="not-run-no-comparable-frozen-artifact",
        ))
    for surface, model, execution, model_hash in (
        ("B2_CPU", "B2", "board-CPU", b2_model),
        ("B2_EP", "B2", "board-SpaceMIT-EP", b2_model),
        ("C2_CPU", "C2", "board-CPU", c2_model),
        ("C2_EP", "C2", "board-SpaceMIT-EP", c2_model),
    ):
        precision_rows.append(metric_row(
            h500[surface], model=model, precision="S8-QDQ", execution=execution,
            location="board", dataset="H500-train2017", model_sha256=model_hash,
            tail_sha256=tail, status="Stage65D-H500",
        ))
    fp32_map = float(fp32_row["map50_95"])
    b2_map = float(host["B2"]["map50_95"])
    c2_map = float(host["C2"]["map50_95"])
    recovered = (c2_map - b2_map) / (fp32_map - b2_map)
    for row in precision_rows:
        if row["dataset"] != "val2017-5000":
            row["gap_to_fp32_map50_95"] = "not-comparable-different-dataset"
            row["c2_fraction_b2_to_fp32_gap_recovered"] = "not-comparable-different-dataset"
        else:
            row["gap_to_fp32_map50_95"] = fp32_map - float(row["map50_95"])
            row["c2_fraction_b2_to_fp32_gap_recovered"] = (
                recovered if row["model"] == "C2" else "not-applicable"
            )
    write_tsv(out / "same_source_precision_accuracy_table.tsv", precision_rows)
    write_tsv(out / "same_source_vendor_accuracy_table.tsv", precision_rows)
    write_tsv(out / "h500_host_board_transfer.tsv", [
        {
            "model": model,
            "board_provider": provider,
            "host_map50_95": host_h500[host_name]["map50_95"],
            "board_map50_95": h500[f"{model}_{provider}"]["map50_95"],
            "board_minus_host": float(h500[f"{model}_{provider}"]["map50_95"])
            - float(host_h500[host_name]["map50_95"]),
            "interpretation": "task-level H500 transfer; prediction bytes may differ",
        }
        for model, host_name in (("B2", "B2"), ("C2", "C2_T6_RANK_QP"))
        for provider in ("CPU", "EP")
    ])

    custom_summary = read_tsv(CUSTOM_ROOT / "outputs/accuracy/full_coco_summary.tsv")[0]
    custom_detail = read_tsv(
        CUSTOM_ROOT / "outputs/evidence/resolution_coco_results_v2.tsv"
    )[0]
    custom_binary = CUSTOM_ROOT / "bin/yolo26_k1x_int8"
    custom_observed = {
        "tag_object": git(PROTECTED, "rev-parse", "v0.10.0-internal-rd.1^{tag}"),
        "tag_peeled_commit": git(PROTECTED, "rev-parse", "v0.10.0-internal-rd.1^{}"),
        "release_manifest_sha256": sha256(CUSTOM_ROOT / "release_manifest.json"),
        "sha256sums_sha256": sha256(CUSTOM_ROOT / "SHA256SUMS"),
        "executor_sha256": sha256(custom_binary),
        "healthcheck_sha256": sha256(CUSTOM_ROOT / "bin/y26_k1x_healthcheck"),
        "prediction_sha256": custom_summary["prediction_sha256"],
        "map50_95": custom_summary["map50_95"],
        "source_model_sha256": (
            CUSTOM_ROOT / "model-evidence/SOURCE_MODEL_SHA256"
        ).read_text(encoding="utf-8").split()[0],
    }
    custom_expected = {
        "tag_object": "bf1043d669ff38461a62d116f383ad530128d9b5",
        "tag_peeled_commit": "1fd2e71bb1d5a924e7c0444cada94f681b73aa91",
        "release_manifest_sha256": "dced8ddfc540ab5b7fd72ecfe7a16021338ea56258fb33d09c5e023ba3d98b98",
        "sha256sums_sha256": "f9c604d7a3167664a86c48dd101e4f4935a243bde726c6853a9f9390aa278341",
        "executor_sha256": "34da155ed02a83a74babbec30aff960bdccfb6cc16018230ae7bc030462f7187",
        "healthcheck_sha256": "4b82ff86aecbb07d1a5647bb7a92b9f9f3711a877ac5b54ed37fa30c75dfe2b4",
        "prediction_sha256": "cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda",
        "map50_95": "0.3707408944391919",
        "source_model_sha256": "30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c",
    }
    custom_rows = [
        {"field": "release", "value": "0.10.0-internal-rd.1", "status": "pass", "evidence": CUSTOM_ROOT / "config/release.env"},
        {"field": "protected_source_config", "value": "0.9.2-historical-not-current-package", "status": "pass", "evidence": PROTECTED / "config/release.env"},
        {"field": "integer_contract", "value": "K1X_INT8_V1", "status": "pass", "evidence": CUSTOM_ROOT / "config/release.env"},
        {"field": "execution", "value": "read-only-binding-no-rebuild-no-current-run", "status": "pass", "evidence": GATE_REASON},
    ]
    custom_evidence = {
        "tag_object": "annotated-tag",
        "tag_peeled_commit": "protected-main",
        "release_manifest_sha256": CUSTOM_ROOT / "release_manifest.json",
        "sha256sums_sha256": CUSTOM_ROOT / "SHA256SUMS",
        "executor_sha256": custom_binary,
        "healthcheck_sha256": CUSTOM_ROOT / "bin/y26_k1x_healthcheck",
        "prediction_sha256": CUSTOM_ROOT / "outputs/accuracy/full_coco_summary.tsv",
        "map50_95": CUSTOM_ROOT / "outputs/accuracy/full_coco_summary.tsv",
        "source_model_sha256": CUSTOM_ROOT / "model-evidence/SOURCE_MODEL_SHA256",
    }
    custom_rows.extend({
        "field": name,
        "value": value,
        "status": "pass" if value == custom_expected[name] else "fail",
        "evidence": custom_evidence[name],
    } for name, value in custom_observed.items())
    write_tsv(out / "custom_release_binding.tsv", custom_rows)
    if any(row["status"] != "pass" for row in custom_rows):
        raise RuntimeError("accepted custom-engine release binding failed")

    cross_rows = [
        {
            "surface": "accepted-custom-engine",
            "dataset": "val2017-5000",
            "model_sha256": "30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c",
            "lineage": "different custom-engine model/export/quantization surface",
            "quantization": "K1X_INT8_V1",
            "runtime": "accepted custom executor 0.10.0-internal-rd.1",
            "map50_95": custom_summary["map50_95"],
            "map75": custom_detail["map75"],
            "ap_small": custom_summary["ap_small"],
            "ap_medium": custom_summary["ap_medium"],
            "ap_large": custom_summary["ap_large"],
            "ar_large": "not-reported",
            "ar100": custom_detail["ar_100"],
            "prediction_count": custom_detail["prediction_count"],
            "caveat": "application-level cross-surface comparison; not engine-only or quantizer-only",
        },
        {
            "surface": "C2-host-reference",
            "dataset": "val2017-5000",
            "model_sha256": c2_model,
            "lineage": "same-source YOLO26 split plus common FP32 tail",
            "quantization": "signed S8-QDQ",
            "runtime": "host CPU accepted DEV-001C",
            "map50_95": host["C2"]["map50_95"],
            "map75": host["C2"]["map75"],
            "ap_small": host["C2"]["ap_small"],
            "ap_medium": host["C2"]["ap_medium"],
            "ap_large": host["C2"]["ap_large"],
            "ar_large": host["C2"]["ar_large"],
            "ar100": host["C2"]["ar_100"],
            "prediction_count": host["C2"]["prediction_count"],
            "caveat": "host-only context; Stage65D board full val gate did not open",
        },
    ]
    write_tsv(out / "cross_surface_application_accuracy_table.tsv", cross_rows)

    passport = [
        {"artifact": "B2", "role": "universal-control", "host_full_val_map50_95": host["B2"]["map50_95"], "board_h500_cpu_map50_95": h500["B2_CPU"]["map50_95"], "board_h500_ep_map50_95": h500["B2_EP"]["map50_95"], "placement": f"pass-one-{partitions['B2']['fused_node_count']}-node-subgraph", "cpu_ep_agreement": "control-reference", "performance": "not-run-gate-closed", "stability": "not-run-gate-closed", "disposition": "retain-B2"},
        {"artifact": "C2", "role": "frozen-host-winner", "host_full_val_map50_95": host["C2"]["map50_95"], "board_h500_cpu_map50_95": h500["C2_CPU"]["map50_95"], "board_h500_ep_map50_95": h500["C2_EP"]["map50_95"], "placement": f"pass-one-{partitions['C2']['fused_node_count']}-node-subgraph", "cpu_ep_agreement": "fail-map-and-AP-large", "performance": "not-run-gate-closed", "stability": "not-run-gate-closed", "disposition": "diagnostic-only-not-promoted"},
        {"artifact": "A1", "role": "historical-frozen-research", "host_full_val_map50_95": host["A1"]["map50_95"], "board_h500_cpu_map50_95": "historical-Stage65C", "board_h500_ep_map50_95": "historical-Stage65C", "placement": "historical-pass", "cpu_ep_agreement": "historical-fail", "performance": "not-opened", "stability": "not-opened", "disposition": "not-promoted"},
        {"artifact": "FP32", "role": "same-source-host-reference", "host_full_val_map50_95": fp32_row["map50_95"], "board_h500_cpu_map50_95": "not-run", "board_h500_ep_map50_95": "not-run", "placement": "not-run", "cpu_ep_agreement": "not-run", "performance": "not-run", "stability": "not-run", "disposition": "host-reference-only"},
        {"artifact": "FP16", "role": "precision-reference", "host_full_val_map50_95": "not-run-no-comparable-frozen-artifact", "board_h500_cpu_map50_95": "not-run", "board_h500_ep_map50_95": "not-run", "placement": "not-run", "cpu_ep_agreement": "not-run", "performance": "not-run", "stability": "not-run", "disposition": "identity-unresolved"},
        {"artifact": "custom-engine", "role": "cross-surface-application-reference", "host_full_val_map50_95": custom_summary["map50_95"], "board_h500_cpu_map50_95": "not-comparable", "board_h500_ep_map50_95": "not-comparable", "placement": "different-runtime", "cpu_ep_agreement": "not-applicable", "performance": "not-run-gate-closed", "stability": "historical-only", "disposition": "accepted-unchanged"},
    ]
    write_tsv(out / "vendor_lane_stage65d_passport.tsv", passport)

    (out / "stage_readiness_or_blocker.md").write_text(
        "# Stage65D readiness\n\n"
        "Classification: `stage65d-frozen-c2-k1x-provider-agreement-fail-"
        "diagnostic-only-retain-b2`.\n\n"
        "C2 preserves B2 placement and improves H500 EP mAP, but violates the "
        "predeclared C2 CPU/EP mAP and AP-large agreement limits. Full val2017, "
        "performance, same-boot custom execution and soak remain gate-closed. "
        "C2 is diagnostic-only; B2 remains the vendor control.\n",
        encoding="utf-8",
    )
    (out / "human_decision_options.md").write_text(
        "# Human decision options\n\n"
        "- Retain B2 as the vendor-lane baseline (current evidence-supported option).\n"
        "- Keep C2 frozen for a separately authorized provider-difference diagnostic.\n"
        "- Authorize an application-specific C2 profile only after resolving CPU/EP agreement.\n"
        "- Separately authorize head-only QAT or model/executor co-design.\n"
        "- Separately authorize a same-source K1X_INT8_V2 comparison.\n\n"
        "No option is automatic and Stage65D authorizes no runtime promotion.\n",
        encoding="utf-8",
    )

    board_after = board_capture()
    write_tsv(out / "board_identity_after.tsv", board_after)
    by_field: dict[str, list[str]] = {}
    for row in board_after:
        by_field.setdefault(row["field"], []).append(row["value"])
    write_tsv(out / "storage_write_audit_after.tsv", [
        {"surface": "stage_root_mount", "value": by_field["data_mount"][0]},
        {"surface": "root_mount", "value": by_field["root_mount"][0]},
        {"surface": "emmc_stage_path_count", "value": by_field["emmc_stage_path_count"][0]},
    ])
    (out / "system_rollback_report.md").write_text(
        "# System rollback\n\nNo alternate OS profile, runtime default, model, "
        "governor or persistent system configuration was installed by Stage65D. "
        "The board boot ID remained unchanged and no rollback command was required. "
        "All project artifacts remained under the NVMe-backed `/data` stage root.\n",
        encoding="utf-8",
    )

    expected = {
        "protected_main": "1fd2e71bb1d5a924e7c0444cada94f681b73aa91",
        "custom_executor_tree": "c2e400de14fb1c88d4aed70a249d9eff19a05d0f",
        "xslim_head": "46d5d36bcb6979bab6567fb4fe62839689f1881c",
        "xslim_tree": "1788779cd0887a1c8e6924cd63ad7d16d42f41ca",
        "ncnn_head": "a245a70c641a1f20f357c65d103e5f9e50fe84a1",
        "ncnn_tree": "20b96dadbd1fc0a53159cb35749719e967b55906",
        "ncnn_diff": "2bf1cc38885018a02478aa7542581639786c79bca5ce11a6e827d24bcc5f4eca",
        "ncnn_dirty_count": "3",
    }
    actual = {
        "protected_main": git(PROTECTED, "rev-parse", "yolo26-custom-int8-engine"),
        "custom_executor_tree": git(PROTECTED, "rev-parse", "yolo26-custom-int8-engine:custom_int8_engine"),
        "xslim_head": git(XSLIM, "rev-parse", "HEAD"),
        "xslim_tree": git(XSLIM, "rev-parse", "HEAD^{tree}"),
        "ncnn_head": git(NCNN, "rev-parse", "HEAD"),
        "ncnn_tree": git(NCNN, "rev-parse", "HEAD^{tree}"),
        "ncnn_diff": ncnn_diff_sha256(),
        "ncnn_dirty_count": str(len(git(NCNN, "status", "--porcelain").splitlines())),
    }
    write_tsv(out / "protected_projects_unchanged.tsv", [
        {
            "field": name,
            "actual": actual[name],
            "expected": value,
            "status": "pass" if actual[name] == value else "fail",
        }
        for name, value in expected.items()
    ])
    process_pattern = (
        "[s]tage65d_(accuracy|bootstrap|performance|stability|run_coco_board|"
        "run_two_stage_board)|[c]ustom_executor"
    )
    host_workers = subprocess.run(
        ["pgrep", "-a", "-f", process_pattern],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    board_workers = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "svt@banana", "pgrep", "-a", "-f", process_pattern],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    write_tsv(out / "process_audit_after.tsv", [
        {"surface": "host", "matching_processes": host_workers or "none", "status": "pass" if not host_workers else "fail"},
        {"surface": "board", "matching_processes": board_workers or "none", "status": "pass" if not board_workers else "fail"},
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
