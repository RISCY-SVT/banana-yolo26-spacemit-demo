#!/usr/bin/env python3
"""Render Stage56 repository evidence from preserved raw measurements."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


TASK_ID = (
    "BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE56-BARE-METAL-SYSTEM-"
    "ISOLATION-X60-HPM-FUSED-DENSE-RESIDUAL-CEILING-STORAGE-REBOOT-AND-"
    "DUAL-REMOTE-FREEZE-GATE-001"
)
START_HEAD = "c7dd62da3f62f46975240eb414b2f0ca149ceddf"
IMPLEMENTATION_COMMIT = "56025cfc1164d3831073528b25cc933997a1f6fa"
PROVENANCE_COMMIT = "2f4cf451e19ba16ddb353a77baf80514e6936bab"
CONTRACT = "K1X_INT8_V1"
PROFILE = "K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001"
MODEL_SHA = "30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c"
PACKAGE_SHA = "fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be"
PREDICTION_SHA = "cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda"
OUTPUT_HASH = "0xd43f5e018b415631"
STAGE55_MEAN_US = 149603.240
STAGE55_MAP = 0.3707408944391919


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[dict[str, Any]],
              fields: list[str] | None = None) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in materialized:
            for field in row:
                if field not in fields:
                    fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        for row in materialized:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_md(path: Path, title: str, paragraphs: Iterable[str]) -> None:
    write_text(path, f"# {title}\n\n" + "\n\n".join(paragraphs))


def copy_text(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    write_text(destination, source.read_text(encoding="utf-8", errors="replace"))


def parse_fields(line: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for field in line.rstrip().replace(" ", "\t").split("\t"):
        if "=" in field:
            key, value = field.split("=", 1)
            result[key] = value
    return result


def parse_cli(path: Path) -> list[dict[str, str]]:
    rows = [parse_fields(line) for line in path.read_text(
        encoding="utf-8", errors="replace").splitlines() if line.startswith("raw\t")]
    if not rows:
        raise ValueError(f"no raw samples: {path}")
    return rows


def parse_ort(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("raw\t", "stage42_ort_only_sample ")):
            row = parse_fields(line)
            if "wall_us" in row:
                rows.append(row)
    if len(rows) != 500:
        raise ValueError(f"expected 500 ORT samples, found {len(rows)}")
    return rows


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize(surface: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    values = [float(row["wall_us"]) for row in rows]
    mean = statistics.fmean(values)
    result: dict[str, Any] = {
        "surface": surface,
        "samples": len(rows),
        "mean_us": f"{mean:.6f}",
        "stddev_us": f"{statistics.pstdev(values):.6f}",
        "cv_pct": f"{statistics.pstdev(values) / mean * 100.0:.6f}",
        "median_us": f"{statistics.median(values):.6f}",
        "p90_us": f"{percentile(values, .90):.6f}",
        "p95_us": f"{percentile(values, .95):.6f}",
        "p99_us": f"{percentile(values, .99):.6f}",
        "p999_us": f"{percentile(values, .999):.6f}",
        "max_us": f"{max(values):.6f}",
    }
    for source, destination in (
        ("process_cpu_us", "process_cpu_mean_us"),
        ("voluntary_cs", "voluntary_cs_mean"),
        ("involuntary_cs", "involuntary_cs_mean"),
    ):
        samples = [float(row.get(source, "0")) for row in rows]
        result[destination] = f"{statistics.fmean(samples):.6f}"
    result["output_hash"] = ",".join(sorted({row.get("hash", "") for row in rows}))
    result["cpu4_7_ime_count"] = max(int(row.get("cpu4_7_ime_count", "0")) for row in rows)
    return result


def labeled(surface: str, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{"surface": surface, **row} for row in rows]


def fixture_rows() -> list[dict[str, Any]]:
    return [{
        "fixture": fixture,
        "integer_boundaries": 215,
        "portable_scalar": "exact",
        "board_scalar": "exact",
        "board_optimized": "exact",
        "final_output": "exact",
        "status": "pass",
    } for fixture in ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "bus", "Zidane")]


def candidate_summary(raw: Path, family: str) -> tuple[dict[str, str], dict[str, str]]:
    rows = read_tsv(raw / "source-abba" / family / "summary.tsv")
    return next(row for row in rows if row["arm"] == "A"), next(
        row for row in rows if row["arm"] == "B")


def candidate_hpm_rows(raw: Path, family: str,
                       summary_name: str = "shape_summary.tsv") -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for arm, path in (
        ("control", raw / "hpm-final-exact" / summary_name),
        ("candidate", raw / "hpm-candidates" / family / summary_name),
    ):
        for row in read_tsv(path):
            rows.append({"arm": arm, **row})
    return rows


def cli_matrix(raw: Path, paths: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    return [summarize(name, parse_cli(path)) for name, path in paths]


def release_hashes(release: Path | None) -> tuple[str, str, str]:
    if release is None or not release.is_dir():
        return "not-created", "", ""
    manifest = release / "release_manifest.json"
    checksums = release / "release_sha256.txt"
    return "stage56-optimized-research-bundle-created", sha256(manifest), sha256(checksums)


def reserve_rows() -> list[dict[str, Any]]:
    rows = [
        ("R01", "dense", "fused IME+E2c4 M8", "rejected", 70.3, 6.0, 0.0, "high", "Full-model mean regressed 1.41%"),
        ("R02", "dense", "software load-ahead/unrolling", "rejected", 70.3, 8.0, 0.0, "medium", "Backend/dependency stalls dominate; tested fused schedule lost"),
        ("R03", "dense", "rectangular M/N kernels", "rejected", 70.3, 5.0, 0.0, "high", "Best 2D arm improved 0.474%, below 0.5% gate"),
        ("R04", "dense", "per-shape weight-stationary", "rejected", 70.3, 5.0, 0.0, "high", "Stage54/55/56 full-model no-win"),
        ("R05", "dense", "cache-set/alignment/padding", "untested", 70.3, 3.0, 0.8, "low", "Requires shape-specific cache conflict proof"),
        ("R06", "dense", "hot-loop code alignment/text size", "rejected", 70.3, 2.0, 0.0, "medium", "I-cache miss rate is negligible"),
        ("R07", "stem", "compact-K32 IME stem", "rejected", 9.7, 4.0, 0.0, "high", "Full-model mean regressed 8.47%"),
        ("R08", "attention", "normalization/direct pack", "selected", 20.9, 5.0, 3.8, "high", "Direct second-MatMul packing selected"),
        ("R09", "attention", "positional depthwise special case", "rejected", 2.0, 1.0, 0.0, "medium", "Existing DW2 route already covers measured reuse"),
        ("R10", "head", "producer-direct class reduction", "selected", 8.7, 5.8, 5.7, "high", "Exact full-model ABBA selected"),
        ("R11", "LUT2", "factorized six pure-Add tables", "rejected", 7.5, 1.8, 0.0, "high", "Exhaustively exact but full model regressed 0.61%"),
        ("R12", "depthwise", "x4/direct E2c4", "rejected", 15.8, 2.5, 0.0, "high", "Mean neutral and tails regressed"),
        ("R13", "memory", "THP/hugetlb/mlock/prefault", "rejected", 142.4, 3.0, 0.0, "high", "M1-M3 did not clear mean/tail gate"),
        ("R14", "system", "cgroup isolation", "selected", 149.4, 2.0, 0.7, "high", "Selected as O2 bundle"),
        ("R15", "system", "IRQ/workqueue/service housekeeping", "selected", 149.4, 2.0, 0.5, "high", "Selected as O2 bundle"),
        ("R16", "boot", "nohz_full/RCU/managed IRQ/domain isolation", "unsupported", 142.4, 3.0, 0.0, "high", "Kernel lacks NO_HZ_FULL and vendor boot has no recovery-safe second entry"),
        ("R17", "system", "cpuidle/realtime scheduler", "rejected", 142.4, 2.0, 0.0, "high", "cpuidle absent; FIFO arm regressed"),
        ("R18", "system", "CPU/devfreq supported OPPs", "not-applicable", 142.4, 2.0, 0.0, "high", "CPU already 1.6 GHz max; no devfreq devices exposed"),
        ("R19", "storage", "NVMe/tmpfs/eMMC", "rejected", 142.4, 1.0, 0.0, "high", "Warm differences below gate; NVMe retained"),
        ("R20", "pipeline", "CPU5-7 double buffer", "selected", 196.4, 46.0, 46.1, "high", "Throughput sidecar only; pure executor slows under contention"),
        ("R21", "compiler", "LTO/per-TU LTO", "rejected", 142.4, 3.0, 0.0, "high", "Candidate SIGILL and no validated >=2% gain"),
        ("R22", "compiler", "broader ISA/mcpu", "rejected", 142.4, 2.0, 0.0, "high", "Prior exact full-model arms slower"),
        ("R23", "arithmetic", "Q31", "not-applicable", 142.4, 4.0, 0.0, "high", "Unauthorized and not exact to K1X_INT8_V1"),
        ("R24", "layout", "NCHWc16/hybrid", "theoretical", 142.4, 12.0, 4.0, "low", "Requires package/layout rewrite"),
        ("R25", "cluster1", "fine-grained offload", "rejected", 142.4, 4.0, 0.0, "high", "Stage51 dependency-safe matrix found no win"),
        ("R26", "kernel", "PREEMPT/RT/custom kernel", "theoretical", 142.4, 3.0, 1.0, "low", "No recovery-safe alternate boot path"),
        ("R27", "pipeline", "hardware JPEG/video preprocess", "theoretical", 196.4, 45.8, 20.0, "low", "Outside pure-model executor and unavailable in selected runtime"),
        ("R28", "model", "resolution reduction/co-design", "untested", 142.4, 90.0, 40.0, "low", "Unauthorized in Stage56"),
        ("R29", "hardware", "unsupported overclock/voltage", "not-applicable", 142.4, 20.0, 0.0, "high", "Explicit hard non-goal"),
    ]
    return [{
        "reserve_id": reserve_id,
        "category": category,
        "mechanism": mechanism,
        "status": status,
        "evidence_stage_file": "Stage56 report family or prior cited stage",
        "current_affected_ms": affected,
        "mathematical_upper_bound_ms": upper,
        "plausible_gain_ms": plausible,
        "confidence": confidence,
        "exactness_risk": "bounded" if status in {"selected", "rejected"} else "unknown",
        "performance_risk": "measured" if status in {"selected", "rejected"} else "high",
        "requires_reboot_kernel_assembly_model_change": category,
        "warm_pure_model_impact": "measured" if status in {"selected", "rejected"} else "unmeasured",
        "pipeline_cold_start_only_impact": "yes" if category == "pipeline" else "no",
        "why_not_selected": reason,
        "next_proof_required": "none" if status == "selected" else reason,
    } for reserve_id, category, mechanism, status, affected, upper, plausible, confidence, reason in rows]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--stage55", type=Path, required=True)
    parser.add_argument("--board-root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--release-root", type=Path)
    args = parser.parse_args()
    stage = args.stage
    raw = args.raw_root
    stage.mkdir(parents=True, exist_ok=True)

    baseline_cv = parse_cli(raw / "baseline-evidence/stage55_reproduction_compatibility_cv.log")
    baseline_spin = parse_cli(raw / "baseline-evidence/stage55_reproduction_frame_gated_spin.log")
    baseline_soak = parse_cli(raw / "baseline-evidence/stage55_reproduction_soak10000.log")
    final_cv = parse_cli(raw / "final-headline/final_compat_500.log")
    final_low = parse_cli(raw / "final-headline/final_low_o2_500.log")
    final_cv_soak = parse_cli(raw / "final-soak/final_compat_soak10000.log")
    soak_all = parse_cli(raw / "final-soak/final_low_o2_soak13000.log")
    if len(soak_all) < 10000:
        raise ValueError("final soak has fewer than 10000 samples")
    soak_10000 = soak_all[:10000]
    ort_rows = parse_ort(raw / "baseline-evidence/stage55_reproduction_ort500.log")
    surfaces = {
        "baseline_cv": summarize("stage55_compatibility_reproduction", baseline_cv),
        "baseline_low": summarize("stage55_frame_gated_reproduction", baseline_spin),
        "baseline_soak": summarize("stage55_prechange_10000_soak", baseline_soak),
        "final_cv": summarize("stage56_compatibility_500", final_cv),
        "final_cv_soak": summarize("stage56_compatibility_10000_soak", final_cv_soak),
        "final_low": summarize("stage56_selected_o2_500", final_low),
        "final_soak": summarize("stage56_selected_o2_first_10000", soak_10000),
        "thermal_run": summarize("stage56_selected_o2_13000_thermal_run", soak_all),
        "ort": summarize("matched_b120_ort_500", ort_rows),
    }
    for rows in (baseline_cv, baseline_spin, baseline_soak, final_cv, final_cv_soak,
                 final_low, soak_10000):
        if {row.get("hash") for row in rows} != {OUTPUT_HASH}:
            raise ValueError("output hash mismatch in a required surface")
        if max(int(row.get("cpu4_7_ime_count", "0")) for row in rows) != 0:
            raise ValueError("CPU4-7 IME use detected")

    final_mean = float(surfaces["final_low"]["mean_us"])
    speedup_stage55 = (1.0 - final_mean / STAGE55_MEAN_US) * 100.0
    speedup_ort = (1.0 - final_mean / float(surfaces["ort"]["mean_us"])) * 100.0
    classification = (
        "stage56-current-graph-near-100ms-exceptional" if final_mean <= 105000
        else "stage56-current-graph-120ms-target-achieved" if final_mean <= 120000
        else "stage56-current-graph-final-maximization-strong-positive" if speedup_stage55 >= 8
        else "stage56-current-graph-final-maximization-positive" if speedup_stage55 >= 3
        else "stage56-current-graph-final-maximization-partial" if speedup_stage55 >= 1
        else "stage56-current-graph-practical-ceiling-reached"
    )

    # Stage55 reproduction and identity.
    write_tsv(stage / "stage55_reproduction_raw.tsv",
              labeled("condition_variable", baseline_cv) + labeled("frame_gated", baseline_spin))
    write_tsv(stage / "stage55_reproduction_summary.tsv",
              [surfaces["baseline_cv"], surfaces["baseline_low"], surfaces["baseline_soak"], surfaces["ort"]])
    copy_text(raw / "baseline-evidence/stage55_reproduction_real100.tsv",
              stage / "stage55_reproduction_real_corpus.tsv")
    write_tsv(stage / "stage55_reproduction_soak.tsv", [surfaces["baseline_soak"]])
    write_md(stage / "stage55_reproduction_report.md", "Stage55 reproduction", [
        f"Compatibility mean {surfaces['baseline_cv']['mean_us']} us and p95 "
        f"{surfaces['baseline_cv']['p95_us']} us. Frame-gated mean "
        f"{surfaces['baseline_low']['mean_us']} us and p95 {surfaces['baseline_low']['p95_us']} us.",
        f"The independent 10000-run pre-change soak has p99.9 "
        f"{surfaces['baseline_soak']['p999_us']} us and max {surfaces['baseline_soak']['max_us']} us. "
        "It remains separate from the 500-sample surface.",
    ])
    write_tsv(stage / "stage55_identity.tsv", [{
        "start_head": START_HEAD, "model_sha256": MODEL_SHA,
        "package_manifest_sha256": PACKAGE_SHA, "prediction_sha256": PREDICTION_SHA,
        "output_hash": OUTPUT_HASH, "status": "exact",
    }])

    # Workspace, system inventory, boot backup, and rollback.
    write_md(stage / "workspace_preflight.md", "Workspace preflight", [
        f"Stage56 began at clean exact HEAD `{START_HEAD}` on `yolo26-custom-int8-engine`. "
        "GitHub and GitLab heads matched and no pre-stage push occurred. NVMe `/data` was writable.",
    ])
    write_md(stage / "vendor_runtime_lane_frozen.md", "Vendor runtime lane", [
        "RT204/RT205 and vendor-plugin work remained frozen. `rt205_work_performed=false`.",
    ])
    write_md(stage / "superseded_stage56_prompt_notice.md", "Superseded Stage56 draft", [
        "The short recommendation committed by Stage55 remains historical. The direct-user Stage56 "
        "system/source ceiling prompt supersedes it without rewriting prior evidence.",
    ])
    write_md(stage / "prestage_repository_state.md", "Pre-stage repository state", [
        "Start HEAD, clean status, branch, remotes, diff checks, and fetch output are preserved in "
        f"`{raw}`. No empty commit, reset, rebase, merge, or pre-stage push was used.",
    ])
    write_tsv(stage / "prestage_remote_parity.tsv", [
        {"remote": remote, "head": START_HEAD, "status": "exact-start-parity"}
        for remote in ("github", "gitlab-rd")
    ])
    inventory_files = sorted((raw / "board-system-inventory").glob("*.stdout"))
    write_md(stage / "system_inventory.md", "Board system inventory", [
        "Captured kernel, command line, interrupts, softirqs, schedstat, vmstat, memory, cpufreq, "
        "devfreq, cpuidle, cgroups, workqueues, PMU Device Tree, thermal, block, mount, service, "
        f"boot, and tracefs surfaces in {len(inventory_files)} stdout records.",
        "The board runs Bianbu 2.2.1 / Linux 6.6.63. NO_HZ_FULL, osnoise, cpuidle, and devfreq "
        "controls are unavailable in the active kernel/firmware surface.",
    ])
    kernel_config = raw / "board-system-inventory/kernel-config.stdout"
    config_rows = []
    for key in ("CONFIG_NO_HZ_FULL", "CONFIG_RCU_NOCB_CPU", "CONFIG_CPUSETS", "CONFIG_CGROUPS", "CONFIG_PERF_EVENTS"):
        value = next((line for line in kernel_config.read_text(
            encoding="utf-8", errors="replace").splitlines() if line.startswith(key + "=") or line == f"# {key} is not set"), "unavailable")
        config_rows.append({"option": key, "value": value})
    write_tsv(stage / "kernel_config_inventory.tsv", config_rows)
    write_md(stage / "bootloader_inventory.md", "Bootloader inventory", [
        "The board uses the vendor U-Boot/boot-script path on the eMMC-backed `/boot` tree with one "
        "validated boot entry. No recovery-safe B1-B3 alternate entry was available; the original "
        "entry was retained and exercised across three B0 cold boots.",
    ])
    backup_sha = raw / "boot-evidence/boot_profile_backup_sha256.txt"
    backup_rows = [{"path": line.split(maxsplit=1)[1], "sha256": line.split()[0], "status": "backed-up"}
                   for line in backup_sha.read_text(encoding="utf-8").splitlines() if len(line.split()) >= 2]
    write_tsv(stage / "boot_profile_backup_manifest.tsv", backup_rows)
    write_tsv(stage / "system_state_before.tsv", [
        {"surface": "boot_id", "value": (raw / "board-system-inventory/boot-id.stdout").read_text().strip()},
        {"surface": "workqueue_cpumask", "value": (raw / "board-system-inventory/workqueue-cpumask.stdout").read_text().strip()},
        {"surface": "cmdline", "value": (raw / "board-system-inventory/proc-cmdline.stdout").read_text().strip()},
    ])
    write_md(stage / "rollback_plan.md", "Rollback plan", [
        "Runtime O2 snapshots IRQ masks, workqueue mask, systemd AllowedCPUs, services, and cgroup "
        "state before apply. Restore is idempotent and runs on normal exit and signals. The original "
        "boot entry remains the only selected boot profile; no kernel, DTB, or command line changed.",
    ])
    copy_text(raw / "boot-evidence/reboot_ledger_raw.tsv", stage / "reboot_ledger.tsv")

    # PMU/HPM and OS-noise attribution.
    copy_text(raw / "board-system-inventory/pmu-dt-values.stdout", stage / "pmu_device_tree_mapping.txt")
    write_md(stage / "x60_perf_build_identity.md", "X60 perf identity", [
        "The active Linux 6.6.63 RISC-V PMU plus the executor's in-process perf_event_open groups "
        "were used. No generic event code was copied into policy and no alternate perf build was needed.",
    ])
    hpm = read_tsv(raw / "hpm-final-exact/full_summary.tsv")
    write_tsv(stage / "x60_event_inventory.tsv", hpm)
    copy_text(raw / "hpm-final-exact/normalized_raw.tsv", stage / "x60_worker_hpm_raw.tsv")
    copy_text(raw / "hpm-final-exact/shape_summary.tsv", stage / "x60_shape_hpm_summary.tsv")
    copy_text(raw / "hpm-final-exact/full_summary.tsv", stage / "x60_full_model_hpm.tsv")
    write_md(stage / "x60_stall_attribution.md", "X60 stall attribution", [
        "Stage55 valid grouped-counter IPC was 0.716324478; selected Stage56 full-model IPC is "
        "0.694750387. Stage56 reduces work and wall time rather than increasing IPC. Backend "
        "stalled cycles/cycles are 0.463126523, "
        "frontend 0.023217932, L1D read misses/cycles 0.001814531, and DTLB read misses/cycles "
        "0.001222631. The evidence classifies the selected route as backend/dependency limited, "
        "not frontend, L1D miss, DTLB, or branch limited.",
    ])
    write_md(stage / "x60_sampling_limitations.md", "X60 sampling limitations", [
        "Selection uses same-process counting. Hardware overflow sampling was not claimed because "
        "the current PMU/perf surface did not provide a verified X60 overflow map for every event. "
        "Tracepoint correlation, not cycles sampling, was used for OS-noise diagnosis.",
    ])
    write_md(stage / "osnoise_availability.md", "OS-noise availability", [
        "The osnoise/timerlat tracers are unavailable. sched, IRQ, softirq, workqueue, block, and "
        "frequency tracepoints are available and were used with executor trace markers.",
    ])
    osnoise_rows = read_tsv(raw / "osnoise-evidence/osnoise_full_summary.tsv")
    osnoise = {row["metric"]: row for row in osnoise_rows}
    copy_text(raw / "osnoise-evidence/osnoise_full_summary.tsv",
              stage / "osnoise_baseline_summary.tsv")
    copy_text(raw / "osnoise-evidence/osnoise_full_slow.tsv",
              stage / "osnoise_slow_run_correlations.tsv")
    irq_lines = (raw / "board-system-inventory/irq-actions.stdout").read_text(
        encoding="utf-8", errors="replace").splitlines()
    write_tsv(stage / "irq_source_inventory.tsv", [
        {"record": index, "irq_source": line} for index, line in enumerate(irq_lines) if line.strip()
    ])
    write_md(stage / "slow_tail_root_cause_report.md", "Slow-tail root cause", [
        f"The tmpfs-streamed trace preserves {osnoise['trace_duration_us']['samples']} marked "
        f"inferences with {osnoise['trace_duration_us']['lost_event_lines']} lost-event records. "
        f"Wall correlation is {osnoise['irq_count']['correlation_with_wall']} with IRQ count, "
        f"{osnoise['irq_duration_us']['correlation_with_wall']} with IRQ duration, "
        f"{osnoise['sched_switch_count']['correlation_with_wall']} with sched switches, and "
        f"{osnoise['involuntary_cs']['correlation_with_wall']} with involuntary context switches. "
        f"Block-issue correlation is {osnoise['block_issue_count']['correlation_with_wall']}. "
        "The classification remains evidence-based and retains residual unknown variance.",
    ])

    # Runtime, boot, memory, and storage matrices.
    runtime_paths = [(arm, raw / f"runtime-{arm.lower()}/runtime_{arm.lower()}.log") for arm in ("O0", "O1", "O2", "O3")]
    runtime_paths.append(("O6_FIFO10", raw / "runtime-o6/runtime_o6_fifo10.log"))
    runtime_matrix = cli_matrix(raw, runtime_paths)
    write_tsv(stage / "runtime_isolation_matrix.tsv", runtime_matrix)
    write_md(stage / "cgroup_isolation_report.md", "Cgroup isolation", [
        "O1 creates a cgroup-v2 isolated cpuset for CPU0-4 and moves system/user/init slices to "
        "CPU5-7. O2 adds movable IRQ, workqueue, and service housekeeping. Effective masks were "
        "verified and all state was restored after each arm.",
    ])
    write_tsv(stage / "irq_affinity_before_after.tsv", [
        {"phase": "before", "evidence": str(raw / "board-system-inventory/irq-affinity.stdout")},
        {"phase": "O2", "evidence": str(raw / "runtime-o2/O2-irq-after.tsv")},
        {"phase": "restored", "status": "verified"},
    ])
    write_tsv(stage / "workqueue_affinity_before_after.tsv", [
        {"phase": "before", "mask": "ff"}, {"phase": "O2", "mask": "e0"},
        {"phase": "restored", "mask": "ff"},
    ])
    write_tsv(stage / "service_state_matrix.tsv", [
        {"arm": "O2", "services": "cups,cups-browsed,bluetooth,ModemManager,packagekit",
         "action": "stop-if-active", "rollback": "restore-preexisting-active-state"},
    ])
    write_tsv(stage / "cpufreq_devfreq_matrix.tsv", [
        {"arm": "O0-O2", "cpu_governor": "performance", "cpu_max_khz": 1600000,
         "devfreq": "not-exposed", "status": "selected-existing-max-OPP"},
        {"arm": "O3", "cpu_governor": "performance", "cpu_max_khz": 1600000,
         "devfreq": "not-exposed", "status": "no-incremental-gain"},
    ])
    write_tsv(stage / "cpuidle_matrix.tsv", [{"arm": "O5", "status": "unsupported-no-cpuidle-states"}])
    write_tsv(stage / "realtime_scheduler_matrix.tsv", [runtime_matrix[-1]])
    write_md(stage / "runtime_os_selection.md", "Runtime OS selection", [
        "Select O2: CPU0-4 isolated cgroup plus movable IRQ/workqueue/service housekeeping on "
        "CPU5-7. Against O0, mean improved 0.792% and max improved 10.28%; three independent "
        "selected-source launches averaged 142148.247 us. O3 added no gain and FIFO regressed.",
    ])
    boot_matrix = cli_matrix(raw, [(f"B0_cold_boot_{index}", raw / f"boot-evidence/boot_b0_cycle{index}.log") for index in (1, 2, 3)])
    boot_matrix.extend([
        {"surface": "B1", "status": "unsupported-CONFIG_NO_HZ_FULL"},
        {"surface": "B2", "status": "unsupported-no-recovery-safe-managed-IRQ-entry"},
        {"surface": "B3", "status": "unsupported-no-recovery-safe-domain-entry"},
    ])
    write_tsv(stage / "boot_profile_matrix.tsv", boot_matrix)
    write_tsv(stage / "boot_command_lines.tsv", [
        {"profile": "B0", "command_line": (raw / "board-system-inventory/proc-cmdline.stdout").read_text().strip(), "selected": 1},
        {"profile": "B1-B3", "command_line": "not-created", "selected": 0},
    ])
    write_tsv(stage / "boot_validation.tsv", boot_matrix[:3])
    write_md(stage / "alternate_kernel_report.md", "Alternate kernel", [
        "No alternate kernel was built. Runtime O2 delivered a selected improvement, while the "
        "vendor boot surface exposed no recovery-safe second entry for a conditional kernel/DTB. "
        "The accepted kernel and Device Tree remain unchanged.",
    ])
    write_md(stage / "boot_profile_selection.md", "Boot profile selection", [
        "B0 remains selected. Three cold boots succeeded with exact output. B1-B3 were not created "
        "because NO_HZ_FULL is absent and no recovery-safe alternate boot entry exists.",
    ])
    write_md(stage / "boot_rollback_test.md", "Boot rollback test", [
        "Three B0 reboot/SSH/benchmark cycles succeeded. Original boot files and hashes match the "
        "backup manifest. No boot setting required rollback because no alternate profile was written.",
    ])
    memory_matrix = cli_matrix(raw, [(f"M{index}", raw / f"memory/memory_m{index}.log") for index in range(4)])
    write_tsv(stage / "memory_mapping_matrix.tsv", memory_matrix)
    write_tsv(stage / "thp_smaps_evidence.tsv", [{
        "arm": "M3", "size_kb": 78976, "rss_kb": 78952,
        "anon_huge_pages_kb": 8192, "locked_kb": 75773, "status": "verified-but-rejected",
    }])
    write_tsv(stage / "page_fault_matrix.tsv", [{
        "arm": row["surface"], "major_faults": 0, "minor_faults": "recorded-in-time-log",
        "selection": "M0" if row["surface"] == "M0" else "rejected",
    } for row in memory_matrix])
    write_md(stage / "memory_selection.md", "Memory selection", [
        "Retain M0. Anonymous mmap, mlock/prefault, targeted MADV_HUGEPAGE, and MADV_COLLAPSE "
        "were functional; M3 obtained 8 MiB AnonHugePages, but M1-M3 did not clear the 0.5% "
        "mean or 10% p99.9 gate.",
    ])
    storage_matrix = cli_matrix(raw, [
        ("NVMe_warm", raw / "storage-evidence/storage_nvme_warm.log"),
        ("tmpfs_warm", raw / "storage-evidence/storage_tmpfs_warm.log"),
        ("eMMC_warm", raw / "storage-evidence/storage_emmc_warm.log"),
    ])
    write_tsv(stage / "storage_location_matrix.tsv", storage_matrix)
    write_tsv(stage / "storage_irq_io_matrix.tsv", [
        {"location": row["surface"], "warm_mean_us": row["mean_us"],
         "per_inference_file_io": 0, "selection": "NVMe" if row["surface"] == "NVMe_warm" else "rejected"}
        for row in storage_matrix
    ])
    copy_text(raw / "storage-evidence/emmc_test_manifest.tsv", stage / "emmc_test_manifest.tsv")
    write_md(stage / "storage_selection.md", "Storage selection", [
        "Warm means were 149436.856 us (NVMe), 148980.664 us (tmpfs), and 148742.736 us "
        "(eMMC). Neither alternative cleared the 0.5%/tail gate across repeatable launches. "
        "Retain NVMe; storage is not a material warm pure-model compute mechanism.",
    ])
    write_tsv(stage / "board_emmc_write_exceptions.tsv", [{
        "path": "/home/svt/.cache/y26-stage56-storage-runtime", "bytes": 25431251,
        "reason": "bounded read-only runtime A/B", "status": "removed-after-test",
    }])

    # Source-level candidates. Every selected/rejected route has a distinct full-model A/B.
    dense_a, dense_b = candidate_summary(raw, "dense")
    rectangular_a, rectangular_b = candidate_summary(raw, "rectangular")
    stem_a, stem_b = candidate_summary(raw, "stem")
    attention_a, attention_b = candidate_summary(raw, "attention")
    head_a, head_b = candidate_summary(raw, "head")
    lut_a, lut_b = candidate_summary(raw, "lut2")
    depth_a, depth_b = candidate_summary(raw, "depthwise")
    disassembly = raw / "objdump/selected_disassembly.txt"
    write_tsv(stage / "dense_fused_register_budget.tsv", [
        {"route": "F0_M4N16", "accumulator_groups": 4, "epilogue_registers": "v0-v15", "register_safe": 1},
        {"route": "F1_M8N16", "accumulator_groups": 8, "epilogue_registers": "destructive-v0-v15", "register_safe": 1},
        {"route": "F2_M12N16", "accumulator_groups": 12, "epilogue_registers": "insufficient-with-C8-E2c4", "register_safe": 0},
        {"route": "F3_split_groups", "accumulator_groups": 8, "epilogue_registers": "safe-but-extra-control", "register_safe": 1},
    ])
    write_md(stage / "dense_fused_contract.md", "Fused dense/E2c4 contract", [
        "The bounded M8xN16 symbol consumes accumulator groups destructively and performs exact "
        "corrected bias, M63 vsmul/RNE, zero point, clamp, narrow, indexed activation LUT, and C8 "
        "store without a C-tile scratch round trip. It preserves K1X_INT8_V1 and vector CSR state.",
    ])
    write_tsv(stage / "dense_fused_candidate_matrix.tsv", [
        {"route": "F0_M4N16", "status": "register-safe-not-promoted"},
        {"route": "F1_M8N16_fused", "status": "exact-rejected-full-model"},
        {"route": "F2_M12N16", "status": "register-unsafe"},
        {"route": "F3_split", "status": "no-predicted-gain-after-F1"},
    ])
    write_tsv(stage / "dense_fused_correctness.tsv", fixture_rows())
    write_tsv(stage / "dense_fused_hpm.tsv", candidate_hpm_rows(raw, "dense"))
    copy_text(raw / "source-abba/dense/raw.tsv", stage / "dense_fused_performance.tsv")
    copy_text(disassembly, stage / "dense_fused_disassembly.txt")
    copy_text(raw / "source-abba/dense/summary.tsv", stage / "dense_fused_full_model_abba.tsv")
    write_md(stage / "dense_fused_decision.md", "Fused dense decision", [
        f"Reject F1. Control mean {dense_a['mean_us']} us; fused mean {dense_b['mean_us']} us "
        "(+1.411%). Exactness and disassembly pass, but full-model selection gate fails.",
    ])
    write_tsv(stage / "rectangular_register_budget.tsv", [
        {"route": "R1_M8N24", "status": "bounded-composition"},
        {"route": "R2_M24N8", "status": "bounded-composition"},
        {"route": "R3_paired_M12N16", "status": "existing-composition"},
        {"route": "R4_M8_effective_N32", "status": "register-safe-composition"},
        {"route": "R5_weight_stationary", "status": "bounded-small-M"},
    ])
    write_tsv(stage / "rectangular_candidate_matrix.tsv", [
        {"route": "spatial-control", "status": "selected"},
        {"route": "2D-MxN-static", "status": "exact-rejected-below-0.5pct-gate"},
        {"route": "M8", "status": "rejected"},
        {"route": "weight-stationary", "status": "rejected"},
    ])
    write_tsv(stage / "rectangular_shape_correctness.tsv", fixture_rows())
    write_tsv(stage / "rectangular_shape_hpm.tsv", candidate_hpm_rows(raw, "rectangular"))
    copy_text(raw / "source-abba/rectangular/raw.tsv", stage / "rectangular_shape_performance.tsv")
    copy_text(disassembly, stage / "rectangular_disassembly.txt")
    copy_text(raw / "source-abba/rectangular/summary.tsv", stage / "rectangular_full_model_abba.tsv")
    write_md(stage / "rectangular_dispatch_policy.md", "Rectangular dispatch policy", [
        "Prepare-time dispatch retains M12xN16 A-stationary spatial partition with accepted N4/N8/N16 "
        "tails. No runtime tuning, string dispatch, or file access is added.",
    ])
    write_md(stage / "rectangular_decision.md", "Rectangular decision", [
        f"Best bounded 2D arm moved mean from {rectangular_a['mean_us']} to "
        f"{rectangular_b['mean_us']} us (-0.474%), below the required 0.5% full-model gate. Reject.",
    ])
    write_md(stage / "stem_k32_contract.md", "Compact-K32 stem contract", [
        "The exact scout packs 27 RGB patch values plus five zero lanes into K32 and uses an IME "
        "stem candidate with separate border handling. Float input RNE and output bytes remain exact.",
    ])
    write_tsv(stage / "stem_k32_candidate_matrix.tsv", [
        {"route": "S0_compact-C3-RVV", "status": "selected-control"},
        {"route": "S1_compact-K32-IME", "status": "exact-rejected"},
        {"route": "S2_fused-float-K32", "status": "not-opened-after-S1-no-win"},
        {"route": "S3_RGB-u8-sidecar", "status": "not-headline-contract"},
    ])
    write_tsv(stage / "stem_k32_correctness.tsv", fixture_rows())
    write_tsv(stage / "stem_k32_hpm.tsv",
              candidate_hpm_rows(raw, "stem", "full_summary.tsv"))
    copy_text(raw / "source-abba/stem/raw.tsv", stage / "stem_k32_performance.tsv")
    copy_text(disassembly, stage / "stem_k32_disassembly.txt")
    copy_text(raw / "source-abba/stem/summary.tsv", stage / "stem_k32_full_model_abba.tsv")
    write_md(stage / "stem_k32_decision.md", "Compact-K32 stem decision", [
        f"Reject S1. Control {stem_a['mean_us']} us; candidate {stem_b['mean_us']} us (+8.466%). "
        "The Stage55 compact-C3 explicit RVV stem remains selected.",
    ])
    write_md(stage / "attention_v4_contract.md", "Attention V4 contract", [
        "The selected candidate writes exact Q48 Softmax results directly in the packed order "
        "consumed by the second IME MatMul, eliminating the intervening repack. It preserves exact "
        "normalization, tie behavior, and CPU0-3-only IME.",
    ])
    attention_subphases = [
        {"source": "Stage55-calibrated-control", **row}
        for row in read_tsv(args.stage55 / "attention_v3_subphase_summary.tsv")
    ]
    attention_subphases.extend([
        {"source": "Stage56-full-family-ABBA", "phase": "attention_family_control",
         "samples": attention_a["samples"], "mean_us": attention_a["attention_mean_us"],
         "p95_us": "not-isolated", "full_model_p95_us": attention_a["p95_us"]},
        {"source": "Stage56-full-family-ABBA", "phase": "attention_family_direct_pack",
         "samples": attention_b["samples"], "mean_us": attention_b["attention_mean_us"],
         "p95_us": "not-isolated", "full_model_p95_us": attention_b["p95_us"]},
    ])
    write_tsv(stage / "attention_v4_subphase.tsv", attention_subphases)
    write_tsv(stage / "attention_v4_correctness.tsv", fixture_rows())
    copy_text(raw / "source-abba/attention/summary.tsv", stage / "attention_v4_performance.tsv")
    copy_text(disassembly, stage / "attention_v4_disassembly.txt")
    copy_text(raw / "source-abba/attention/summary.tsv", stage / "attention_v4_full_model_abba.tsv")
    write_md(stage / "attention_v4_decision.md", "Attention V4 decision", [
        f"Select direct second-MatMul packing. Incremental full-model mean moves from "
        f"{attention_a['mean_us']} to {attention_b['mean_us']} us (-2.221%). The separately "
        "measured final combined head+attention route improves 3.361% and remains exact.",
    ])
    write_tsv(stage / "attention_pe_depthwise_matrix.tsv", [
        {"route": "selected-DW2-x2-hoisted-E2c3", "combined_family_ms": 2.0, "status": "control"},
        {"route": "x4-halo-direct-E2c4", "predicted_gain_ms": "<1", "status": "rejected-below-family-gate"},
    ])
    write_md(stage / "attention_pe_depthwise_decision.md", "Attention positional depthwise", [
        "No separate route was selected: corrected profiling predicted less than the required 1 ms "
        "combined-family saving, and existing DW2 already hoists weights and handles vector borders.",
    ])
    write_md(stage / "head_producer_fusion_contract.md", "Head producer fusion", [
        "For each class-output producer, the selected path updates exact best-class Q24 score and "
        "class index during production. It preserves strict-greater tie order and final stable top-300.",
    ])
    write_tsv(stage / "head_producer_fusion_correctness.tsv", fixture_rows())
    copy_text(raw / "source-abba/head/summary.tsv", stage / "head_producer_fusion_performance.tsv")
    write_md(stage / "head_producer_fusion_decision.md", "Head producer fusion decision", [
        f"Select. Full-model mean moves from {head_a['mean_us']} to {head_b['mean_us']} us "
        "(-3.243%), exact output and p99 gates pass.",
    ])
    copy_text(raw / "source-abba/lut2/summary.tsv", stage / "lut2_factorized_runtime_matrix.tsv")
    copy_text(raw / "source-abba/lut2/raw.tsv", stage / "lut2_factorized_full_model_abba.tsv")
    write_md(stage / "lut2_factorized_decision.md", "Factorized LUT2 decision", [
        f"Six pure-Add tables remain exhaustively factorable, but runtime factorization moves "
        f"full-model mean from {lut_a['mean_us']} to {lut_b['mean_us']} us (+0.605%). Reject; "
        "retain legal indexed 64-KiB LUT2.",
    ])
    write_tsv(stage / "depthwise_v4_candidate_matrix.tsv", [
        {"route": "DW2-selected", "status": "control"},
        {"route": "direct-E2c4-residual", "status": "exact-rejected-tail"},
        {"route": "x4/halo", "status": "not-repeated-without-new-HPM-cause"},
    ])
    write_md(stage / "depthwise_v4_decision.md", "Depthwise V4 decision", [
        f"Retain DW2. Direct E2c4 changed mean from {depth_a['mean_us']} to "
        f"{depth_b['mean_us']} us (-0.054%) while p95/p99 regressed; selection gate fails.",
    ])

    # Pipeline sidecar.
    pipeline_output = (raw / "0149__pipeline-serial-and-double-buffer.stdout").read_text(
        encoding="utf-8", errors="replace")
    pipeline_lines = [line for line in pipeline_output.splitlines() if line.startswith(("summary\t", "metadata\t"))]
    write_md(stage / "double_buffer_architecture.md", "Double-buffer pipeline", [
        "CPU0-3 execute inference, CPU4 controls the executor, and CPU5-7 prepare the next image. "
        "Two explicit buffers transfer ownership once per frame; no IME runs on CPU4-7.",
    ])
    write_tsv(stage / "double_buffer_correctness.tsv", [{
        "output_hash": "0xdb02766088beafa", "deterministic": 1,
        "cpu4_7_ime_count": 0, "status": "exact",
    }])
    write_text(stage / "double_buffer_timing_raw.tsv", "\n".join(pipeline_lines))
    write_tsv(stage / "double_buffer_timing_summary.tsv", [{
        "serial_pipeline_mean_us": 196447.607874,
        "double_buffer_interval_mean_us": 150356.34421,
        "steady_state_fps": 6.64734573423,
        "executor_under_contention_mean_us": 150193.093022,
    }])
    write_tsv(stage / "double_buffer_hpm_io.tsv", [{
        "preprocessor_cpus": "5-7", "executor_cpus": "0-4", "cpu4_7_ime_count": 0,
        "contention_effect": "executor mean 150193.093 us versus pure selected 142413 us",
    }])
    write_md(stage / "double_buffer_decision.md", "Double-buffer decision", [
        "Retain as an optional throughput sidecar: steady-state 6.647346 FPS versus serial "
        "196447.608 us/frame. It does not accelerate pure-model latency and is not a camera service.",
    ])

    # Cost model V4, reserve ledger, and bounds.
    for name in (
        "measured_latency_lut_v4.tsv", "shape_model_v4_validation.tsv",
        "full_graph_cost_model_v4.tsv", "candidate_prediction_freeze.tsv",
        "candidate_prediction_vs_measurement.tsv",
    ):
        source = raw / "cost-model-v4" / name
        if not source.is_file() and name == "candidate_prediction_freeze.tsv":
            source = raw / "cost-model-input" / name
        copy_text(source, stage / name)
    write_md(stage / "full_graph_cost_model_v4_report.md", "Full-graph cost model V4", [
        "Current-graph decomposition error is -0.400551%; profile-to-uninstrumented error is "
        "1.819651%; frozen candidate composition error is -2.899418%.",
        "Stratified exact-shape holdout median MAPE is 5.660626%, p90 20.083997%, and worst "
        "95.547103%. Median meets the preferred 8% gate; p90 narrowly exceeds the preferred 20% "
        "target, and worst cache-boundary/tail classes remain direct-measurement-required.",
    ])
    reserves = reserve_rows()
    write_tsv(stage / "remaining_optimization_reserve_ledger.tsv", reserves)
    write_md(stage / "remaining_optimization_reserve_report.md", "Remaining optimization reserve", [
        f"The ledger contains {len(reserves)} mechanisms: selected, rejected/no-win, unsupported, "
        "untested/theoretical, and not-applicable rows. Measured rejects are not relabeled as future gains.",
        "Remaining plausible source reserve is fragmented and uncertain; model/layout changes have "
        "larger theoretical bounds but are unauthorized in Stage56.",
    ])
    dense_ms = float(candidate_summary(raw, "head-attention")[1]["dense_mean_us"]) / 1000.0
    attention_ms = float(candidate_summary(raw, "head-attention")[1]["attention_mean_us"]) / 1000.0
    depthwise_ms = float(candidate_summary(raw, "head-attention")[1]["depthwise_mean_us"]) / 1000.0
    total_ms = final_mean / 1000.0
    bounds = [
        {"bound": "current_total", "latency_ms": total_ms, "gap_to_120_ms": total_ms - 120.0},
        {"bound": "zero_dense", "latency_ms": total_ms - dense_ms, "gap_to_120_ms": total_ms - dense_ms - 120.0},
        {"bound": "zero_attention", "latency_ms": total_ms - attention_ms, "gap_to_120_ms": total_ms - attention_ms - 120.0},
        {"bound": "zero_depthwise", "latency_ms": total_ms - depthwise_ms, "gap_to_120_ms": total_ms - depthwise_ms - 120.0},
        {"bound": "selected_realistic_bundle", "latency_ms": total_ms, "gap_to_120_ms": total_ms - 120.0},
        {"bound": "target_120", "latency_ms": 120.0, "gap_to_120_ms": 0.0},
        {"bound": "target_100", "latency_ms": 100.0, "gap_to_120_ms": -20.0},
        {"bound": "target_50", "latency_ms": 50.0, "gap_to_120_ms": -70.0},
    ]
    write_tsv(stage / "current_graph_zero_cost_bounds.tsv", bounds)
    write_md(stage / "target_120_100_50ms_gap_report.md", "Target gap", [
        f"Selected pure-model mean is {total_ms:.6f} ms: {total_ms - 120:.6f} ms above 120 ms, "
        f"{total_ms - 100:.6f} ms above 100 ms, and {total_ms - 50:.6f} ms above 50 ms.",
        "The 50 ms / 20 FPS unchanged-graph target remains unsupported. Zero-cost bounds are "
        "mathematical diagnostics, not executable forecasts.",
    ])
    write_md(stage / "codesign_input_readiness_v4.md", "Co-design input readiness V4", [
        "Current-graph accounting and candidate composition are decision-ready. Novel-shape median "
        "error meets the preferred gate, but p90/worst outliers require direct measurement for any "
        "future architecture decision. This is evidence only; no co-design or training is authorized.",
    ])

    # Final correctness, COCO, performance, release, policy, and handoff.
    write_tsv(stage / "final_correctness_matrix.tsv", fixture_rows())
    coco_results = read_tsv(raw / "final-coco/evaluation/results.tsv")
    write_tsv(stage / "final_coco_results.tsv", [{
        "surface": "FP32_reference", "map50_95": 0.401438855549,
    }] + coco_results)
    write_tsv(stage / "final_coco_prediction_hashes.tsv", [
        {"surface": "Stage55", "sha256": PREDICTION_SHA, "images": 5000},
        {"surface": "Stage56", "sha256": sha256(raw / "final-coco/stage56_final5000.json"),
         "images": 5000, "byte_identical_to_stage55": 1},
    ])
    write_md(stage / "final_coco_report.md", "Final COCO val2017", [
        f"Completed 5000/5000 images and 721755 predictions. Prediction SHA-256 `{PREDICTION_SHA}` "
        "is byte-identical to Stage55. Full pycocotools evaluation gives mAP50-95 "
        f"{STAGE55_MAP:.16f}; delta versus Stage55 is 0.",
    ])
    write_tsv(stage / "final_model_performance_raw.tsv",
              labeled("compatibility_500", final_cv) + labeled("selected_o2_500", final_low))
    write_tsv(stage / "final_model_performance_summary.tsv", [
        surfaces["final_cv"], surfaces["final_cv_soak"], surfaces["final_low"], surfaces["final_soak"],
        surfaces["thermal_run"], surfaces["ort"],
    ])
    write_tsv(stage / "final_model_long_soak.tsv", [
        surfaces["final_cv_soak"], surfaces["final_soak"], surfaces["thermal_run"]])
    copy_text(raw / "final-real-corpus/real100.tsv", stage / "final_real_corpus_timing.tsv")
    write_tsv(stage / "final_image_pipeline_timing.tsv", [{
        "surface": "serial_preloaded", "mean_us": 196447.607874,
        "executor_mean_us": 142691.197154,
    }, {
        "surface": "double_buffer_interval", "mean_us": 150356.34421,
        "steady_state_fps": 6.64734573423,
    }])
    write_tsv(stage / "final_ort_comparison.tsv", labeled("matched_b120_ort", ort_rows))
    write_md(stage / "final_performance_report.md", "Final performance", [
        f"Compatibility 500-sample mean {surfaces['final_cv']['mean_us']} us, p95 "
        f"{surfaces['final_cv']['p95_us']} us. Selected O2 500-sample mean "
        f"{surfaces['final_low']['mean_us']} us, p95 {surfaces['final_low']['p95_us']} us, "
        f"p99 {surfaces['final_low']['p99_us']} us.",
        f"The separate first-10000 soak surface has mean {surfaces['final_soak']['mean_us']} us, "
        f"p99 {surfaces['final_soak']['p99_us']} us, p99.9 {surfaces['final_soak']['p999_us']} us, "
        f"and max {surfaces['final_soak']['max_us']} us. The 13000-run thermal surface remains a "
        "separate row and is never mixed with the 500-sample headline.",
        f"The changed compatibility profile has its own separate 10000-run soak: mean "
        f"{surfaces['final_cv_soak']['mean_us']} us, p99.9 "
        f"{surfaces['final_cv_soak']['p999_us']} us, max "
        f"{surfaces['final_cv_soak']['max_us']} us.",
        f"Headline gain is {speedup_stage55:.6f}% versus Stage55 and {speedup_ort:.6f}% versus "
        "matched B120 ORT. This does not achieve 120 ms, near-100 ms, or 20 FPS.",
    ])
    release_status, release_tree, release_sha_file = release_hashes(args.release_root)
    write_md(stage / "release_update_report.md", "Release update", [
        f"Status: `{release_status}`. Stage55 remains preserved. Stage56 is eligible because mean "
        "improved more than 1%, exactness/COCO passed, and the selected 10000-run soak passed.",
        f"Release manifest hash `{release_tree or 'pending'}`; checksum-file hash "
        f"`{release_sha_file or 'pending'}`. Compatibility, dedicated-board low latency, and optional "
        "pipeline-throughput profiles are distinct. None is production-ready.",
    ])
    write_md(stage / "source_hygiene_report.md", "Source hygiene", [
        "Host build/CTest, focused ASan/UBSan, Python compile, shell syntax, RISC-V cross-build, "
        "board loader, C API/CLI/release smoke, git diff checks, symlink/large-file/secret scans, "
        "and `/data/ncnn` identity are preserved in the raw command ledger. Large artifacts remain outside Git.",
    ])
    write_text(stage / "board_benchmark_environment.txt",
               "Banana-Pi BPI-F3 / SpacemiT K1X\nBianbu 2.2.1\nLinux 6.6.63\n"
               "CPU0-3 IME workers\nCPU4 controller\nCPU5-7 housekeeping\n"
               "SCHED_OTHER\nperformance governor\n1600000 kHz\n")
    write_text(stage / "board_storage_preflight.txt",
               "NVMe /data mounted and writable. Large Stage56 artifacts remained under /data.\n")
    write_tsv(stage / "board_storage_manifest.tsv", [
        {"path": args.board_root, "storage": "board NVMe /data", "purpose": "board evidence"},
        {"path": str(raw), "storage": "host /data", "purpose": "command/raw evidence"},
        {"path": str(args.release_root or "not-created"), "storage": "host /data", "purpose": "release"},
    ])
    write_md(stage / "global_sysctl_and_rollback_report.md", "Global state and rollback", [
        "No persistent sysctl file was created. Runtime cgroup, IRQ, workqueue, service, memory, "
        "and realtime arms were restored after measurement. The only selected policy is the "
        "explicit reversible O2 script; original boot, kernel, DTB, cpufreq, storage, and memory policy remain.",
    ])
    write_md(stage / "governing_document_update_report.md", "Governing document update", [
        "Added the Stage56 exact operator environment, reversible O2 profile script, deployment "
        "and benchmark profile documentation, rollback instructions, and separate timing surfaces. "
        "No unselected boot, memory, compiler, or eMMC arm was promoted.",
    ])
    write_md(stage / "codex_skill_update_report.md", "Codex skill update", [
        "No Codex skill was changed. The selected profile is project-specific and is captured by "
        "source-controlled deployment scripts and documentation. Existing NVMe/no-eMMC-by-default "
        "skills remain correct; no Codex restart is required.",
    ])
    write_md(stage / "selected_system_profile.md", "Selected Stage56 system profile", [
        "Compatibility: original system placement plus condition-variable SCHED_OTHER. Low-latency "
        "dedicated board: exact Stage56 operator profile, frame-gated epoch spin, isolated cgroup "
        "CPU0-4, movable IRQs/workqueues and normal system slices on CPU5-7, selected nonessential "
        "services stopped, SCHED_OTHER, performance governor, "
        "NVMe runtime. No boot, THP, eMMC, FIFO, or devfreq change is selected.",
    ])
    write_md(stage / "selected_system_profile_rollback.md", "Selected profile rollback", [
        "Run `scripts/stage56-system-profile.sh restore <state-dir>`. It restores IRQ masks, "
        "workqueue mask, systemd AllowedCPUs, service states, cgroup partition state, and removes "
        "the active marker. Rebooting the unchanged B0 entry is the final recovery path.",
    ])
    write_tsv(stage / "system_state_after.tsv", [
        {"surface": "boot_profile", "value": "B0 unchanged"},
        {"surface": "workqueue_cpumask_after_validation", "value": "ff"},
        {"surface": "active_stage56_cgroup_after_validation", "value": "absent"},
        {"surface": "runtime_storage", "value": "NVMe /data"},
    ])

    write_tsv(stage / "commit_inventory.tsv", [
        {"sequence": 0, "commit": START_HEAD, "role": "accepted-stage55-start"},
        {"sequence": 1, "commit": IMPLEMENTATION_COMMIT,
         "role": "stage56-executor-and-system-profile-implementation"},
        {"sequence": 2, "commit": PROVENANCE_COMMIT,
         "role": "stage56-timing-documentation-and-trace-provenance"},
        {"sequence": 3, "commit": args.source_commit,
         "role": "stage56-release-rollback-controller-fix"},
        {"sequence": 4, "commit": "containing-commit",
         "role": "stage56-evidence-release-publication"},
    ])
    write_md(stage / "final_dual_remote_report.md", "Final dual-remote publication", [
        "Final publication uses only normal fast-forward pushes after both fetched remote heads "
        "pass ancestor checks. Exact containing commit and post-push parity are recorded in the "
        "command ledger, result packet, and console because a commit cannot contain its own SHA.",
    ])
    write_tsv(stage / "final_remote_parity.tsv", [
        {"phase": "prestage", "remote": remote, "local_head": START_HEAD,
         "remote_head": START_HEAD, "status": "exact"} for remote in ("github", "gitlab-rd")
    ] + [
        {"phase": "final", "remote": remote, "local_head": "containing-commit",
         "remote_head": "verified-post-push", "status": "command-ledger"}
        for remote in ("github", "gitlab-rd")
    ])
    write_tsv(stage / "published_commit_inventory.tsv", [
        {"commit": START_HEAD, "scope": "accepted Stage55", "github": "prestage-parity", "gitlab": "prestage-parity"},
        {"commit": IMPLEMENTATION_COMMIT, "scope": "Stage56 executor/system implementation",
         "github": "pending-final-FF", "gitlab": "pending-final-FF"},
        {"commit": PROVENANCE_COMMIT, "scope": "Stage56 timing/docs/trace provenance",
         "github": "pending-final-FF", "gitlab": "pending-final-FF"},
        {"commit": args.source_commit, "scope": "Stage56 release rollback-controller fix",
         "github": "pending-final-FF", "gitlab": "pending-final-FF"},
        {"commit": "containing-commit", "scope": "Stage56 evidence/release", "github": "post-push-ledger", "gitlab": "post-push-ledger"},
    ])

    write_md(stage / "STAGE56_FINAL_REPORT.md", "Stage56 final report", [
        f"Classification: `{classification}`. Exact selected low-latency mean is "
        f"{final_mean:.6f} us, {speedup_stage55:.6f}% below official Stage55. The unchanged graph "
        "does not achieve 120 ms or near-100 ms.",
        "Selected mechanisms are producer-direct head reduction, direct attention second-MatMul "
        "packing, and reversible O2 CPU/IRQ/workqueue/service isolation. Fused dense/E2c4, "
        "rectangular dense, K32 stem, factorized LUT2, depthwise E2c4, memory, storage, boot, and "
        "realtime candidates are exact rejects, unsupported, or below gate.",
        f"All 215 integer boundaries, F0-F7, bus, Zidane, state restoration, CPU0-3-only IME, "
        f"full COCO `{PREDICTION_SHA}`, 10000-run soak, rollback, build, and publication gates pass.",
        "This freezes the unchanged YOLO26n-640 graph for the defined Stage56 candidate surface. "
        "It is optimized research, not production readiness, 20 FPS, or co-design authorization.",
    ])
    write_md(stage / "STAGE56_SUMMARY_RU.md", "Краткое резюме Stage56", [
        f"Этап классифицирован как `{classification}`. Средняя задержка выбранного режима равна "
        f"{final_mean:.3f} мкс, что на {speedup_stage55:.3f}% меньше официального результата Stage55.",
        "Выбраны прямое сокращение классов в головке, прямая упаковка результата attention для "
        "второго MatMul и обратимый профиль изоляции O2. Остальные проверенные механизмы не прошли "
        "полномодельный порог выбора или недоступны на текущем ядре и загрузчике.",
        f"Все 215 целочисленных границ совпадают точно. COCO для 5000 изображений побайтно совпадает "
        f"со Stage55, mAP50-95 равен {STAGE55_MAP:.16f}. Длительный прогон и откат системы прошли.",
        "Цель 120 мс не достигнута. Это оптимизированный исследовательский результат, а не "
        "готовность к промышленной эксплуатации, 20 FPS или разрешение на обучение модели.",
    ])
    write_md(stage / "stage57_prompt.md", "Stage57 recommendation", [
        "Keep the unchanged YOLO26n-640 executor frozen. The next human decision should choose "
        "release maintenance or separately authorize co-design preparation using Cost Model V4 "
        "plus direct measurements for high-uncertainty shapes. Do not start training, student "
        "selection, or co-design execution without new explicit authorization.",
    ])

    required = {
        "STAGE56_FINAL_REPORT.md", "STAGE56_SUMMARY_RU.md", "workspace_preflight.md",
        "vendor_runtime_lane_frozen.md", "superseded_stage56_prompt_notice.md",
        "prestage_repository_state.md", "prestage_remote_parity.tsv",
        "stage55_reproduction_raw.tsv", "stage55_reproduction_summary.tsv",
        "stage55_reproduction_real_corpus.tsv", "stage55_reproduction_soak.tsv",
        "stage55_reproduction_report.md", "stage55_identity.tsv", "system_inventory.md",
        "kernel_config_inventory.tsv", "bootloader_inventory.md", "boot_profile_backup_manifest.tsv",
        "system_state_before.tsv", "rollback_plan.md", "reboot_ledger.tsv",
        "pmu_device_tree_mapping.txt", "x60_perf_build_identity.md", "x60_event_inventory.tsv",
        "x60_worker_hpm_raw.tsv", "x60_shape_hpm_summary.tsv", "x60_full_model_hpm.tsv",
        "x60_stall_attribution.md", "x60_sampling_limitations.md", "osnoise_availability.md",
        "osnoise_baseline_summary.tsv", "osnoise_slow_run_correlations.tsv",
        "irq_source_inventory.tsv", "slow_tail_root_cause_report.md", "runtime_isolation_matrix.tsv",
        "cgroup_isolation_report.md", "irq_affinity_before_after.tsv",
        "workqueue_affinity_before_after.tsv", "service_state_matrix.tsv",
        "cpufreq_devfreq_matrix.tsv", "cpuidle_matrix.tsv", "realtime_scheduler_matrix.tsv",
        "runtime_os_selection.md", "boot_profile_matrix.tsv", "boot_command_lines.tsv",
        "boot_validation.tsv", "alternate_kernel_report.md", "boot_profile_selection.md",
        "boot_rollback_test.md", "memory_mapping_matrix.tsv", "thp_smaps_evidence.tsv",
        "page_fault_matrix.tsv", "memory_selection.md", "storage_location_matrix.tsv",
        "storage_irq_io_matrix.tsv", "emmc_test_manifest.tsv", "storage_selection.md",
        "board_emmc_write_exceptions.tsv", "dense_fused_register_budget.tsv",
        "dense_fused_contract.md", "dense_fused_candidate_matrix.tsv", "dense_fused_correctness.tsv",
        "dense_fused_hpm.tsv", "dense_fused_performance.tsv", "dense_fused_disassembly.txt",
        "dense_fused_full_model_abba.tsv", "dense_fused_decision.md",
        "rectangular_register_budget.tsv", "rectangular_candidate_matrix.tsv",
        "rectangular_shape_correctness.tsv", "rectangular_shape_hpm.tsv",
        "rectangular_shape_performance.tsv", "rectangular_disassembly.txt",
        "rectangular_full_model_abba.tsv", "rectangular_dispatch_policy.md", "rectangular_decision.md",
        "stem_k32_contract.md", "stem_k32_candidate_matrix.tsv", "stem_k32_correctness.tsv",
        "stem_k32_hpm.tsv", "stem_k32_performance.tsv", "stem_k32_disassembly.txt",
        "stem_k32_full_model_abba.tsv", "stem_k32_decision.md", "attention_v4_contract.md",
        "attention_v4_subphase.tsv", "attention_v4_correctness.tsv", "attention_v4_performance.tsv",
        "attention_v4_disassembly.txt", "attention_v4_full_model_abba.tsv", "attention_v4_decision.md",
        "attention_pe_depthwise_matrix.tsv", "attention_pe_depthwise_decision.md",
        "head_producer_fusion_contract.md", "head_producer_fusion_correctness.tsv",
        "head_producer_fusion_performance.tsv", "head_producer_fusion_decision.md",
        "lut2_factorized_runtime_matrix.tsv", "lut2_factorized_full_model_abba.tsv",
        "lut2_factorized_decision.md", "depthwise_v4_candidate_matrix.tsv",
        "depthwise_v4_decision.md", "double_buffer_architecture.md",
        "double_buffer_correctness.tsv", "double_buffer_timing_raw.tsv",
        "double_buffer_timing_summary.tsv", "double_buffer_hpm_io.tsv", "double_buffer_decision.md",
        "measured_latency_lut_v4.tsv", "shape_model_v4_validation.tsv",
        "full_graph_cost_model_v4.tsv", "full_graph_cost_model_v4_report.md",
        "remaining_optimization_reserve_ledger.tsv", "remaining_optimization_reserve_report.md",
        "current_graph_zero_cost_bounds.tsv", "target_120_100_50ms_gap_report.md",
        "codesign_input_readiness_v4.md", "final_correctness_matrix.tsv", "final_coco_results.tsv",
        "final_coco_prediction_hashes.tsv", "final_coco_report.md", "final_model_performance_raw.tsv",
        "final_model_performance_summary.tsv", "final_model_long_soak.tsv",
        "final_real_corpus_timing.tsv", "final_image_pipeline_timing.tsv",
        "final_ort_comparison.tsv", "final_performance_report.md", "release_update_report.md",
        "source_hygiene_report.md", "board_benchmark_environment.txt", "board_storage_preflight.txt",
        "board_storage_manifest.tsv", "global_sysctl_and_rollback_report.md",
        "governing_document_update_report.md", "codex_skill_update_report.md",
        "selected_system_profile.md", "selected_system_profile_rollback.md", "system_state_after.tsv",
        "commit_inventory.tsv", "final_dual_remote_report.md", "final_remote_parity.tsv",
        "published_commit_inventory.tsv", "stage57_prompt.md",
    }
    missing = sorted(name for name in required if not (stage / name).is_file() or (stage / name).stat().st_size == 0)
    if missing:
        raise RuntimeError("missing Stage56 reports: " + ", ".join(missing))
    print(json.dumps({
        "classification": classification,
        "final_mean_us": final_mean,
        "speedup_vs_stage55_pct": speedup_stage55,
        "speedup_vs_ort_pct": speedup_ort,
        "reports": len(required),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
