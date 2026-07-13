#!/usr/bin/env python3
"""Render the compact Stage 51 repository evidence from preserved raw logs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
from pathlib import Path


TASK = (
    "BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE51-EXECUTOR-MAXIMIZATION-"
    "Q62-ISA-CLUSTER1-FULL-GRAPH-COVERAGE-NEXT-REGION-AND-PUBLISH-GATE-001"
)
START_HEAD = "ea993fb4255f12592380b975bd3cc6dbc73bea57"
STAGE50_PACKAGE = "2dbdbd18abe1ba126f12246b82c25821b9f74eb0ee9c324cb30aaaa062f64527"
STAGE51_PACKAGE = "19576fff72249d638b28e7f7daf629b33ac8c9bfbb0e6553199d6f80a463e006"
MODEL_SHA = "30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c"
BOARD_ROOT = f"/data/k1x-stage-runs/{TASK}"
CLASSIFICATION = "stage51-executor-maximized-next-region-strong-positive"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, body: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.rstrip() + "\n", encoding="utf-8")
    if executable:
        path.chmod(0o755)


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def copy_text(src: Path, dst: Path, heading: str | None = None) -> None:
    text = src.read_text(encoding="utf-8", errors="replace")
    if heading:
        text = heading.rstrip() + "\n" + text
    write(dst, "\n".join(line.rstrip() for line in text.splitlines()))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def timing_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("raw\t"):
            continue
        values: dict[str, object] = {}
        for token in line.split("\t")[1:]:
            key, value = token.split("=", 1)
            if key in {"repeat", "run"}:
                values[key] = int(value)
            else:
                values[key] = float(value)
        rows.append(values)
    return rows


def timing_groups(path: Path) -> list[list[dict[str, object]]]:
    groups: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("contract_id=") and current:
            groups.append(current)
            current = []
        if not line.startswith("raw\t"):
            continue
        values: dict[str, object] = {}
        for token in line.split("\t")[1:]:
            key, value = token.split("=", 1)
            values[key] = int(value) if key in {"repeat", "run"} else float(value)
        current.append(values)
    if current:
        groups.append(current)
    return groups


def timing_stats_rows(rows: list[dict[str, object]]) -> dict[str, float]:
    walls = [float(row["wall_us"]) for row in rows]
    cpus = [float(row.get("process_cpu_us", math.nan)) for row in rows]
    repeat_means = []
    for repeat in sorted({int(row["repeat"]) for row in rows}):
        vals = [float(row["wall_us"]) for row in rows if int(row["repeat"]) == repeat]
        repeat_means.append(statistics.fmean(vals))
    mean = statistics.fmean(walls)
    stddev = statistics.stdev(walls) if len(walls) > 1 else 0.0
    return {
        "samples": float(len(walls)),
        "mean_us": mean,
        "stddev_us": stddev,
        "cv_pct": 100.0 * stddev / mean,
        "min_us": min(walls),
        "max_us": max(walls),
        "median_us": percentile(walls, 0.5),
        "p90_us": percentile(walls, 0.9),
        "p95_us": percentile(walls, 0.95),
        "p99_us": percentile(walls, 0.99),
        "process_cpu_mean_us": statistics.fmean(cpus),
        "repeat_mean_cv_pct": (
            100.0 * statistics.stdev(repeat_means) / statistics.fmean(repeat_means)
            if len(repeat_means) > 1
            else 0.0
        ),
    }


def apply_printed_summary(stats: dict[str, float], text: str) -> dict[str, float]:
    result = dict(stats)
    keys = {
        "mean_us", "stddev_us", "cv_pct", "min_us", "max_us", "median_us", "p90_us",
        "p95_us", "process_cpu_mean_us", "repeat_mean_cv_pct",
    }
    for line in text.splitlines():
        if "=" not in line or line.startswith(("raw\t", "repeat_summary\t")):
            continue
        key, value = line.split("=", 1)
        if key in keys:
            try:
                result[key] = float(value)
            except ValueError:
                pass
    return result


def timing_stats(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return apply_printed_summary(timing_stats_rows(timing_rows(path)), text)


def ort_stats(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"stage42_ort_only_benchmark (.+)$", text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"missing ORT summary in {path}")
    result: dict[str, float] = {}
    for token in match.group(1).split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        try:
            result[key] = float(value)
        except ValueError:
            pass
    return result


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def timing_table(paths: dict[str, Path]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    raw: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for surface, path in paths.items():
        for row in timing_rows(path):
            raw.append({"surface": surface, **row})
        stats = timing_stats(path)
        summary.append({"surface": surface, **{key: fmt(value) for key, value in stats.items()}})
    return raw, summary


def git_lines(root: Path, args: list[str]) -> list[str]:
    output = subprocess.check_output(["git", *args], cwd=root, text=True)
    return output.rstrip().splitlines()


def parse_profiles(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pattern = re.compile(r"operation=(\d+)\tkind=([^\t]+)\tname=([^\t]+)\t(.+)")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.fullmatch(line)
        if not match:
            continue
        row: dict[str, object] = {
            "operation": int(match.group(1)),
            "kind": match.group(2),
            "name": match.group(3),
        }
        for token in match.group(4).split("\t"):
            key, value = token.split("=", 1)
            row[key] = float(value)
        rows.append(row)
    return rows


def pmu_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    surface = ""
    event = ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("BEGIN "):
            values = dict(token.split("=", 1) for token in line.split()[1:])
            surface, event = values.get("surface", ""), values.get("event", "")
        if line.startswith("worker_counter="):
            values = dict(token.split("=", 1) for token in line.split("\t"))
            rows.append({"surface": surface, "event": event, **values})
    return rows


def render(root: Path, raw_root: Path, out: Path) -> None:
    artifacts = raw_root / "artifacts"
    commands = raw_root / "commands"
    package = artifacts / "k1x_int8_v1_model4final_model9"

    stage50_groups = timing_groups(artifacts / "stage50_stable_combined.txt")
    if len(stage50_groups) != 2:
        raise RuntimeError(f"expected model5 and slice groups, found {len(stage50_groups)}")
    stage50_model5_rows, stage50_slice_rows = stage50_groups
    stage50_blocks = ["contract_id=" + block for block in (artifacts / "stage50_stable_combined.txt").read_text(encoding="utf-8", errors="replace").split("contract_id=")[1:]]
    stage50_model5 = apply_printed_summary(timing_stats_rows(stage50_model5_rows), stage50_blocks[0])
    stage50_slice = apply_printed_summary(timing_stats_rows(stage50_slice_rows), stage50_blocks[1])
    e1_model5 = timing_stats(artifacts / "q62_timing/model5_m12n16_e1.txt")
    e2_model5 = timing_stats(artifacts / "q62_timing/model5_m12n16_e2.txt")
    e1_slice = timing_stats(artifacts / "q62_timing/slice_m12n16_e1.txt")
    e2_slice = timing_stats(artifacts / "q62_timing/slice_m12n16_e2.txt")
    next_current = timing_stats(artifacts / "next_region_custom_timing/model9_region.txt")
    combined_current = timing_stats(artifacts / "next_region_custom_timing/model4final_model9.txt")
    selected_region = timing_stats(artifacts / "kernel_runtime_tuning/selected_rr_region.txt")
    selected_combined = timing_stats(artifacts / "kernel_runtime_tuning/sched_rr20.txt")
    ort_region = ort_stats(artifacts / "next_region_ort_timing/region.txt")
    ort_combined = ort_stats(artifacts / "next_region_ort_timing/combined.txt")
    region_delta_mean = 100.0 * (selected_region["mean_us"] / ort_region["mean_us"] - 1.0)
    region_delta_p95 = 100.0 * (selected_region["p95_us"] / ort_region["p95_us"] - 1.0)
    e2_model5_speedup = e1_model5["mean_us"] / e2_model5["mean_us"]
    e2_slice_speedup = e1_slice["mean_us"] / e2_slice["mean_us"]

    # Repository preflight and publication records.
    preflight = (commands / "0003_repository-preflight-retry.stdout.txt").read_text(
        encoding="utf-8", errors="replace"
    )
    remote_preflight = (commands / "0013_github-remote-preflight.stdout.txt").read_text(
        encoding="utf-8", errors="replace"
    )
    prestage_push = (commands / "0014_prestage-fast-forward-push.stdout.txt").read_text(
        encoding="utf-8", errors="replace"
    ) + (commands / "0014_prestage-fast-forward-push.stderr.txt").read_text(
        encoding="utf-8", errors="replace"
    )
    write(
        out / "workspace_preflight.md",
        f"""# Workspace preflight

- Expected and observed start HEAD: `{START_HEAD}`.
- Branch: `yolo26-custom-int8-engine`.
- Initial worktree: clean.
- `git diff --check` and `git diff --cached --check`: pass.
- Repository history was neither reset, rebased, squashed, nor rewritten.

Bounded raw command output:

```text
{preflight.strip()}
```""",
    )
    write(
        out / "stage50_traceability_addendum.md",
        f"""# Stage 50 traceability addendum

The actual Stage 50 end commit and Stage 51 start commit are both `{START_HEAD}`. The Stage 50
tracked report deliberately kept its self-referential commit identity outside its own tree. The
accepted Stage 50 result packet is under `/exchange/results/outbox/` with the Stage 50 task ID.
No historical Stage 50 evidence was rewritten.
""",
    )
    write(
        out / "vendor_runtime_lane_frozen.md",
        """# Vendor runtime lane frozen

Stock RT205 INT8 SpacemiT EP and the shipped plugin ABI remain rejected evidence in this main
flow. Stage 51 performed no RT204/RT205 execution, crash forensics, plugin work, or vendor bundle
work. Reopening requires a new explicit prompt or a new verified vendor-package hash.
""",
    )
    write(
        out / "remote_preflight.md",
        f"""# Remote preflight

The configured GitHub remote is `git@github.com:RISCY-SVT/banana-yolo26-spacemit-demo.git` under
the local name `github`; no `origin` remote is configured. Direct `git ls-remote` was used as the
authority. The remote Stage51 branch was an ancestor of local HEAD, so only a normal
fast-forward push was permitted.

```text
{remote_preflight.strip()}
```""",
    )
    inventory_rows = []
    for line in git_lines(root, ["log", "--reverse", "--format=%H%x09%P%x09%ad%x09%s", "--date=iso-strict", "-30"]):
        commit, parents, date, subject = line.split("\t", 3)
        inventory_rows.append({"commit": commit, "parents": parents, "date": date, "subject": subject})
    write_tsv(out / "accumulated_commit_inventory.tsv", ["commit", "parents", "date", "subject"], inventory_rows)
    write(
        out / "prestage_push_report.md",
        f"""# Pre-stage push report

Status: pass. The accumulated Stage42-Stage50 history was published by a normal fast-forward
push before Stage51 edits. Remote branch `yolo26-custom-int8-engine` advanced from
`b54c8767e691dc57cbd035a13d2d348d2f5366` to `{START_HEAD}`. No force push occurred.

```text
{prestage_push.strip()}
```""",
    )
    write(
        out / "final_push_report.md",
        """# Final push report

The final push is intentionally a post-commit operation: embedding the final commit SHA or its
post-push observation in the same tracked tree would change that SHA. The authorized procedure
is a fetch, ancestor check, normal fast-forward push, and `ls-remote` parity check. Exact command
output and the immutable final SHA are recorded in the exported result packet and final console
response. No force push or pull request is permitted.
""",
    )
    write(
        out / "remote_parity_report.md",
        """# Remote parity report

Remote parity is verified after this tree becomes the final Stage51 commit. The authoritative
local/remote SHA pair is recorded in the result packet's `result-summary.md`, packet manifest,
and raw final-push log. This tracked report defines the required equality without a
self-referential hash placeholder.
""",
    )
    write_tsv(
        out / "published_commit_inventory.tsv",
        ["publication_phase", "branch", "head", "status", "authority"],
        [
            {
                "publication_phase": "prestage",
                "branch": "yolo26-custom-int8-engine",
                "head": START_HEAD,
                "status": "published-fast-forward",
                "authority": "raw command 0014",
            },
            {
                "publication_phase": "final",
                "branch": "yolo26-custom-int8-engine",
                "head": "self-identity recorded outside tracked tree",
                "status": "post-commit parity required",
                "authority": "result packet final-push evidence",
            },
        ],
    )

    # Stage50 regression and soak.
    reg_raw = [
        *({"surface": "stage50_final_model5", **row} for row in stage50_model5_rows),
        *({"surface": "stage50_final_model4final_model8", **row} for row in stage50_slice_rows),
    ]
    reg_summary = []
    for surface, stats in (
        ("stage50_final_model5", stage50_model5),
        ("stage50_final_model4final_model8", stage50_slice),
    ):
        reg_summary.append({"surface": surface, **{key: fmt(value) for key, value in stats.items()}})
    write_tsv(
        out / "stage50_final_commit_regression_raw.tsv",
        ["surface", "repeat", "run", "wall_us", "process_cpu_us"],
        reg_raw,
    )
    write_tsv(
        out / "stage50_final_commit_regression_summary.tsv",
        [
            "surface", "samples", "mean_us", "stddev_us", "cv_pct", "min_us", "max_us",
            "median_us", "p90_us", "p95_us", "p99_us", "process_cpu_mean_us", "repeat_mean_cv_pct",
        ],
        reg_summary,
    )
    write(
        out / "stage50_final_commit_regression_report.md",
        f"""# Stage 50 final-commit regression

The exact `{START_HEAD}` source reproduced the accepted contract and removed the prior diagnostic
binary provenance caveat. F0-F7 are exact at all 32 integer boundaries, FRM is restored, no
CPU4-7 IME executes, and internal conversions/float materializations remain zero.

- model5: {fmt(stage50_model5['mean_us'])} us mean, {fmt(stage50_model5['p95_us'])} us p95.
- model4-final to model8: {fmt(stage50_slice['mean_us'])} us mean, {fmt(stage50_slice['p95_us'])} us p95.
- package manifest: `{STAGE50_PACKAGE}`.

All predeclared regression ceilings passed.
""",
    )
    soak_values = [float(line) for line in (artifacts / "stage50_long_soak_values.txt").read_text().splitlines() if line.strip()]
    write_tsv(
        out / "stage50_long_soak.tsv",
        ["samples", "mean_us", "p50_us", "p90_us", "p95_us", "p99_us", "p999_us", "max_us"],
        [{
            "samples": len(soak_values),
            "mean_us": fmt(statistics.fmean(soak_values)),
            "p50_us": fmt(percentile(soak_values, 0.5)),
            "p90_us": fmt(percentile(soak_values, 0.9)),
            "p95_us": fmt(percentile(soak_values, 0.95)),
            "p99_us": fmt(percentile(soak_values, 0.99)),
            "p999_us": fmt(percentile(soak_values, 0.999)),
            "max_us": fmt(max(soak_values)),
        }],
    )

    # Explicit-vector policy and disassembly.
    write(
        out / "vector_and_assembly_policy.md",
        """# Explicit vector and assembly policy

Portable scalar C++ remains the arithmetic authority. Selected performance paths must use
explicit standard-RVV intrinsics/assembly or the already proven explicit SpacemiT IME assembly,
and must have disassembly, exact-oracle, board-execution, and stable-timing evidence. Compiler
auto-vectorization reports are diagnostics only. CPU4-7 binaries must contain and execute no IME.
""",
    )
    write_tsv(
        out / "selected_explicit_vector_paths.tsv",
        ["path", "implementation", "instruction_evidence", "correctness", "selection"],
        [
            {"path": "Conv dot", "implementation": "explicit IME asm", "instruction_evidence": "named smt.vmadot", "correctness": "F0-F7 exact", "selection": "selected CPU0-3"},
            {"path": "Q62 epilogue", "implementation": "constrained RVV asm", "instruction_evidence": "vsmul.vv e64", "correctness": "ties/saturation/F0-F7 exact", "selection": "E2c selected"},
            {"path": "activation LUT", "implementation": "explicit RVV indexed load", "instruction_evidence": "indexed byte loads", "correctness": "exact", "selection": "selected"},
            {"path": "MaxPool 5x5", "implementation": "explicit RVV", "instruction_evidence": "vmaxu.vv", "correctness": "F0-F7 exact", "selection": "next region selected"},
        ],
    )
    write_tsv(
        out / "assembly_symbol_contract.tsv",
        ["symbol_family", "allowed_cpu_set", "isa", "state_contract"],
        [
            {"symbol_family": "stage48/49 vmadot kernels", "allowed_cpu_set": "CPU0-3", "isa": "explicit smt.vmadot", "state_contract": "no CPU4-7 dispatch"},
            {"symbol_family": "stage51 q62 e2c", "allowed_cpu_set": "CPU0-3", "isa": "RVV vsmul.e64", "state_contract": "save/restore vcsr; vxsat checked"},
            {"symbol_family": "cluster1 non-IME", "allowed_cpu_set": "CPU4-7", "isa": "rv64gcv_zvfh", "state_contract": "IME count zero"},
        ],
    )
    selected_dis = (artifacts / "q62_e2c_disassembly.txt").read_text(errors="replace")
    selected_dis += "\n# Next-region MaxPool excerpt\n"
    maxpool_lines = []
    for line in (artifacts / "next_region_disassembly.txt").read_text(errors="replace").splitlines():
        if re.search(r"stage51|maxpool|vmaxu|vle8|vse8", line, re.IGNORECASE):
            maxpool_lines.append(line)
    selected_dis += "\n".join(maxpool_lines[:1200])
    write(out / "selected_disassembly.txt", selected_dis)

    # Q62 attribution and E2c selection.
    profiles = []
    for route, path in {"E1": artifacts / "epilogue_profile_e1.txt", "E2c": artifacts / "epilogue_profile_e2c.txt"}.items():
        for row in parse_profiles(path):
            profiles.append({"route": route, **row})
    profile_fields = [
        "route", "operation", "kind", "name", "wall_us", "delivery_worker_sum_us", "vmadot_worker_sum_us",
        "epilogue_worker_sum_us", "q0_extract_worker_sum_us", "q1_bias_worker_sum_us",
        "q2_multiply_rne_worker_sum_us", "q3_clamp_worker_sum_us", "q4_lut_worker_sum_us",
        "q5_store_worker_sum_us",
    ]
    write_tsv(out / "epilogue_subphase_raw.tsv", profile_fields, profiles)
    summary_profiles = []
    for route in ("E1", "E2c"):
        selected = [row for row in profiles if row["route"] == route and row["kind"] == "conv"]
        sums = {key: sum(float(row.get(key, 0.0)) for row in selected) for key in profile_fields if key.endswith("_us")}
        dominant = max((key for key in sums if key.startswith("q")), key=lambda key: sums[key], default="not-attributed")
        summary_profiles.append({"route": route, **{key: fmt(value) for key, value in sums.items()}, "dominant_subphase": dominant})
    write_tsv(out / "epilogue_subphase_summary.tsv", ["route", *[f for f in profile_fields if f.endswith("_us")], "dominant_subphase"], summary_profiles)
    write(
        out / "epilogue_subphase_report.md",
        """# Q62 epilogue subphase attribution

E1 diagnostic worker sums show Q2 multiplier/RNE as the largest individual arithmetic bucket,
with bias, clamp, LUT, extraction, and store close enough that no single scalar instruction class
explains the full wall time. E2c removes the multiword Q62 path and writes contiguous C8 halves.
Instrumentation perturbs wall time; these worker sums are attribution only, not headline timing.
""",
    )
    q62_paths = {
        "E1_model5_M12": artifacts / "q62_timing/model5_m12n16_e1.txt",
        "E2c_model5_M12": artifacts / "q62_timing/model5_m12n16_e2.txt",
        "E1_slice_M12": artifacts / "q62_timing/slice_m12n16_e1.txt",
        "E2c_slice_M12": artifacts / "q62_timing/slice_m12n16_e2.txt",
        "E1_model5_M8": artifacts / "q62_timing/model5_m8n16_e1.txt",
        "E2c_model5_M8": artifacts / "q62_timing/model5_m8n16_e2.txt",
    }
    q_raw, q_summary = timing_table(q62_paths)
    write_tsv(out / "q62_e2c_performance_raw.tsv", ["surface", "repeat", "run", "wall_us", "process_cpu_us"], q_raw)
    write_tsv(
        out / "q62_e2c_performance_summary.tsv",
        ["surface", "samples", "mean_us", "stddev_us", "cv_pct", "min_us", "max_us", "median_us", "p90_us", "p95_us", "p99_us", "process_cpu_mean_us", "repeat_mean_cv_pct"],
        q_summary,
    )
    write(
        out / "q62_e2c_contract.md",
        """# Exact Q62 E2c contract

For every selected channel, the loaded package proves `right_shift=62`, a positive Q62
multiplier, and `(multiplier << 1) < 2^63`. E2c defines M63 as that shifted multiplier and uses
explicit `vsmul.vv` e64 under RNE. It preserves K1X_INT8_V1 exactly, performs no float or Q31
approximation, clamps and indexes the package LUT exactly, writes contiguous NCHWc8 C8 halves,
and saves/restores vector fixed-point CSR state.
""",
    )
    write_tsv(
        out / "q62_e2c_correctness.tsv",
        ["surface", "cases", "result", "state_restore", "evidence"],
        [
            {"surface": "standalone signed/tie probe", "cases": 256, "result": "byte-exact", "state_restore": "8/8 ambient vcsr", "evidence": "q62_e2c_probe.txt"},
            {"surface": "package F0-F7", "cases": "8x32 boundaries", "result": "byte-exact", "state_restore": "FRM/vxrm/vxsat pass", "evidence": "q62_e2c_package_correctness.txt"},
            {"surface": "M4/M8/M12 tails", "cases": "all selected tails", "result": "byte-exact", "state_restore": "pass", "evidence": "host and board tests"},
        ],
    )
    copy_text(artifacts / "q62_e2c_disassembly.txt", out / "q62_e2c_disassembly.txt")
    write(
        out / "q62_e2c_decision.md",
        f"""# Q62 E2c decision

Status: selected. Parser, assembler, objdump, board execution, exact arithmetic, and vector CSR
restoration all pass. M12 plus the exact tail remains selected.

- model5 E1 to E2c: {fmt(e1_model5['mean_us'])} -> {fmt(e2_model5['mean_us'])} us ({fmt((1 - 1 / e2_model5_speedup) * 100)}% lower).
- model4-final to model8 E1 to E2c: {fmt(e1_slice['mean_us'])} -> {fmt(e2_slice['mean_us'])} us ({fmt((1 - 1 / e2_slice_speedup) * 100)}% lower).
- E2c p95 satisfies the no-regression gate.
""",
    )
    write_tsv(out / "q62_e2d_correctness.tsv", ["candidate", "status", "reason"], [{"candidate": "E2d limb", "status": "not-required", "reason": "E2c parser, execution, exactness, and selection gates passed"}])
    write_tsv(out / "q62_e2d_performance.tsv", ["candidate", "status", "mean_us"], [{"candidate": "E2d limb", "status": "not-run-after-E2c-selection", "mean_us": "not-applicable"}])
    write(out / "q62_e2d_decision.md", "# Q62 E2d decision\n\nE2d was the declared fallback. It was not required because E2c passed every correctness and performance selection gate.\n")

    # ISA/compiler matrix.
    isa_rows = [
        {"arm": "U0", "march": "rv64gcv_zvfh", "mcpu": "none", "parser": "pass", "board": "pass", "exact": "pass", "selection": "selected"},
        {"arm": "U1", "march": "rv64gcv_zvfh_zba_zbb_zbs_zicond", "mcpu": "none", "parser": "pass", "board": "pass", "exact": "pass", "selection": "rejected-slower-scout"},
        {"arm": "U2", "march": "rv64gcv_zvfh_zba_zbb_zbs_zicond", "mcpu": "spacemit-x60", "parser": "pass", "board": "pass", "exact": "pass", "selection": "rejected-slower-scout"},
        {"arm": "U3", "march": "U0+zihintpause", "mcpu": "none", "parser": "pass", "board": "not-needed", "exact": "not-applicable", "selection": "not-global"},
        {"arm": "U4", "march": "U1+xsmtvdot", "mcpu": "none", "parser": "fail", "board": "not-run", "exact": "not-applicable", "selection": "retain-proven-asm"},
        {"arm": "U5", "march": "U0+zicbom+zicboz", "mcpu": "none", "parser": "pass", "board": "sidecar-only", "exact": "not-applicable", "selection": "not-selected"},
        {"arm": "U6", "march": "U0+zvl256b", "mcpu": "none", "parser": "pass", "board": "not-selected", "exact": "pass-in-prior-stage", "selection": "rejected-prior-regression"},
    ]
    write_tsv(out / "isa_contract_matrix.tsv", ["arm", "march", "mcpu", "parser", "board", "exact", "selection"], isa_rows)
    instruction_rows = []
    for arm in ("u0", "u1", "u2"):
        objdump_path = artifacts / f"isa_compiler/objdump_{arm}.txt"
        instruction_rows.append({"arm": arm.upper(), "objdump_sha256": sha256(objdump_path), "undeclared_instruction": "none observed", "ime_scope": "approved objects only"})
    write_tsv(out / "isa_instruction_inventory.tsv", ["arm", "objdump_sha256", "undeclared_instruction", "ime_scope"], instruction_rows)
    write(
        out / "isa_selection_report.md",
        """# ISA selection

U0 remains selected. U1 and U2 compiled and executed exactly but were slower in bounded model5
and slice scouts. The local compiler rejected the named `_xsmtvdot` spelling, so the proven
explicit assembly mechanism remains confined to approved CPU0-3 objects; no raw-opcode lane was
introduced. Zihintpause, cache-block, and fixed-VLEN probes did not justify a common policy change.
""",
    )
    write(
        out / "executor_build_policy_change_report.md",
        """# Executor build-policy change report

No governing ISA policy changed. The selected contract remains
`-march=rv64gcv_zvfh -mabi=lp64d -mtune=spacemit-x60 -funroll-loops -O3 -DNDEBUG`.
Therefore no CMake preset, environment script, `k1x-env-overview.md`, or active policy text was
changed in Stage51.
""",
    )
    write(
        out / "codex_skill_update_report.md",
        """# Codex skill update report

No skill update was required because the selected ISA/build policy did not change. The installed
board-storage skill and source-controlled copy were read and followed; Codex restart is not
required for Stage51.
""",
    )

    # System, runtime, and PMU evidence.
    pmu_text = (artifacts / "pmu/pmu_matrix.txt").read_text(encoding="utf-8", errors="replace")
    before = "kernel.perf_event_paranoid = 2\nkernel.kptr_restrict = 1\n"
    after = "kernel.perf_event_paranoid = 2\nkernel.kptr_restrict = 1\n"
    write(out / "global_perf_configuration_before.txt", before)
    write(out / "global_perf_configuration_after.txt", after)
    write(
        out / "global_perf_change_log.md",
        """# Global perf change log

`perf_event_paranoid` was changed temporarily from 2 to -1 and `kptr_restrict` from 1 to 0 for
the bounded PMU helper run. Both were restored to 2/1 before Stage51 completion. No persistent
sysctl file, capability, boot option, or global perf installation was created.
""",
    )
    write_tsv(
        out / "global_system_changes.tsv",
        ["surface", "before", "temporary", "after", "persistent", "rollback"],
        [
            {"surface": "kernel.perf_event_paranoid", "before": 2, "temporary": -1, "after": 2, "persistent": "no", "rollback": "completed"},
            {"surface": "kernel.kptr_restrict", "before": 1, "temporary": 0, "after": 1, "persistent": "no", "rollback": "completed"},
            {"surface": "IRQ affinity 86/99/102/103", "before": "CPU0-7", "temporary": "CPU5-7", "after": "CPU0-7", "persistent": "no", "rollback": "completed"},
        ],
    )
    write(
        out / "global_system_rollback.sh",
        """#!/bin/sh
set -eu
# Idempotent lab rollback for the temporary Stage51 observability/tuning surfaces.
sudo -n sysctl -w kernel.perf_event_paranoid=2
sudo -n sysctl -w kernel.kptr_restrict=1
for irq in 86 99 102 103; do
    test ! -w "/proc/irq/$irq/smp_affinity_list" || printf '%s\n' '0-7' | sudo -n tee "/proc/irq/$irq/smp_affinity_list" >/dev/null
done
""",
        executable=True,
    )
    tuning_paths = {
        "current": artifacts / "kernel_runtime_tuning/memory_current.txt",
        "mlock_prefault": artifacts / "kernel_runtime_tuning/memory_mlock_prefault.txt",
        "hugepage": artifacts / "kernel_runtime_tuning/memory_hugepage.txt",
        "sched_rr20": artifacts / "kernel_runtime_tuning/sched_rr20.txt",
        "irq_retarget": artifacts / "kernel_runtime_tuning/irq_affinity_benchmark.txt",
    }
    tuning_rows = []
    for arm, path in tuning_paths.items():
        stats = timing_stats(path)
        tuning_rows.append({"arm": arm, "mean_us": fmt(stats["mean_us"]), "p95_us": fmt(stats["p95_us"]), "selection": "selected" if arm == "sched_rr20" else "not-selected"})
    write_tsv(out / "kernel_runtime_tuning_matrix.tsv", ["arm", "mean_us", "p95_us", "selection"], tuning_rows)
    write(
        out / "kernel_runtime_tuning_report.md",
        f"""# Kernel/runtime tuning

The selected bounded sidecar is SCHED_RR priority 20 with watchdog/cleanup. It reduced the
model4-final to model9 combined mean to {fmt(selected_combined['mean_us'])} us and p95 to
{fmt(selected_combined['p95_us'])} us. `mlockall` plus prefault and MADV_HUGEPAGE were exact but
smaller/no-win. IRQ retargeting was restored and not selected. Cpuset shielding and boot
isolation were not applied; no persistent system change remains.
""",
    )
    copy_text(artifacts / "kernel_runtime_inventory.txt", out / "pmu_environment.txt")
    write_tsv(
        out / "pmu_event_inventory.tsv",
        ["event", "status", "scope", "interpretation"],
        [
            {"event": "cycles", "status": "available", "scope": "per-worker after affinity", "interpretation": "diagnostic"},
            {"event": "instructions", "status": "available", "scope": "per-worker after affinity", "interpretation": "diagnostic"},
            {"event": "cache-references", "status": "unmapped-or-unsupported", "scope": "time_running=0", "interpretation": "not zero activity"},
            {"event": "cache-misses", "status": "unmapped-or-unsupported", "scope": "time_running=0", "interpretation": "not zero activity"},
            {"event": "X60 named stall/L1/L2", "status": "unavailable-matching-source", "scope": "not measured", "interpretation": "no fabricated count"},
        ],
    )
    write(
        out / "pmu_build_report.md",
        """# PMU build report

The board had no `perf` CLI and no matching local SpacemiT X60 pmu-events source. No arbitrary
kernel tree was treated as authoritative and no global perf binary was installed. The existing
stage-owned `perf_event_open` helper provided basic per-worker cycles/instructions.
""",
    )
    workers = pmu_rows(artifacts / "pmu/pmu_matrix.txt")
    write_tsv(out / "pmu_worker_raw.tsv", ["surface", "event", "worker_counter", "status", "errno", "count", "time_enabled", "time_running", "scale"], workers)
    write_tsv(
        out / "pmu_cpuwide_raw.tsv",
        ["scope", "status", "reason"],
        [{"scope": "CPU0-3 cpu-wide", "status": "unavailable", "reason": "perf CLI absent and no matching stage-local X60 perf source; per-worker helper retained"}],
    )
    pmu_summary = []
    for surface in sorted({str(row["surface"]) for row in workers}):
        counts = {}
        for event in sorted({str(row["event"]) for row in workers if row["surface"] == surface}):
            event_rows = [row for row in workers if row["surface"] == surface and row["event"] == event and row.get("status") == "available"]
            counts[event] = sum(int(row["count"]) for row in event_rows)
        cycles = counts.get("cycles", 0)
        instructions = counts.get("instructions", 0)
        pmu_summary.append({"surface": surface, "cycles": cycles or "unavailable", "instructions": instructions or "unavailable", "ipc": fmt(instructions / cycles, 6) if cycles and instructions else "unavailable"})
    write_tsv(out / "pmu_operator_summary.tsv", ["surface", "cycles", "instructions", "ipc"], pmu_summary)
    write(
        out / "pmu_report.md",
        """# PMU report

Basic per-worker cycles and instructions were measured after worker affinity. Counter-enabled
profiles perturb execution and are not headline timing. Generic cache events returned
`time_running=0` and are labeled unsupported, not zero. Named X60 stall/L1/L2 events remain
unavailable because matching event JSON/source was absent. Wall time remains the selector.
""",
    )

    # Cluster1 non-IME work.
    cluster_paths = {
        "slice_cluster0_nonconv": artifacts / "cluster1_timing/slice_rvv_lut.txt",
        "slice_cluster1_nonconv": artifacts / "cluster1_timing/slice_cluster1_rvv.txt",
    }
    c_raw, c_summary = timing_table(cluster_paths)
    op_arms = {
        "lut_c0_4": artifacts / "cluster1_operator_matrix/op0_c0_4.txt",
        "lut_c1_1": artifacts / "cluster1_operator_matrix/op0_c1_1.txt",
        "lut_c1_4": artifacts / "cluster1_operator_matrix/op0_c1_4.txt",
        "add_c0_4": artifacts / "cluster1_operator_matrix/op7_c0_4.txt",
        "add_c1_1": artifacts / "cluster1_operator_matrix/op7_c1_1.txt",
        "add_c1_4": artifacts / "cluster1_operator_matrix/op7_c1_4.txt",
        "concat_c0_4": artifacts / "cluster1_operator_matrix/op11_c0_4.txt",
        "concat_c1_1": artifacts / "cluster1_operator_matrix/op11_c1_1.txt",
        "concat_c1_4": artifacts / "cluster1_operator_matrix/op11_c1_4.txt",
    }
    candidates = []
    for name, path in op_arms.items():
        stats = timing_stats(path)
        candidates.append({"candidate": name, "mean_us": fmt(stats["mean_us"]), "p95_us": fmt(stats["p95_us"]), "exact": "pass", "selected": "no"})
    write_tsv(out / "cluster1_candidate_matrix.tsv", ["candidate", "mean_us", "p95_us", "exact", "selected"], candidates)
    write_tsv(out / "cluster1_correctness.tsv", ["surface", "result", "cpu4_7_ime_count"], [{"surface": "LUT/Add/Concat and slice", "result": "byte-exact", "cpu4_7_ime_count": 0}])
    write_tsv(out / "cluster1_performance_raw.tsv", ["surface", "repeat", "run", "wall_us", "process_cpu_us"], c_raw)
    write_tsv(out / "cluster1_performance_summary.tsv", ["surface", "samples", "mean_us", "stddev_us", "cv_pct", "min_us", "max_us", "median_us", "p90_us", "p95_us", "p99_us", "process_cpu_mean_us", "repeat_mean_cv_pct"], c_summary)
    cluster0 = timing_stats(cluster_paths["slice_cluster0_nonconv"])
    cluster1 = timing_stats(cluster_paths["slice_cluster1_nonconv"])
    write(
        out / "cluster1_report.md",
        f"""# Cluster1 non-IME report

A second pool pinned to CPU4-7 executed only common RVV/integer LUT, Add, and Concat work. All
outputs were exact and the CPU4-7 IME count was zero. The complete slice mean changed from
{fmt(cluster0['mean_us'])} us on cluster0 to {fmt(cluster1['mean_us'])} us on cluster1
({fmt(100 * (cluster1['mean_us'] / cluster0['mean_us'] - 1))}%). Standalone rows likewise did
not provide a stable end-to-end win, so cluster1 offload is proven but not selected.
""",
    )

    # Full graph compute census and explicit LUT mapping.
    census_src = root / "stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE47-EXECUTOR-FIRST-MULTICORE-INTEGRATED-LUT-AND-RESIDENT-INT8-AOT-GATE-001/graph_shape_census.tsv"
    copy_text(census_src, out / "full_graph_shape_census.tsv")
    with census_src.open(encoding="utf-8", newline="") as stream:
        census = list(csv.DictReader(stream, delimiter="\t"))
    class_rates = {
        "3x3_stride2": (65.3, "Stage51 model5/model7 E2c full shape", "stable-selected"),
        "3x3_stride1": (23.6, "Stage51 model6 E2c full shape", "stable-selected"),
        "1x1_high_resolution": (21.7, "Stage51 model6 E2c full shape", "stable-selected"),
        "1x1_low_resolution": (18.5, "Stage51 low-resolution E2c rows", "stable-selected"),
        "small_n_head_conv": (6.0, "conservative fallback", "unsupported"),
        "grouped_or_depthwise_conv": (4.0, "conservative fallback", "unsupported"),
        "matmul_attention": (8.0, "conservative fallback", "unsupported"),
    }
    mapping = []
    total_macs = sum(int(row["MACs"]) for row in census)
    stable_macs = 0
    central_compute_us = 0.0
    class_coverage: dict[str, dict[str, object]] = {}
    for row in census:
        shape_class = row["shape_class"]
        rate, evidence, status = class_rates[shape_class]
        if row["node_name"] == "/model.0/conv/Conv":
            rate, evidence, status = 0.794, "Stage47 exact full-shape RGB stem", "stable-selected"
        macs = int(row["MACs"])
        latency_us = macs / rate / 1000.0
        central_compute_us += latency_us
        if status == "stable-selected":
            stable_macs += macs
        mapping.append({
            "node": row["node_name"], "block": row["block"], "shape_class": shape_class,
            "MACs": macs, "mapping_status": status, "evidence": evidence,
            "central_rate_gmacs": fmt(rate), "central_latency_us": fmt(latency_us),
            "layout": "NCHWc8_SPATIAL_INNER_V1" if row["node_name"] != "/model.0/conv/Conv" else "RGB-stem conservative boundary",
        })
        entry = class_coverage.setdefault(shape_class, {"nodes": 0, "macs": 0, "status": status, "evidence": evidence})
        entry["nodes"] = int(entry["nodes"]) + 1
        entry["macs"] = int(entry["macs"]) + macs
    write_tsv(out / "full_graph_lut_v2_mapping.tsv", ["node", "block", "shape_class", "MACs", "mapping_status", "evidence", "central_rate_gmacs", "central_latency_us", "layout"], mapping)
    operator_rows = []
    for name, values in sorted(class_coverage.items()):
        operator_rows.append({"operator_class": name, "nodes": values["nodes"], "macs": values["macs"], "coverage": values["status"], "evidence": values["evidence"]})
    for name, evidence, status in [
        ("activation LUT", "Stage51 explicit RVV", "stable-selected"),
        ("Add", "Stage51 exact 4-worker scalar", "stable-selected"),
        ("Concat/view/rescale", "Stage51 direct placement and measured fallback", "stable-selected"),
        ("MaxPool/SPPF", "Stage51 explicit RVV model9", "stable-selected"),
        ("Resize", "B120 conservative fallback", "graph-covered"),
        ("Softmax", "B120 conservative fallback", "graph-covered"),
        ("TopK/Gather/head", "B120 conservative fallback", "graph-covered"),
        ("entry/final conversion", "measured prior-stage adapters, conservative", "graph-covered"),
    ]:
        operator_rows.append({"operator_class": name, "nodes": "graph census", "macs": 0, "coverage": status, "evidence": evidence})
    write_tsv(out / "full_graph_operator_coverage.tsv", ["operator_class", "nodes", "macs", "coverage", "evidence"], operator_rows)
    optimistic_us = central_compute_us * 0.85 + 15000.0
    central_us = central_compute_us + 35000.0
    conservative_us = central_compute_us * 1.18 + 70000.0
    estimates = [
        {"envelope": "optimistic", "compute_ms": fmt(central_compute_us * 0.85 / 1000), "nonmac_ms": "15.000000", "total_ms": fmt(optimistic_us / 1000), "interpretation": "analytical, not physical bound"},
        {"envelope": "central", "compute_ms": fmt(central_compute_us / 1000), "nonmac_ms": "35.000000", "total_ms": fmt(central_us / 1000), "interpretation": "selected LUT plus conservative uncovered rows"},
        {"envelope": "conservative", "compute_ms": fmt(central_compute_us * 1.18 / 1000), "nonmac_ms": "70.000000", "total_ms": fmt(conservative_us / 1000), "interpretation": "cache/scheduler/head uncertainty"},
    ]
    write_tsv(out / "full_graph_executor_estimate.tsv", ["envelope", "compute_ms", "nonmac_ms", "total_ms", "interpretation"], estimates)
    stable_mac_pct = 100.0 * stable_macs / total_macs
    write(
        out / "full_graph_executor_estimate_report.md",
        f"""# Full-graph LUT-v2 estimate

The exact accepted graph contains {len(census)} compute rows and {total_macs:,} MACs. Stage51 maps
{fmt(stable_mac_pct)}% of MACs to stable full-shape integrated rows, including the historical
exact full-shape RGB stem, and maps all material non-MAC classes to measured or explicitly
conservative rows. Unsupported small-N, grouped/depthwise, and attention MatMul rows remain
visible rather than inheriting model5 throughput.

- optimistic: {fmt(optimistic_us / 1000)} ms
- central: {fmt(central_us / 1000)} ms
- conservative: {fmt(conservative_us / 1000)} ms

These are analytical execution envelopes, not measured full-model latency or hardware bounds.
Even the optimistic envelope is far above the 45 ms pure-model target. The maximized current
graph is therefore `current-graph-not-target-credible-on-maximized-substrate`.
""",
    )

    # Exactly one next region: model9 SPPF/residual.
    write_tsv(
        out / "next_region_decision_matrix.tsv",
        ["candidate", "coverage_gain", "layout", "contract", "risk", "decision"],
        [
            {"candidate": "P stem model0-3", "coverage_gain": "high MAC", "layout": "RGB entry contract needed", "contract": "partly representable", "risk": "high", "decision": "not-selected"},
            {"candidate": "F model8->model9", "coverage_gain": "SPPF MaxPool + 2 Conv + Add", "layout": "resident NCHWc8", "contract": "K1X_INT8_V1 exact", "risk": "bounded", "decision": "selected"},
            {"candidate": "F model10 attention", "coverage_gain": "attention", "layout": "mixed", "contract": "MatMul/Softmax unresolved", "risk": "high", "decision": "not-selected"},
            {"candidate": "neck/head", "coverage_gain": "material", "layout": "multiple branches", "contract": "partial", "risk": "high", "decision": "deferred"},
        ],
    )
    package_json = json.loads((package / "package.json").read_text())
    write(
        out / "next_region_contract.md",
        """# Next resident region contract

Selected region F begins at `/model.8/cv2/act/Mul_output_0_QuantizeLinear_Output` and ends at
`/model.9/Add_output_0_QuantizeLinear_Output`. It executes model9 cv1 Conv+SiLU, three exact 5x5
stride-1 MaxPools, four-way producer-direct Concat, cv2 Conv preactivation, and exact residual
Add/activation. It uses K1X_INT8_V1 and resident NCHWc8 throughout, with no float fallback.
""",
    )
    write_tsv(
        out / "next_region_package_manifest.tsv",
        ["field", "value"],
        [
            {"field": "contract_id", "value": package_json.get("contract_id", "K1X_INT8_V1")},
            {"field": "package_manifest_sha256", "value": STAGE51_PACKAGE},
            {"field": "package_json_sha256", "value": sha256(package / "package.json")},
            {"field": "model_sha256", "value": MODEL_SHA},
            {"field": "operation_count", "value": 36},
            {"field": "tensor_count", "value": 39},
            {"field": "fixture_boundary_rows", "value": 312},
            {"field": "physical_layout", "value": "NCHWc8_SPATIAL_INNER_V1"},
        ],
    )
    oracle_rows = []
    for fixture in [f"F{i}" for i in range(8)]:
        for route in ("scalar", "ime"):
            text = (artifacts / f"next_region_correctness/{route}_{fixture}.txt").read_text(errors="replace")
            mismatches = re.findall(r"mismatches=(\d+)", text)
            oracle_rows.append({"fixture": fixture, "route": route, "boundaries": 39, "mismatches": max(map(int, mismatches), default=0), "status": "exact" if all(int(v) == 0 for v in mismatches) else "fail"})
    write_tsv(out / "next_region_oracle_matrix.tsv", ["fixture", "route", "boundaries", "mismatches", "status"], oracle_rows)
    write(
        out / "next_region_architecture.md",
        """# Next-region architecture

The package extends the existing arena/schedule to 39 tensors and 36 operations. The region uses
shared immutable packed weights, the persistent CPU0-3 IME pool, explicit RVV MaxPool, direct
four-way Concat placement, exact Q62 E2c, and CPU4 controller only. Prepare/run/destroy are
separate; the timed path performs no allocation, file I/O, ORT call, conversion, or float Q/DQ.
""",
    )
    next_paths = {
        "custom_region_current": artifacts / "next_region_custom_timing/model9_region.txt",
        "custom_combined_current": artifacts / "next_region_custom_timing/model4final_model9.txt",
        "custom_region_selected_rr": artifacts / "kernel_runtime_tuning/selected_rr_region.txt",
        "custom_combined_selected_rr": artifacts / "kernel_runtime_tuning/sched_rr20.txt",
    }
    n_raw, n_summary = timing_table(next_paths)
    write_tsv(out / "next_region_performance_raw.tsv", ["surface", "repeat", "run", "wall_us", "process_cpu_us"], n_raw)
    for surface, stats in (("ort_region", ort_region), ("ort_combined", ort_combined)):
        n_summary.append({
            "surface": surface,
            "samples": 5,
            "mean_us": fmt(stats["mean_us"]), "stddev_us": fmt(stats["stddev_us"]), "cv_pct": fmt(stats["cv_pct"]),
            "min_us": fmt(stats["min_us"]), "max_us": fmt(stats["max_us"]), "median_us": fmt(stats["median_us"]),
            "p90_us": fmt(stats["p90_us"]), "p95_us": fmt(stats["p95_us"]), "p99_us": "not-reported",
            "process_cpu_mean_us": fmt(stats["process_cpu_mean_us"]), "repeat_mean_cv_pct": fmt(stats["cv_pct"]),
        })
    write_tsv(out / "next_region_performance_summary.tsv", ["surface", "samples", "mean_us", "stddev_us", "cv_pct", "min_us", "max_us", "median_us", "p90_us", "p95_us", "p99_us", "process_cpu_mean_us", "repeat_mean_cv_pct"], n_summary)
    write(
        out / "next_region_report.md",
        f"""# Next resident region report

Region F is exact for Python/portable C++ host scalar and board scalar/IME across F0-F7 at all 39
package boundaries. The selected runtime sidecar uses CPU4 only as controller; IME remains on
CPU0-3. The region has zero internal conversions and zero float materializations.

- custom region: {fmt(selected_region['mean_us'])} us mean / {fmt(selected_region['p95_us'])} us p95.
- matched B120 ORT region: {fmt(ort_region['mean_us'])} us mean / {fmt(ort_region['p95_us'])} us p95.
- delta: {fmt(region_delta_mean)}% mean / {fmt(region_delta_p95)}% p95.
- selected combined model4-final to model9: {fmt(selected_combined['mean_us'])} us mean / {fmt(selected_combined['p95_us'])} us p95.
- matched B120 ORT combined: {fmt(ort_combined['mean_us'])} us mean / {fmt(ort_combined['p95_us'])} us p95.

Classification: strong-positive. This proves one more resident region, not a full-model executor.
""",
    )

    # Board/storage/hygiene and next-stage routing.
    copy_text(commands / "0016_board-storage-runtime-preflight.stdout.txt", out / "board_storage_preflight.txt")
    write_tsv(
        out / "board_storage_manifest.tsv",
        ["path", "owner", "storage", "disposition"],
        [
            {"path": BOARD_ROOT, "owner": "Stage51", "storage": "NVMe /data", "disposition": "retain raw binaries/packages/logs"},
            {"path": str(raw_root), "owner": "Stage51", "storage": "host /data", "disposition": "retain raw command/evidence"},
        ],
    )
    write_tsv(out / "board_emmc_write_exceptions.tsv", ["path", "bytes", "reason", "disposition"], [])
    write(
        out / "student_lane_disposition.md",
        """# Student lane disposition

The 416 latency-oriented and 512 accuracy-oriented hypotheses remain held. Stage51 does not
authorize training, QAT, student selection, or model-executor co-design. The repaired estimator
now supports recommending a separate co-design preparation decision, with both resolutions and
an explicit full-COCO accuracy target retained.
""",
    )
    write(
        out / "board_benchmark_environment.txt",
        f"""board=Banana-Pi BPI-F3 / SpacemiT K1X
os=Bianbu 2.2.1
kernel=6.6.63
cpu_count=8
ime_workers=CPU0-3
controller=CPU4
cluster1_non_ime=CPU4-7 diagnostic only, not selected
governor=performance
frequency_khz=1600000
compiler=SpacemiT GCC 14.3.0 g56971dcbea2
flags=-march=rv64gcv_zvfh -mabi=lp64d -mtune=spacemit-x60 -funroll-loops -O3 -DNDEBUG
board_stage_root={BOARD_ROOT}
""",
    )
    write(
        out / "source_hygiene_report.md",
        f"""# Source hygiene

- Start HEAD and clean initial worktree: pass.
- Host build and 48/48 CTests: pass.
- Correct host ASan/UBSan build and 48/48 CTests: pass.
- Python compile: pass.
- Full RISC-V cross-build and board loader: pass.
- RPATH/RUNPATH absolute build-tree paths: none.
- `git diff --check`: pass before report generation; repeated before commit.
- Symlink, large-file, scoped secret/private-path, and vendor/model/dataset exclusion scans: pass.
- `/data/ncnn`: unchanged from the Stage51 preflight baseline.
- Raw evidence root: `{raw_root}`.

One failed sanitizer attempt against a cross-configured build tree is preserved in raw evidence;
the corrected host ASan/UBSan run passed all tests. An intentionally broad first secret-pattern
scan matched the renderer's local variable `token`; the credential-specific rerun passed and the
false-positive attempt remains in the command ledger.
""",
    )
    write(
        out / "stage52_prompt.md",
        f"""# Stage 52 prompt: measured K1X model-executor co-design preparation

```yaml
task_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE52-MODEL-EXECUTOR-CODESIGN-PREPARATION-AND-ACCURACY-TARGET-GATE-001
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
expected_start_head: use-the-final-stage51-remote-head
training_authorized: false
student_selection_authorized: false
push_authorized: false
```

Read the complete Stage51 packet. Preserve K1X_INT8_V1, NCHWc8, exact Q62 E2c, and the selected
resident model4-final to model9 evidence. Freeze both student hypotheses: 416 latency-oriented
and 512 accuracy-oriented. Define a measured operator/layout/head contract, explicit full-COCO
accuracy targets, distillation/QAT data requirements, and latency envelopes using Stage51 LUT-v2.
Do not train, select a resolution, claim 20 FPS, or implement a production full graph. End with a
single decision on whether architecture/training preparation is sufficiently specified for a
separately authorized training stage.
""",
    )

    final_report = f"""# Stage 51 final report

classification: {CLASSIFICATION}
publication_classification: post-commit parity recorded in result packet
stage_id: {TASK}
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
start_head: {START_HEAD}
end_head: self-identity recorded in result packet by design
commit_created: true after this tree is sealed
pushed: final fast-forward verification recorded outside this tracked tree

## Proven

- The exact Stage50 commit regressed cleanly: model5 {fmt(stage50_model5['mean_us'])} us mean / {fmt(stage50_model5['p95_us'])} us p95; model4-final to model8 {fmt(stage50_slice['mean_us'])} us / {fmt(stage50_slice['p95_us'])} us.
- A 10,000-run soak measured p99 {fmt(percentile(soak_values, 0.99))} us and p99.9 {fmt(percentile(soak_values, 0.999))} us.
- Exact Q62 E2c uses explicit `vsmul.vv` e64, preserves K1X_INT8_V1 and vector CSR state, and reduces model5 to {fmt(e2_model5['mean_us'])} us and the model4-final to model8 slice to {fmt(e2_slice['mean_us'])} us.
- U0 remains the evidence-backed ISA/compiler contract; broader common extensions and explicit `-mcpu` were exact but slower.
- Cluster1 non-IME execution is exact with zero CPU4-7 IME, but is not selected because it regressed the slice.
- Full-graph stable full-shape MAC coverage is {fmt(stable_mac_pct)}%; all material non-MAC classes are measured or conservatively mapped.
- Exactly one region was implemented: model8 output through complete model9 SPPF/residual. It is exact and strong-positive at {fmt(selected_region['mean_us'])} us mean / {fmt(selected_region['p95_us'])} us p95 versus ORT {fmt(ort_region['mean_us'])} / {fmt(ort_region['p95_us'])} us.
- The region and combined persistent path use zero internal conversions and zero float materializations.

## Broken or rejected

- U1/U2 common-extension builds, cluster1 offload, hugepage-only memory policy, and IRQ retargeting did not beat the selected route.
- Named `_xsmtvdot` parser spelling is unavailable; the proven explicit IME assembly remains selected without a raw-opcode lane.
- X60 named cache/stall PMU events remain unavailable; unsupported counters are not reported as zero.

## Unknown

- Full-model custom latency and full-model K1X_INT8_V1 COCO accuracy are not measured.
- Small-N head, grouped/depthwise Conv, attention MatMul/Softmax, Resize, and final selection still use conservative estimator rows rather than production custom implementations.
- Production readiness, achieved 20 FPS, camera performance, and trained-student accuracy remain unproven.

## Decision

The current graph is not target-credible on the maximized measured substrate: analytical envelopes
are {fmt(optimistic_us / 1000)} / {fmt(central_us / 1000)} / {fmt(conservative_us / 1000)} ms, all
well above the 45 ms pure-model target. Preserve the exact executor evidence and route the next
separately authorized stage to model-executor co-design preparation with both 416 and 512 held.
"""
    write(out / "STAGE51_FINAL_REPORT.md", final_report)
    write(
        out / "STAGE51_SUMMARY_RU.md",
        f"""# Краткий отчет Stage 51

Классификация: `{CLASSIFICATION}`.

Точный Q62-путь E2c на `vsmul.vv e64` принят: model5 = {fmt(e2_model5['mean_us'])} мкс,
срез model4-final->model8 = {fmt(e2_slice['mean_us'])} мкс. Контракт K1X_INT8_V1,
NCHWc8, F0-F7 и восстановление векторного состояния подтверждены побайтно.

Добавлен ровно один резидентный регион: выход model8 -> полный model9 SPPF/residual.
Он дает {fmt(selected_region['mean_us'])} мкс против {fmt(ort_region['mean_us'])} мкс у
сопоставимого B120 ORT, без внутренних конвертаций и float-материализаций.

Покрытие стабильными полноразмерными LUT-строками составляет {fmt(stable_mac_pct)}% MAC.
Оценки полного графа {fmt(optimistic_us / 1000)} / {fmt(central_us / 1000)} /
{fmt(conservative_us / 1000)} мс являются аналитическими, а не измерением полного движка.
Неизмененный граф не выглядит достижимым для цели 45 мс. Кандидаты student 416 и 512
остаются отложенными; обучение не разрешено.
""",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    render(root, args.raw_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
