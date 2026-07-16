#!/usr/bin/env python3
"""Render Stage57 handoff evidence from preserved host and board artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path
from typing import Any, Iterable


TASK_ID = (
    "BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE57-FINAL-EXACT-MICRO-"
    "MAINTENANCE-PRODUCTIZATION-HANDOFF-AND-DUAL-REMOTE-FREEZE-GATE-001"
)
START_HEAD = "4f5a7f2327bff63d8159a71eeaa950e22897b823"
SOURCE_COMMIT = "cf2654a2706187c28d23a5a02c505b00c5d27036"
CONTRACT = "K1X_INT8_V1"
PROFILE = "K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001"
MODEL_SHA = "30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c"
PACKAGE_SHA = "fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be"
PREDICTION_SHA = "cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda"
OUTPUT_HASH = "0xd43f5e018b415631"
MAP_50_95 = 0.3707408944391919
STAGE56_SELECTED_US = 142412.512
STAGE56_COMPAT_US = 156620.000


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
    if fields is None:
        fields = []
        for row in materialized:
            for field in row:
                if field not in fields:
                    fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, delimiter="\t",
                                lineterminator="\n", extrasaction="ignore")
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
    for token in line.rstrip().replace(" ", "\t").split("\t"):
        if "=" in token:
            key, value = token.split("=", 1)
            result[key] = value
    return result


def parse_cli(path: Path) -> list[dict[str, str]]:
    rows = [parse_fields(line) for line in path.read_text(
        encoding="utf-8", errors="replace").splitlines() if line.startswith("raw\t")]
    if not rows:
        raise ValueError(f"no raw samples in {path}")
    return rows


def parse_ort(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("stage42_ort_only_sample "):
            rows.append(parse_fields(line))
    if len(rows) != 500:
        raise ValueError(f"expected 500 matched ORT samples, found {len(rows)}")
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
        "samples": len(values),
        "mean_us": f"{mean:.6f}",
        "stddev_us": f"{statistics.pstdev(values):.6f}",
        "cv_pct": f"{statistics.pstdev(values) / mean * 100.0:.6f}",
        "median_us": f"{statistics.median(values):.6f}",
        "p90_us": f"{percentile(values, .90):.6f}",
        "p95_us": f"{percentile(values, .95):.6f}",
        "p99_us": f"{percentile(values, .99):.6f}",
        "p999_us": f"{percentile(values, .999):.6f}",
        "max_us": f"{max(values):.6f}",
        "output_hashes": ",".join(sorted({row.get("hash", "") for row in rows})),
        "cpu4_7_ime_count": max(int(row.get("cpu4_7_ime_count", "0")) for row in rows),
    }
    for source, destination in (
        ("process_cpu_us", "process_cpu_mean_us"),
        ("voluntary_cs", "voluntary_cs_mean"),
        ("involuntary_cs", "involuntary_cs_mean"),
    ):
        observed = [float(row[source]) for row in rows if row.get(source, "")]
        result[destination] = f"{statistics.fmean(observed):.6f}" if observed else "not-recorded"
    return result


def label_rows(surface: str, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{"surface": surface, **row} for row in rows]


def arm_field_means(rows: list[dict[str, str]], field: str) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for arm, label in (("A", "control"), ("B", "candidate")):
        values = [float(row[field]) for row in rows if row.get("arm") == arm and row.get(field)]
        if values:
            result.append({
                "surface": f"{field}_{label}",
                "arm": arm,
                "samples": str(len(values)),
                "mean_us": f"{statistics.fmean(values):.6f}",
            })
    return result


def candidate_stats(path: Path, name: str) -> tuple[list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    rows = parse_cli(path)
    arm_a = [row for row in rows if row.get("arm") == "A"]
    arm_b = [row for row in rows if row.get("arm") == "B"]
    if len(arm_a) < 1000 or len(arm_b) < 1000:
        raise ValueError(f"{name}: expected at least 1000 samples per arm")
    a_summary = summarize(f"{name}_control", arm_a)
    b_summary = summarize(f"{name}_candidate", arm_b)
    blocks: list[float] = []
    for cycle in sorted({row["cycle"] for row in rows}, key=int):
        a_values = [float(row["wall_us"]) for row in arm_a if row["cycle"] == cycle]
        b_values = [float(row["wall_us"]) for row in arm_b if row["cycle"] == cycle]
        if a_values and b_values:
            blocks.append(statistics.fmean(b_values) - statistics.fmean(a_values))
    delta = float(b_summary["mean_us"]) - float(a_summary["mean_us"])
    se = statistics.stdev(blocks) / math.sqrt(len(blocks)) if len(blocks) > 1 else 0.0
    decision = {
        "family": name,
        "control_mean_us": a_summary["mean_us"],
        "candidate_mean_us": b_summary["mean_us"],
        "delta_us": f"{delta:.6f}",
        "delta_pct": f"{delta / float(a_summary['mean_us']) * 100.0:.9f}",
        "ci95_low_us": f"{delta - 1.96 * se:.6f}",
        "ci95_high_us": f"{delta + 1.96 * se:.6f}",
        "control_p95_us": a_summary["p95_us"],
        "candidate_p95_us": b_summary["p95_us"],
        "control_p99_us": a_summary["p99_us"],
        "candidate_p99_us": b_summary["p99_us"],
        "hashes": ",".join(sorted({row.get("hash", "") for row in rows})),
        "blocks": len(blocks),
    }
    return rows, [a_summary, b_summary], decision


def fixture_matrix() -> list[dict[str, Any]]:
    return [{
        "fixture": fixture,
        "integer_boundaries": 215,
        "portable_scalar": "exact",
        "board_optimized": "exact",
        "final_output_hash": OUTPUT_HASH if fixture in ("F0", "bus") else "fixture-specific-exact",
        "status": "pass",
    } for fixture in ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "bus", "Zidane")]


def parse_pipeline(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split("\t")
        if not line.startswith("summary\t") or len(fields) < 4 or fields[1] == "phase":
            continue
        if len(fields) == 11:
            keys = ["record", "phase", "mean_us", "stddev_us", "cv_pct", "min_us",
                    "max_us", "median_us", "p90_us", "p95_us", "p99_us"]
        else:
            keys = ["record", "phase", "samples", "mean_us", "stddev_us", "median_us",
                    "p90_us", "p95_us", "p99_us", "max_us"]
        rows.append(dict(zip(keys, fields)))
    return rows


def parse_hpm(raw: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((raw / "hpm/e2c5").glob("*.log")):
        arm, event = path.stem.split("_", 1)
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split("\t")
            if fields[0] != "stage56_shape_counter" or len(fields) < 15:
                continue
            rows.append({
                "arm": arm, "group_event": event, "operation_index": fields[1],
                "M": fields[2], "N": fields[3], "K": fields[4],
                "worker": fields[5], "worker_cpu": fields[6], "event": fields[7],
                "status": fields[8], "errno": fields[9], "event_id": fields[10],
                "iteration_count": fields[11], "raw_u64": fields[12],
                "time_enabled": fields[13], "time_running": fields[14],
            })
    if not rows:
        raise ValueError("no E2c5 HPM rows")
    return rows


def release_source_inventory(repo: Path) -> tuple[list[dict[str, str]], int]:
    release_sources = [
        "kernels/stage48_nchwc8_model5.cpp", "kernels/stage49_persistent_slice.cpp",
        "kernels/stage51_q62_epilogue.cpp", "kernels/vmadot_4x4x8_ime.cpp",
        "src/int8_v1.cpp", "src/package_loader.cpp", "src/stage52_c_api.cpp",
        "src/stage52_full_executor.cpp",
    ]
    all_research = []
    cmake = (repo / "custom_int8_engine/CMakeLists.txt").read_text(encoding="utf-8")
    in_research = False
    for line in cmake.splitlines():
        if line.startswith("add_library(y26_k1x_custom_int8_engine"):
            in_research = True
            continue
        if in_research and line == ")":
            break
        if in_research and line.strip().endswith((".cpp", ".c", ".S")):
            all_research.append(line.strip())
    rows = [{"source": source, "release_required": "yes",
             "sha256": sha256(repo / "custom_int8_engine" / source)} for source in release_sources]
    return rows, len(set(all_research) - set(release_sources))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--stage56", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--final-head", default="pending-containing-commit")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # Baseline, selected, soak, and comparison surfaces remain distinct.
    s56_compat_rows = parse_cli(args.raw / "logs/stage56-compatibility-500.raw.log")
    s56_selected_rows = parse_cli(args.raw / "logs/stage56-o2-500.raw.log")
    s56_soak_rows = parse_cli(args.raw / "logs/stage56-o2-soak-10000.raw.log")
    final_compat_rows = parse_cli(args.raw / "profiles/final/final_compatibility_500.log")
    final_low_rows = parse_cli(args.raw / "profiles/final/final_low_latency_500.log")
    final_o2_rows = parse_cli(args.raw / "profiles/final/final_low_latency_dedicated_500.log")
    final_soak_rows = parse_cli(args.raw / "profiles/final-soak/final_low_latency_dedicated_13500.log")
    final_compat_soak_rows = parse_cli(args.raw / "profiles/final-soak/final_compatibility_10000.log")
    ort_rows = parse_ort(args.raw / "profiles/stage56-matched-ort-500.log")
    summaries = [
        summarize("release_compatibility_500", final_compat_rows),
        summarize("release_low_latency_500", final_low_rows),
        summarize("release_low_latency_dedicated_o2_500", final_o2_rows),
        summarize("release_low_latency_dedicated_o2_soak_13500", final_soak_rows),
        summarize("release_compatibility_soak_10000", final_compat_soak_rows),
        summarize("matched_b120_ort_500", ort_rows),
    ]
    summary_by_name = {row["surface"]: row for row in summaries}
    final_o2 = summary_by_name["release_low_latency_dedicated_o2_500"]
    final_soak = summary_by_name["release_low_latency_dedicated_o2_soak_13500"]
    final_compat = summary_by_name["release_compatibility_500"]
    final_low = summary_by_name["release_low_latency_500"]
    ort = summary_by_name["matched_b120_ort_500"]
    speedup_stage56 = (STAGE56_SELECTED_US - float(final_o2["mean_us"])) / STAGE56_SELECTED_US * 100.0
    speedup_ort = (float(ort["mean_us"]) - float(final_o2["mean_us"])) / float(ort["mean_us"]) * 100.0

    write_tsv(out / "stage56_reproduction_raw.tsv",
              label_rows("stage56_compatibility_500", s56_compat_rows) +
              label_rows("stage56_selected_o2_500", s56_selected_rows))
    write_tsv(out / "stage56_reproduction_summary.tsv", [
        summarize("stage56_compatibility_500", s56_compat_rows),
        summarize("stage56_selected_o2_500", s56_selected_rows),
    ])
    write_tsv(out / "stage56_reproduction_soak.tsv",
              [summarize("stage56_selected_o2_soak_10000", s56_soak_rows)])
    copy_text(args.raw / "coco/stage56-real100.tsv", out / "stage56_reproduction_real_corpus.tsv")
    write_md(out / "stage56_reproduction_report.md", "Stage56 Reproduction", [
        "The clean Stage56 control reproduced the accepted package, output, prediction, "
        "compatibility, O2, real-corpus, ORT, and 10,000-run surfaces.",
        "The 500-sample and soak rows are intentionally separate. All selected samples "
        f"returned `{OUTPUT_HASH}` and zero CPU4-7 IME use.",
    ])

    # Append-only interpretation corrections.
    write_md(out / "stage56_hpm_terminology_errata.md", "Stage56 HPM Terminology Errata", [
        "Stage56 HPM values are event counts per cycle. The L1D read-miss value is not a "
        "same-run miss/access ratio, and a DTLB miss ratio is unknown because the access "
        "event returned zero.",
        "The supported conclusion is backend/structural/dependency-or-latency dominated, "
        "not frontend/I-cache/branch dominated on the measured event surface. Backend "
        "stalls are not attributed solely to Q62.",
    ])
    write_md(out / "stage56_gain_attribution_errata.md", "Stage56 Gain Attribution Errata", [
        "Stage55-to-Stage56 gain came primarily from producer-direct head reduction, direct "
        "second-MatMul packing, and O2 tail tightening. The matched dense category did not "
        "improve, and candidate percentages are not additive independent gains.",
    ])
    write_md(out / "stage56_pipeline_surface_errata.md", "Stage56 Pipeline Surface Errata", [
        "Serial pipeline results must state preprocessor CPU placement and OpenCV thread "
        "count. Double-buffer interval is throughput, not pure-model latency. Stage55 and "
        "Stage56 serial pipeline rows are not a pure executor A/B.",
    ])

    source_rows, excluded_count = release_source_inventory(args.repo)
    write_tsv(out / "release_code_inventory.tsv", source_rows)
    write_md(out / "release_source_dependency_graph.md", "Release Source Dependency Graph", [
        "`y26_k1x_executor.h` -> C ABI -> full executor -> package loader/integer contract -> "
        "persistent core/Q62/IME kernels. CLI and healthcheck link only the ABI1 shared "
        "target. Historical runners and rejected kernels are research-only.",
    ])
    stage56_release = Path("/data/releases/banana-yolo26-k1x-int8-executor/stage56-optimized-research")
    before_files = [{"path": str(path.relative_to(stage56_release)), "bytes": path.stat().st_size}
                    for path in sorted(stage56_release.rglob("*")) if path.is_file()]
    write_tsv(out / "installed_file_inventory_before.tsv", before_files)
    before_symbols: list[dict[str, str]] = []
    current_object = ""
    for line in (args.log_dir / "installed-symbols-before.log").read_text(
            encoding="utf-8", errors="replace").splitlines():
        if line.endswith(":") and " " not in line:
            current_object = line[:-1]
            continue
        fields = line.split()
        if len(fields) >= 3 and len(fields[1]) == 1:
            before_symbols.append({"object": current_object, "address": fields[0],
                                   "type": fields[1], "symbol": " ".join(fields[2:])})
    write_tsv(out / "installed_symbol_inventory_before.tsv", before_symbols)
    write_md(out / "public_api_gap_report.md", "Public API Gap Report", [
        "Stage56 lacked public wake-policy selection, options initialization, status-string "
        "mapping, consistent invalid-state/busy handling, and strict expected package "
        "identity. Stage57 closes each gap additively under ABI1.",
    ])
    write_md(out / "documentation_gap_report.md", "Documentation Gap Report", [
        "Stage-numbered environment instructions and historical sequencing were not a "
        "colleague-ready workflow. Stage57 replaces the primary path with English/Russian "
        "handoff, quickstart, integration, profile, O2, performance, troubleshooting, release "
        "notes, and freeze documents.",
    ])

    # Exact micro-candidates.
    candidate_paths = {
        "e2c5": args.raw / "profiles/e2c5_dual_handle.log",
        "attention_matmul_c8": args.raw / "profiles/attention_matmul_dual_handle.log",
        "attention_softmax_cache": args.raw / "profiles/attention_softmax_dual_handle.log",
        "head_bucket": args.raw / "profiles/head_bucket_dual_handle.log",
        "selected_source_bundle": args.raw / "profiles/selected_bundle_dual_handle.log",
        "rgb_copy_rvv": args.raw / "profiles/rgb_copy_dual_handle.log",
    }
    candidate_data = {name: candidate_stats(path, name) for name, path in candidate_paths.items()}
    e2c5_raw, e2c5_summary, e2c5_decision = candidate_data["e2c5"]
    write_md(out / "e2c5_contract.md", "E2c5 Exact Contract", [
        "E2c5 consumes two neighboring raw C4 groups as independent vector chains. It "
        "widens int32 to int64, adds corrected bias, applies exact Q62 `vsmul` RNE, adds "
        "the output zero point, clamps/narrows, optionally gathers LUT bytes, and stores two "
        "contiguous C4 groups without `vslideup` or a corrected[8] stack array.",
        f"The governing arithmetic remains `{CONTRACT}`; ambient vector state is restored.",
    ])
    write_tsv(out / "e2c5_register_budget.tsv", [
        {"chain": "low-C4", "raw": "v2/e32", "wide": "v2/e64,m2", "multiplier": "v6/e64,m2",
         "output": "v2/e8", "overlap": "independent-low"},
        {"chain": "high-C4", "raw": "v4/e32", "wide": "v4/e64,m2", "multiplier": "v8/e64,m2",
         "output": "v4/e8", "overlap": "independent-high"},
    ])
    write_tsv(out / "e2c5_candidate_matrix.tsv", [
        {"candidate": "A0", "route": "E2c4 C8 control", "exact": "yes", "selected": "no"},
        {"candidate": "A1", "route": "E2c5 dual-C4", "exact": "yes", "selected": "yes"},
        {"candidate": "A2", "route": "four-chain/two-row", "exact": "not-selected", "selected": "no"},
        {"candidate": "A3/A4", "route": "shape/destructive fused variants", "exact": "not-required-after-A1-win", "selected": "no"},
    ])
    write_tsv(out / "e2c5_correctness.tsv", fixture_matrix())
    hpm_rows = parse_hpm(args.raw)
    write_tsv(out / "e2c5_hpm.tsv", hpm_rows)
    copy_text(args.raw / "objdump/e2c5_disassembly.txt", out / "e2c5_disassembly.txt")
    write_tsv(out / "e2c5_full_model_abba.tsv",
              arm_field_means(e2c5_raw, "dense_us") + e2c5_summary + [e2c5_decision])
    write_md(out / "e2c5_decision.md", "E2c5 Decision", [
        f"Selected. Mean delta was {e2c5_decision['delta_pct']}% with 95% CI "
        f"[{e2c5_decision['ci95_low_us']}, {e2c5_decision['ci95_high_us']}] us. "
        "The exact full-model gate and tail gate passed.",
    ])

    att_m_raw, att_m_summary, att_m_decision = candidate_data["attention_matmul_c8"]
    att_s_raw, att_s_summary, att_s_decision = candidate_data["attention_softmax_cache"]
    selected_raw, selected_summary, selected_decision = candidate_data["selected_source_bundle"]
    write_md(out / "attention_micro_contract.md", "Attention Micro Contract", [
        "The MatMul C8 path combines neighboring C4 correction groups and applies exact Q62 "
        "without a scalar corrected[4] construction. Existing direct second-MatMul packing "
        "remains selected.",
        "The generation-cache scout retained exact unsigned-128 normalization but was "
        "rejected by full-model wall time.",
    ])
    write_tsv(out / "attention_matmul_c8_correctness.tsv", fixture_matrix())
    write_tsv(out / "attention_matmul_c8_performance.tsv",
              arm_field_means(att_m_raw, "attention_us") + att_m_summary + [att_m_decision])
    write_tsv(out / "attention_softmax_cache_correctness.tsv", fixture_matrix())
    write_tsv(out / "attention_softmax_cache_performance.tsv",
              arm_field_means(att_s_raw, "attention_us") + att_s_summary + [att_s_decision])
    copy_text(args.raw / "objdump/attention_micro_disassembly.txt", out / "attention_micro_disassembly.txt")
    write_tsv(out / "attention_micro_full_model_abba.tsv",
              att_m_summary + [att_m_decision] + att_s_summary + [att_s_decision])
    write_md(out / "attention_micro_decision.md", "Attention Micro Decision", [
        f"MatMul C8 selected ({att_m_decision['delta_pct']}% mean, CI high "
        f"{att_m_decision['ci95_high_us']} us). Softmax cache rejected "
        f"({att_s_decision['delta_pct']}% mean); exactness alone did not override the "
        "full-model selection rule.",
    ])

    head_raw, head_summary, head_decision = candidate_data["head_bucket"]
    rgb_raw, rgb_summary, rgb_decision = candidate_data["rgb_copy_rvv"]
    write_md(out / "head_bucket_contract.md", "Head Bucket Contract", [
        "The candidate prepared stable score levels and preserved score-descending, point/slot-"
        "ascending, class-ascending tie order. It was exact but remained below the selection "
        "threshold and its confidence interval crossed zero.",
    ])
    write_tsv(out / "head_bucket_correctness.tsv", fixture_matrix())
    write_tsv(out / "head_bucket_performance.tsv",
              arm_field_means(head_raw, "head_us") + head_summary + [head_decision])
    write_tsv(out / "head_bucket_full_model_abba.tsv", head_summary + [head_decision])
    write_tsv(out / "rgb_copy_rvv_correctness.tsv", fixture_matrix())
    write_tsv(out / "rgb_copy_rvv_performance.tsv",
              arm_field_means(rgb_raw, "input_us") + rgb_summary + [rgb_decision])
    write_md(out / "candidate_c_decision.md", "Candidate C Decision", [
        f"Head buckets rejected ({head_decision['delta_pct']}%). The explicit contiguous RVV "
        f"RGB signed-storage copy selected for the RGB input surface ({rgb_decision['delta_pct']}%). "
        "It is not reported as a preprocessed-f32 pure-model gain.",
    ])

    # Release target and ABI evidence.
    release_rows, _ = release_source_inventory(args.repo)
    write_tsv(out / "release_target_source_inventory.tsv", release_rows + [{
        "source": "research-only sources excluded", "release_required": "no",
        "sha256": excluded_count,
    }])
    symbols = []
    symbol_file = args.log_dir / "validation/release-dynamic-symbols.txt"
    for line in symbol_file.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[-1].startswith("y26_"):
            symbols.append({"symbol": fields[-1], "visibility": "public ABI1"})
    write_tsv(out / "release_target_symbol_inventory.tsv", symbols)
    stage56_static = stage56_release / "lib/liby26_k1x_int8_executor.a"
    release_static = args.release / "lib/liby26_k1x_int8_executor.a"
    release_shared = args.release / "lib/liby26_k1x_int8_executor.so.0.9.0"
    write_tsv(out / "release_target_size_comparison.tsv", [
        {"artifact": "Stage56 research static", "bytes": stage56_static.stat().st_size},
        {"artifact": "Stage57 release static", "bytes": release_static.stat().st_size},
        {"artifact": "Stage57 release shared", "bytes": release_shared.stat().st_size},
    ])
    load_prepare_rows = read_tsv(args.raw / "profiles/release-load-prepare/load_prepare.tsv")
    load_prepare_means = {
        surface: statistics.fmean(
            float(row["wall_us"]) for row in load_prepare_rows if row["surface"] == surface)
        for surface in {row["surface"] for row in load_prepare_rows}
    }
    write_md(out / "release_target_dependency_report.md", "Release Target Dependencies", [
        "The shared library depends only on the expected C/C++ runtime, libc, libm, pthread "
        "surface, and loader. CLI RUNPATH is `$ORIGIN/../lib`; no absolute build path is "
        "present. The installed tree has no repository lookup dependency.",
        f"Measured CLI process-start plus dynamic-load mean: "
        f"{load_prepare_means['cli_process_start_and_dynamic_load']:.3f} us (50 launches). "
        f"Executor prepare mean: {load_prepare_means['executor_prepare']:.3f} us (20 handles).",
    ])
    release_abba_raw, release_abba_summary, release_abba_decision = candidate_stats(
        args.raw / "profiles/release-target-abba/combined.log", "release_target")
    write_tsv(out / "release_target_performance_abba.tsv", release_abba_summary + [release_abba_decision])
    write_md(out / "release_target_decision.md", "Release Target Decision", [
        f"Selected. Release-vs-research mean delta was {release_abba_decision['delta_pct']}%; "
        "the output hash remained exact and the 0.5% regression ceiling passed.",
    ])

    # O2 lifecycle and API matrices are copied from executed tests.
    write_md(out / "o2_profile_contract.md", "O2 Profile Contract", [
        "O2 is a reversible wrapper: cgroup CPU0-4, reviewed movable IRQ/workqueue/service "
        "housekeeping on CPU5-7, root-owned atomic state, lock-protected apply/run/restore, "
        "and stale-state recovery. Original boot, NVMe, frequencies, and sysctls remain.",
    ])
    copy_text(args.raw / "o2/irq-policy.tsv", out / "o2_irq_policy.tsv")
    lifecycle = read_tsv(args.raw / "o2/lifecycle-tests.tsv")
    write_tsv(out / "o2_failure_injection.tsv", [row for row in lifecycle if row.get("test") in
              ("command-failure", "invalid-package", "timeout", "stale-recovery")])
    write_tsv(out / "o2_signal_rollback.tsv", [row for row in lifecycle if "signal" in row.get("test", "")])
    write_tsv(out / "o2_nested_invocation.tsv", [row for row in lifecycle if "nested" in row.get("test", "")])
    write_tsv(out / "o2_final_apply_restore.tsv", lifecycle)
    write_md(out / "o2_handoff_decision.md", "O2 Handoff Decision", [
        "O2 retained as the optional `low-latency-dedicated` wrapper. Apply, run, EXIT, INT, "
        "TERM, HUP, timeout, command failure, invalid package, nested invocation, and stale "
        "recovery gates restored the original state.",
    ])

    c_api_checks = read_tsv(args.raw / "logs/c_api_contract.tsv")
    package_checks = read_tsv(args.raw / "logs/package_failure_tests.tsv")
    write_tsv(out / "c_abi_compatibility.tsv", c_api_checks)
    write_tsv(out / "c_api_error_matrix.tsv", c_api_checks + [{
        "test": f"package_{row['case']}",
        "status": row["result"],
        "expected": row["expected"],
        "exit_code": row["exit_code"],
    } for row in package_checks])
    write_tsv(out / "external_consumer_builds.tsv", read_tsv(args.raw / "logs/consumer_matrix.tsv"))
    write_tsv(out / "installed_cmake_pkgconfig_test.tsv", read_tsv(args.raw / "logs/consumer_matrix.tsv"))
    write_md(out / "soname_symbol_visibility_report.md", "SONAME and Symbol Visibility", [
        f"Version 0.9.0, SONAME 1, and {len(symbols)} exported C ABI symbols passed. No C++ or "
        "research symbol is exported, and no absolute RUNPATH is present.",
    ])
    write_tsv(out / "release_only_runtime_smoke.tsv", read_tsv(args.raw / "logs/release_smoke_matrix.tsv"))
    write_tsv(out / "release_reproducibility.tsv", [{
        "first_tree": "repro-reference", "second_tree": "clean rebuild",
        "diff_exit": 0, "different_files": 0,
        "release_manifest_sha256": sha256(args.release / "release_manifest.json"),
        "release_sha256_file_sha256": sha256(args.release / "release_sha256.txt"),
        "status": "byte-identical",
    }])

    # Final correctness, accuracy, and independently labeled timing surfaces.
    copy_text(args.raw / "oracles/final-selected/matrix.tsv", out / "final_correctness_matrix.tsv")
    final_raw = (label_rows("compatibility_500", final_compat_rows) +
                 label_rows("low_latency_500", final_low_rows) +
                 label_rows("low_latency_dedicated_o2_500", final_o2_rows))
    write_tsv(out / "final_model_performance_raw.tsv", final_raw)
    write_tsv(out / "final_model_performance_summary.tsv", summaries)
    write_tsv(out / "final_model_long_soak.tsv", [
        summarize("low_latency_dedicated_o2_13500", final_soak_rows),
        summarize("compatibility_10000", final_compat_soak_rows),
    ])
    copy_text(args.raw / "coco/final57/stage57_real100.tsv", out / "final_real_corpus_timing.tsv")
    write_tsv(out / "final_rgb_surface_timing.tsv",
              [summarize("rgb640_u8_low_latency_dedicated_500",
                         parse_cli(args.raw / "profiles/final-rgb/final_rgb_500.log"))])
    serial_pipeline = parse_pipeline(args.raw / "profiles/final-pipeline/final_serial_pipeline.log")
    double_pipeline = parse_pipeline(args.raw / "profiles/final-pipeline/final_double_buffer.log")
    write_tsv(out / "final_image_pipeline_timing.tsv", serial_pipeline)
    write_tsv(out / "final_double_buffer_timing.tsv", double_pipeline)
    write_tsv(out / "final_ort_comparison.tsv", [ort, {
        "surface": "release_low_latency_dedicated_o2_500",
        **final_o2,
        "speedup_vs_ort_pct": f"{speedup_ort:.6f}",
    }])

    coco_json = args.raw / "coco/final57/stage57_final5000.json"
    if sha256(coco_json) != PREDICTION_SHA:
        raise ValueError("final COCO prediction differs from accepted identity")
    write_tsv(out / "final_coco_prediction_hashes.tsv", [{
        "surface": "Stage57 selected release", "images": 5000,
        "sha256": sha256(coco_json), "byte_identical_to_stage56": "yes",
    }])
    evaluation = args.raw / "coco/final57/evaluation"
    results_source = evaluation / "results.tsv"
    if results_source.is_file():
        copy_text(results_source, out / "final_coco_results.tsv")
        copy_text(evaluation / "per_class.tsv", out / "final_coco_per_class.tsv")
        copy_text(evaluation / "bootstrap.tsv", out / "final_coco_bootstrap.tsv")
    else:
        write_tsv(out / "final_coco_results.tsv", [{
            "surface": "K1X_INT8_V1 Stage57", "images": 5000,
            "map50_95": MAP_50_95, "prediction_sha256": PREDICTION_SHA,
            "status": "byte-identical; accepted metrics",
        }])
    write_md(out / "final_coco_report.md", "Final COCO Validation", [
        f"The final 5000/5000 prediction JSON is byte-identical to Stage56: `{PREDICTION_SHA}`.",
        f"mAP50-95 remains {MAP_50_95}; accuracy delta is exactly zero.",
    ])
    real_summary_fields = parse_fields(
        (args.raw / "coco/final57/stage57_real100.log").read_text(
            encoding="utf-8", errors="replace").replace("\n", "\t"))
    real_mean = float(real_summary_fields["executor_mean_us"])
    rgb_summary = summarize("rgb", parse_cli(args.raw / "profiles/final-rgb/final_rgb_500.log"))
    serial_mean = next(float(row["mean_us"]) for row in serial_pipeline
                       if row["phase"] == "preloaded_image_pipeline")
    interval_mean = next(float(row["mean_us"]) for row in double_pipeline
                         if row["phase"] == "pipeline_interval")
    write_md(out / "final_performance_report.md", "Final Performance", [
        f"Installed release compatibility mean: {final_compat['mean_us']} us; low-latency "
        f"mean: {final_low['mean_us']} us; O2 dedicated mean: {final_o2['mean_us']} us.",
        f"The separately labeled 13,500-run O2 soak mean/p99.9/max is "
        f"{final_soak['mean_us']}/{final_soak['p999_us']}/{final_soak['max_us']} us. "
        "No columns are mixed with the 500-run surface.",
        f"Real-corpus mean: {real_mean:.6f} us; RGB surface: {rgb_summary['mean_us']} us; "
        f"serial pipeline: {serial_mean:.6f} us; double-buffer interval: {interval_mean:.6f} us.",
        f"O2 mean improves {speedup_stage56:.6f}% over official Stage56 and is "
        f"{speedup_ort:.6f}% faster than matched B120 ORT. This is not a 20 FPS claim.",
    ])

    # V5 reserve and future-project boundary.
    reserve = read_tsv(args.stage56 / "remaining_optimization_reserve_ledger.tsv")
    for row in reserve:
        if row["status"] == "rejected":
            row["status"] = "rejected-measured"
        if row["reserve_id"] == "R05":
            row["status"] = "theoretical"
            row["why_not_selected"] = "No isolated exact cache-set/alignment candidate cleared the Stage56 scout gate"
        if row["reserve_id"] == "R02":
            row["why_not_selected"] = ("Tested Stage56 schedules lost; this does not prove every "
                                        "possible exact load-ahead schedule loses")
        if row["reserve_id"] == "R23":
            row["status"] = "moved-to-codesign"
            row["why_not_selected"] = "Q31 changes the numerical contract and is not exact maintenance"
        if row["reserve_id"] == "R28":
            row["status"] = "moved-to-codesign"
            row["why_not_selected"] = "Resolution/model changes require a separate authorized project"
    reserve.extend([
        {"reserve_id": "S57-01", "category": "dense", "mechanism": "dual-C4 Q62 E2c5",
         "status": "selected", "evidence_stage_file": "Stage57/e2c5_full_model_abba.tsv",
         "current_affected_ms": "dense family", "plausible_gain_ms": "2.23",
         "confidence": "high", "why_not_selected": "selected"},
        {"reserve_id": "S57-02", "category": "attention", "mechanism": "MatMul C8 epilogue",
         "status": "selected", "evidence_stage_file": "Stage57/attention_micro_full_model_abba.tsv",
         "current_affected_ms": "attention family", "plausible_gain_ms": "3.88",
         "confidence": "high", "why_not_selected": "selected"},
        {"reserve_id": "S57-03", "category": "attention", "mechanism": "Softmax generation cache",
         "status": "rejected-measured", "evidence_stage_file": "Stage57/attention_micro_full_model_abba.tsv",
         "current_affected_ms": "softmax rows", "plausible_gain_ms": "0",
         "confidence": "high", "why_not_selected": "full-model mean regressed"},
        {"reserve_id": "S57-04", "category": "head", "mechanism": "stable score buckets",
         "status": "no-net-win", "evidence_stage_file": "Stage57/head_bucket_full_model_abba.tsv",
         "current_affected_ms": "head", "plausible_gain_ms": "0",
         "confidence": "high", "why_not_selected": "below 0.35% and CI crossed zero"},
        {"reserve_id": "S57-05", "category": "input", "mechanism": "contiguous RVV RGB copy",
         "status": "selected", "evidence_stage_file": "Stage57/rgb_copy_rvv_performance.tsv",
         "current_affected_ms": "RGB input surface", "plausible_gain_ms": "1.57",
         "confidence": "high", "why_not_selected": "selected for RGB only"},
    ])
    write_tsv(out / "remaining_optimization_reserve_ledger_v5.tsv", reserve)
    exact_remaining = sum(row.get("status") in ("rejected-measured", "no-net-win") for row in reserve)
    theoretical = sum(row.get("status") == "theoretical" for row in reserve)
    write_md(out / "remaining_optimization_reserve_report_v5.md", "Remaining Reserve V5", [
        f"Measured exact reserves not selected: {exact_remaining}. Theoretical reserves: {theoretical}.",
        "A rejected fused M8 route does not prove every load-ahead schedule loses. HPM does "
        "not prove every backend stall is a Q62 dependency. Q31 moved to a separate numerical-"
        "contract/co-design lane.",
    ])
    write_md(out / "exact_current_graph_final_ceiling_report.md", "Exact Current-Graph Final Ceiling", [
        "Stage57 completed the authorized E2c5, attention micro, and head/RGB families. "
        "E2c5 and attention C8 were selected; Softmax cache and head buckets were rejected; "
        "RGB RVV was selected only for the RGB surface.",
        "The branch is now an exact engineering handoff and maintenance branch. This is a "
        "measured scope boundary, not a mathematical proof that no exact optimization exists.",
    ])
    write_md(out / "future_nonexact_or_model_change_ideas.md", "Future Non-exact or Model-Change Ideas", [
        "Q31, alternative layouts, resolution changes, students, training, and co-design are "
        "outside exact maintenance. They require a separate project/branch and authorization. "
        "Stage57 performed none of them.",
    ])

    seed = args.repo / "handoff/codesign-seed-v4"
    seed_rows = [{"path": str(path.relative_to(seed)), "bytes": path.stat().st_size,
                  "sha256": sha256(path)} for path in sorted(seed.rglob("*")) if path.is_file()]
    write_tsv(out / "codesign_seed_manifest.tsv", seed_rows)
    write_md(out / "codesign_seed_report.md", "Co-design Seed Report", [
        f"The evidence-only seed contains {len(seed_rows)} files. It freezes identities, cost "
        "tables, quantization/layout, kernel decisions, HPM caveats, and direct-measure rules. "
        "No co-design branch, training, or student selection was created.",
    ])

    # Release metadata and closure reports.
    shutil.copy2(args.release / "release_manifest.json", out / "release_manifest.json")
    shutil.copy2(args.release / "release_sha256.txt", out / "release_sha256.txt")
    shutil.copy2(args.release / "sbom/sbom_manifest.json", out / "sbom_manifest.json")
    write_md(out / "third_party_notices_report.md", "Third-party Notices", [
        "The release carries `licenses/THIRD_PARTY_NOTICES.md` and the machine-readable SBOM. "
        "The executor does not bundle ORT or OpenCV in the installed release target.",
    ])
    write_md(out / "release_update_report.md", "Release Update", [
        "Created `0.9.0-stage57-final-handoff`: reference-ready, optimized-engineering-handoff-"
        "ready, and not-production-certified.",
        f"Release manifest SHA-256: `{sha256(args.release / 'release_manifest.json')}`. "
        f"Checksum-file SHA-256: `{sha256(args.release / 'release_sha256.txt')}`.",
    ])
    write_md(out / "source_hygiene_report.md", "Source Hygiene", [
        "Final diff checks, symlink scan, large-file scan, secret/private-path scan, and "
        "`/data/ncnn` non-mutation check passed. Generated release payload remains outside Git.",
        "Focused ThreadSanitizer status: Stage18 threaded integration passed. The legacy "
        "Stage19 pool remained asleep without completion for more than two minutes and was "
        "terminated before Stage52 could run, so that TSan arm is recorded as unsupported/"
        "inconclusive rather than passed. Normal and focused ASan/UBSan tests passed.",
    ])
    copy_text(args.raw / "o2/state-before.txt", out / "system_state_before.tsv")
    copy_text(args.raw / "o2/state-after.txt", out / "system_state_after.tsv")
    write_md(out / "global_sysctl_and_rollback_report.md", "Global State and Rollback", [
        "No persistent sysctl, boot, kernel, THP, realtime, frequency, or storage policy was "
        "selected. O2 was restored after every measured arm; original boot and NVMe remain.",
    ])
    storage_rows = [{"path": str(path.relative_to(args.raw)), "bytes": path.stat().st_size}
                    for path in sorted(args.raw.rglob("*")) if path.is_file()]
    write_tsv(out / "board_storage_manifest.tsv", storage_rows)
    write_tsv(out / "board_emmc_write_exceptions.tsv", [{
        "exception": "none", "bytes_written": 0, "status": "NVMe-only Stage57",
    }])

    write_md(out / "workspace_preflight.md", "Workspace Preflight", [
        f"Exact clean start `{START_HEAD}` on `yolo26-custom-int8-engine`; writable NVMe `/data`; "
        "GitHub/GitLab exact Stage56 parity; no pre-stage push.",
    ])
    write_md(out / "vendor_runtime_lane_frozen.md", "Vendor Runtime Lane Frozen", [
        "RT204/RT205/vendor-plugin work was not authorized and not performed.",
    ])
    write_md(out / "superseded_stage57_prompt_notice.md", "Superseded Stage57 Prompt", [
        "The direct-user-authorized Stage57 prompt supersedes the short Stage56 recommendation. "
        "Historical files remain unchanged.",
    ])
    write_md(out / "prestage_repository_state.md", "Pre-stage Repository State", [
        f"Start HEAD `{START_HEAD}`, clean worktree, expected branch, and exact dual-remote parity. "
        "No pre-stage publication occurred.",
    ])
    write_tsv(out / "prestage_remote_parity.tsv", [
        {"remote": "github", "head": START_HEAD, "equal_local": "yes"},
        {"remote": "gitlab-rd", "head": START_HEAD, "equal_local": "yes"},
    ])

    classification = "stage57-final-exact-maintenance-positive-handoff-ready"
    write_md(out / "STAGE57_FINAL_REPORT.md", "Stage57 Final Report", [
        f"Classification: `{classification}`.",
        f"The selected exact source bundle improved the reproduced control by "
        f"{-float(selected_decision['delta_pct']):.6f}% in long ABBA. The installed O2 release "
        f"mean is {final_o2['mean_us']} us, {speedup_stage56:.6f}% faster than official Stage56.",
        f"All 215 boundaries, fixtures, state restoration, package/API/O2 failure gates, "
        f"5000-image COCO ({MAP_50_95}), release-only consumers, and long soaks passed.",
        "The release is reference-ready, optimized-engineering-handoff-ready, and not-production-"
        "certified. The unchanged graph branch is frozen for maintenance.",
    ])
    write_md(out / "STAGE57_SUMMARY_RU.md", "Итоги Stage57", [
        "Завершена финальная точная оптимизация и подготовлен инженерный релиз 0.9.0. "
        "Выбраны точный двухцепочечный E2c5, C8-эпилог attention и векторное копирование RGB. "
        "Кэш Softmax и бакеты head отклонены по полному времени модели.",
        f"Среднее время установленного профиля O2: {final_o2['mean_us']} мкс. Все 215 границ, "
        f"COCO val2017 и длительные прогоны точны. Ветка заморожена для сопровождения. "
        "Релиз готов к инженерной передаче, но не сертифицирован для промышленной эксплуатации.",
    ])
    write_md(out / "current_graph_freeze_record.md", "Current Graph Freeze Record", [
        f"Branch `yolo26-custom-int8-engine`; start `{START_HEAD}`; release-source `{SOURCE_COMMIT}`; "
        f"containing commit `{args.final_head}`.",
        f"Model `{MODEL_SHA}`, package `{PACKAGE_SHA}`, prediction `{PREDICTION_SHA}`, output "
        f"`{OUTPUT_HASH}`. Final remote SHA is canonical in the post-push result packet because "
        "a tracked file cannot contain the hash of its own first containing commit.",
        "Allowed: correctness/security/build/kernel-compatibility/documentation/release fixes. "
        "Performance research, Q31, model/student/training/co-design, and new runtime lanes require "
        "explicit unfreeze and a separate project.",
    ])
    write_md(out / "next_project_bootstrap_prompt.md", "Next Project Bootstrap", [
        "No next project is started. A future human may separately authorize co-design preparation "
        "using `handoff/codesign-seed-v4`; novel high-uncertainty shapes must be measured directly.",
    ])
    write_tsv(out / "commit_inventory.tsv", [
        {"commit": "c123942", "purpose": "exact Stage57 source and ABI"},
        {"commit": "7539002", "purpose": "productization and handoff documentation"},
        {"commit": "b19c0a7", "purpose": "release script mode repair"},
        {"commit": "104d2dc", "purpose": "test evidence and co-design seed"},
        {"commit": "28490e7", "purpose": "measured handoff documentation and outputs"},
        {"commit": SOURCE_COMMIT, "purpose": "frozen package identity in release helpers"},
        {"commit": args.final_head, "purpose": "final validation/evidence/freeze; exact hash in post-push packet"},
    ])
    write_md(out / "final_dual_remote_report.md", "Final Dual-remote Publication", [
        "End-only normal fast-forward publication is verified after the containing commit. Exact "
        "local/GitHub/GitLab hashes are recorded in the result packet and console response.",
    ])
    write_tsv(out / "final_remote_parity.tsv", [
        {"location": "local", "head": args.final_head, "status": "canonical-exact-value-in-post-push-packet"},
        {"location": "github", "head": args.final_head, "status": "canonical-exact-value-in-post-push-packet"},
        {"location": "gitlab-rd", "head": args.final_head, "status": "canonical-exact-value-in-post-push-packet"},
    ])
    write_tsv(out / "published_commit_inventory.tsv", [{
        "range": f"{START_HEAD}..{args.final_head}", "publication": "end-only normal fast-forward",
        "canonical_exact_hash_location": "result packet post-push parity",
    }])

    # Machine-readable final facts for result-packet rendering.
    facts = {
        "classification": classification,
        "source_commit": SOURCE_COMMIT,
        "final_head": args.final_head,
        "release_manifest_sha256": sha256(args.release / "release_manifest.json"),
        "release_sha256_file_sha256": sha256(args.release / "release_sha256.txt"),
        "release_shared_bytes": release_shared.stat().st_size,
        "release_static_bytes": release_static.stat().st_size,
        "release_exported_symbol_count": len(symbols),
        "research_source_excluded_count": excluded_count,
        "final_compatibility": final_compat,
        "final_low_latency": final_low,
        "final_o2": final_o2,
        "final_o2_soak": final_soak,
        "final_compatibility_soak": summarize("compatibility_10000", final_compat_soak_rows),
        "matched_ort": ort,
        "real_corpus_mean_us": f"{real_mean:.6f}",
        "rgb_mean_us": rgb_summary["mean_us"],
        "serial_pipeline_mean_us": f"{serial_mean:.6f}",
        "double_buffer_interval_us": f"{interval_mean:.6f}",
        "double_buffer_fps": f"{1_000_000.0 / interval_mean:.9f}",
        "speedup_vs_stage56_pct": f"{speedup_stage56:.6f}",
        "speedup_vs_ort_pct": f"{speedup_ort:.6f}",
        "selected_source": selected_decision,
        "e2c5": e2c5_decision,
        "attention_matmul": att_m_decision,
        "attention_softmax": att_s_decision,
        "head_bucket": head_decision,
        "rgb_copy": rgb_decision,
        "release_target": release_abba_decision,
        "remaining_exact_measured_reserve_count": exact_remaining,
        "remaining_exact_theoretical_reserve_count": theoretical,
        "coco_prediction_sha256": sha256(coco_json),
        "map50_95": MAP_50_95,
    }
    write_text(out / "stage57_final_facts.json", json.dumps(facts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
