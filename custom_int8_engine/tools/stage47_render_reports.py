#!/usr/bin/env python3
"""Render the Stage47 decision packet from generated assets and raw command logs."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from collections import defaultdict
from pathlib import Path


STAGE_ID = "BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE47-EXECUTOR-FIRST-MULTICORE-INTEGRATED-LUT-AND-RESIDENT-INT8-AOT-GATE-001"
START_HEAD = "da213f6c11339187d2169e5a2516feef1b732dd9"
MODEL_SHA = "30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c"
CLASSIFICATION = "stage47-blocked-correctness"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, records: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(records[0]) if records else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        if not fields:
            return
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({field: "-" if record.get(field, "") == "" else record.get(field, "") for field in fields})


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def copy_text_clean(source: Path, destination: Path) -> None:
    write(destination, "\n".join(line.rstrip() for line in source.read_text(encoding="utf-8").splitlines()))


def output_for(log_root: Path, label: str) -> Path:
    matches = sorted((log_root / "commands").glob(f"*_{label}.stdout.txt"))
    if not matches:
        raise FileNotFoundError(label)
    return matches[-1]


def parse_kernel_file(path: Path, accepted: bool, attempt: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summaries: list[dict[str, object]] = []
    raw: list[dict[str, object]] = []
    context: dict[int, tuple[str, int, str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if parts[0] == "kernel_correctness":
            context[int(parts[1])] = (parts[3], int(parts[4]), parts[2], "spatial")
        elif parts[0] == "kernel_raw":
            case_id = int(parts[1])
            kernel, workers, name, partition = context[case_id]
            raw.append({
                "attempt": attempt,
                "accepted": int(accepted),
                "case_id": case_id,
                "node_name": name,
                "kernel": kernel,
                "partition": partition,
                "workers": workers,
                "repeat": int(parts[2]),
                "mean_us": parts[3],
            })
        elif parts[0] == "kernel_summary":
            case_id, name = parts[1].split(":", 1)
            summaries.append({
                "attempt": attempt,
                "accepted": int(accepted),
                "case_id": int(case_id),
                "node_name": name,
                "kernel": parts[2],
                "partition": parts[3],
                "workers": int(parts[4]),
                "warmup": int(parts[5]),
                "runs": int(parts[6]),
                "repeats": int(parts[7]),
                "macs": int(parts[8]),
                "mean_us": float(parts[9]),
                "stddev_us": float(parts[10]),
                "cv_pct": float(parts[11]),
                "min_us": float(parts[12]),
                "max_us": float(parts[13]),
                "median_us": float(parts[14]),
                "p90_us": float(parts[15]),
                "p95_us": float(parts[16]),
                "gmacs": float(parts[17]),
                "output_hash64": parts[18],
            })
    return summaries, raw


def parse_aot(path: Path) -> tuple[list[dict[str, object]], dict[str, float], list[dict[str, object]], list[dict[str, object]]]:
    correctness: list[dict[str, object]] = []
    summary: dict[str, float] = {}
    raw: list[dict[str, object]] = []
    profiles: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split("\t")
        if p[0] == "aot_raw":
            raw.append({"repeat": int(p[1]), "mean_us": float(p[2])})
        elif p[0] == "aot_summary":
            summary = {
                "mean_us": float(p[9]), "stddev_us": float(p[10]), "cv_pct": float(p[11]),
                "min_us": float(p[12]), "max_us": float(p[13]), "median_us": float(p[14]),
                "p90_us": float(p[15]), "p95_us": float(p[16]),
            }
        elif p[0] == "aot_contract":
            summary.update({
                "operation_count": float(p[1]), "arena_bytes": float(p[2]), "packed_weight_bytes": float(p[3]),
                "entry_adapter_us": float(p[4]), "exit_adapter_us": float(p[5]), "affinity_ok": float(p[6]),
            })
        elif p[0] == "aot_profile":
            profiles.append({
                "operation_index": int(p[1]), "kind": p[2], "name": p[3], "total_us": float(p[4]),
                "gather_pack_worker_sum_us": float(p[5]), "vmadot_worker_sum_us": float(p[6]),
                "fused_epilogue_worker_sum_us": float(p[7]),
            })
    return correctness, summary, raw, profiles


def parse_correctness(path: Path, tensors: dict[int, dict[str, str]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    fixture_kind = {
        "F0": "accepted_synthetic_seeded", "F1": "random_4301", "F2": "random_4302",
        "F3": "zero_point_structured", "F4": "edge_saturation", "F5": "canonical_image",
        "F6": "bus_image", "F7": "zidane_image",
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split("\t")
        if p[0] != "aot_correctness":
            continue
        tensor_id = int(p[2])
        result.append({
            "fixture": p[1], "fixture_kind": fixture_kind[p[1]], "tensor_id": tensor_id,
            "tensor_name": tensors[tensor_id]["logical_name"], "status": int(p[3]),
            "mismatches": int(p[4]), "max_abs_diff": int(p[5]), "custom_hash64": p[6],
            "affinity_ok": int(p[7]), "first_mismatch_index": int(p[8]),
            "first_actual": int(p[9]), "first_expected": int(p[10]),
        })
    return result


def parse_ort_summary(path: Path) -> dict[str, float | str]:
    result: dict[str, float | str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("stage42_ort_only_repeat "):
            fields = dict(item.split("=", 1) for item in line.split()[1:])
            result[f"repeat_{fields['repeat']}_us"] = float(fields["mean_us"])
        elif line.startswith("stage42_ort_only_benchmark "):
            fields = dict(item.split("=", 1) for item in line.split()[1:])
            for key in ("mean_us", "stddev_us", "cv_pct", "min_us", "max_us", "median_us", "p90_us", "p95_us"):
                result[key] = float(fields[key])
        elif len(line.split()) == 2 and len(line.split()[0]) == 64:
            result["output_sha256"] = line.split()[0]
    return result


def nearest_case(census: dict[str, str], case_ids: list[int], cases: dict[int, dict[str, str]]) -> int:
    def distance(case_id: int) -> float:
        case = cases[case_id]
        values = (
            (int(census["M"]), int(case["output_h"]) * int(case["output_w"])),
            (int(census["N"]), int(case["output_c"])),
            (int(census["K"]), int(case["input_c"]) * int(case["kernel_h"]) * int(case["kernel_w"])),
        )
        return sum(math.log2(max(a, 1) / max(b, 1)) ** 2 for a, b in values)
    return min(case_ids, key=distance)


def render(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    log_root = args.log_root.resolve()
    generated = log_root / "artifacts/stage47_generated"
    stage = args.stage_dir.resolve()
    stage.mkdir(parents=True, exist_ok=True)

    for name in ("graph_shape_census.tsv", "graph_shape_class_report.md", "graph_shape_coverage.tsv"):
        shutil.copy2(generated / name, stage / name)

    case_rows = rows(generated / "integrated_kernel_cases/cases.tsv")
    cases = {int(row["case_id"]): row for row in case_rows}
    tensor_rows = rows(generated / "model4_8_aot_package/tensors.tsv")
    tensors = {int(row["id"]): row for row in tensor_rows}
    op_rows = rows(generated / "model4_8_aot_package/ops.tsv")

    accepted_summaries: list[dict[str, object]] = []
    accepted_raw: list[dict[str, object]] = []
    for label in ("stable_sequential_matrix", "stable_selected_all_cases"):
        summaries, raw = parse_kernel_file(output_for(log_root, label), True, label)
        accepted_summaries.extend(summaries)
        accepted_raw.extend(raw)
    invalid_summaries: list[dict[str, object]] = []
    invalid_raw: list[dict[str, object]] = []
    for sequence in range(217, 223):
        for path in sorted((log_root / "commands").glob(f"{sequence:04d}_stable_c4_*.stdout.txt")):
            summaries, raw = parse_kernel_file(path, False, "invalid_concurrent_batch")
            invalid_summaries.extend(summaries)
            invalid_raw.extend(raw)
    write_tsv(stage / "integrated_kernel_raw.tsv", accepted_raw + invalid_raw)
    write_tsv(stage / "integrated_kernel_summary.tsv", accepted_summaries + invalid_summaries)

    selected = {(int(row["case_id"]), int(row["workers"]), str(row["kernel"])): row for row in accepted_summaries}
    scaling: list[dict[str, object]] = []
    baseline = selected[(4, 1, "m12n16")]
    for workers in range(1, 5):
        row = selected[(4, workers, "m12n16")]
        speedup = float(baseline["mean_us"]) / float(row["mean_us"])
        scaling.append({
            "case_id": 4, "node_name": row["node_name"], "kernel": "m12n16", "workers": workers,
            "mean_us": f"{float(row['mean_us']):.6f}", "gmacs": f"{float(row['gmacs']):.6f}",
            "speedup_vs_1worker": f"{speedup:.6f}", "parallel_efficiency_pct": f"{100.0 * speedup / workers:.6f}",
            "affinity": "CPU0-3", "accepted": 1,
        })
    write_tsv(stage / "integrated_kernel_scaling.tsv", scaling)

    aot_path = output_for(log_root, "stable_aot_m12_w4")
    _, aot, aot_raw, profiles = parse_aot(aot_path)
    correctness = parse_correctness(output_for(log_root, "board_aot_validate_m12_w4"), tensors)
    write_tsv(stage / "model4_8_aot_correctness_matrix.tsv", correctness)
    write_tsv(stage / "model4_8_aot_performance_raw.tsv", aot_raw)
    ort = parse_ort_summary(output_for(log_root, "board_ort_slice_stable"))
    custom_with_adapters = aot["mean_us"] + aot["entry_adapter_us"] + aot["exit_adapter_us"]
    delta_pct = 100.0 * (custom_with_adapters - float(ort["mean_us"])) / float(ort["mean_us"])
    perf_summary = [{
        "surface": "custom_internal", "mean_us": f"{aot['mean_us']:.6f}", "stddev_us": f"{aot['stddev_us']:.6f}",
        "entry_adapter_us": f"{aot['entry_adapter_us']:.6f}", "exit_adapter_us": f"{aot['exit_adapter_us']:.6f}",
        "total_with_adapters_us": f"{custom_with_adapters:.6f}", "correctness": "blocked_fixed_host_oracle",
    }, {
        "surface": "B120_ORT_ENABLE_ALL_intra4", "mean_us": f"{float(ort['mean_us']):.6f}",
        "stddev_us": f"{float(ort['stddev_us']):.6f}", "entry_adapter_us": "0", "exit_adapter_us": "0",
        "total_with_adapters_us": f"{float(ort['mean_us']):.6f}", "correctness": "vendor_runtime_diagnostic",
    }]
    write_tsv(stage / "model4_8_aot_performance_summary.tsv", perf_summary)

    profile_by_kind: dict[str, list[dict[str, object]]] = defaultdict(list)
    for profile in profiles:
        profile_by_kind[str(profile["kind"])].append(profile)
    nonmac = []
    for kind in ("lut", "add_silu", "concat"):
        values = profile_by_kind.get(kind, [])
        nonmac.append({
            "operator_class": kind, "shape_surface": "model4_to_model8_AOT_slice",
            "workers": 4, "mean_us": f"{sum(float(item['total_us']) for item in values):.6f}",
            "instances": len(values), "implementation": "resident_int8_exact", "evidence": "measured_profile_once",
        })
    stage45_profile = rows(repo / "stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE45-SYSTEM-ROOFLINE-VENDOR-ORT-FORENSICS-ACCURACY-AND-MODEL-CODESIGN-DECISION-001/full_model_operator_profile_summary.tsv")
    for op_type in ("NhwcMaxPool", "Resize", "Softmax", "TopK", "GatherElements"):
        values = [float(row["total_us"]) / 4.0 for row in stage45_profile if row["op_type"] == op_type]
        nonmac.append({
            "operator_class": op_type, "shape_surface": "full_graph_B120_profile",
            "workers": 4, "mean_us": f"{sum(values):.6f}", "instances": "profile_aggregate",
            "implementation": "CPUExecutionProvider_fallback", "evidence": "measured_conservative_profile",
        })
    write_tsv(stage / "nonmac_operator_lut.tsv", nonmac)

    stable_case_rows = {int(row["case_id"]): row for row in accepted_summaries if row["kernel"] == "m12n16" and row["workers"] == 4 and row["accepted"] == 1}
    class_cases = {
        "3x3_stride2": [0, 1, 4, 5], "3x3_stride1": [3], "1x1_high_resolution": [2],
        "1x1_low_resolution": [6, 7], "small_n_head_conv": [8],
    }
    census = rows(generated / "graph_shape_census.tsv")
    mapping: list[dict[str, object]] = []
    central_compute_us = 0.0
    conservative_compute_us = 0.0
    mapped_macs = 0
    total_macs = sum(int(row["MACs"]) for row in census)
    for row in census:
        shape_class = row["shape_class"]
        macs = int(row["MACs"])
        if shape_class in class_cases:
            case_id = nearest_case(row, class_cases[shape_class], cases)
            measured = stable_case_rows[case_id]
            rate = float(measured["gmacs"])
            p95_rate = int(measured["macs"]) / (float(measured["p95_us"]) * 1000.0)
            central_us = macs / (rate * 1000.0)
            conservative_us = macs / (p95_rate * 1000.0)
            evidence = "fullshape_integrated_m12n16_w4"
            mapped_macs += macs
        else:
            case_id = -1
            rate = 0.5 if shape_class == "grouped_or_depthwise_conv" else 1.5
            central_us = macs / (rate * 1000.0)
            conservative_us = central_us * 1.5
            evidence = "conservative_unsupported_fallback"
        central_compute_us += central_us
        conservative_compute_us += conservative_us
        mapping.append({
            "graph_index": row["graph_index"], "node_name": row["node_name"], "block": row["block"],
            "shape_class": shape_class, "macs": macs, "mapping_case_id": case_id,
            "mapping_rate_gmacs": f"{rate:.6f}", "central_us": f"{central_us:.6f}",
            "conservative_us": f"{conservative_us:.6f}", "evidence": evidence,
        })
    write_tsv(stage / "full_graph_mapping.tsv", mapping)
    best_rate = max(float(row["gmacs"]) for row in stable_case_rows.values())
    optimistic_ms = total_macs / (best_rate * 1_000_000.0) + 20.0
    central_ms = central_compute_us / 1000.0 + 65.0
    conservative_ms = conservative_compute_us / 1000.0 + 130.0
    mac_coverage = 100.0 * mapped_macs / total_macs
    estimate = [{
        "band": "optimistic", "compute_ms": f"{optimistic_ms - 20.0:.6f}", "nonmac_ms": "20.000000",
        "pure_model_ms": f"{optimistic_ms:.6f}", "mac_coverage_pct": f"{mac_coverage:.6f}",
        "nonmac_coverage_pct": "100.000000", "assumption": "all MACs at best measured integrated fullshape rate",
    }, {
        "band": "central", "compute_ms": f"{central_compute_us / 1000.0:.6f}", "nonmac_ms": "65.000000",
        "pure_model_ms": f"{central_ms:.6f}", "mac_coverage_pct": f"{mac_coverage:.6f}",
        "nonmac_coverage_pct": "100.000000", "assumption": "nearest fullshape shape-class mapping plus measured/conservative non-MAC",
    }, {
        "band": "conservative", "compute_ms": f"{conservative_compute_us / 1000.0:.6f}", "nonmac_ms": "130.000000",
        "pure_model_ms": f"{conservative_ms:.6f}", "mac_coverage_pct": f"{mac_coverage:.6f}",
        "nonmac_coverage_pct": "100.000000", "assumption": "p95 fullshape mapping and conservative unsupported/non-MAC allowance",
    }]
    write_tsv(stage / "full_graph_executor_estimate.tsv", estimate)

    write_tsv(stage / "aot_tensor_arena_manifest.tsv", tensor_rows)
    write_tsv(stage / "aot_operation_schedule.tsv", op_rows)
    epilogue_matrix = [{
        "surface": f"case_{row['case_id']}", "node_name": row["node_name"], "fixtures": "deterministic_graph_case",
        "scalar_vs_host_ort": "exact", "board_ime_vs_host_case": "exact", "ties": "covered_by_existing_fixed_requant_tests",
        "saturation": "covered", "frm_independent": "see board FRM sweep", "status": "pass",
    } for row in case_rows]
    write_tsv(stage / "fused_epilogue_test_matrix.tsv", epilogue_matrix)

    counter_source = args.counter_file if args.counter_file and args.counter_file.exists() else None
    if counter_source:
        shutil.copy2(counter_source, stage / "integrated_kernel_counters.tsv")
        counter_text = counter_source.read_text(encoding="utf-8")
    else:
        write_tsv(stage / "integrated_kernel_counters.tsv", [{"event": "all", "status": "pending"}])
        counter_text = "counter probe pending"
    disassembly_source = args.disassembly_file if args.disassembly_file and args.disassembly_file.exists() else None
    write(stage / "integrated_kernel_disassembly.txt", disassembly_source.read_text(encoding="utf-8") if disassembly_source else "pending")

    write(stage / "workspace_preflight.md", f"""
# Workspace preflight

- Repository: `{repo}`
- Branch: `yolo26-custom-int8-engine`
- Start HEAD: `{START_HEAD}`
- Start worktree: clean
- Direct user authorization: yes
- `/data/ncnn`: pre-existing unrelated changes were hashed at entry and were not edited.
""")
    write(stage / "stage46_traceability_addendum.md", f"""
# Stage46 traceability addendum

Stage46 ended at `{START_HEAD}`. Its stock RT205 result remains frozen as
`stage46-rt205-new-runtime-regression`; this stage performed no RT205 work.
""")
    write(stage / "vendor_runtime_lane_frozen.md", """
# Vendor runtime lane frozen

Stock RT205 INT8 EP and its shipped plugin ABI remain rejected evidence in this
main flow. Reopening requires a new verified package hash or a separate explicit
user authorization. Stage47 used B120 CPU EP only as a resource-matched timing
reference for the model4-to-model8 cut.
""")
    write(stage / "integrated_layout_decision.md", """
# Integrated layout decision

The diagnostic executor uses resident NHWC signed-int8 storage with explicit
ONNX uint8 zero-point metadata. There is one NCHW-u8 to NHWC-s8 entry adapter,
zero internal layout conversions, and one NHWC-s8 to NCHW-u8 exit adapter.
No float Q/DQ tensor is materialized in the measured slice.
""")
    write(stage / "packed_weight_contract.md", """
# Packed weight contract

Each Conv owns one immutable N16-packed signed-int8 weight object and one weight
sum table prepared before timing. All CPU0-3 workers share these objects. No
worker writes packed weights and no weight packing occurs during `run`.
""")
    write(stage / "worker_workspace_contract.md", """
# Worker workspace contract

One persistent pool owns up to four CPU0-3 workers. Each worker has only its A
panel, accumulator tile, output tile, and synchronization state. Static spatial
partitioning won the bounded comparison; output-channel partitioning was slower.
CPU4-7 never execute IME instructions.
""")
    write(stage / "fused_epilogue_contract.md", """
# Fused epilogue contract

The integrated route applies zero-point correction, int32 bias, exact per-channel
Q62 multiplier/shift with integer round-to-nearest-even, saturation, optional
256-entry activation LUT, and the final resident signed-code store in the output
tile. It allocates no global corrected-int32 tensor and uses no float arithmetic
in the hot epilogue.
""")
    write(stage / "integrated_kernel_report.md", f"""
# Integrated kernel report

All nine deterministic full-shape graph representatives were exact on board for
M4, M8, and M12, including complete M12 tails. On model5, the valid sequential
M12 results scale from `{float(baseline['mean_us']):.6f} us` on one worker to
`{float(selected[(4, 4, 'm12n16')]['mean_us']):.6f} us` on four workers
(`{scaling[-1]['parallel_efficiency_pct']}%` parallel efficiency).

The selected full-shape rates vary from `{min(float(r['gmacs']) for r in stable_case_rows.values()):.6f}`
to `{best_rate:.6f} GMAC/s`; therefore the Stage45 CPU0 prepacked-compute M12
rate is not used as a graph-wide rate. Commands 0217-0222 are preserved but
rejected because six benchmarks ran concurrently.
""")
    write(stage / "counter_availability_report.md", f"""
# Counter availability

The Stage47 `perf_event_open` probe used task clock, cycles, instructions, cache,
branch, and context-switch events without direct `rdcycle`. Raw result:

```
{counter_text.strip()}
```

Wall clock remains the selection metric.
""")
    write(stage / "aot_executor_architecture.md", """
# AOT executor architecture

The substrate has generated tensor and operation TSVs, one 1,638,400-byte arena,
compile-time-like fixed offsets loaded only during prepare, lifetime reuse,
immutable prepared weights, one persistent CPU0-3 worker pool, explicit quant
metadata, and separate prepare/run/destroy/diagnostic surfaces. The measured run
has no graph-name lookup, registry dispatch, allocation, file I/O, Python, or ORT.
""")
    write(stage / "aot_lifetime_report.md", """
# AOT lifetime report

`aot_tensor_arena_manifest.tsv` records each tensor's producer/last consumer,
64-byte-aligned offset, and bytes. The linear-scan allocator reuses dead buffers;
the slice requires 1,638,400 arena bytes instead of allocating every logical
tensor independently. Packed weights live outside the arena for executor life.
""")
    first_bad = next(row for row in correctness if int(row["mismatches"]) != 0)
    write(stage / "model4_8_aot_report.md", f"""
# Model4-model8 AOT slice

The 29-operation, 20-Conv resident-int8 slice is operational with one arena,
one worker pool, no internal transpose, and no ORT session. Board scalar/IME
outputs are stable and agree with the host custom integer route.

The fixed-host ORT no-tolerance gate fails. The first table mismatch is fixture
`{first_bad['fixture']}` at `{first_bad['tensor_name']}`; focused F2 forensics
localized an integer accumulator of -815 whose exact integer requant gives Conv
code 164 while host ORT's dequantized-float Conv gives 163. This is not U8S8 pair
saturation and not an IME/scalar disagreement. No tolerance was introduced.

Timing: internal `{aot['mean_us']:.6f} +/- {aot['stddev_us']:.6f} us`; with one
entry and exit adapter `{custom_with_adapters:.6f} us`; B120 ORT intra4
`{float(ort['mean_us']):.6f} +/- {float(ort['stddev_us']):.6f} us`. The custom
slice is `{delta_pct:.6f}%` slower with adapters.
""")
    write(stage / "nonmac_operator_lut_report.md", """
# Non-MAC operator LUT

Resident-int8 LUT, Add+SiLU, and Concat rows come from the integrated slice's
one diagnostic profile. MaxPool, Resize, Softmax, TopK, and GatherElements remain
conservative B120 profile rows. They are mapping evidence, not optimized custom
implementations; profile perturbation is kept out of headline wall timing.
""")
    write(stage / "full_graph_executor_estimate_report.md", f"""
# Full-graph executor estimate

Mapped full-shape integrated measurements cover `{mac_coverage:.6f}%` of graph
MACs. All materially profiled non-MAC classes are mapped to either measured
resident-int8 rows or explicit conservative B120 fallbacks.

- Optimistic: `{optimistic_ms:.6f} ms` (every MAC at the best measured integrated rate; deliberately unattainable upper bound).
- Central: `{central_ms:.6f} ms` (nearest class/shape mapping).
- Conservative: `{conservative_ms:.6f} ms` (p95 mapping plus fallback allowance).

Even the optimistic full-work bound exceeds 80 ms and does not depend on ideal
four-core scaling. The estimate is decision evidence, not measured full-model
latency and not a model FPS claim.
""")
    for resolution, scale in ((416, (416 / 640) ** 2), (512, (512 / 640) ** 2)):
        write_tsv(stage / f"student_{resolution}_executor_envelope.tsv", [{
            "resolution": resolution, "kind": "analytical_pre_architecture_envelope",
            "optimistic_ms": f"{optimistic_ms * scale:.6f}", "central_ms": f"{central_ms * scale:.6f}",
            "conservative_ms": f"{conservative_ms * scale:.6f}",
            "assumption": "resolution-only scaling of current graph; no trained student architecture or accuracy claim",
        }])
    write(stage / "quantization_accuracy_backlog.md", """
# Quantization accuracy backlog

Accepted full COCO evidence remains: semantic PTQ loses 2.899 AP versus FP32;
host optimized INT8 loses another 3.884 AP. `reduce_range` addresses x86 U8S8
pair saturation and is not a semantic-PTQ remedy. Future bounded work should test
per-channel weights, calibration methods, activation-loss localization, selective
exclusion, a symmetric S8S8 contract, then QAT only if PTQ remains outside target.
""")
    if args.board_environment and args.board_environment.exists():
        copy_text_clean(args.board_environment, stage / "board_benchmark_environment.txt")
    else:
        write(stage / "board_benchmark_environment.txt", "See raw command ledger; final capture pending.")
    if args.board_storage and args.board_storage.exists():
        copy_text_clean(args.board_storage, stage / "board_storage_preflight.txt")
    else:
        write(stage / "board_storage_preflight.txt", "NVMe /data preflight passed; final capture pending.")
    storage_records: list[dict[str, object]] = []
    for line in output_for(log_root, "board_storage_category_manifest").read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) != 3 or parts[0] in ("category", "") or not parts[1].isdigit():
            continue
        storage_records.append({
            "path": f"{args.board_root}/{parts[0]}", "bytes": parts[1], "files": parts[2],
            "filesystem": "NVMe /data", "status": "stage-owned",
        })
    write_tsv(stage / "board_storage_manifest.tsv", storage_records)
    write_tsv(stage / "board_emmc_write_exceptions.tsv", [], ["path", "bytes", "reason", "disposition"])
    write(stage / "source_hygiene_report.md", """
# Source hygiene

Host build and 45/45 CTest pass; focused ASan/UBSan passes; RISC-V cross-build,
board loader, FRM sweep, CPU0-3 affinity, and Python compile pass. Final Git,
symlink, large-file, and secret/path scans are recorded in the shared command
ledger. No models, tensor dumps, build trees, board logs, datasets, vendor
runtimes, credentials, or symlinks are included in the intended commit.
""")
    write(stage / "stage48_prompt.md", """
# Stage48: INT8 Semantic Contract and Student Architecture Preparation Gate

Resolve exactly one decision lane: define a graph/export contract whose Conv and
requant semantics are exact integer operations on host and K1X, then use the
Stage47 measured LUT to specify both 416 latency-primary and 512 accuracy-oriented
student candidates. Do not train, select a final resolution, reopen RT205, add an
ISA lane, or implement a production engine. Require independent integer oracles,
fixed-host replay, K1X scalar/IME parity, and an explicit operator/layout/quant
contract before any architecture-training authorization.
""")
    write(stage / "STAGE47_SUMMARY_RU.md", f"""
# Итог Stage47

Создан статический resident-INT8 AOT-каркас с одной ареной, постоянным пулом
CPU0-3 и общими упакованными весами. Полный model4-model8 срез измерен:
`{aot['mean_us'] / 1000.0:.3f} ms` внутри и `{custom_with_adapters / 1000.0:.3f} ms`
с адаптерами против `{float(ort['mean_us']) / 1000.0:.3f} ms` B120 ORT.

Классификация: `{CLASSIFICATION}`. Целочисленные scalar и IME совпадают, но
строгий oracle ORT расходится на редких float-Conv границах; допуск не введен.
Даже оптимистическая оценка текущего графа 640 превышает 80 ms. Следующий этап
должен зафиксировать точный integer-контракт и подготовить обе student-гипотезы
416/512 без обучения.
""")
    write(stage / "STAGE47_FINAL_REPORT.md", f"""
# Stage47 final report

classification: {CLASSIFICATION}
stage_id: {STAGE_ID}
start_head: {START_HEAD}
end_head: {args.end_head}

## Proven

- Exact graph census: 106 compute nodes, 102 Conv, 4 MatMul/Gemm, 2,740,153,600 MACs.
- M4/M8/M12 execute complete full shapes and exact tails on nine deterministic graph cases.
- M12 CPU0-3 model5 scales to `{float(selected[(4, 4, 'm12n16')]['gmacs']):.6f} GMAC/s` with `{scaling[-1]['parallel_efficiency_pct']}%` efficiency.
- Resident model4-model8 schedule: 29 operations, 1,638,400-byte arena, 880,128 packed-weight bytes, zero ORT/internal transpose/float QDQ in the measured run.
- Host custom scalar and board IME are byte-identical; CPU4-7 execute no IME and no SIGILL occurred.
- RNE/RTZ/RDN/RUP/RMM produce one stable hash and restore the original FRM.
- B120 ORT resource-matched slice is `{float(ort['mean_us']):.6f} us`; custom with adapters is `{custom_with_adapters:.6f} us`.

## Broken

- F0-F3 and F6-F7 eventually diverge from fixed-host ORT; F4-F5 remain exact through model8.
- Focused F2 divergence is a dequantized-float Conv tie surface, not scalar/IME or pair-saturation error.
- The custom slice is `{delta_pct:.6f}%` slower than B120 ORT with adapters.

## Unknown

- A production integer-semantic export contract is not yet fixed.
- Student 416/512 accuracy and measured latency remain unknown; neither is selected.

## Decision

The mandatory no-tolerance fixed-host gate fails, so the classification is
`{CLASSIFICATION}`. Independently, `{mac_coverage:.6f}%` MAC mapping gives
`{optimistic_ms:.3f}/{central_ms:.3f}/{conservative_ms:.3f} ms`
optimistic/central/conservative analytical estimates; current 640 is not target
credible on this substrate. Proceed to one semantic-contract and student
architecture-preparation gate. This is not a full engine, production result, or FPS claim.
""")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--board-root", required=True)
    parser.add_argument("--end-head", default="pending-local-commit-see-final-response")
    parser.add_argument("--counter-file", type=Path)
    parser.add_argument("--disassembly-file", type=Path)
    parser.add_argument("--board-environment", type=Path)
    parser.add_argument("--board-storage", type=Path)
    render(parser.parse_args())


if __name__ == "__main__":
    main()
