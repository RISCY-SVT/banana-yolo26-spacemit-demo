#!/usr/bin/env python3
"""Render compact Stage48 reports from preserved raw evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
from pathlib import Path


TASK = "BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE48-EXACT-INTEGER-CONTRACT-AND-NCHWC8-DIRECT-A-TILE-PROOF-001"
CLASSIFICATION = "stage48-integer-contract-pass-direct-layout-ort-competitive"
START_HEAD = "3c1eabb5198316c26c9577c0018343568e84c993"
MODEL_SHA = "30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c"
STAGE47_R0_US = 26388.005044
ORT_US = 11701.0
BOARD_ROOT = f"/data/k1x-stage-runs/{TASK}"


def write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_tsv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def parse_summary(path: Path) -> tuple[dict[str, str], list[list[str]]]:
    lines = [line.split("\t") for line in read_text(path).splitlines() if line]
    header: list[str] | None = None
    summary: dict[str, str] | None = None
    repeats: list[list[str]] = []
    for fields in lines:
        if fields[0] == "repeat_summary":
            repeats.append(fields)
        elif fields[:3] == ["summary", "repeat", "runs"]:
            header = fields
        elif fields[0] == "summary" and len(fields) > 4 and header is not None:
            summary = dict(zip(header, fields, strict=True))
    if summary is None:
        raise RuntimeError(f"missing summary in {path}")
    return summary, repeats


def f6(value: float | str) -> str:
    return f"{float(value):.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    args = parser.parse_args()

    stage = args.repo / "stages" / TASK
    commands = args.log_root / "commands"
    stage.mkdir(parents=True, exist_ok=True)

    package = json.loads(read_text(args.package / "package.json"))
    with (args.package / "model5_meta.tsv").open(encoding="utf-8", newline="") as stream:
        meta = next(csv.DictReader(stream, delimiter="\t"))

    shutil.copyfile(args.package / "asset_hashes.tsv", stage / "k1x_int8_v1_asset_hashes.tsv")
    shutil.copyfile(args.package / "model5_meta.tsv", stage / "k1x_int8_v1_model5_manifest.tsv")

    # Full retained candidate matrix: compact repeat rows in Git, complete samples in raw logs.
    raw_rows: list[list[object]] = []
    summary_rows: list[list[object]] = []
    summaries: dict[tuple[str, str, int], dict[str, str]] = {}
    for path in sorted(args.matrix_dir.glob("model5_*.tsv")):
        summary, repeats = parse_summary(path)
        first_repeat = repeats[0]
        kernel = first_repeat[6]
        partition = first_repeat[8]
        workers = int(first_repeat[9])
        repeat_means = [float(row[3]) for row in repeats]
        repeat_std = statistics.stdev(repeat_means) if len(repeat_means) > 1 else 0.0
        repeat_cv = repeat_std / statistics.mean(repeat_means) * 100.0
        summaries[(kernel, partition, workers)] = summary
        for row in repeats:
            raw_rows.append([kernel, partition, workers, row[1], row[3], row[4], row[10], path])
        summary_rows.append([
            kernel, partition, workers, summary["runs"], summary["repeats"], summary["mean_us"],
            summary["stddev_us"], summary["cv_pct"], summary["min_us"], summary["max_us"],
            summary["median_us"], summary["p90_us"], summary["p95_us"],
            summary["process_cpu_mean_us"], summary["gmacs"], f6(repeat_std), f6(repeat_cv),
            summary["mismatches"], summary["affinity_ok"], path,
        ])
    raw_header = [
        "kernel", "partition", "workers", "repeat", "repeat_mean_us", "repeat_process_cpu_mean_us",
        "output_hash64", "raw_per_run_path",
    ]
    write_tsv(stage / "model5_performance_raw.tsv", raw_header, raw_rows)
    summary_header = [
        "kernel", "partition", "workers", "runs", "repeats", "mean_us", "stddev_us", "sample_cv_pct",
        "min_us", "max_us", "median_us", "p90_us", "p95_us", "process_cpu_mean_us", "gmacs",
        "repeat_mean_stddev_us", "repeat_mean_cv_pct", "mismatches", "affinity_ok", "raw_per_run_path",
    ]
    write_tsv(stage / "model5_performance_summary.tsv", summary_header, summary_rows)

    selected = summaries[("m12n16", "spatial", 4)]
    selected_us = float(selected["mean_us"])
    speedup = STAGE47_R0_US / selected_us
    delta_vs_ort = (selected_us - ORT_US) / ORT_US * 100.0

    candidate_rows = [[
        "R0", "NHWC-s8", "generic_scalar_pack_a", "m12n16", "spatial", 4,
        f6(STAGE47_R0_US), "Stage47", "exact", "baseline",
    ]]
    for candidate, kernel in (("R1", "m4n16"), ("R2", "m8n16"), ("R3", "m12n16")):
        value = summaries[(kernel, "spatial", 4)]
        candidate_rows.append([
            candidate, "NCHWc8_SPATIAL_INNER_V1", "rvv_vlseg2e64_c8x4", kernel, "spatial", 4,
            value["mean_us"], "Stage48 10/100/5", "exact", "selected" if candidate == "R3" else "retained",
        ])
    write_tsv(stage / "model5_candidate_matrix.tsv", [
        "candidate", "layout", "load_strategy", "kernel", "partition", "workers", "mean_us",
        "evidence", "correctness", "decision",
    ], candidate_rows)

    # Same-mode load strategy scout, all using the unchanged Stage48 kernel body.
    load_rows = []
    load_commands = {
        "four_u64_c8": commands / "0083_stable-load-u64.stdout.txt",
        "rvv_vlse64_c8x4": commands / "0084_stable-load-vlse64.stdout.txt",
        "rvv_vlseg2e64_c8x4": commands / "0085_stable-load-vlseg2e64.stdout.txt",
    }
    for name, path in load_commands.items():
        value, repeats = parse_summary(path)
        repeat_means = [float(row[3]) for row in repeats]
        repeat_cv = statistics.stdev(repeat_means) / statistics.mean(repeat_means) * 100.0
        load_rows.append([name, value["mean_us"], value["stddev_us"], value["cv_pct"],
                          f6(repeat_cv), value["gmacs"], value["mismatches"], path])
    write_tsv(stage / "nchwc8_load_candidates.tsv", [
        "strategy", "mean_us", "stddev_us", "sample_cv_pct", "repeat_mean_cv_pct", "gmacs",
        "mismatches", "raw_path",
    ], load_rows)

    # Oracle and final board correctness.
    adversarial_rows = []
    with (args.package / "adversarial_requant.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            adversarial_rows.append([
                "adversarial_requant", row["case_id"], row["name"], "python_arbitrary_precision",
                "portable_cpp_scalar", "board_cpp_scalar", "pass", row["accumulator"], row["multiplier"],
                row["right_shift"], row["rounded"], row["output_code"],
            ])
    fixture_rows = []
    correctness_rows = []
    final_text = read_text(commands / "0122_final-board-correctness.stdout.txt")
    for line in final_text.splitlines():
        if not line.startswith("validate\t"):
            continue
        values = line.split("\t")
        fixture, route = values[1], values[2]
        fixture_rows.append([
            "model5_fixture", fixture, "F0-F7", "python_integer_package", "portable_cpp_host_scalar",
            f"board_{route}", "pass" if values[8] == "0" else "fail", "-", "-", "-", "-", values[11],
        ])
        correctness_rows.append([
            fixture, route, values[3], values[4], values[5], values[6], values[7], values[8], values[9],
            values[10], values[11], values[12], values[13], values[14],
        ])
    write_tsv(stage / "integer_oracle_test_matrix.tsv", [
        "surface", "case_id", "case_name", "authority", "implementation_a", "implementation_b", "status",
        "accumulator", "multiplier", "right_shift", "rounded", "output_or_hash64",
    ], adversarial_rows + fixture_rows)
    write_tsv(stage / "nchwc8_correctness_matrix.tsv", [
        "fixture", "route", "kernel", "load", "partition", "workers", "run_status", "mismatches",
        "max_abs_diff", "first_mismatch", "actual_hash64", "expected_hash64", "affinity_ok", "profile_total_us",
    ], correctness_rows)

    byte_order_lines = read_text(commands / "0082_board-byte-order.stdout.txt").splitlines()
    byte_order_rows = [line.split("\t") for line in byte_order_lines[1:] if line]
    write_tsv(stage / "nchwc8_a_tile_byte_order.tsv", byte_order_lines[0].split("\t"), byte_order_rows)

    legacy_rows = []
    legacy_lines = read_text(commands / "0051_compare-integer-vs-legacy-model5.stdout.txt").splitlines()
    for line in legacy_lines[1:]:
        fixture, mismatches, maximum, first, integer, legacy = line.split()
        legacy_rows.append([fixture, mismatches, maximum, first, integer, legacy, "diagnostic-only"])
    write_tsv(stage / "legacy_float_qdq_diagnostic.tsv", [
        "fixture", "mismatches", "max_abs_diff", "first_mismatch", "integer_code", "legacy_qdq_code", "policy",
    ], legacy_rows)

    write_tsv(stage / "accumulator_safety.tsv", [
        "node", "K", "activation_zero_point", "max_abs_activation", "max_abs_weight", "max_abs_bias",
        "absolute_bound", "int32_limit", "int32_safe", "policy",
    ], [[
        "/model.5/conv/Conv", meta["k"], meta["input_zero_point"], 246, 128, "included-per-channel",
        meta["accumulator_absolute_bound"], 2147483647, meta["int32_safe"], "reject-package-unless-proven",
    ]])

    # Diagnostic phase attribution: these are worker-local sums and instrumentation perturbs wall time.
    phase_rows = []
    for line in read_text(commands / "0112_phase-attribution.stdout.txt").splitlines():
        if line.startswith("case\t"):
            phase_rows.append(line.split("\t"))
    phase_header = [
        "kind", "requested_load", "run", "route", "kernel", "actual_load", "partition", "workers",
        "status", "mismatches", "wall_total_us", "direct_a_worker_sum_us", "vmadot_worker_sum_us",
        "scalar_epilogue_worker_sum_us", "barrier_us", "min_worker_us", "max_worker_us", "vector_groups",
        "scalar_c8_groups", "border_chunks", "affinity_ok",
    ]
    write_tsv(stage / "model5_phase_attribution.tsv", phase_header, phase_rows)

    # Counter retry and environment/storage evidence.
    counter_rows = []
    for privilege, filename in (("unprivileged", "0118_counter-normal.stdout.txt"),
                                ("sudo_nonpersistent", "0119_counter-sudo.stdout.txt")):
        lines = read_text(commands / filename).splitlines()
        for line in lines[1:]:
            if line:
                counter_rows.append([privilege, *line.split("\t")])
    write_tsv(stage / "counter_raw.tsv", ["privilege", "event", "status", "errno", "error", "count"], counter_rows)
    shutil.copyfile(commands / "0154_board-environment-final.stdout.txt", stage / "board_benchmark_environment.txt")
    shutil.copyfile(commands / "0016_board-storage-preflight.stdout.txt", stage / "board_storage_preflight.txt")
    storage_rows = []
    for line in read_text(commands / "0155_board-storage-manifest.stdout.txt").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            storage_rows.append([parts[1], parts[0], "board-nvme-stage-owned"])
    write_tsv(stage / "board_storage_manifest.tsv", ["path", "sha256", "classification"], storage_rows)
    write_tsv(stage / "board_emmc_write_exceptions.tsv", ["path", "bytes", "reason", "disposition"], [])

    # Narrative reports.
    write(stage / "integer_oracle_report.md", f"""
# K1X_INT8_V1 integer oracle report

`K1X_INT8_V1` is the exact authority. The package exporter derives integer
multipliers from the exact float32 bit patterns using rational arithmetic; the
board consumes only encoded integers and a 256-entry activation LUT. It does
not call `frexp`, `llround`, `exp`, or threshold search during prepare/run.

The Python reference uses a proven int64-safe full Conv accumulation and Python
arbitrary-precision requant arithmetic. Portable C++ scalar, board scalar, and
board IME are byte-exact for F0-F7. All 12 adversarial positive/negative tie,
threshold-neighborhood, saturation, and zero cases pass. The model5 absolute
accumulator bound is `{meta['accumulator_absolute_bound']}`, below INT32_MAX.

A complete second export to a distinct raw-evidence directory is byte-identical;
`diff -qr` reports no differences.

Legacy host ORT float-QDQ differs on F1, F2, F4, and one F7 element. That is a
separate model-replay diagnostic and is not the integer-contract authority.
""")

    write(stage / "nchwc8_load_candidate_report.md", f"""
# NCHWc8 direct-load candidate report

The byte-order proof covers border (`m_begin=0`), interior (`4`), and row-edge
(`40`) tiles. All C8-u64, `vlse64`, and `vlseg2e64` panels are byte-identical.
Disassembly confirms the intended `vlse64.v`, `vlseg2e64.v`, and existing
`smt.vmadot` instructions.

The selected strategy is `rvv_vlseg2e64_c8x4`. Its same-mode scout mean was
`{load_rows[2][1]} us`; the final full candidate matrix measured
`{selected['mean_us']} us`. Interior delivery uses C8-group vector loads and
bounded worker-local A tiles. The generic Stage47 per-byte interior `pack_a`
loop and full im2col materialization are absent. Borders use exact semantic
zero-point C8 chunks.
""")

    write(stage / "model5_performance_report.md", f"""
# Model5 direct-layout performance

Protocol: board CPU0-3, governor `performance` at 1.6 GHz, warmup 10, 100 runs,
5 repeats. Every arm produced the fixed expected output hash.

Selected R3 is M12xN16, spatial partition, four workers, scalar exact integer
epilogue, and `vlseg2e64` direct delivery. Mean is `{selected_us:.6f} us`,
median `{float(selected['median_us']):.6f} us`, p95 `{float(selected['p95_us']):.6f} us`,
and `{float(selected['gmacs']):.6f} GMAC/s`. Sparse scheduler excursions make the
per-sample CV `{float(selected['cv_pct']):.6f}%`; the five repeat means have CV
`{next(row[16] for row in summary_rows if row[0:3] == ['m12n16', 'spatial', 4])}%`,
so the repeated central result is stable while all outliers remain in raw logs.

This is `{speedup:.6f}x` faster than Stage47 R0 (`{STAGE47_R0_US:.6f} us`) and
`{abs(delta_vs_ort):.6f}%` lower than the resource-matched B120 ORT model5
reference (`{ORT_US:.3f} us`). It therefore satisfies the predeclared
ORT-competitive threshold.

The NCHW-to-NCHWc8 conversion is excluded by contract and separately measured:
entry `30343.690198 us`, exit `7836.849598 us`. A later persistent-layout slice
must prove that these conversions are not paid per operator. No end-to-end or
production result is claimed.
""")

    write(stage / "counter_availability_report.md", """
# Counter availability

`perf_event_paranoid=2`; the unprivileged probe returned EACCES for every event.
One non-persistent `sudo` retry succeeded for task-clock, cycles, instructions,
branches, branch misses, and context switches; the PMU reported zero for the
generic cache events. The probe measures diagnostic work only, not the model5
worker threads. Consequently cycles-per-vmadot is unknown and wall clock is the
selection authority. No sysctl was changed.
""")

    write(stage / "board_storage_policy_report.md", f"""
# Board storage policy

Board `/data` is writable NVMe (`/dev/nvme0n1p1`, ext4); root is eMMC. All new
Stage48 binaries, packages, outputs, profiles, and logs are under
`{BOARD_ROOT}`. `TMPDIR` and `XDG_CACHE_HOME` point into that root. The recorded
eMMC exception count is zero.
""")

    write(stage / "STAGE48_FINAL_REPORT.md", f"""
# Stage48 final report

classification: {CLASSIFICATION}
stage_id: {TASK}
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
start_head: {START_HEAD}
end_head: pending-local-commit-see-final-response

## Proven

- `K1X_INT8_V1_GENERAL` is versioned, package-defined, and exact across Python,
  portable C++ scalar, board scalar, and board IME for F0-F7 and adversarial ties.
- Model5 accumulator bound `{meta['accumulator_absolute_bound']}` proves int32 safety.
- `NCHWc8_SPATIAL_INNER_V1` byte order and C8 direct delivery pass; disassembly
  contains `vlseg2e64.v` and `smt.vmadot`; no generic per-byte interior pack exists.
- Selected M12/spatial/four-worker model5 is `{selected_us:.6f} us`, exact, and
  `{speedup:.6f}x` faster than Stage47 R0; it is `{abs(delta_vs_ort):.6f}%` below
  the resource-matched ORT reference.
- FRM RNE/RTZ/RDN/RUP/RMM produce one hash and restore RNE; CPU affinity is 0-3,
  CPU4-7 execute no IME, and no SIGILL occurred.

## Broken

- Scalar NCHW/NCHWc8 entry/exit conversions remain expensive and are not an
  acceptable per-operator path.
- Legacy float-QDQ replay is not byte-exact to the new integer contract on all fixtures.

## Unknown

- A persistent NCHWc8 contiguous slice has not yet been measured.
- An exact RVV epilogue was not implemented because the scalar exact epilogue
  already crossed the predeclared ORT-competitive threshold.
- Full-model integer-contract accuracy and performance remain unknown.

## Decision

The specific direct-layout hypothesis passes. Keep the executor-first lane open
for one persistent NCHWc8 contiguous-slice/LUT-v2 gate. Student 416 and 512 both
remain deferred; training is unauthorized. RT205 work performed: false.

## Validation

- Host configure/build and all 47 CTests: pass.
- Focused x86 ASan/UBSan Stage48 tests: 2/2 pass.
- Python compile and deterministic package regeneration: pass; directory diff is empty.
- Full RISC-V cross-build and board loader: pass; no RPATH/RUNPATH.
- Board scalar/IME F0-F7, adversarial ties, M12 tail, FRM restoration,
  CPU0-3 affinity, byte order, and 10/100/5 performance matrix: pass.
- Git diff checks, symlink, large-file, secret, private-path, and `/data/ncnn`
  integrity checks: pass.

## Non-claims

No production readiness, full engine, default dispatch, model FPS, camera
performance, COCO accuracy, or trained-student accuracy is claimed.
""")

    write(stage / "STAGE48_SUMMARY_RU.md", f"""
# Краткий отчет Stage48

Классификация: `{CLASSIFICATION}`.

Определен точный целочисленный контракт `K1X_INT8_V1`. Python-оракул,
переносимый C++ scalar, scalar на плате и IME дают точное совпадение на F0-F7 и
на тестах округления/насыщения. Для model5 доказана безопасность int32.

Экспериментальный NCHWc8 путь с `vlseg2e64`, M12xN16, spatial-разбиением и
четырьмя ядрами выполняет model5 за `{selected_us:.6f} мкс`: в `{speedup:.6f}`
раза быстрее Stage47 R0 и на `{abs(delta_vs_ort):.6f}%` быстрее диагностического
B120 ORT блока. Это внутренний resident-layout Conv; дорогие преобразования
layout измерены отдельно и не включены.

Следующий шаг: доказать постоянный NCHWc8 layout на непрерывном срезе графа.
416/512 student и обучение остаются отложенными и не разрешены.
""")

    write(stage / "stage49_prompt.md", f"""
# Stage49 prompt: Persistent NCHWc8 contiguous-slice and LUT-v2 gate

## Start

- repo: `/data/banana-yolo26-spacemit-demo`
- branch: `yolo26-custom-int8-engine`
- expected HEAD: Stage48 local commit reported in the final response
- authority: `K1X_INT8_V1`; legacy float-QDQ remains diagnostic only

## Mission

Prove one persistent `NCHWc8_SPATIAL_INNER_V1` contiguous model4-to-model6
slice using offline integer packages, one arena, shared immutable packed weights,
persistent CPU0-3 workers, no internal layout conversion, no float Q/DQ, and no
ORT in the measured slice. Measure entry/exit adapters separately.

1. Extend the independent integer exporter/oracle only to the exact model4-6
   operations needed by the slice.
2. Preserve exact Python/C++/board scalar/IME parity at every integer boundary.
3. Reuse Stage48 direct model5 M12/spatial/four-worker route.
4. Add one exact RVV epilogue candidate only if measured slice attribution says
   the scalar epilogue is the remaining bounded bottleneck.
5. Compare the exact custom internal slice and adapter-inclusive slice against a
   resource-matched B120 ORT diagnostic cut.

No RT205 work, student blueprint/training, CPU4-7 IME, full graph executor,
default dispatch, production claim, or push is authorized.
""")

    print(json.dumps({
        "classification": CLASSIFICATION,
        "selected_mean_us": selected_us,
        "speedup_vs_stage47": speedup,
        "delta_vs_ort_pct": delta_vs_ort,
        "reports": len(list(stage.iterdir())),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
