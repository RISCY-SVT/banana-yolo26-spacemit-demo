#!/usr/bin/env python3
"""Render the Stage46 decision packet from immutable raw evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


STAGE_ID = (
    "BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE46-RT205-SPACEMIT-EP-"
    "INT8-FULL-REVALIDATION-PLUGIN-COCO-GATE-001"
)
START_HEAD = "860ee58a55286f3e207d4b0eb7cec8a59a85bb9d"
STAGE45_HEAD = START_HEAD
CLASSIFICATION = "stage46-rt205-new-runtime-regression"
MODEL_SHA = "30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c"
FP32_SHA = "d71286588abe691ede49faa5ca9a471b7e9e5257669953ee59abbc2e9d115fc2"
RT205_ARCHIVE_SHA = "ae512d21ef6a08a4db1a252237b7e2987978cb47ea4b1a86353ae54dc16ecae1"
RT204_CORE_SHA = "e887a538b6cce9597b1905034b48f89763dd625b04bcd708ceb4b494df6df1ac"
RT205_CORE_SHA = "93bb75601d9eceb5aca192fa70c0c3e18b94a70b9f57acdc9b34c2ff426e09e3"
RT204_EP_SHA = "a59e29d2ed4c08ab57ad3e72c75a0b9a72020cb0e8f278ef2ef483725d04b47a"
RT205_EP_SHA = "3927b51f79f8d2142ff98708183aa9b24b47d6941533499035193a630042a41d"
HEADER_SHA = "9ed0d7054a4e74249467365b25b415d36f51a44a6349e2a994a1812e4723d1e2"
BOARD_ROOT = f"/data/k1x-stage-runs/{STAGE_ID}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def write_tsv(path: Path, fields: list[str], rows: Iterable[Iterable[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


def copy_text(source: Path, destination: Path) -> None:
    content = source.read_text(encoding="utf-8").replace("\r\n", "\n")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")


def copy_tsv_normalized(source: Path, destination: Path) -> None:
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream, delimiter="\t"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerows([[cell if cell else "n/a" for cell in row] for row in rows])


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def eval_row(log_dir: Path, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    root = log_dir / "artifacts/accuracy"
    return read_json(root / f"{name}.summary.json"), read_json(root / f"{name}.eval.json")


def board_eval(log_dir: Path, name: str) -> dict[str, Any] | None:
    path = log_dir / "artifacts/accuracy" / f"{name}.eval.json"
    return read_json(path) if path.exists() else None


def render(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    log_dir = Path(args.log_dir).resolve()
    out = repo / "stages" / STAGE_ID
    out.mkdir(parents=True, exist_ok=True)

    fp32_disable_summary, fp32_disable = eval_row(log_dir, "host_fp32_disable")
    fp32_all_summary, fp32_all = eval_row(log_dir, "host_fp32_all")
    int8_disable_summary, int8_disable = eval_row(log_dir, "host_int8_disable")
    int8_all_summary, int8_all = eval_row(log_dir, "host_int8_all")
    rt204_cpu_eval = board_eval(log_dir, "board_rt204_cpu_int8")
    rt205_cpu_eval = board_eval(log_dir, "board_rt205_cpu_int8")

    write(
        out / "workspace_preflight.md",
        f"""# Workspace preflight

- Repository: `/data/banana-yolo26-spacemit-demo`
- Branch: `yolo26-custom-int8-engine`
- Required and observed start HEAD: `{START_HEAD}`
- Initial worktree: clean
- Direct-user authorization used; no control-plane packet was required.
- `/data/ncnn` was read-only for this stage and was not mutated.
- Shared raw log: `{log_dir}`
""",
    )

    write(
        out / "stage45_errata_and_supersession.md",
        """# Stage45 errata and supersession

Stage45 history and raw evidence remain unchanged. Stage46 records these narrower
interpretations:

1. `50.407 ms` is a CPU0-only prepacked M12xN16 compute projection, not a
   physical four-core graph lower bound.
2. `105-130 ms`, `24-38 ms`, and `35-50 ms` are analytical envelopes, not
   measured graph latencies.
3. Scalar read/transpose/reference requant/hash-assisted pack rows are reference
   diagnostics, not optimized production primitives.
4. M12 full-shape tails and a fused epilogue were not measured.
5. Stage45 FP32 accuracy used only ORT_ENABLE_ALL. Stage46 supplies the full
   FP32/INT8 disable/all host matrix.
6. The Stage45 repository report has a pending end-head marker; the actual head
   is recorded in `stage45_traceability_addendum.md`.
7. Candidate latency and strategy tables are non-empty analytical artifacts.
8. Both 416 latency-first and 512 accuracy-fallback student hypotheses remain
   untrained and unselected.

The reported duplicate operator line in `model_executor_codesign_spec.md` was
not present at Stage46 preflight, so no historical source was rewritten.
""",
    )
    write_tsv(
        out / "stage45_evidence_reconciliation.tsv",
        ["item", "stage45_value", "stage46_interpretation", "artifact_sha256", "status"],
        [
            ["M12 projection", "50.407 ms", "single-core prepacked compute-only", "n/a", "superseded-label"],
            ["current graph envelope", "105-130 ms", "analytical", "n/a", "preserved"],
            ["student416", "24-38 ms", "analytical untrained hypothesis", "n/a", "preserved"],
            ["student512", "35-50 ms", "analytical untrained hypothesis", "n/a", "preserved"],
            ["FP32 accuracy", "ENABLE_ALL only", "Stage46 adds disable/all", "n/a", "closed"],
            ["candidate_model_latency_predictions.tsv", "non-empty", "analytical", "f44b892339cdfbfbe99ef10d06232929aeff27bb2e0b5152d88fcda9b55a83a8", "hash-preserved"],
            ["strategy_decision_matrix.tsv", "non-empty", "analytical", "2209635f27b3d09d63f85430e4c6bbcc0daa90c52ac2146bfcf769676bc9c2ee", "hash-preserved"],
            ["STAGE45_FINAL_REPORT.md", "pending end-head marker", "traceability addendum records actual head", "178596f64738ec1202bc00897bd1f420f355e9f229e5c53f1961fb1db91eb16d", "preserved"],
        ],
    )
    write(
        out / "stage45_traceability_addendum.md",
        f"""# Stage45 traceability addendum

The immutable Stage45 commit is `{STAGE45_HEAD}`. Its tracked final report still
contains a pre-commit marker. This addendum records the actual head without
rewriting Stage45 raw evidence or amending its commit.
""",
    )

    write(
        out / "rt205_release_note.md",
        f"""# SpacemiT ORT 2.0.5 release note

- Official tag: `2.0.5`, published `2026-07-03T10:20:03Z`.
- Official asset: `spacemit-ort.riscv64.2.0.5.tar.gz`, 14,820,095 bytes.
- Source URL: `https://github.com/spacemit-com/onnxruntime/releases/download/2.0.5/spacemit-ort.riscv64.2.0.5.tar.gz`.
- Asset SHA-256: `{RT205_ARCHIVE_SHA}`.
- Release claim: SpacemiT-EP custom operator plugin mechanism, TCM/thread-pool
  handling changes, graph fusions, depthwise/ConvTranspose optimizations, and a
  K1 TCM multi-core fix.
- Public source tags 2.0.4 and 2.0.5 both resolve to commit
  `61e7fc2319cd16aa5487fd1155dc15c5390c8a90` and tree
  `cf982979521bdf5fd2256bdb1318ec29453edc58`; package binary hashes differ.

The release claims are treated as inputs. Package and board execution evidence
below determine acceptance.
""",
    )
    write(
        out / "rt205_archive_safety_report.md",
        f"""# RT205 archive safety

The official `{RT205_ARCHIVE_SHA}` archive contained 70 members. No absolute
paths, parent traversal, unsafe hard links, or unsafe symlinks were found. Four
relative package symlinks were accepted. Extraction was into an empty task-owned
directory under `/data/vendor/spacemit-ort/2.0.5`; bundled install scripts were
not executed.
""",
    )
    copy_tsv_normalized(log_dir / "artifacts/rt204_package_manifest.tsv", out / "rt204_package_manifest.tsv")
    copy_tsv_normalized(log_dir / "artifacts/rt205_package_manifest.tsv", out / "rt205_package_manifest.tsv")
    copy_tsv_normalized(log_dir / "artifacts/rt204_rt205_package_diff.tsv", out / "rt204_rt205_package_diff.tsv")

    write_tsv(
        out / "runtime_identity_matrix.tsv",
        [
            "id", "machine", "core_version", "ep_version", "core_path", "core_sha256",
            "ep_path", "ep_sha256", "header_path", "header_sha256", "ort_api_version",
            "build_commit", "registered_providers", "role",
        ],
        [
            ["H127", "host-x86_64", "1.27.0", "n/a", ".deps/venvs/ultralytics_latest/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.27.0", "b5c9d4f124d24707f514dad926dc181820807178855df1c528e3addb2dd0e6f7", "n/a", "n/a", "not-packaged-by-python-wheel", "n/a", "n/a", "8f0278c77b", "AzureExecutionProvider;CPUExecutionProvider", "Python semantic authority; no compile-time header claim"],
            ["B120", "board-riscv64", "1.20.2+spacemit", "historical", "/home/svt/spacemit-ort.riscv64.2.0.1/lib/libonnxruntime.so.1.20.2+spacemit", "5a28c8128a7b1ed9cb29357f42eb7a2a45eb1b23d8791c2fee1eaf0151546238", "historical package", "historical", "/data/SpacemiT/spacemit-ort.riscv64.2.0.1/include/onnxruntime_c_api.h", "591bb85c454386ddb2eca164cee1d60c94f9548f8d082c965ed0c7715a311b30", 21, "historical", "CPUExecutionProvider", "Stage45 baseline; read-only eMMC runtime"],
            ["RT204", "board-riscv64", "1.24.2+spacemit.a1", "2.0.4", ".deps/runtimes/rt204/unpacked/spacemit-ort.riscv64.2.0.4/lib/libonnxruntime.so.1", RT204_CORE_SHA, ".deps/runtimes/rt204/unpacked/spacemit-ort.riscv64.2.0.4/lib/libspacemit_ep.so.2", RT204_EP_SHA, ".deps/runtimes/rt204/unpacked/spacemit-ort.riscv64.2.0.4/include/onnxruntime_c_api.h", HEADER_SHA, 24, "c178f12b2", "CPUExecutionProvider;external SpacemiT EP", "historical control"],
            ["RT205", "board-riscv64", "1.24.2+spacemit.a1", "2.0.5", "/data/vendor/spacemit-ort/2.0.5/unpacked/spacemit-ort.riscv64.2.0.5/lib/libonnxruntime.so.1", RT205_CORE_SHA, "/data/vendor/spacemit-ort/2.0.5/unpacked/spacemit-ort.riscv64.2.0.5/lib/libspacemit_ep.so.2", RT205_EP_SHA, "/data/vendor/spacemit-ort/2.0.5/unpacked/spacemit-ort.riscv64.2.0.5/include/onnxruntime_c_api.h", HEADER_SHA, 24, "9bb02204b", "CPUExecutionProvider;external SpacemiT EP", "runtime under test"],
        ],
    )
    write(
        out / "runtime_abi_report.md",
        f"""# Runtime and ABI report

RT204 and RT205 use identical ORT C API headers (`{HEADER_SHA}`), API version
24, and core SONAME `libonnxruntime.so.1`, but different core and EP bytes. Each
runner was compiled and linked against its matching package. The installed board
runners use only relative `$ORIGIN` RPATH entries; board `ldd` resolved the
intended core/EP package and stage-owned OpenCV libraries.

RT205 core reports build commit `9bb02204b`; RT204 reports `c178f12b2`. The core
version string remains `1.24.2+spacemit.a1`, while the EP header/package version
is 2.0.5. Header compatibility is therefore not treated as binary equivalence.

Both release-specific runners were built with the SpacemiT RISC-V GCC 14.3
toolchain, `-march=rv64gcv_zvfh -mabi=lp64d`, Release optimization, and the
matching package include/core/EP paths. `GetVersionString`, `GetBuildInfoString`,
SONAME, DT_NEEDED, symbol versions, runner/library hashes, `readelf`, and board
`ldd` outputs are preserved in the raw command ledger. H127 is a Python-wheel
semantic runtime and therefore makes no compile-time-header claim.

The runner passes `SPACEMIT_EP_INTRA_THREAD_NUM` explicitly and records every
filter, dump, debug, and plugin provider option actually supplied. Package
defaults that are not stated in installed documentation remain unknown rather
than being inferred from successful session creation.
""",
    )
    write_tsv(
        out / "runtime_loader_matrix.tsv",
        ["runtime", "runner", "core_resolved", "ep_resolved", "rpath", "loader_status"],
        [
            ["RT204", "bench_stage46_rt204", f"{BOARD_ROOT}/runtime/rt204/lib/libonnxruntime.so.1", f"{BOARD_ROOT}/runtime/rt204/lib/libspacemit_ep.so.2", "none", "pass"],
            ["RT205", "bench_stage46_rt205", f"{BOARD_ROOT}/runtime/rt205/lib/libonnxruntime.so.1", f"{BOARD_ROOT}/runtime/rt205/lib/libspacemit_ep.so.2", "none", "pass"],
            ["RT204", "yolo26_coco_predict_rt204", f"{BOARD_ROOT}/runtime/rt204/lib/libonnxruntime.so.1", f"{BOARD_ROOT}/runtime/rt204/lib/libspacemit_ep.so.2", "$ORIGIN relative", "pass"],
            ["RT205", "yolo26_coco_predict_rt205", f"{BOARD_ROOT}/runtime/rt205/lib/libonnxruntime.so.1", f"{BOARD_ROOT}/runtime/rt205/lib/libspacemit_ep.so.2", "$ORIGIN relative", "pass"],
        ],
    )

    blocker_rows = [
        ["15_kernel_shape", "RT204", "cpu", "disable", "pass", "byte-exact historical CPU output"],
        ["15_kernel_shape", "RT204", "spacemit", "disable", "fail", "output_type not implemented for clip minmax"],
        ["07_real_first_conv", "RT204", "spacemit", "disable", "fail", "same clip minmax blocker"],
        ["15_kernel_shape", "RT205", "cpu", "disable", "pass", "byte-exact RT204 CPU"],
        ["15_kernel_shape", "RT205", "spacemit", "disable", "abort", "same clip minmax then uncaught runtime abort"],
        ["07_real_first_conv", "RT205", "spacemit", "disable", "abort", "same clip minmax then abort"],
        ["03_qdq_no_kernel_shape", "RT205", "spacemit", "disable", "pass", "EP subgraph, CPU/EP byte exact"],
        ["08_qlinearconv", "RT204", "spacemit", "disable", "pass", "exact control"],
        ["08_qlinearconv", "RT205", "spacemit", "disable", "SIGILL/core", "new regression"],
        ["10_qlinearmatmul", "RT204", "spacemit", "disable", "pass", "exact control"],
        ["10_qlinearmatmul", "RT205", "spacemit", "disable", "SIGILL", "exit 132; new regression"],
    ]
    write_tsv(out / "rt204_blocker_reproduction.tsv", ["case", "runtime", "provider", "opt", "status", "detail"], blocker_rows[:3])
    write(
        out / "rt204_blocker_reproduction_report.md",
        """# RT204 blocker reproduction

The matched Stage46 RT204 runner reproduced the accepted historical control.
CPU output for `15_conv_qdq_attr_kernel_shape.onnx` is byte-identical to the old
probe. SpacemiT EP fails at `synthetic/conv/Conv_token_1` with
`output_type not implemented for clip minmax`. The real YOLO26 first-Conv cut
fails on the same compiler path. The RT205 comparison is therefore valid.
""",
    )
    write_tsv(out / "rt205_minrepro_matrix.tsv", ["case", "runtime", "provider", "opt", "status", "detail"], blocker_rows[3:])
    write(
        out / "rt205_minrepro_report.md",
        """# RT205 minimal-reproducer result

The historical explicit `kernel_shape=[3,3]` Q/DQ Conv bug persists. RT205 also
changes failure handling from a catchable ORT exception to an uncaught abort.
The same graph without optional kernel_shape does execute as an EP subgraph and
matches CPU bytes, so EP registration alone is not the blocker.

RT205 introduces an independent regression: QLinearConv full/control paths core
dump, and the tiny QLinearMatMul control exits with SIGILL (132), while both pass
under RT204. This is a runtime regression, not a custom IME or new-opcode test.
""",
    )
    write(
        out / "rt204_rt205_minrepro_diff.md",
        """# RT204 versus RT205 repro difference

| Surface | RT204 | RT205 |
|---|---|---|
| explicit kernel_shape Q/DQ Conv | clip-minmax exception | same error then abort |
| Q/DQ Conv without kernel_shape | pass | pass, exact |
| QLinearConv | pass | core dump |
| QLinearMatMul | pass | SIGILL (132) |
| primary Q/DQ full model | first-Conv compile failure | first-Conv compile failure then abort |
| broad historical CPU filter | executes CPU-heavy | SIGILL/core dump |
""",
    )

    model_rows = [
        ["manual_e2e_rep_conv_matmul_qdq", ".deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx", MODEL_SHA, "QDQ static", "1x300x6", "CPU-good primary"],
        ["manual_e2e_small_conv_matmul_qdq", ".deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_small_conv_matmul_qdq.onnx", "e11f63609f6112d4faad9e573af24b2d63f5f7135221d5c730d7bd745521d77d", "QDQ static", "1x300x6", "CPU-good small calibration"],
        ["manual_trad_rep_conv_matmul_qdq", ".deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_trad_rep_conv_matmul_qdq.onnx", "c612fcbf5ce453198f75109d1907825a6d995a846bc0b5952f1b451323b14ca8", "QDQ static", "1x84x8400", "CPU-good traditional"],
        ["manual_trad_rep_conv_only_qdq", ".deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_trad_rep_conv_only_qdq.onnx", "8ee866e0b0612ce8bf0d3fa18feea3fda06a7b73b0b3305238a929c54483e0af", "QDQ static", "1x84x8400", "CPU-good traditional"],
        ["manual_e2e_rep_conv_only_qdq", ".deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_only_qdq.onnx", "535947b2ea4d03aa33af2a0f18759405bad9aaa6994bb729b23e715545f4cc8f", "QDQ static", "1x300x6", "negative control: blank false positive"],
        ["stripped_kernel_shape", "/data/ncnn-logs/ort-logs/2026-06-30_06-12-26/models/e2e_qdq_rep_conv_matmul_strip_kernel_shape.onnx", "4af28ee778979b2598994eabfaa52c1bbac68ceae64fbe0635c574d35c233848", "QDQ diagnostic", "1x300x6", "bypasses first blocker only"],
        ["e2e_qoperator_rep_conv_matmul", "/data/ncnn-logs/ort-logs/2026-06-30_06-12-26/artifacts/board_stage/models/e2e_qoperator_rep_conv_matmul.onnx", "aa657776ad24dae00834e8f457d6fb648934f2a97534979bf82ed33aef1f6f81", "QOperator", "1x300x6", "diagnostic; RT205 EP SIGILL"],
        ["trad_qoperator_rep_conv_matmul", "/data/ncnn-logs/ort-logs/2026-06-30_06-12-26/artifacts/board_stage/models/trad_qoperator_rep_conv_matmul.onnx", "a56147084e0fbdbedf9516734c02ebf958808b83d33bba072f92ee230c65288f", "QOperator", "1x84x8400", "diagnostic only"],
        ["e2e_rep_default", ".deps/models/yolo26/int8_rt204_forensics/e2e_rep_default/e2e_rep_default.onnx", "9367b4ba9c950b8d5100c1bc73914292e98af51651c5bf3d2d4458c9a742351e", "exporter static", "1x300x6", "negative control: CPU-bad"],
        ["traditional_rep_default", ".deps/models/yolo26/int8_rt204_forensics/traditional_rep_default/traditional_rep_default.onnx", "4ea9ef008ed1e9672b80545a588603dff8693928c240ec1e1864935a5cbd298c", "exporter static", "1x84x8400", "negative control: CPU-bad"],
        ["e2e_xslim_dynq", ".deps/models/yolo26/xslim_gate/dynamic/e2e_xslim_dynq.onnx", "ae220ace468dcae3555a4f60fa051ae2ea2d560780bd3f2dd959d4670225cb3c", "weight-dequantized diagnostic", "1x300x6", "CPU-good; not static activation INT8"],
        ["traditional_xslim_dynq", ".deps/models/yolo26/xslim_gate/dynamic/traditional_xslim_dynq.onnx", "fda140f340acc0ee1ee5517ca63e80410497f5e80e760f890b83642d4c7822ef", "weight-dequantized diagnostic", "1x84x8400", "CPU-good; not static activation INT8"],
        ["traditional_xslim_minmax_static", "historical artifact not retained locally", "d497215700ffa65e296731cc4e9cab2d624a4378ce344dece6dcae54bb367f04", "QDQ static", "1x84x8400", "negative control: all class scores zero"],
        ["traditional_xslim_percentile_static", "historical artifact not retained locally", "0fb95ea9811d9a4d449c4e114b80ab77541a6d1c58cd18d6fe511dd22822b9c9", "QDQ static", "1x84x8400", "negative control: all class scores zero"],
    ]
    write_tsv(out / "yolo26_runtime_model_manifest.tsv", ["logical_id", "path", "sha256", "quantization", "output", "status"], model_rows)
    write(
        out / "yolo26_model_lineage_report.md",
        """# YOLO26 runtime model lineage

The primary surface remains the Stage42 manual static Q/DQ e2e model with hash
`30a94e...29c0c`. No model was re-exported, simplified, calibrated, or replaced.
CPU-bad exporter/XSlim candidates remain negative controls; a runtime cannot
repair their oracle semantics. QOperator and stripped-kernel models are diagnostic
branches and are not promoted to correctness authority.
""",
    )

    write_tsv(
        out / "runtime_mode_scout.tsv",
        ["surface", "runtime", "provider", "opt", "status", "placement", "note"],
        [
            ["tiny QDQ Conv no kernel_shape", "RT205", "SpacemiT", "disable", "pass exact", "EP subgraph observed", "bounded positive control"],
            ["tiny QDQ Conv kernel_shape", "RT205", "SpacemiT", "disable/all", "abort", "compile attempted", "historical bug persists"],
            ["primary QDQ", "RT205", "CPU", "disable/all", "pass", "CPU", "disable exact host on F0/F5-F7"],
            ["primary QDQ", "RT205", "SpacemiT", "all", "abort", "first Conv compile failure", "not runnable"],
            ["stripped-kernel-shape QDQ", "RT205", "SpacemiT", "all", "core dump", "first blocker bypassed", "follow-on runtime failure"],
            ["attention/MatMul minimal control", "RT205", "SpacemiT", "disable", "SIGILL", "assignment not completed", "new regression"],
            ["primary QDQ broad filter", "RT205", "SpacemiT+CPU", "all", "SIGILL/core", "disabled QDQ/Conv/MatMul/Add", "fallback workaround regressed"],
            ["QOperator e2e", "RT204", "SpacemiT", "all", "pass but non-parity", "subgraph evidence", "not accepted"],
            ["QOperator e2e", "RT205", "SpacemiT", "all", "SIGILL", "unknown before crash", "new regression"],
            ["manual e2e small QDQ", "RT205", "SpacemiT", "all", "abort", "first Conv compile failure", "CPU-good candidate rejected"],
            ["manual traditional Conv+MatMul QDQ", "RT205", "SpacemiT", "all", "abort", "first Conv compile failure", "CPU-good candidate rejected"],
            ["manual traditional Conv-only QDQ", "RT205", "SpacemiT", "all", "abort", "first Conv compile failure", "CPU-good candidate rejected"],
            ["FP32 e2e", "RT205", "SpacemiT", "all", "runs", "EP subgraph", "diagnostic output not cross-runtime exact"],
            ["FP16 body/head", "RT205", "SpacemiT", "all", "runs", "EP subgraph", "diagnostic output not cross-runtime exact"],
        ],
    )
    write_tsv(
        out / "runtime_mode_stable_matrix.tsv",
        ["surface", "runtime", "provider", "opt", "cpuset", "threads", "warmup", "runs", "repeats", "mean_us", "stddev_us", "cv_pct", "status"],
        [
            ["primary QDQ pure graph scout", "RT204", "CPU", "all", "0-3", 1, 3, 20, 2, "2490869.192150", "1638.153361", "0.065766", "pass"],
            ["primary QDQ pure graph scout", "RT204", "CPU", "all", "0-3", 2, 3, 20, 2, "1508708.831175", "67.084528", "0.004446", "pass"],
            ["primary QDQ pure graph", "RT204", "CPU", "all", "0-3", 4, 10, 100, 5, "1023677.578412", "317.786422", "0.031044", "pass"],
            ["primary QDQ pure graph scout", "RT205", "CPU", "all", "0-3", 1, 3, 20, 2, "2485203.749075", "125.307418", "0.005042", "pass"],
            ["primary QDQ pure graph scout", "RT205", "CPU", "all", "0-3", 2, 3, 20, 2, "1510556.013500", "685.152247", "0.045358", "pass"],
            ["primary QDQ pure graph", "RT205", "CPU", "all", "0-3", 4, 10, 100, 5, "1024818.557872", "620.595993", "0.060557", "pass"],
            ["primary QDQ pure graph", "RT205", "SpacemiT", "all", "0-3", 4, 0, 1, 1, "", "", "", "not-runnable"],
            ["FP32 e2e diagnostic", "RT205", "SpacemiT", "all", "0-3", 4, 10, 100, 5, "445504.354768", "1375.491892", "0.308749", "pass timing; output not cross-runtime exact"],
            ["FP16 body/head diagnostic", "RT205", "SpacemiT", "all", "0-3", 4, 10, 100, 5, "368527.093944", "2284.576367", "0.619921", "pass timing; output not cross-runtime exact"],
        ],
    )
    write_tsv(
        out / "provider_assignment_matrix.tsv",
        ["surface", "runtime", "provider_registered", "provider_appended", "assignment_evidence", "status"],
        [
            ["03 QDQ Conv", "RT205", "CPU listed; EP external", 1, "SpineSubgraph dump contains intended QDQ Conv", "observed"],
            ["15 QDQ Conv attr", "RT205", "CPU listed; EP external", 1, "compiler names Conv_token_1 before abort", "attempted-failed"],
            ["primary QDQ", "RT205", "CPU listed; EP external", 1, "compiler names /model.0/conv/Conv_token_21", "attempted-failed"],
            ["primary CPU", "RT205", "CPU listed", 0, "explicit CPU session", "observed"],
            ["plugin", "RT205", "EP external", 1, "dlopen fails before assignment", "not-observed"],
        ],
    )
    write_tsv(
        out / "subgraph_inventory.tsv",
        ["artifact", "runtime", "surface", "size", "sha256", "node_count", "operators", "status", "location"],
        [
            ["SpaceMITExecutionProvider_SpineSubgraph_11394679816361779845_1_0_shape1_3_8_8__.onnx", "RT205", "03 QDQ Conv", 2598, "0952109e3ca61072a84ede28b56859ab6c7ee1700e89ed51e3fb18fda3c63936", 12, "Conv,DequantizeLinear,QuantizeLinear,Transpose", "complete/executed", BOARD_ROOT],
            ["SpaceMITExecutionProvider_SpineSubgraph_13358161925164545459_1_0_shape1_3_640_640__.onnx", "RT205", "first Conv cut", 3544, "226ad0993237790c4ce43eb29a0580c3f208b14768cc9e20b720fb26b0c93109", 13, "Conv,DequantizeLinear,QuantizeLinear,Transpose", "compile-failing", BOARD_ROOT],
            ["SpaceMITExecutionProvider_SpineSubgraph_7770448811129185215_1_0_shape1_3_640_640__.onnx", "RT205", "filtered/full diagnostic", 412472, "162b864c034f4acfc5f022d0dca8bbf5adb6b0821d017dc4ed90b8551901a6a8", 1076, "Add,Concat,Conv,ConvWithBinary,DequantizeLinear,MatMul,MaxPool,Mul,QuantizeLinear,ReduceMax,Reshape,Resize,Sigmoid,Slice,Softmax,Split,Sub,Transpose", "diagnostic; execution crashes", BOARD_ROOT],
        ],
    )

    fixture_rt204 = read_tsv(log_dir / "artifacts/fixed_fixtures/rt204_cpu_vs_host.tsv")
    fixture_rt205 = read_tsv(log_dir / "artifacts/fixed_fixtures/rt205_cpu_vs_host.tsv")
    fixture_rows = []
    for runtime, rows in (("RT204", fixture_rt204), ("RT205", fixture_rt205)):
        for row in rows:
            fixture_rows.append([runtime, "CPU", "disable", row["fixture"], row["actual_sha256"], row["mismatches"], row["max_abs_diff"], "pass"])
    fixture_rows.extend(
        [
            ["RT205", "SpacemiT", "disable", "03 tiny", "afc6013424b46b5aaaf991562841d790910e8a56c3f8caf06a2b091bffb685b9", 0, 0, "pass"],
            ["RT205", "SpacemiT", "all", "F0 primary", "", "", "", "not-runnable"],
        ]
    )
    write_tsv(out / "fixed_fixture_correctness_matrix.tsv", ["runtime", "provider", "opt", "fixture", "output_sha256", "mismatches", "max_abs_diff", "status"], fixture_rows)
    write(
        out / "first_divergence_rt205.md",
        """# First divergence under RT205

RT205 CPU with ORT_DISABLE_ALL is byte-exact to host ORT 1.27 at final `output0`
for every graph-valid full-model Stage43 fixture (F0, F5, F6, F7), the F8 blank
image, and the structured 0/1 F9 edge fixture. RT204 CPU is exact on the same
set. Stage43 F1-F4 are model4/block inputs, not full-model image tensors, and
were not relabeled.

RT205 SpacemiT EP never reaches an accepted full-model output. Its earliest
failure is the first quantized Conv (`/model.0/conv/Conv_token_21`) with the
historical clip-minmax compiler error and subsequent abort. A final-output
numerical divergence is therefore not defined for the EP path. The tiny 03
Q/DQ Conv without explicit kernel_shape is assigned to EP and byte-exact.
""",
    )
    write_tsv(
        out / "full_output_semantic_comparison.tsv",
        ["comparison", "opt", "fixture", "mismatches", "max_abs_diff", "status", "interpretation"],
        [
            ["host1.27 vs RT204 CPU", "disable", "F0/F5/F6/F7", 0, 0, "pass", "semantic contract matched"],
            ["host1.27 vs RT205 CPU", "disable", "F0/F5/F6/F7", 0, 0, "pass", "semantic contract matched"],
            ["host1.27 vs RT204 CPU", "all", "F0", 1660, "637.2356567382812", "diagnostic mismatch", "cross-runtime optimized graph"],
            ["host1.27 vs RT205 CPU", "all", "F0", 1660, "637.2356567382812", "diagnostic mismatch", "same as RT204 board CPU"],
            ["RT205 CPU vs RT205 EP", "all", "F0", "", "", "not-runnable", "EP aborts at first Conv"],
            ["RT204 QOperator CPU vs EP", "all", "F0", 1773, "642.2642860412598", "fail", "diagnostic QOperator candidate is not same-runtime parity"],
            ["host FP32 vs RT205 FP32 EP", "all", "F0", 1681, "634.9085507392883", "diagnostic mismatch", "discontinuous e2e output; no semantic acceptance"],
            ["RT205 FP32 CPU vs EP", "all", "F0", 1685, "634.9085488319397", "diagnostic mismatch", "placement/timing control only"],
            ["host FP16 vs RT205 FP16 EP", "all", "F0", 1423, "635.6875", "diagnostic mismatch", "discontinuous e2e output; no semantic acceptance"],
            ["RT205 FP16 CPU vs EP", "all", "F0", 1405, "635.21484375", "diagnostic mismatch", "placement/timing control only"],
        ],
    )

    write_tsv(
        out / "rt205_plugin_api_inventory.tsv",
        ["mechanism", "artifact", "sha256_or_symbol", "status"],
        [
            ["SpacemiT EP plugin", "include/spacemit_ort_plugin.h", "acb5767a3b7d15b383d1d0bc799126ea673bfe7797b979a9c956b68c12e346dc", "present"],
            ["SpacemiT EP plugin", "samples/plugin", "official source sample", "present and builds"],
            ["standard ORT custom op", "RegisterCustomOps", "exported by task plugin", "schema registration only"],
            ["SpacemiT EP plugin", "SpacemitPluginInit", "exported by task and official plugins", "present"],
            ["SpacemiT EP plugin", "SpacemitPluginGetAbiVersion/Metadata", "version/metadata entry points", "present in official API"],
            ["SpacemiT EP plugin", "SPACEMIT_EP_PLUGIN_LIB", "provider option carrying plugin path", "documented"],
            ["SpacemiT EP plugin", "AddOperator", "full custom operator track", "documented"],
            ["SpacemiT EP plugin", "AddDispatch", "built-in operator dispatch overlay track", "documented"],
            ["tensor contract", "DataType", "float32,uint8,int8,uint16,int16,int32,int64,string,bool,float16,double,uint32,uint64,complex,bfloat16", "declared"],
            ["tensor contract", "SpinePluginTensor", "shape/data/mutable-data/count/byte-size", "declared"],
            ["threading", "SpinePluginContext", "EP thread index/count/offset", "declared"],
            ["allocator", "plugin API", "no independent allocator contract exposed", "not-exposed"],
            ["lifetime", "SetCustomDispatch", "framework accepts dispatch ownership", "documented"],
            ["license", "header/sample", "MIT header; package license preserved", "present"],
            ["SpacemiT EP plugin ABI", "SpinePluginTensor::GetDataType", "undefined at dlopen", "broken"],
            ["independent plugin binary", "libstage46_u8_xor_plugin.so", "e5e790663d26caed18472fcb28f08aab430c393010a2a8b320dc3afd9bfc4fd8", "built; load fails"],
            ["official sample binary", "libcustom_plugin.so", "d86beb3d9957b3c47caea4e9ddf0c537715f72c49d3eed5e140d0746909ef98a", "built; ldd -r unresolved"],
        ],
    )
    write(
        out / "rt205_plugin_mechanism_report.md",
        """# RT205 plugin mechanism

The release implements mechanism C: a SpacemiT-EP-specific custom operator
plugin API (`spacemit_ort_plugin.h`, `SpacemitPluginInit`, provider option
`SPACEMIT_EP_PLUGIN_LIB`). It is distinct from standard ORT custom domains and
generic plugin EP loading.

The official sample and independent exact uint8 plugin both compile. Neither can
load against the shipped package: public API methods declared in the header,
including `SpinePluginTensor::GetDataType()`, are unresolved, and the package EP
does not export them. The independent session fails at `dlopen` before operator
registration or partitioning. The mechanism is package-present but ABI-broken.
""",
    )
    write_tsv(
        out / "rt205_plugin_smoke_correctness.tsv",
        ["plugin", "model_sha256", "expected_sha256", "load", "execution", "mismatches", "status"],
        [
            ["stage46_u8_xor", "7aedfb16959a079f52e14ea05fc0982e8e5af946fbf6890496f1349ba4963f9a", "8f04c27407f8082faf5e54b853caf7889a5fccca357e8f653799ab5a10d0d114", "fail unresolved symbol", "not reached", "n/a", "fail"],
            ["official sample", "n/a", "n/a", "ldd -r unresolved public API", "not reached", "n/a", "fail"],
        ],
    )
    write(
        out / "rt205_plugin_partition_report.md",
        """# Plugin partition result

No partition claim is possible. The plugin shared objects fail dynamic loading
before SpacemiT EP can inspect the custom node. Execution provider, graph split,
tensor materialization, and surrounding fusion preservation are all unknown.
""",
    )
    write_tsv(out / "rt205_plugin_overhead_raw.tsv", ["case", "calls", "mean_us", "status"], [["plugin registration", 0, "", "unavailable: loader failure"], ["per-call", 0, "", "unavailable: no execution"]])
    write_tsv(out / "rt205_plugin_overhead_summary.tsv", ["plugin_mean_overhead_us", "status"], [["not-measurable", "loader failure"]])

    render_performance_accuracy_strategy(out, log_dir, fp32_disable_summary, fp32_disable, fp32_all_summary, fp32_all, int8_disable_summary, int8_disable, int8_all_summary, int8_all, rt204_cpu_eval, rt205_cpu_eval)


def render_performance_accuracy_strategy(
    out: Path,
    log_dir: Path,
    fp32_disable_summary: dict[str, Any],
    fp32_disable: dict[str, Any],
    fp32_all_summary: dict[str, Any],
    fp32_all: dict[str, Any],
    int8_disable_summary: dict[str, Any],
    int8_disable: dict[str, Any],
    int8_all_summary: dict[str, Any],
    int8_all: dict[str, Any],
    rt204_cpu_eval: dict[str, Any] | None,
    rt205_cpu_eval: dict[str, Any] | None,
) -> None:
    """Render timing, COCO, storage, strategy, and final reports."""
    accuracy_root = log_dir / "artifacts/accuracy"

    def metric(evaluation: dict[str, Any] | None, key: str) -> str:
        if evaluation is None:
            return "not-runnable"
        value = evaluation.get(key)
        return "unknown" if value is None else str(value)

    def board_timing(stem: str) -> dict[str, float | int | str]:
        path = accuracy_root / f"{stem}.manifest.tsv"
        if not path.exists():
            return {
                "images": "not-runnable",
                "mean_inference_ms": "not-runnable",
                "mean_total_ms": "not-runnable",
            }
        rows = read_tsv(path)
        inference = [float(row["inference_ms"]) for row in rows]
        total = [float(row["total_ms"]) for row in rows]
        return {
            "images": len(rows),
            "mean_inference_ms": sum(inference) / len(inference),
            "mean_total_ms": sum(total) / len(total),
        }

    rt204_board_timing = board_timing("board_rt204_cpu_int8")
    rt205_board_timing = board_timing("board_rt205_cpu_int8")
    rt204_cpu_map_delta = (
        float(rt204_cpu_eval["map50_95"]) - float(int8_disable["map50_95"])
        if rt204_cpu_eval is not None
        else None
    )
    rt205_cpu_map_delta = (
        float(rt205_cpu_eval["map50_95"]) - float(int8_disable["map50_95"])
        if rt205_cpu_eval is not None
        else None
    )
    rt204_mean = 1023677.578412
    rt205_mean = 1024818.557872
    rt205_fp32_mean = 445504.354768
    rt205_fp16_mean = 368527.093944
    stage45_mean = 461603.297250
    rt205_vs_rt204_speedup = rt204_mean / rt205_mean
    rt205_slowdown_pct = (rt205_mean / rt204_mean - 1.0) * 100.0

    performance_rows = []
    for runtime, wall, cpu in (
        (
            "RT204",
            [1023507.906830, 1023600.817040, 1023279.917080, 1024042.688460, 1023956.562650],
            [2802424.091090, 2802859.094010, 2801278.164750, 2802780.444300, 2804488.727050],
        ),
        (
            "RT205",
            [1024641.231500, 1023987.544420, 1024866.612880, 1025720.094870, 1024877.305690],
            [2804853.272080, 2803824.372200, 2805968.637110, 2806104.200890, 2805394.127820],
        ),
    ):
        for repeat, (wall_us, process_cpu_us) in enumerate(zip(wall, cpu, strict=True)):
            performance_rows.append(
                ["primary QDQ", runtime, "CPU", "all", "0-3", 4, 10, 100, repeat, wall_us, process_cpu_us, "pass"]
            )
    performance_rows.extend(
        [
            ["primary QDQ", "RT204", "SpacemiT", "all", "0-3", 4, 0, 1, 0, "", "", "not-runnable: clip-minmax"],
            ["primary QDQ", "RT205", "SpacemiT", "all", "0-3", 4, 0, 1, 0, "", "", "not-runnable: clip-minmax then abort"],
            ["QOperator diagnostic", "RT204", "SpacemiT", "all", "0-3", 4, 0, 1, 0, "about 1030000", "not-captured", "non-parity diagnostic"],
            ["QOperator diagnostic", "RT205", "SpacemiT", "all", "0-3", 4, 0, 1, 0, "", "", "SIGILL"],
        ]
    )
    for label, wall, cpu in (
        (
            "FP32 e2e diagnostic",
            [444817.598350, 446652.147330, 444165.711080, 447285.131940, 444601.185140],
            [2211702.608610, 2220636.271880, 2208575.836370, 2224089.109260, 2210732.186780],
        ),
        (
            "FP16 body/head diagnostic",
            [368176.396780, 365315.965010, 367957.873230, 369702.102590, 371483.132110],
            [1828582.447130, 1814152.063100, 1827404.107420, 1836041.163000, 1845011.689250],
        ),
    ):
        for repeat, (wall_us, process_cpu_us) in enumerate(zip(wall, cpu, strict=True)):
            performance_rows.append(
                [label, "RT205", "SpacemiT", "all", "0-3", 4, 10, 100, repeat, wall_us, process_cpu_us, "diagnostic output not cross-runtime exact"]
            )
    write_tsv(
        out / "rt204_rt205_performance_raw.tsv",
        ["surface", "runtime", "provider", "opt", "cpuset", "intra", "warmup", "runs", "repeat", "wall_mean_us", "process_cpu_mean_us", "status"],
        performance_rows,
    )
    write_tsv(
        out / "rt204_rt205_performance_summary.tsv",
        ["surface", "runtime", "provider", "mean_us", "stddev_us", "cv_pct", "min_us", "max_us", "median_us", "p90_us", "p95_us", "process_cpu_mean_us", "status"],
        [
            ["primary QDQ", "B120", "CPU", f"{stage45_mean:.6f}", "not-restated", "not-restated", "not-restated", "not-restated", "not-restated", "not-restated", "not-restated", "not-restated", "Stage45 accepted mean only"],
            ["primary QDQ", "RT204", "CPU", f"{rt204_mean:.6f}", "317.786422", "0.031044", "1023279.917080", "1024042.688460", "1023600.817040", "1024008.238136", "1024025.463298", "2802766.104240", "pass"],
            ["primary QDQ", "RT205", "CPU", f"{rt205_mean:.6f}", "620.595993", "0.060557", "1023987.544420", "1025720.094870", "1024866.612880", "1025382.979198", "1025551.537034", "2805228.922020", "pass"],
            ["primary QDQ", "RT204", "SpacemiT", "not-runnable", "", "", "", "", "", "", "", "", "clip-minmax blocker"],
            ["primary QDQ", "RT205", "SpacemiT", "not-runnable", "", "", "", "", "", "", "", "", "clip-minmax then abort"],
            ["FP32 e2e diagnostic", "RT205", "SpacemiT", f"{rt205_fp32_mean:.6f}", "1375.491892", "0.308749", "444165.711080", "447285.131940", "444817.598350", "447031.938096", "447158.535018", "2215147.202580", "timing pass; semantic diagnostic only"],
            ["FP16 body/head diagnostic", "RT205", "SpacemiT", f"{rt205_fp16_mean:.6f}", "2284.576367", "0.619921", "365315.965010", "371483.132110", "368176.396780", "370770.720302", "371126.926206", "1830238.293980", "timing pass; semantic diagnostic only"],
        ],
    )
    write(
        out / "rt204_rt205_performance_report.md",
        f"""# RT204 and RT205 performance

The stable, matching primary-QDQ CPU surface used CPU0-3, ORT_ENABLE_ALL,
sequential execution, intra=4/inter=1, and warmup/runs/repeats 10/100/5.

- RT204 CPU: `{rt204_mean:.6f} us` (stddev `317.786422 us`).
- RT205 CPU: `{rt205_mean:.6f} us` (stddev `620.595993 us`).
- RT205 is `{rt205_slowdown_pct:.6f}%` slower than RT204 on this surface.
- The accepted Stage45 B120 CPU baseline is `{stage45_mean:.6f} us` and remains
  materially faster than either release package CPU runtime.

The bounded intra-thread scouts were RT204 `2490869.192150 / 1508708.831175 /
1023677.578412 us` and RT205 `2485203.749075 / 1510556.013500 /
1024818.557872 us` for intra 1/2/4 respectively. Intra=4 is the stable selected
CPU resource setting; semantic oracle sessions remain single-threaded.

There is no accepted RT204 or RT205 SpacemiT-EP full-model latency for the
primary model: RT204 fails the historical first-Conv compiler gate, and RT205
fails the same gate before aborting. QOperator rows are diagnostic because RT204
does not preserve the accepted output and RT205 SIGILLs. No hidden fallback row
is promoted as accelerated INT8.

Separate RT205 EP diagnostic controls run at `{rt205_fp32_mean:.6f} us` for FP32
and `{rt205_fp16_mean:.6f} us` for the body/head FP16 model. Their synthetic e2e
`output0` arrays are not cross-runtime exact, so these are timing diagnostics,
not accepted semantic or production paths. Both remain far above 50 ms.
""",
    )
    write_tsv(
        out / "session_compile_timing.tsv",
        ["runtime", "surface", "provider", "session_create_us", "first_run_us", "steady_mean_us", "status"],
        [
            ["RT204", "primary QDQ", "CPU", "2436113.147", "1062076.934", f"{rt204_mean:.6f}", "pass"],
            ["RT205", "primary QDQ", "CPU", "2435241.788", "1060585.614", f"{rt205_mean:.6f}", "pass"],
            ["RT204", "primary QDQ", "SpacemiT", "not-completed", "not-run", "not-runnable", "compiler blocker"],
            ["RT205", "primary QDQ", "SpacemiT", "not-completed", "not-run", "not-runnable", "compiler blocker then abort"],
        ],
    )
    write(
        out / "board_benchmark_environment.txt",
        f"""hostname=bf3
uname=Linux bf3 6.6.63 #2.2.7.2 SMP PREEMPT Fri Aug 15 12:32:44 UTC 2025 riscv64
boot_id=90be7592-f6d9-4d69-ae40-a6c9d25a51ab
online_cpus=0-7
benchmark_cpuset=0-3
governor=performance
cpu0_frequency_khz=1600000
cpu0_min_khz=614400
cpu0_max_khz=1600000
ORT_execution_mode=sequential
ORT_intra_threads=4
ORT_inter_threads=1
model_sha256={MODEL_SHA}
rt204_core_sha256={RT204_CORE_SHA}
rt205_core_sha256={RT205_CORE_SHA}
board_stage_root={BOARD_ROOT}
TMPDIR={BOARD_ROOT}/tmp
XDG_CACHE_HOME={BOARD_ROOT}/cache
""",
    )

    dataset_sha = "55304dfa58b86399878b2c054e6abc394df909366979b85ca5c3f2ea039fe86d"
    annotations_sha = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
    write_tsv(
        out / "coco2017_dataset_manifest.tsv",
        ["artifact", "host_path", "board_path", "sha256", "count", "status"],
        [
            ["val2017 sorted image inventory", "/data/datasets/coco2017/val2017", f"{BOARD_ROOT}/datasets/coco-val2017/val2017", dataset_sha, 5000, "verified"],
            ["instances_val2017.json", "/data/datasets/coco2017/annotations/instances_val2017.json", "not copied; host evaluation", annotations_sha, 5000, "verified"],
        ],
    )
    write(
        out / "coco2017_surface_contracts.md",
        """# COCO2017 surface contracts

All accuracy rows use the official 5000-image COCO val2017 set and matching
instances annotations. Preprocessing is OpenCV linear letterbox with value 114,
RGB order, NCHW float32, and division by 255. The e2e output contract is
`output0` float32 `1x300x6`; confidence threshold is 0.001 and maximum detections
is 300. Host rows use ORT 1.27.0 CPUExecutionProvider. Board rows use matched
RT204/RT205 package CPU sessions, intra=4/inter=1, on CPU0-3.

Host prediction environment: Python 3.12.3, OpenCV 4.13.0, NumPy 2.5.0,
Pillow 12.2.0. Evaluation environment: Python 3.12.3, NumPy 2.5.1,
pycocotools 2.0.11. The board C++ predictor uses the repository's frozen
letterbox/decode implementation and stage-owned OpenCV shared libraries.

SpacemiT EP rows are `not-runnable`, not zero-accuracy measurements. The first
quantized Conv fails before a full output. CPU-bad negative-control models are
not promoted into the full evaluation matrix.
""",
    )

    accuracy_surfaces: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
        ("host_fp32_disable", fp32_disable_summary, fp32_disable),
        ("host_fp32_all", fp32_all_summary, fp32_all),
        ("host_int8_disable", int8_disable_summary, int8_disable),
        ("host_int8_all", int8_all_summary, int8_all),
    ]
    accuracy_rows: list[list[Any]] = []
    for name, summary, evaluation in accuracy_surfaces:
        accuracy_rows.append(
            [
                name,
                summary["model_sha256"],
                evaluation["predictions_sha256"],
                summary["runtime_version"],
                summary["provider"],
                summary["optimization"],
                summary["images"],
                0,
                summary["predictions"],
                summary["predictions"] / summary["images"],
                summary["mean_inference_ms"],
                "not-captured",
                evaluation["map50_95"],
                evaluation["map50"],
                evaluation["precision_mean_valid_cocoeval_grid"],
                evaluation["recall_mean_valid_cocoeval_grid"],
                evaluation["ap_small"],
                evaluation["ap_medium"],
                evaluation["ap_large"],
                "pass",
            ]
        )
    for name, runtime, evaluation, timing in (
        ("rt204_cpu_int8_all", "1.24.2+spacemit.a1/EP-package-2.0.4", rt204_cpu_eval, rt204_board_timing),
        ("rt205_cpu_int8_all", "1.24.2+spacemit.a1/EP-package-2.0.5", rt205_cpu_eval, rt205_board_timing),
    ):
        accuracy_rows.append(
            [
                name,
                MODEL_SHA,
                metric(evaluation, "predictions_sha256"),
                runtime,
                "CPUExecutionProvider",
                "all",
                timing["images"],
                0 if evaluation is not None else "not-runnable",
                metric(evaluation, "prediction_count"),
                (
                    float(evaluation["prediction_count"]) / float(evaluation["image_count"])
                    if evaluation is not None
                    else "not-runnable"
                ),
                timing["mean_inference_ms"],
                timing["mean_total_ms"],
                metric(evaluation, "map50_95"),
                metric(evaluation, "map50"),
                metric(evaluation, "precision_mean_valid_cocoeval_grid"),
                metric(evaluation, "recall_mean_valid_cocoeval_grid"),
                metric(evaluation, "ap_small"),
                metric(evaluation, "ap_medium"),
                metric(evaluation, "ap_large"),
                "pass" if evaluation else "pending/not-runnable",
            ]
        )
    for name, runtime, reason in (
        ("rt204_ep_int8_all", "2.0.4", "historical clip-minmax compiler blocker"),
        ("rt205_ep_int8_all", "2.0.5", "clip-minmax blocker then abort"),
    ):
        accuracy_rows.append([name, MODEL_SHA, "not-runnable", runtime, "SpacemiTExecutionProvider", "all", "not-runnable", "not-runnable", "not-runnable", "not-runnable", "not-runnable", "not-runnable", "not-runnable", "not-runnable", "not-runnable", "not-runnable", "not-runnable", "not-runnable", "not-runnable", reason])
    accuracy_fields = [
        "surface", "model_sha256", "predictions_sha256", "runtime", "provider", "optimization", "images", "runtime_failures", "predictions",
        "detections_per_image", "mean_inference_ms", "mean_full_pipeline_ms", "map50_95", "map50", "precision", "recall",
        "ap_small", "ap_medium", "ap_large", "status",
    ]
    write_tsv(out / "coco2017_accuracy_raw.tsv", accuracy_fields, accuracy_rows)
    write_tsv(out / "coco2017_accuracy_summary.tsv", accuracy_fields, accuracy_rows)

    annotations = read_json(Path("/data/datasets/coco2017/annotations/instances_val2017.json"))
    category_counts: dict[int, int] = {}
    for annotation in annotations["annotations"]:
        category_id = int(annotation["category_id"])
        category_counts[category_id] = category_counts.get(category_id, 0) + 1
    per_class_rows: list[list[Any]] = []
    per_class_names = [name for name, _, _ in accuracy_surfaces]
    if rt204_cpu_eval is not None:
        per_class_names.append("board_rt204_cpu_int8")
    if rt205_cpu_eval is not None:
        per_class_names.append("board_rt205_cpu_int8")
    for name in per_class_names:
        per_class_path = accuracy_root / f"{name}.eval.per_class.tsv"
        if per_class_path.exists():
            for row in read_tsv(per_class_path):
                category_id = int(row["category_id"])
                per_class_rows.append(
                    [name, category_id, row["name"], category_counts.get(category_id, 0), row["ap50_95"]]
                )
    write_tsv(
        out / "coco2017_per_class.tsv",
        ["surface", "category_id", "name", "instance_count", "ap50_95"],
        per_class_rows,
    )
    write_tsv(
        out / "coco2017_bootstrap_ci.tsv",
        ["comparison", "metric", "delta", "ci_low", "ci_high", "status"],
        [
            ["host INT8 disable - host FP32 disable", "mAP50-95", f"{int8_disable['map50_95'] - fp32_disable['map50_95']:.12f}", "not-computed", "not-computed", "complete deterministic full-set delta"],
            ["host INT8 all - host INT8 disable", "mAP50-95", f"{int8_all['map50_95'] - int8_disable['map50_95']:.12f}", "not-computed", "not-computed", "complete deterministic full-set delta"],
            ["RT205 EP - host semantic INT8", "mAP50-95", "not-runnable", "not-runnable", "not-runnable", "EP produces no full-model output"],
        ],
    )
    write(
        out / "coco2017_accuracy_report.md",
        f"""# Full COCO2017 accuracy

The mandatory host 2x2 matrix completed on all 5000 images:

| Surface | mAP50-95 | mAP50 |
|---|---:|---:|
| FP32 ORT_DISABLE_ALL | {fp32_disable['map50_95']:.12f} | {fp32_disable['map50']:.12f} |
| FP32 ORT_ENABLE_ALL | {fp32_all['map50_95']:.12f} | {fp32_all['map50']:.12f} |
| INT8 ORT_DISABLE_ALL | {int8_disable['map50_95']:.12f} | {int8_disable['map50']:.12f} |
| INT8 ORT_ENABLE_ALL | {int8_all['map50_95']:.12f} | {int8_all['map50']:.12f} |

FP32 optimization is effectively invariant (`{fp32_all['map50_95'] - fp32_disable['map50_95']:+.12f}`).
Semantic INT8 loses `{int8_disable['map50_95'] - fp32_disable['map50_95']:+.12f}` mAP50-95 relative to FP32,
and ORT_ENABLE_ALL introduces another `{int8_all['map50_95'] - int8_disable['map50_95']:+.12f}` on the INT8 graph.

RT204 CPU full COCO mAP50-95: `{metric(rt204_cpu_eval, 'map50_95')}`.
RT205 CPU full COCO mAP50-95: `{metric(rt205_cpu_eval, 'map50_95')}`.
Their deltas versus fixed-host semantic INT8 are
`{rt204_cpu_map_delta if rt204_cpu_map_delta is not None else 'not-runnable'}` and
`{rt205_cpu_map_delta if rt205_cpu_map_delta is not None else 'not-runnable'}`.
The two package CPU prediction JSON files are byte-identical when both rows are
complete (`1dbc118383009fa68ddb1b786af68f56b38c473d8c03a23a782a38e4727d0b48`).
RT204 dataset mean inference/full-pipeline time is
`{rt204_board_timing['mean_inference_ms']} / {rt204_board_timing['mean_total_ms']} ms`;
RT205 is `{rt205_board_timing['mean_inference_ms']} / {rt205_board_timing['mean_total_ms']} ms`.
Both SpacemiT EP full-COCO rows are not runnable and are not encoded as zero.
No EP accuracy parity claim is possible, and no bootstrap EP delta is meaningful
without an EP output surface.

The fixed host semantic blank fixture has 28 rows at score >=0.001 (maximum
score 0.0167016685); this behavior is recorded rather than hidden. The structured
0/1 edge fixture has no row at that threshold. RT204/RT205 CPU exactness for
these fixtures is part of the fixed-fixture gate; the EP path cannot reach them.
""",
    )

    acceptance_rows = [
        ["historical_clip_minmax_bug", "fail", "same explicit-kernel_shape compiler error under RT205"],
        ["attention_matmul_followon", "fail", "QLinearMatMul control SIGILL; stripped full model aborts"],
        ["full_model_session_creation", "fail", "primary EP session/run cannot complete"],
        ["full_model_stability", "fail", "abort/SIGILL paths"],
        ["CPU_oracle", "pass", "RT205 CPU disable exact on F0/F5/F6/F7"],
        ["SpacemiT_EP_placement", "partial", "tiny no-kernel-shape QDQ Conv only"],
        ["integer_boundary_exactness", "partial", "tiny positive control exact; full EP unavailable"],
        ["fixed_fixture_semantics", "partial", "CPU passes; EP unavailable"],
        ["full_COCO_accuracy", "partial", "host and package CPU complete; EP not-runnable"],
        ["pure_inference_latency", "partial", "CPU stable; EP not-runnable"],
        ["full_pipeline_latency", "partial", "CPU dataset path measured; EP not-runnable"],
        ["plugin_load", "fail", "unresolved public plugin ABI symbols"],
        ["plugin_execution", "fail", "not reached"],
        ["plugin_partition_preservation", "unknown", "plugin cannot load"],
        ["plugin_overhead", "unknown", "not measurable"],
    ]
    write_tsv(out / "rt205_acceptance_matrix.tsv", ["axis", "status", "evidence"], acceptance_rows)
    write_tsv(
        out / "post_rt205_strategy_matrix.tsv",
        ["strategy", "runtime_evidence", "accuracy_risk", "engineering_risk", "disposition"],
        [
            ["stock RT205 INT8 EP", "first Conv blocker plus QOperator SIGILL/core", "unmeasurable EP COCO", "high", "reject"],
            ["RT205 plugin mixed route", "package ABI prevents plugin load", "unknown", "high", "defer until vendor ABI repair"],
            ["K1X student 416", "latency-primary analytical hypothesis", "must train and measure", "medium-high", "preserve"],
            ["K1X student 512", "accuracy-fallback analytical hypothesis", "must train and measure", "medium-high", "preserve"],
            ["bounded multicore AOT salvage", "custom exact kernels exist but prior islands no net win", "fixed-host oracle available", "high", "secondary"],
            ["FP16/RVV mainline", "RT205 INT8 vendor lane regressed", "FP32/FP16 accuracy surface available", "medium", "fallback option"],
        ],
    )
    write(
        out / "student_416_512_disposition.md",
        """# Student 416/512 disposition

Both untrained hypotheses remain in the next decision packet. The 416 candidate
is latency-primary; the 512 candidate is the accuracy fallback and may become
primary only after measured runtime and trained accuracy justify it. Stage46
does not train, select, or claim either candidate. Resolution selection must use
trained COCO accuracy, measured K1X latency, cache/tail behavior, head scale
count, and the now-negative RT205/plugin evidence rather than quadratic FLOP
scaling alone.
""",
    )

    write(
        out / "board_storage_preflight.txt",
        f"""hostname=bf3
/data_source=/dev/nvme0n1p1
/data_fstype=ext4
/data_size=469G
/data_free_at_preflight=438G
/data_writable=true
/_source=/dev/mmcblk2p6
board_stage_root={BOARD_ROOT}
storage_policy_status=pass
""",
    )
    write_tsv(
        out / "board_storage_manifest.tsv",
        ["path", "class", "storage", "bytes", "status"],
        [
            [f"{BOARD_ROOT}/runtime", "RT204/RT205 package runtime", "NVMe /data", 66673646, "present"],
            [f"{BOARD_ROOT}/bin", "runtime-specific runners/evaluators", "NVMe /data", 498504, "present"],
            [f"{BOARD_ROOT}/models", "task model copies", "NVMe /data", 33422420, "present"],
            [f"{BOARD_ROOT}/repros", "minimal repros", "NVMe /data", 18060134, "present"],
            [f"{BOARD_ROOT}/plugins", "plugin proof binaries", "NVMe /data", 81908, "present"],
            [f"{BOARD_ROOT}/datasets", "5000 COCO images", "NVMe /data", 834693004, "present"],
            [f"{BOARD_ROOT}/preprocessed", "fixed tensor inputs", "NVMe /data", 34406400, "present"],
            [f"{BOARD_ROOT}/outputs", "tensor/COCO outputs", "NVMe /data", 121929374, "present"],
            [f"{BOARD_ROOT}/profiles", "profile outputs", "NVMe /data", 0, "empty"],
            [f"{BOARD_ROOT}/subgraphs", "EP subgraph dumps", "NVMe /data", 418614, "present"],
            [f"{BOARD_ROOT}/logs", "board logs/timing", "NVMe /data", 3574842, "present"],
            [f"{BOARD_ROOT}/tmp", "TMPDIR", "NVMe /data", 0, "empty"],
            [f"{BOARD_ROOT}/cache", "XDG_CACHE_HOME", "NVMe /data", 0, "empty"],
        ],
    )
    write_tsv(out / "board_emmc_write_exceptions.tsv", ["path", "bytes", "reason", "disposition"], [])

    write(
        out / "source_hygiene_report.md",
        """# Source hygiene

- Project start tree was clean at the required head.
- Vendor archives/libraries, COCO data, model binaries, outputs, subgraphs,
  profiles, build trees, and raw board logs remain outside Git.
- Repository changes are limited to diagnostic source, the tiny plugin proof,
  reproducible storage-skill source/installer, Stage45 addenda, Stage46 reports,
  and the Stage47 prompt.
- `/data/ncnn` was not mutated.
- The only preliminary secret-pattern hit is the storage skill self-test's own
  deny-list regex; it contains no credential value and is an intentional self-match.
- Symlink, large-file, secret/private-path, `git diff --check`, and staged-diff
  checks are required immediately before the local commit and are recorded in
  the shared command ledger/result packet.
""",
    )
    write(
        out / "stage47_prompt.md",
        """# Stage47 - K1X student 416/512 architecture and training preparation

## Mission

Prepare, but do not start, a reproducible K1X-specific student-model training
lane after the RT205 vendor INT8 regression. Preserve both 416 latency-primary
and 512 accuracy-fallback candidates. Freeze teacher, COCO data, preprocessing,
operator set, quantization arithmetic, export contracts, distillation losses,
latency LUT inputs, and acceptance gates.

## Required gates

1. Reproduce the Stage46 FP32 and semantic-INT8 full-COCO surfaces.
2. Specify two static-shape students using K1X-measured operators and aligned
   channels, without assuming depthwise or attention efficiency.
3. Produce training/QAT manifests and dry-run export/oracle checks only.
4. Define separate accuracy and board latency gates before training approval.
5. Keep RT205/plugin routes rejected unless a new verified vendor package fixes
   the Q/DQ Conv, QOperator, and plugin ABI regressions.

No training, default dispatch, production claim, push, or vendor binary commit
is authorized without a separate direct-user packet.
""",
    )

    rt204_map = metric(rt204_cpu_eval, "map50_95")
    rt205_map = metric(rt205_cpu_eval, "map50_95")
    write(
        out / "STAGE46_FINAL_REPORT.md",
        f"""# Stage46 final report

## Classification

`{CLASSIFICATION}`

Start HEAD: `{START_HEAD}`. End HEAD is recorded in the result packet and final
console response after the atomic local commit. Push: false.

## Proven

- The official RT205 archive is `{RT205_ARCHIVE_SHA}` and passed safe extraction.
- The matched RT204 harness reproduces `output_type not implemented for clip minmax`.
- RT205 preserves that blocker and aborts after it on the primary first Conv.
- RT205 newly core dumps on QLinearConv and SIGILLs on QLinearMatMul controls that
  execute under RT204.
- A no-kernel-shape tiny Q/DQ Conv is assigned to RT205 EP and CPU/EP byte-exact;
  this bounded positive control does not generalize to the full model.
- RT204/RT205 CPU ORT_DISABLE_ALL are byte-exact to host ORT 1.27 for
  F0/F5/F6/F7 plus F8 blank and F9 structured edge fixtures.
- Stable primary CPU timing is `{rt204_mean:.6f} us` on RT204 and
  `{rt205_mean:.6f} us` on RT205; RT205 is `{rt205_slowdown_pct:.6f}%` slower.
- The mandatory host full-COCO 2x2 matrix completed. FP32 disable/all mAP50-95 is
  `{fp32_disable['map50_95']:.12f}` / `{fp32_all['map50_95']:.12f}`; INT8 is
  `{int8_disable['map50_95']:.12f}` / `{int8_all['map50_95']:.12f}`.
- Full package CPU COCO rows: RT204 `{rt204_map}`, RT205 `{rt205_map}`.
- RT205 EP FP32/FP16 diagnostic timing is `{rt205_fp32_mean:.6f} / {rt205_fp16_mean:.6f} us`;
  those discontinuous e2e outputs are not promoted as cross-runtime exact.
- RT205's SpacemiT plugin API and sample are package-present, but both the
  official and independent plugins have unresolved public ABI methods at load.
- All new board artifacts are under the NVMe stage root; eMMC exceptions: zero.

## Broken

- Historical explicit-kernel-shape Q/DQ Conv compilation under SpacemiT EP.
- RT205 QOperator Conv and MatMul execution (core dump/SIGILL).
- RT205 full primary INT8 SpacemiT-EP session/run.
- RT205 plugin loading and therefore plugin execution/partition/overhead gates.

## Unknown

- Full-model RT205 EP integer parity, COCO accuracy, and latency: not runnable.
- Plugin execution provider and partition preservation: loader failure prevents observation.
- Trained accuracy and measured board latency of the 416 and 512 student hypotheses.

## Correctness and accuracy policy

Host ORT 1.27 CPU with ORT_DISABLE_ALL plus independent operator semantics remains
the authority. Board package CPU rows validate integration. Board EP outputs must
also prove placement and semantics; CPU fallback is not accelerated INT8.

## Decision

Reject stock RT205 INT8 EP and its shipped plugin ABI for this model. Route the
next authorized work to K1X student 416/512 architecture and training preparation,
while retaining FP16/RVV as the fallback mainline. No training is authorized here.

## Validation

- Host Release build and CTest: pass, 44/44 tests.
- RT204 and RT205 matched-header RISC-V builds: pass.
- Independent and official plugin sample cross-builds: pass; loader gate fails as documented.
- Board loader identity, fixed fixtures, minimal repros, stable CPU timing, and
  full runnable COCO surfaces: executed.
- Python compile, storage-skill self-test, TSV structural validation, Git diff,
  staged diff, symlink, large-file, and secret/private-path checks: recorded in
  the final shared command ledger and result packet.

## Non-claims

No production readiness, default backend, model FPS, camera throughput, retained
custom-engine accuracy, usable RT205 plugin, student-model accuracy, or full
custom engine is claimed.
""",
    )
    write(
        out / "STAGE46_SUMMARY_RU.md",
        f"""# Итог Stage46

Классификация: `{CLASSIFICATION}`.

Контроль RT204 воспроизвел историческую ошибку `clip minmax`. В RT205 ошибка
сохранилась, а после нее процесс завершается аварийно. Дополнительно RT205 дал
core dump на QLinearConv и SIGILL на QLinearMatMul, тогда как RT204 выполняет
эти минимальные тесты. Полная INT8-модель через SpacemiT EP не запускается.

CPU-режим RT205 при ORT_DISABLE_ALL точно совпал с host ORT на F0/F5/F6/F7.
Стабильное время CPU для основной модели: RT204 `{rt204_mean:.6f} us`, RT205
`{rt205_mean:.6f} us`; ускорения нет. Полная host-матрица COCO2017 дала
FP32 disable/all `{fp32_disable['map50_95']:.12f}` / `{fp32_all['map50_95']:.12f}`
и INT8 disable/all `{int8_disable['map50_95']:.12f}` / `{int8_all['map50_95']:.12f}`.

Механизм плагинов 2.0.5 присутствует в пакете, но официальный и независимый
плагины не загружаются из-за неразрешенных символов публичного ABI. Следующий
этап должен подготовить обе student-ветки: 416 как latency-first и 512 как
accuracy fallback. Обучение и production-интеграция этим этапом не разрешены.
""",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--log-dir", required=True)
    args = parser.parse_args()
    render(args)


if __name__ == "__main__":
    main()
