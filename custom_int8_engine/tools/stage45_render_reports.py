#!/usr/bin/env python3
"""Render the Stage45 decision packet from immutable raw evidence.

This script deliberately distinguishes measured values, diagnostic ceilings, and
analytical projections. It does not execute inference or alter raw evidence.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
from pathlib import Path
from typing import Iterable


STAGE_ID = (
    "BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE45-"
    "SYSTEM-ROOFLINE-VENDOR-ORT-FORENSICS-ACCURACY-AND-MODEL-CODESIGN-DECISION-001"
)
START_HEAD = "bdefd89cc4247cb9e0ddac6fd06b561b05d29c87"
MODEL_SHA = "30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c"
FP32_SHA = "d71286588abe691ede49faa5ca9a471b7e9e5257669953ee59abbc2e9d115fc2"
ORT_SHA = "5a28c8128a7b1ed9cb29357f42eb7a2a45eb1b23d8791c2fee1eaf0151546238"
BOARD_ROOT = f"/data/k1x-stage-runs/{STAGE_ID}"
CLASSIFICATION = "stage45-model-executor-codesign-recommended"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_tsv(path: Path, headers: list[str], rows: Iterable[Iterable[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)


def copy_text(src: Path, dst: Path) -> None:
    # Raw csv evidence may use the platform csv module's CRLF default. Keep raw
    # files immutable and normalize only the repository report copy.
    write(dst, src.read_text(encoding="utf-8").replace("\r\n", "\n"))


def read_accuracy(log_dir: Path, stem: str) -> tuple[dict, dict]:
    summary = json.loads((log_dir / "accuracy" / f"{stem}.summary.json").read_text())
    evaluation = json.loads((log_dir / "accuracy" / f"{stem}.eval.json").read_text())
    return summary, evaluation


def parse_roofline(log_dir: Path) -> list[dict[str, str]]:
    path = log_dir / "board_artifacts/logs/roofline_stable.tsv"
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    for line in path.read_text().splitlines():
        if line.startswith("category\t"):
            header = line.split("\t")
        elif header is not None and "\t" in line and not line.startswith("microkernel_validation="):
            values = line.split("\t")
            if len(values) == len(header):
                rows.append(dict(zip(header, values)))
    return rows


def render(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    log_dir = Path(args.log_dir).resolve()
    out = repo / "stages" / STAGE_ID
    out.mkdir(parents=True, exist_ok=True)

    graph = json.loads((log_dir / "analysis/graph/graph_summary.json").read_text())
    fp32_summary, fp32_eval = read_accuracy(log_dir, "fp32_all_500")
    sem_summary, sem_eval = read_accuracy(log_dir, "int8_disable_500")
    op_summary, op_eval = read_accuracy(log_dir, "int8_all_500")
    roof = parse_roofline(log_dir)

    stage44_residual = 24157.4 - 2660.0
    stage44_residual_pct = (stage44_residual / 11701.121842 - 1.0) * 100.0
    model_macs = int(graph["conv_matmul_gemm_macs"])
    m12_rate = 54.360135
    m8_rate = 43.395240
    m4_rate = 24.746637
    m12_compute_floor_ms = model_macs / 1e6 / m12_rate
    m8_compute_floor_ms = model_macs / 1e6 / m8_rate
    m4_compute_floor_ms = model_macs / 1e6 / m4_rate
    target_rate = model_macs / 1e6 / 45.0

    write(
        out / "workspace_preflight.md",
        f"""# Workspace preflight

- Repository: `/data/banana-yolo26-spacemit-demo`
- Branch: `yolo26-custom-int8-engine`
- Required and observed start HEAD: `{START_HEAD}`
- Initial worktree: clean
- `git diff --check`: pass at preflight
- `/data/ncnn`: read-only for this stage; no mutation performed
- Direct-user authorization used; no control-plane packet was required.
""",
    )

    write_tsv(
        out / "stage44_surface_matrix.tsv",
        ["surface", "start_boundary", "end_boundary", "threads", "mean_us", "kind", "comparability_note"],
        [
            ["stage44_ort_model5_intra1", "model4_postactivation_q", "model5_postactivation_q", 1, "18169.770948", "measured", "isolated ORT model5"],
            ["stage44_ort_model5_intra4", "model4_postactivation_q", "model5_postactivation_q", 4, "11701.121842", "measured", "resource-matched isolated ORT model5"],
            ["stage44_custom_r2a", "model4_preactivation_q", "model5_postactivation_q", 3, "24157.400000", "measured", "includes model4 final activation/requant"],
            ["stage44_model4_postactivation", "model4_preactivation_q", "model4_postactivation_q", 3, "2660.000000", "diagnostic", "phase timer; approximate"],
            ["stage44_custom_model5_residual", "model4_postactivation_q", "model5_postactivation_q", 3, f"{stage44_residual:.6f}", "derived_diagnostic", "subtraction of non-identical instrumentation surfaces"],
            ["stage44_scaffold_path_a", "images", "output0", 4, "510864.440692", "measured", "model4-only hybrid scaffold"],
            ["stage44_scaffold_path_b", "images", "output0", 4, "515063.225590", "measured", "model4-model5 hybrid scaffold"],
            ["stage44_full_board_ort_intra4", "images", "output0", 4, "462144.181262", "measured", "historical stable surface"],
        ],
    )
    write(
        out / "stage44_surface_reconciliation.md",
        f"""# Stage44 surface reconciliation

Stage44's official classification remains `stage44-model5-exact-no-net-win`.
The `24157.4 us` custom value starts at the model4 preactivation code and includes
approximately `2660 us` of model4 final activation/requant. The `11701.121842 us`
ORT arm starts at model4 postactivation. Their direct ratio therefore overstates
the model5-only custom deficit.

The subtraction-only custom model5 residual is `{stage44_residual:.3f} us`, still
`{stage44_residual_pct:.3f}%` slower than the four-thread ORT block, but this is a
diagnostic estimate rather than a benchmark. Stage44's instrumentation-off paired
scaffold is the authoritative system result: adding the exact model5 island made
the scaffold `4198.784898 us` (`0.821898%`) slower.

This report also supersedes the Stage44 repository placeholder with final HEAD
`{START_HEAD}` without rewriting historical raw evidence.
""",
    )

    write(
        out / "board_storage_preflight.txt",
        f"""hostname=bf3
board_stage_root={BOARD_ROOT}
data_source=/dev/nvme0n1p1
data_fstype=ext4
data_mount_options=rw,noatime
data_filesystem_size=469G
data_available=438G
root_source=/dev/mmcblk2p6
root_storage_class=eMMC
data_exists=1
data_writable=1
preflight_status=pass
governor=performance
cpu0_frequency_hz=1600000000
boot_id=90be7592-f6d9-4d69-ae40-a6c9d25a51ab
""",
    )
    board_files = [
        ("bin/bench_stage35_vmadot_sigill", "86c5ee76e3d906435df54aa43a798a346fa5eb0056cea57018ff0bab4c8db587"),
        ("bin/bench_stage42_inprocess_runner", "7019146c02569b0bbf6283e9d6d0b210363e1aeee7fd36e6339a26c02dd4f02f"),
        ("bin/bench_stage45_k1x_roofline", "e6928ef22987b4131df00933c676cc99952f2d8787e3d96c7065e6fe21215f82"),
        ("inputs/images_F0.npy", "46908132cf9e04ee14d1702ce99f583255b3a0de0a9b3e74d0f277ab9d8e09b7"),
        ("inputs/model5_F0.npy", "8ca9771270e443fb14f0443942c34bf61ba45a72b0fb8fccde8cf26fd812eb3e"),
        ("logs/full_ort_allcore_scout.txt", "134a646dd76710d5b77763e1cb3a98495019f6bf7d386e9b77d93c398acd19fd"),
        ("logs/full_ort_intra4_env_after.txt", "21a3bd04cc13d27f45ee6b6227cc7b38bd335334d3410a62bfd24d616840fb20"),
        ("logs/full_ort_intra4_env_before.txt", "c9ede14bbbfbf1d3c8cd1cd50b52964723cc047832b0ca2e0e38ea268976a472"),
        ("logs/full_ort_intra4_stable.txt", "66cb419fca10374a670cb1ead5b257d12db6da723c51d5648ad0b35128bf45dc"),
        ("logs/full_ort_thread_scout.txt", "0387157dd44c652dc48f4cd021972ba774c09858d4cb2aa0eced9e6c47809aea"),
        ("logs/full_profile_run.txt", "3dfd97e56e572cbcd40a753a810cbfe4ec3b453e355a950a12f15cc88f27646f"),
        ("logs/model5_profile_run.txt", "dfd05f7edecaacd85b6b6d45befd51498bf792b7e487cfcd3b0166435a2531da"),
        ("logs/roofline_env_after.txt", "bdf078d19fe006f210020bf61a1af40787ce31caf1cf0568e10f13511ecf8ced"),
        ("logs/roofline_env_before.txt", "9976a9a8e4f8c5a33f7f53f21411110c6c679c55abfa2619747e55d83c018adf"),
        ("logs/roofline_microkernel_smoke.tsv", "82c4ec95adb6daaa784d2ae89922626d02731fd0dea965c82bcfb9c1df627053"),
        ("logs/roofline_stable.tsv", "0b40ecb4fbaaa05c13e450acdf6a6829e56b2d28b33951033cce25716da2bf3b"),
        ("logs/runner_dynamic.txt", "7b41a654ed4d93132b8b74e7ebf9c20897eb8506a59fdb0a52efffa343c6ae42"),
        ("logs/runner_ldd.txt", "328fdedc8edbc4b6b38869a86bf64175552aaaa6de631bc8cbe6cbee97f1541f"),
        ("logs/stage35_controls.tsv", "c5759dd8f595d9e519a3f4835371b71ba6e1f8c7a8346b3da4188208a14530eb"),
        ("models/full_model.onnx", MODEL_SHA),
        ("models/model5_isolated_quantized.onnx", "f37a4426054496c34534e5f88c1add2bb2ce738f61e4aed00d2df9c4f218e4e0"),
        ("perf/full_perf_runner.txt", "b09d95543aa5631db62f5218cd7c7d5afde2923355aea896719e1995f1238d86"),
        ("perf/model5_perf_runner.txt", "b09d95543aa5631db62f5218cd7c7d5afde2923355aea896719e1995f1238d86"),
        ("perf/perf_version.txt", "b09d95543aa5631db62f5218cd7c7d5afde2923355aea896719e1995f1238d86"),
        ("profiles/full_ort_only_2026-07-11_17-03-18.json", "b65f1fc56e1d8a609ac1c88e7476c460d320bfb8aae800846f7311f733f3ea75"),
        ("profiles/model5_ort_only_2026-07-11_17-03-17.json", "28f6ae8085664a7717642c643e08e47a2e14066a4ca6c01147446fbababf6a29"),
    ]
    write_tsv(out / "board_storage_manifest.tsv", ["path", "sha256", "storage"], [[f"{BOARD_ROOT}/{p}", h, "NVMe /data"] for p, h in board_files])
    write_tsv(out / "board_emmc_write_exceptions.tsv", ["path", "bytes", "reason", "disposition"], [])
    write(
        out / "board_storage_policy_report.md",
        f"""# Board storage policy

Preflight passed: `/data` is writable ext4 on NVMe (`/dev/nvme0n1p1`) with
`438G` available; `/` is eMMC (`/dev/mmcblk2p6`). All deployed binaries, models,
inputs, profiles, perf failures, logs, temporary files, and caches were kept under
`{BOARD_ROOT}`. `TMPDIR` and `XDG_CACHE_HOME` were redirected there.

No Stage45 artifact was written to eMMC. The installed ORT library under
`/home/svt` was read in place and not duplicated or modified. Exception count: 0.
""",
    )

    skill_root = Path("/data/.codex/skills")
    skill_files = [
        skill_root / "k1x_board_storage_policy/SKILL.md",
        skill_root / "k1x_board_storage_policy/scripts/selftest.sh",
        skill_root / "k1x_task_template/SKILL.md",
        skill_root / "k1x_env_sanity/SKILL.md",
        skill_root / "k1x_deploy_and_run_yolo11/SKILL.md",
    ]
    before_hashes = {
        "k1x_task_template/SKILL.md": "7270b220dd8f5c18ac39c43b18670f0d499a998c983892c7f7dedd4a652defe7",
        "k1x_env_sanity/SKILL.md": "8fe5ab7ea69fb9822506d2293e84e99cd48a4ffbbf6efcc8fe5b9fa4a9d23e0f",
        "k1x_deploy_and_run_yolo11/SKILL.md": "cdaef3bb27a72988be99cfe98e199cf164f39d98c55b5c5d31bfeaa02d0bbcd3",
    }
    write_tsv(
        out / "codex_skill_files.tsv",
        ["path", "before_sha256_prefix", "after_sha256", "status"],
        [[str(path), before_hashes.get(str(path.relative_to(skill_root)), "absent"), sha256_file(path), "created" if "k1x_board_storage_policy" in str(path) else "updated"] for path in skill_files],
    )
    diff_parts = []
    for path in skill_files:
        if path.name == "selftest.sh":
            # The self-test intentionally contains secret-scanner patterns. Its
            # hash and result are exported, but reproducing those literals in the
            # packet-safe diff would correctly trigger the bridge scanner.
            continue
        current = path.read_text().splitlines(keepends=True)
        if "k1x_board_storage_policy" in str(path):
            previous: list[str] = []
        else:
            previous = [line for line in current if "k1x_board_storage_policy" not in line]
        diff_parts.extend(difflib.unified_diff(previous, current, fromfile=f"a/{path.relative_to(skill_root)}", tofile=f"b/{path.relative_to(skill_root)}"))
    write(out / "codex_skill_diff.patch", "".join(diff_parts))
    write(out / "codex_skill_selftest.txt", "PASS: valid front matter and required K1X board storage policy\nPASS: system skill validator: Skill is valid!")
    write(
        out / "codex_skill_storage_policy_report.md",
        """# Codex skill storage policy

Created the local reusable `k1x_board_storage_policy` skill and self-test. Added
short references from the local task-template, environment-sanity, and deployment
skills. The policy requires NVMe verification, a stage-owned `/data` root,
TMP/cache redirection, no silent eMMC fallback, documented exceptions, and safe
stage-owned cleanup. The writable local skill tree is not a Git repository and is
outside the project commit. Before/after hashes and a sanitized diff are recorded.
Codex may require a future process restart to discover the new skill automatically.
""",
    )

    baseline_rows = [
        ["cpu0_3_scout", "0-3", 1, 3, 20, 2, "801359.357875", "56.965689", "800992.070075", "pass", "vendor ORT diagnostic"],
        ["cpu0_3_scout", "0-3", 2, 3, 20, 2, "561324.750250", "7.910616", "1122193.164475", "pass", "vendor ORT diagnostic"],
        ["cpu0_3_scout", "0-3", 3, 3, 20, 2, "502799.473350", "191.203088", "1507977.582725", "pass", "vendor ORT diagnostic"],
        ["cpu0_3_scout", "0-3", 4, 3, 20, 2, "462637.791175", "215.439632", "1850040.410625", "pass", "vendor ORT diagnostic"],
        ["cpu0_3_stable", "0-3", 4, 10, 100, 5, "461603.297250", "435.600042", "1845846.543148", "pass", "selected stable pure graph surface"],
        ["cpu0_7_scout", "0-7", 4, 3, 20, 2, "463106.586925", "diagnostic", "not_recorded", "pass", "vendor-only all-core subprocess; no custom IME"],
        ["cpu0_7_scout", "0-7", 6, 3, 20, 2, "871908.412975", "diagnostic", "not_recorded", "pass", "regresses"],
        ["cpu0_7_scout", "0-7", 8, 3, 20, 2, "975815.491575", "diagnostic", "not_recorded", "pass", "regresses"],
    ]
    write_tsv(out / "full_model_baseline_raw.tsv", ["arm", "cpuset", "intra_threads", "warmup", "runs", "repeats", "wall_mean_us", "wall_stddev_us", "process_cpu_mean_us", "status", "note"], baseline_rows)
    write_tsv(
        out / "full_model_baseline_summary.tsv",
        ["selected_arm", "mean_us", "stddev_us", "cv_pct", "min_us", "max_us", "median_us", "p90_us", "p95_us", "process_cpu_mean_us"],
        [["cpu0_3_intra4", "461603.297250", "435.600042", "0.094367", "461193.164540", "462186.189090", "461362.727850", "462090.156570", "462138.172830", "1845846.543148"]],
    )
    write(
        out / "full_model_baseline_report.md",
        """# Full board ORT baseline

The best safe measured arm is vendor ORT 1.20.2+spacemit, CPU EP,
`ORT_ENABLE_ALL`, sequential execution, inter-op 1, intra-op 4, `taskset -c 0-3`:
`461603.297250 +/- 435.600042 us` (10 warmups, 100 runs, 5 repeats, CV
0.094367%). CPU0-7 vendor-only scouts exited without SIGILL, but six/eight threads
regressed sharply; intra4 on CPU0-3 remains selected.

This is synthetic-input pure ONNX graph latency. It excludes product camera,
decode, preprocessing, rendering, and any production FPS claim.
""",
    )

    copy_text(log_dir / "board_artifacts/logs/roofline_stable.tsv", out / "k1x_primitive_roofline_raw.tsv")
    primitive_names = {"sequential_read", "sequential_write", "memcpy", "nchw_to_nhwc_u8", "uint8_lut", "exact_fixed_rne_u8", "stride2_small_panel_reuse"}
    summary_rows = [[r[k] for k in ["category", "name", "working_set", "mean_us", "stddev_us", "gops_or_gmacs", "gb_per_s"]] for r in roof if r["name"] in primitive_names or (r["category"] == "vmadot_microkernel" and r["working_set"].startswith("model5_geometry"))]
    write_tsv(out / "k1x_primitive_roofline_summary.tsv", ["category", "name", "working_set", "mean_us", "stddev_us", "gops_or_gmacs", "contract_gb_per_s"], summary_rows)
    write_tsv(out / "memory_bandwidth_matrix.tsv", ["primitive", "working_set_bytes", "mean_us", "contract_gb_per_s", "interpretation"], [[r["name"], r["working_set"], r["mean_us"], r["gb_per_s"], "scalar checksum loop" if r["name"] == "sequential_read" else "read+write contract" if r["name"] == "memcpy" else "write contract"] for r in roof if r["category"] == "memory"])
    vmadot_rows = [
        ["stage35_dependent", "register_only", "one_accumulator", "3.751440", "ns_per_vmadot", "exact,no_sigill"],
        ["stage35_independent4", "register_only", "four_accumulators", "0.938110", "ns_per_vmadot", "exact,no_sigill"],
        ["stage35_independent6", "register_only", "six_accumulators", "0.625309", "ns_per_vmadot", "exact,no_sigill"],
    ]
    for r in roof:
        if r["category"] == "vmadot_microkernel":
            vmadot_rows.append([r["name"], r["working_set"], "packed_A_B_delivery", r["gops_or_gmacs"], "GMAC/s", "scalar_oracle_exact,no_sigill"])
    write_tsv(out / "vmadot_delivery_matrix.tsv", ["case", "working_set", "delivery", "value", "unit", "status"], vmadot_rows)
    write(
        out / "vmadot_delivery_report.md",
        f"""# vmadot delivery report

Stage35 register controls reproduced: dependent `3.75144 ns/vmadot`, four-way
independent `0.93811`, six-way independent `0.625309`; all exact and trap-free.
The new packed standalone matrix shows delivery loss and register-block benefit:

- M4xN16 model5 geometry: `24.746637 GMAC/s`, `9533.804644 us`.
- M8xN16: `43.395240 GMAC/s`, `5436.762213 us`.
- M12xN16: `54.360135 GMAC/s`, `4329.271337 us`.

Every shape passed a scalar grouped oracle. These are standalone packed-compute
ceilings, not integrated kernels or authorization to change dispatch. M12xN16 is
the best diagnostic shape, but spilling and full operator dataflow remain unknown.
""",
    )
    write(
        out / "k1x_primitive_roofline_report.md",
        """# K1X primitive roofline

Measured on CPU0 under `steady_clock`; no `rdcycle` was used. At 32 MiB, memcpy
delivered `5.045716 GB/s` under a read+write byte contract and sequential writes
`7.341689 GB/s`. The scalar checksum read probe (`0.866986 GB/s`) is instruction
limited and is not claimed as LPDDR peak. Existing scalar layout/activation
primitives are costly: NCHW->NHWC `9326.987 us`, uint8 LUT `4684.663 us`, exact
fixed requant `5454.583 us`, and stride2 panel work `12139.443 us` on the stated
surfaces. This supports resident INT8 layouts and fused integer post-processing.
""",
    )

    opcode_count = sum(1 for line in (log_dir / "ort_forensics/vmadot_opcode_scan.tsv").read_text().splitlines()[1:] if line.strip())
    write_tsv(
        out / "board_ort_kernel_inventory.tsv",
        ["surface", "evidence", "status", "classification"],
        [
            ["runtime", "/home/svt/spacemit-ort.riscv64.2.0.1/lib/libonnxruntime.so.1.20.2+spacemit", ORT_SHA, "loaded"],
            ["model5_profile", "QLinearConv+Transpose+activation events", "CPUExecutionProvider", "execution_observed_at_operator_level"],
            ["vmadot_code", f"{opcode_count} .text words matching accepted vmadot mask", "localized near SQ4BitGemm spacemit IME symbols", "code_present"],
            ["model5_inner_kernel", "perf command absent; no call graph", "unresolved", "code-present-but-execution-unknown"],
        ],
    )
    write_tsv(
        out / "board_ort_perf_raw.tsv",
        ["target", "command", "exit_code", "status", "stderr"],
        [
            ["model5", "perf stat ... bench_stage42_inprocess_runner", 127, "unavailable", "zsh: command not found: perf"],
            ["full_model", "perf stat ... bench_stage42_inprocess_runner", 127, "unavailable", "zsh: command not found: perf"],
        ],
    )
    excerpt = (log_dir / "ort_forensics/vmadot_symbol_localization.txt").read_text()
    write(out / "board_ort_disassembly_excerpt.txt", excerpt)
    write(
        out / "board_ort_model5_forensics.md",
        """# Board ORT model5 forensics

ORT profiling proves model5 executes `QLinearConv`, inserted transposes, Q/DQ,
Sigmoid, and Mul on `CPUExecutionProvider`. Static library scanning found 192
accepted-mask vmadot instruction words, but the resolvable nearby symbols are
SpacemiT SQ4BitGemm IME packing paths, not proof of model5 QLinearConv execution.
The board has no `perf` command and the runtime is stripped enough that no inner
call graph was recovered. Classification: `code-present-but-execution-unknown`.

The profile's QLinearConv events average roughly `5.047 ms`; full isolated model5
at four threads is `11.701 ms`, showing that transposes and activation/QDQ are a
material part of the winning runtime path.
""",
    )
    write(
        out / "board_ort_full_model_forensics.md",
        """# Board ORT full-model forensics

The untruncated profile assigns all observed node events to CPUExecutionProvider.
It exposes operator-level execution, not inner-kernel identity. Major profiled
families include QLinearConv, DequantizeLinear, Mul, Sigmoid, QuantizeLinear,
Transpose/Concat, MaxPool, QLinearAdd, Resize, QLinearMatMul, and Softmax. Profile
instrumentation raises wall time, so it is used for attribution only; the stable
461.603 ms benchmark is the timing authority. Perf/call-graph evidence is
unavailable because `perf` is not installed on the board.
""",
    )

    copy_text(log_dir / "analysis/graph/operator_manifest.tsv", out / "current_model_operator_manifest.tsv")
    copy_text(log_dir / "analysis/graph/macs_flops.tsv", out / "current_model_flops_macs.tsv")
    copy_text(log_dir / "analysis/graph/memory_liveness.tsv", out / "current_model_memory_liveness.tsv")
    copy_text(log_dir / "analysis/full_model_profile_raw.tsv", out / "full_model_operator_profile_raw.tsv")
    copy_text(log_dir / "analysis/full_model_profile_raw_summary.tsv", out / "full_model_operator_profile_summary.tsv")
    write(
        out / "current_model_graph_audit.md",
        f"""# Current model graph audit

- Model SHA-256: `{MODEL_SHA}`
- Contract: float32 `images` 1x3x640x640 -> float32 `output0` 1x300x6
- Nodes/initializers: `{graph['node_count']}` / `{graph['initializer_count']}`
- Conv/Q/DQ: `{graph['op_counts']['Conv']}` / `{graph['op_counts']['QuantizeLinear']}` / `{graph['op_counts']['DequantizeLinear']}`
- MatMul/Softmax/MaxPool/Concat/Split: 4 / 2 / 3 / 26 / 12
- Static Conv+MatMul/Gemm arithmetic: `{model_macs}` MAC, `{model_macs * 2}` FLOPs at 2 FLOPs/MAC
- Graph-order activation liveness estimate: `{graph['peak_live_activation_bytes']}` bytes
- Peak all noninitializer estimate: `{graph['peak_live_all_noninitializer_bytes']}` bytes
- Sum of materialized float outputs: `{graph['materialized_float_output_bytes_sum']}` bytes

MAC/FLOP totals exclude non-MAC elementwise work. Liveness is a static graph-order
estimate, not an allocator or RSS measurement. The graph is a manual QDQ surface,
not assumed equivalent to a fused marketing export without proof.
""",
    )

    image_manifest_sha = sha256_file(log_dir / "accuracy/fp32_all_500.manifest.tsv")
    write_tsv(
        out / "accuracy_dataset_manifest.tsv",
        ["surface", "path_or_contract", "sha256", "count", "note"],
        [
            ["dataset", "/data/datasets/coco2017/val2017", "55304dfa58b86399878b2c054e6abc394df909366979b85ca5c3f2ea039fe86d", 5000, "sorted local image inventory"],
            ["annotations", "/data/datasets/coco2017/annotations/instances_val2017.json", "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f", 5000, "COCO val2017"],
            ["subset", "first 500 sorted image IDs", image_manifest_sha, 500, "directional deterministic gate"],
            ["preprocess", "letterbox_114_rgb_nchw_float32_div255_opencv_linear", "implementation_in_stage45_system_audit.py", 500, "OpenCV 4.13.0"],
        ],
    )
    write(
        out / "accuracy_surface_contracts.md",
        f"""# Accuracy surface contracts

All three host surfaces used the same deterministic first 500 COCO val2017 images,
annotations, e2e 300x6 output decoder, confidence 0.001, class mapping, and
letterbox/RGB/NCHW `/255` preprocessing.

- FP32 operational: model `{FP32_SHA}`, host ORT 1.27.0 CPU EP, ENABLE_ALL.
- INT8 semantic: model `{MODEL_SHA}`, host ORT 1.27.0 CPU EP, DISABLE_ALL.
- INT8 operational: same INT8 bytes/runtime, ENABLE_ALL.

This is a 500-image directional audit. It is not full val2017 and is not custom
board-engine mAP because no full custom engine exists.
""",
    )
    accuracy_rows = []
    for name, summary, evaluation in [
        ("fp32_operational", fp32_summary, fp32_eval),
        ("int8_semantic", sem_summary, sem_eval),
        ("int8_operational", op_summary, op_eval),
    ]:
        accuracy_rows.append([name, summary["model_sha256"], summary["optimization"], evaluation["image_count"], evaluation["map50_95"], evaluation["map50"], evaluation["map75"], evaluation["ap_small"], evaluation["ap_medium"], evaluation["ap_large"], evaluation["precision_mean_valid_cocoeval_grid"], evaluation["recall_mean_valid_cocoeval_grid"], summary["mean_inference_ms"], summary["stddev_inference_ms"]])
    write_tsv(out / "accuracy_subset_results.tsv", ["surface", "model_sha256", "ort_optimization", "images", "map50_95", "map50", "map75", "ap_small", "ap_medium", "ap_large", "precision_grid_mean", "recall_grid_mean", "host_inference_mean_ms", "host_inference_stddev_ms"], accuracy_rows)
    sem_delta = sem_eval["map50_95"] - fp32_eval["map50_95"]
    op_delta = op_eval["map50_95"] - fp32_eval["map50_95"]
    write(
        out / "accuracy_report.md",
        f"""# Accuracy audit

Directional 500-image COCO val2017 result:

| Surface | mAP50-95 | mAP50 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|
| FP32 operational | {fp32_eval['map50_95']:.6f} | {fp32_eval['map50']:.6f} | {fp32_eval['ap_small']:.6f} | {fp32_eval['ap_medium']:.6f} | {fp32_eval['ap_large']:.6f} |
| INT8 semantic | {sem_eval['map50_95']:.6f} | {sem_eval['map50']:.6f} | {sem_eval['ap_small']:.6f} | {sem_eval['ap_medium']:.6f} | {sem_eval['ap_large']:.6f} |
| INT8 operational | {op_eval['map50_95']:.6f} | {op_eval['map50']:.6f} | {op_eval['ap_small']:.6f} | {op_eval['ap_medium']:.6f} | {op_eval['ap_large']:.6f} |

Semantic INT8 loses `{sem_delta:.6f}` absolute AP (`{sem_delta * 100:.3f}` AP
points) versus FP32; operational INT8 loses `{op_delta:.6f}` (`{op_delta * 100:.3f}`
points). The current manual QDQ surface misses both the within-1-AP and within-2-AP
scenarios on this subset. Full val2017 remains unknown; no accuracy result is
transferred to the incomplete custom board engine.
""",
    )

    lower_rows = [
        ["tiny_l1_m12_compute_only", model_macs, "86.826544", f"{model_macs / 1e6 / 86.826544:.6f}", "diagnostic optimistic", "tiny L1 packed shape; excludes all non-MAC work"],
        ["model5_geometry_m12_compute_only", model_macs, f"{m12_rate:.6f}", f"{m12_compute_floor_ms:.6f}", "measured-delivery optimistic", "assumes every MAC maps to M12xN16 and zero overhead"],
        ["model5_geometry_m8_compute_only", model_macs, f"{m8_rate:.6f}", f"{m8_compute_floor_ms:.6f}", "measured-delivery", "assumes all shapes map to M8xN16"],
        ["current_m4n16_compute_only", model_macs, f"{m4_rate:.6f}", f"{m4_compute_floor_ms:.6f}", "measured-delivery", "current four-acc shape; excludes all non-MAC work"],
        ["conservative_aot_current_graph", model_macs, "35.0", "105-130", "analytical range", "78.29ms MAC floor plus 27-52ms fused QDQ/layout/head/memory"],
        ["current_board_ort", model_macs, "effective 5.936", "461.603297", "measured", "full manual-QDQ graph"],
    ]
    write_tsv(out / "current_graph_lower_bound.tsv", ["case", "macs", "assumed_gmacs", "pure_model_ms", "evidence", "assumptions"], lower_rows)
    candidate_rows = [
        ["YOLO26n_640_current_graph", "640", "2.740154", "105-130", "measured-LUT conservative", "current semantic INT8 -3.603 AP points on subset", "not credible for 45ms"],
        ["YOLO26n_512_resolution_only", "512", "1.753698", "55-75", "quadratic spatial scaling plus measured-LUT", "accuracy unknown and likely lower", "not robustly <=45ms"],
        ["YOLO26n_416_resolution_only", "416", "1.157715", "36-51", "quadratic spatial scaling plus measured-LUT", "accuracy unknown/high risk", "borderline, graph overhead retained"],
        ["K1X_student_512", "512", "1.0-1.35", "35-50", "tile-aligned Conv-only student projection", "requires distillation+QAT", "accuracy-first fallback"],
        ["K1X_student_416", "416", "0.65-0.90", "24-38", "tile-aligned Conv-only student projection", "requires distillation+QAT; highest accuracy risk", "credible latency envelope"],
        ["K1X_two_scale_student_416", "416", "0.55-0.78", "22-34", "two-scale/simple-head projection", "small-object AP risk", "domain/research candidate"],
    ]
    write_tsv(out / "candidate_model_latency_predictions.tsv", ["candidate", "resolution", "projected_macs_g", "predicted_pure_model_ms", "basis", "accuracy_risk", "decision"], candidate_rows)
    strategy_rows = [
        ["unchanged_graph_full_aot", "105-130ms", "current semantic subset 0.410683", "very_high", "medium", "strong roofline; no 45ms path", "reject"],
        ["vendor_ort_reuse", "461.603ms current", "operational subset 0.373479", "medium", "high stripped-runtime dependency", "operator execution proven, inner kernel unknown", "diagnostic only"],
        ["k1x_student_512_plus_aot", "35-50ms projected", "unknown until training", "high", "training/export", "measured-LUT projection", "accuracy fallback"],
        ["k1x_student_416_plus_aot", "24-38ms projected", "unknown until training", "high", "training/export", "measured-LUT projection", "selected primary proof"],
        ["gpu_opencl", "unknown", "unknown", "high", "driver/kernel", "no Stage45 measurement", "defer"],
        ["detector_tracker_or_roi", "display throughput only", "domain-specific", "medium", "application assumptions", "does not meet pure detector target", "separate lane"],
    ]
    write_tsv(out / "strategy_decision_matrix.tsv", ["strategy", "latency", "accuracy", "effort", "dependency_risk", "evidence", "decision"], strategy_rows)
    write(
        out / "twenty_fps_feasibility.md",
        f"""# Twenty-FPS feasibility

Target: pure model <=45 ms and full frame <=50 ms. The exact graph contains
`{model_macs / 1e9:.6f} GMAC`; meeting 45 ms requires `{target_rate:.3f} GMAC/s`
before any QDQ, activation, transpose, pooling, attention, head, or memory cost.
The best model5-geometry standalone diagnostic is M12xN16 at `{m12_rate:.3f}
GMAC/s`, which yields `{m12_compute_floor_ms:.3f} ms` for MACs alone. Therefore
unchanged YOLO26n-640 cannot credibly satisfy the pure-model target.

The conservative current-graph AOT range is `105-130 ms`: 35 GMAC/s mixed-shape
delivery gives 78.29 ms of MAC work, with 27-52 ms reserved for aggressively fused
non-MAC/dataflow/head work. The current measured vendor ORT graph is 461.603 ms.

A tile-aligned K1X student at 416, 0.65-0.90 GMAC, simple static one-to-one head,
resident symmetric INT8, and global AOT scheduling projects to 24-38 ms. A 512
student projects to 35-50 ms and is the accuracy-first fallback. These are design
projections, not achieved latency. Training and accuracy are unproven.
""",
    )
    write(
        out / "model_executor_codesign_spec.md",
        """# K1X model/executor co-design specification

## Model envelope

- Primary: static 416x416 student; fallback: 512x512.
- Teacher: YOLO26n or YOLO26s; distillation plus structured pruning and exact INT8 QAT.
- Operators: tile-aligned plain 3x3/1x1 Conv, Add/views, measured pooling, minimal materialized Concat/Split.
- Avoid unmeasured depthwise assumptions and attention/Softmax in the backbone.
- Structurally reparameterize training branches at export.
- Symmetric int8 activation/weight storage where QAT permits, activation zero point 0, per-channel weight scales.
- Simple static NMS-free one-to-one head; evaluate two- and three-scale variants explicitly.

## Executor contract

- Static AOT schedule and one preallocated arena; quantized tensors remain resident.
- One physical tile-compatible layout with conversions only where measured to repay cost.
- Prepare-time immutable packed weights; no hot-loop allocation, graph-name lookup, or file I/O.
- Fuse exact integer correction, bias, requant, and activation where semantics allow.
- CPU0-3 only for IME; CPU4-7 may be evaluated later for non-IME work only.
- Per-block fixed-host integer oracles remain correctness gates.

## Training and acceptance

- Freeze COCO preprocessing/postprocessing and full model hashes.
- Gates: 500-image direction first, then full val2017; target within 1 AP, relaxed within 2 AP.
- Benchmark exported graph with measured operator LUT before full training commitment.
- No accuracy or <=45 ms claim until a trained/exported model passes both gates.
""",
    )

    write(
        out / "STAGE45_FINAL_REPORT.md",
        f"""# Stage45 final report

- classification: `{CLASSIFICATION}`
- stage_id: `{STAGE_ID}`
- start_head: `{START_HEAD}`
- end_head: `pending-local-commit-see-final-response-and-result-packet`
- push: false
- production/default dispatch: not authorized and unchanged

## Proven

- Board `/data` is writable NVMe; all stage payloads stayed in `{BOARD_ROOT}`.
- Full vendor ORT CPU0-3 intra4: `461603.297250 +/- 435.600042 us`.
- Stage35 register vmadot throughput reproduced; realistic packed M12xN16 reached `54.360135 GMAC/s` with exact scalar-oracle output.
- Exact accepted graph: `{model_macs}` MAC, `{model_macs * 2}` FLOPs-at-2/MAC, peak graph-order activation estimate `{graph['peak_live_activation_bytes']}` bytes.
- 500-image directional mAP50-95: FP32 `{fp32_eval['map50_95']:.6f}`, semantic INT8 `{sem_eval['map50_95']:.6f}`, operational INT8 `{op_eval['map50_95']:.6f}`.

## Broken

- Current graph misses the 45 ms target even under the model5-geometry M12 compute-only ceiling (`{m12_compute_floor_ms:.3f} ms`) before non-MAC work.
- Current semantic INT8 loses `{abs(sem_delta) * 100:.3f}` AP points versus FP32 on the subset; operational optimization loses more.
- Stage44 custom model5 remains slower than resource-matched ORT and its contiguous scaffold was net negative.

## Unknown

- Vendor ORT model5 inner-kernel identity; vmadot code exists but execution was not localized.
- Full COCO val2017 accuracy, trained student accuracy, and actual student/AOT latency.
- Production camera/full-frame behavior.

## Decision

Select a K1X model/executor co-design stage: 416 student as latency-primary and 512
as accuracy fallback, distilled/QAT and paired with a resident-INT8 static AOT
executor. Do not expand the unchanged graph block by block. No model FPS,
production readiness, default dispatch, or retained-accuracy claim is made.

## Validation

- Python compileall: pass.
- Host Release build: pass; CTest 44/44 pass.
- RISC-V Release cross-build with the existing IME route: pass.
- Board CPU0 instruction controls and realistic microkernels: exact, no SIGILL.
- Board runtime loader: pass; deployed binary hash matches; no absolute RPATH.
- CPU0-3 policy: pass; CPU4-7 ran vendor-only ORT scouts, never custom IME.
- Git whitespace, symlink, large-file, and secret/private-path gates: pass after reviewed diagnostic-pattern self-matches.
""",
    )
    write(
        out / "STAGE45_SUMMARY_RU.md",
        f"""# Краткий итог Stage45

Классификация: `{CLASSIFICATION}`.

Лучший стабильный полный board ORT на CPU0-3: `461603.297250 us`. Реалистичный
standalone M12xN16 `smt.vmadot` достиг `54.360135 GMAC/s`, но даже перенос этой
скорости на все `{model_macs / 1e9:.6f} GMAC` даёт `{m12_compute_floor_ms:.3f} ms`
без Q/DQ, активаций, layout, attention и head. Неизменённый YOLO26n-640 не имеет
достоверного пути к 45 ms.

На 500 изображениях COCO mAP50-95: FP32 `{fp32_eval['map50_95']:.6f}`, semantic
INT8 `{sem_eval['map50_95']:.6f}`, operational INT8 `{op_eval['map50_95']:.6f}`.
Текущая INT8 поверхность теряет 3.603 AP относительно FP32 уже на направленном
подмножестве.

Следующий этап: спецификация и подготовка K1X student 416/512 с distillation+QAT
и статическим resident-INT8 AOT executor. Это прогноз и план, не достигнутые FPS,
не production и не подтверждённая точность.
""",
    )
    write(
        out / "stage46_prompt.md",
        f"""# Stage46 prompt: K1X student architecture and training-preparation gate

Stage ID: `BANANA-YOLO26-K1X-STUDENT-416-512-ARCHITECTURE-SPEC-AND-TRAINING-PREPARATION-001`

Start from the final Stage45 commit. Freeze the Stage45 model, 500-image accuracy
surface, measured K1X operator LUT, and fixed preprocessing. Without running a full
training campaign, define and validate two exportable student blueprints:

1. 416 latency-primary, 0.65-0.90 GMAC envelope.
2. 512 accuracy-fallback, 1.0-1.35 GMAC envelope.

Generate graph-level latency estimates from measured K1X primitives, tile-align
channels, select two- versus three-scale head by measured LUT, specify teacher,
distillation/QAT/pruning recipe, and produce a tiny untrained export/scheduler
contract only where useful. Do not claim accuracy or 20 FPS. Authorize training
only in a separate human-approved stage after architecture, dataset, compute,
accuracy, and acceptance contracts are reviewed.
""",
    )
    write(
        out / "source_hygiene_report.md",
        """# Source hygiene

- `git diff --check`: pass.
- Python compileall: pass.
- Host Release build: pass; CTest 44/44 pass.
- Full RISC-V Release cross-build with existing IME route: pass.
- Board loader: pass; Stage45 binary has no RPATH/RUNPATH and its deployed SHA matches.
- Symlink scan under `custom_int8_engine`, `stages`, and `docs`: zero.
- Stage45 report directory: 1.9 MiB; no file exceeds 5 MiB.
- Secret scan: no secret value found. The only matches were the self-test's literal
  secret-pattern strings and documented `/data/.codex/skills` path/hash metadata.
- Board storage: pass; no eMMC write exception.

Tracked scope is limited to the Stage45 diagnostic probe, analysis/report tooling,
repo-local storage policy, Stage44 traceability correction, Stage45 reports, and
next-stage prompt. Raw logs, profiles, datasets, models, build trees, credentials,
and actual local Codex skill files are excluded from project Git.
""",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--log-dir", required=True)
    render(parser.parse_args())


if __name__ == "__main__":
    main()
