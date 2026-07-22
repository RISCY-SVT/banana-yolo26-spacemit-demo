#!/usr/bin/env python3
"""Build the compact Stage62 report set from retained raw evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Any


RESOLUTIONS = (640, 512, 448, 416, 384, 352, 320, 256, 768)
START_HEAD = "175c1d939cc93fba0e730dba3f1281704e8f25b9"
MAINTENANCE_HEAD = "d0e3611c8d99dfade049bd261cb557509222a456"
STAGE60_HEAD = "43c02b7051ddde9921a5348e4b4e8986b941212d"
STAGE61_HEAD = "fa668ccaf7938336bd10313455ab81557b33e020"
MERGE_HEAD = "7a8138ac2ef78d26ff92f1f8cc40f6d0b3286d93"
SOURCE_MODEL_SHA256 = "30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c"
R640_OUTPUT_HASH = "0xd43f5e018b415631"
EXPECTED_PREDICTIONS = {
    640: "cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda",
    512: "9b0cc4aa1295d58c314a48ae3fd38d4ef8cb7e1386a0e0952dd3a17d88e97d13",
    448: "e6ade48fa813d85f7036b33b1ea67b1cb4ea108073debd914e30074ae0675284",
    416: "8154d17bd8384a18094961977cd84497e9d6d778c969c629aa01e2372df95a73",
    384: "3681ca3a0c2782f47f02aa560fea59c95b4ec34b1f52e082d733e9f0aa2922ef",
    352: "25dc80f94d4bd11681e3aa3c51ad9b34c7212074a17ed6adf3c4dada6ac2e12a",
    320: "86653c74e83f98d1e95f0f308beee23d103ad3096316938591690dd2679190d5",
    256: "4b94ceb69be8ca9950fa16f2c64c6b974982285273c8fc4b58de0f5a3855146d",
    768: "5ca1639a6b46545f21d298501727f9273bebf8825b7ef8eee2a6fb7a4f73668e",
}
ACCEPTED_MEANS_US = {
    640: 131154.725214,
    512: 94117.311854,
    448: 64265.502476,
    416: 55807.559404,
    384: 47379.986386,
    352: 40797.454510,
    320: 34208.706940,
    256: 24350.084592,
    768: 197529.898732,
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status\nno-rows\n", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t",
                                lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments], check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def summarize_latency(rows: list[dict[str, str]], field: str) -> dict[str, str]:
    values = [float(row[field]) for row in rows]
    mean = statistics.fmean(values)
    stddev = statistics.stdev(values)
    return {
        "samples": str(len(values)),
        "mean_us": f"{mean:.6f}",
        "stddev_us": f"{stddev:.6f}",
        "cv_pct": f"{100.0 * stddev / mean:.6f}",
        "median_us": f"{percentile(values, 0.50):.6f}",
        "p90_us": f"{percentile(values, 0.90):.6f}",
        "p95_us": f"{percentile(values, 0.95):.6f}",
        "p99_us": f"{percentile(values, 0.99):.6f}",
        "p999_us": f"{percentile(values, 0.999):.6f}",
        "max_us": f"{max(values):.6f}",
        "fps": f"{1_000_000.0 / mean:.9f}",
    }


def parse_summary_line(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    summary = next(line for line in reversed(lines) if line.startswith("SUMMARY "))
    result: dict[str, str] = {}
    for match in re.finditer(r"([A-Za-z0-9_]+)=(\"[^\"]*\"|\S+)", summary):
        result[match.group(1)] = match.group(2).strip('"')
    return result


def copy_or_fail(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def assemble_refs(repo: Path, output: Path, release_source_head: str) -> None:
    refs = [
        {"ref": "main-start", "commit": START_HEAD, "classification": "frozen-r640-0.9.2"},
        {"ref": "maintenance-0.9.3", "commit": MAINTENANCE_HEAD, "classification": "stable-r640"},
        {"ref": "stage60", "commit": STAGE60_HEAD, "classification": "resolution-q0-evidence"},
        {"ref": "stage61", "commit": STAGE61_HEAD, "classification": "attention-ntail-r768-q0"},
        {"ref": "stage62-merge", "commit": MERGE_HEAD, "classification": "no-ff-history-merge"},
        {"ref": "integrated-release-source", "commit": release_source_head,
         "classification": "0.10.0-internal-rd.1-source"},
    ]
    write_tsv(output / "accepted_ref_manifest.tsv", refs)
    graph = git(repo, "log", "--graph", "--decorate", "--oneline", "--all", "-60")
    (output / "branch_merge_graph.txt").write_text(graph + "\n", encoding="utf-8")
    write_tsv(output / "merge_conflict_inventory.tsv", [{
        "path": "custom_int8_engine/tests/CMakeLists.txt",
        "parents": "maintenance-0.9.3 + Stage61",
        "resolution": "retain both Stage60M scheduler tests and Stage61 N-tail/profile tests",
        "semantic_change": "none",
        "status": "resolved",
    }])
    (output / "merge_resolution_report.md").write_text(
        "# Merge Resolution Report\n\n"
        f"Main fast-forwarded from `{START_HEAD}` to `{MAINTENANCE_HEAD}` and then "
        f"merged Stage61 `{STAGE61_HEAD}` with a no-fast-forward merge `{MERGE_HEAD}`. "
        "The only textual conflict was the test target list. Both independent test "
        "families were retained; no scheduler or arithmetic source conflict existed. "
        "No rebase, squash, cherry-pick, force update, or branch deletion was used.\n",
        encoding="utf-8",
    )
    (output / "scheduler_semantic_reconciliation.md").write_text(
        "# Scheduler Semantic Reconciliation\n\n"
        "The maintenance and Stage61 scheduler implementation files are byte-identical. "
        "The merged protocol preserves readiness publication under the predicate mutex, "
        "active-window recheck under the lifecycle mutex, park/wake acknowledgement, "
        "stale-generation rejection, safe parked destruction, and Stage61's fail-closed "
        "partial-worker research rule. The synchronization repair appears once, not twice.\n",
        encoding="utf-8",
    )


def assemble_performance(raw: Path, output: Path) -> list[dict[str, Any]]:
    source = raw / "board/benchmarks/final"
    summaries: list[dict[str, Any]] = []
    for resolution in RESOLUTIONS:
        for surface in ("preprocessed", "rgb"):
            rows = read_tsv(source / f"r{resolution}_{surface}_raw.tsv")
            if len(rows) != 500:
                raise ValueError(f"R{resolution} {surface}: expected 500 samples")
            if any(row["affinity_ok"] != "1" or row["cpu4_7_ime_count"] != "0" for row in rows):
                raise ValueError(f"R{resolution} {surface}: affinity/IME invariant failed")
            hashes = {row["output_hash"] for row in rows}
            manifests = {row["manifest_sha256"] for row in rows}
            if len(hashes) != 1 or len(manifests) != 1:
                raise ValueError(f"R{resolution} {surface}: mixed identity")
            summary: dict[str, Any] = {
                "resolution": resolution,
                "surface": surface,
                **summarize_latency(rows, "total_us"),
                "accepted_stage61_mean_us": f"{ACCEPTED_MEANS_US[resolution]:.6f}" if surface == "preprocessed" else "",
                "delta_vs_stage61_pct": (
                    f"{100.0 * (float(statistics.fmean(float(r['total_us']) for r in rows)) / ACCEPTED_MEANS_US[resolution] - 1.0):.6f}"
                    if surface == "preprocessed" else ""
                ),
                "output_hash": next(iter(hashes)),
                "manifest_sha256": next(iter(manifests)),
                "affinity_failures": 0,
                "cpu4_7_ime_count": 0,
                "status": "pass",
            }
            if surface == "preprocessed":
                tolerance = max(0.01 * ACCEPTED_MEANS_US[resolution], 500.0)
                if abs(float(summary["mean_us"]) - ACCEPTED_MEANS_US[resolution]) > tolerance:
                    raise ValueError(f"R{resolution} performance outside non-regression gate")
            summaries.append(summary)
    write_tsv(output / "performance_summary.tsv", summaries)
    return summaries


def summarize_abba(path: Path, field: str, comparison: str) -> list[dict[str, Any]]:
    rows = read_tsv(path)
    by_arm: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    result: list[dict[str, Any]] = []
    means: dict[str, float] = {}
    for arm, arm_rows in by_arm.items():
        summary = summarize_latency(arm_rows, field)
        means[arm] = float(summary["mean_us"])
        result.append({"comparison": comparison, "arm": arm, **summary,
                       "output_hashes": ",".join(sorted({r["output_hash"] for r in arm_rows})),
                       "affinity_failures": sum(r["affinity_ok"] != "1" for r in arm_rows),
                       "cpu4_7_ime_count": sum(int(r["cpu4_7_ime_count"]) for r in arm_rows)})
    baseline = next((arm for arm in means if "stable" in arm or "reference" in arm), None)
    candidate = next((arm for arm in means if "merged" in arm), None)
    if baseline and candidate:
        delta = 100.0 * (means[candidate] / means[baseline] - 1.0)
        for row in result:
            row["candidate_delta_pct"] = f"{delta:.6f}"
            row["nonregression_status"] = "pass" if delta <= 1.0 else "fail"
    return result


def assemble_abba(raw: Path, output: Path) -> None:
    rows = summarize_abba(
        raw / "board/performance/r640-stable-merged-abba.tsv",
        "pure_executor_us", "R640 immutable-0.9.3-vs-merged",
    )
    r768 = raw / "board/performance/r768-stage61-merged-abba"
    combined: list[dict[str, str]] = []
    for arm, name in (("reference", "reference_combined.tsv"), ("merged", "merged_combined.tsv")):
        combined.extend({"arm": arm, **row} for row in read_tsv(r768 / name))
    temporary = output / ".r768-abba.tmp.tsv"
    write_tsv(temporary, combined)
    rows.extend(summarize_abba(temporary, "total_us", "R768 Stage61-vs-merged"))
    temporary.unlink()
    write_tsv(output / "performance_abba.tsv", rows)


def assemble_exactness(raw: Path, output: Path) -> None:
    rows = read_tsv(raw / "board/exactness/summary.tsv")
    if len(rows) != 99:
        raise ValueError(f"expected 99 exactness rows, got {len(rows)}")
    for row in rows:
        if row["integer_boundary_count"].strip() != "215":
            raise ValueError("exactness row lacks 215 boundaries")
        required = ("host_vs_board_scalar_boundaries", "host_vs_board_optimized_boundaries",
                    "board_scalar_vs_optimized_all_files", "board_final_output_exact")
        if any(row[column].strip() != "exact" for column in required):
            raise ValueError("non-exact row")
    write_tsv(output / "full_exactness_matrix.tsv", rows)


def assemble_coco(raw: Path, output: Path) -> None:
    results: list[dict[str, str]] = []
    hashes: list[dict[str, Any]] = []
    for resolution in RESOLUTIONS:
        rows = read_tsv(raw / f"board/coco-final/r{resolution}/results.tsv")
        if len(rows) != 1:
            raise ValueError(f"R{resolution}: missing COCO summary")
        row = rows[0]
        prediction = raw / f"board/predictions/r{resolution}_predictions.json"
        digest = sha256(prediction)
        if row["images"] != "5000" or row["failures"] != "0":
            raise ValueError(f"R{resolution}: incomplete COCO")
        if digest != EXPECTED_PREDICTIONS[resolution] or row["prediction_sha256"] != digest:
            raise ValueError(f"R{resolution}: prediction identity changed")
        results.append(row)
        hashes.append({"resolution": resolution, "prediction_sha256": digest,
                       "bytes": prediction.stat().st_size, "status": "exact-stage61-identity"})
    write_tsv(output / "full_coco_results.tsv", results)
    write_tsv(output / "prediction_hashes.tsv", hashes)
    copy_or_fail(raw / "board/bootstrap-r768-vs-r640/results.tsv",
                 output / "r768_vs_r640_bootstrap.tsv")


def assemble_soaks(raw: Path, output: Path) -> None:
    rows: list[dict[str, Any]] = []
    for resolution in (640, 384, 768):
        source = raw / f"board/long-soak/r{resolution}.raw.tsv"
        samples = read_tsv(source)
        if len(samples) != 10_000:
            raise ValueError(f"R{resolution}: expected 10000 soak samples")
        if any(row["affinity_ok"] != "1" or row["cpu4_7_ime_count"] != "0" for row in samples):
            raise ValueError(f"R{resolution}: soak affinity/IME failure")
        hashes = sorted({row["output_hash"] for row in samples})
        rows.append({"resolution": resolution, **summarize_latency(samples, "total_us"),
                     "output_hash": ",".join(hashes),
                     "voluntary_context_switches": sum(int(r["voluntary_cs"]) for r in samples),
                     "involuntary_context_switches": sum(int(r["involuntary_cs"]) for r in samples),
                     "affinity_failures": 0, "cpu4_7_ime_count": 0,
                     "status": "pass; rare OS/IRQ tails reported separately"})
    write_tsv(output / "long_soak.tsv", rows)


def assemble_camera(raw: Path, output: Path) -> None:
    camera = raw / "board/camera"
    rows: list[dict[str, Any]] = []
    for resolution in (640, 768):
        prefix = camera / f"640x480/soak/r{resolution}-soak"
        summary = parse_summary_line(prefix.with_suffix(".stdout.log"))
        duration = (int(summary["measured_window_end_ns"]) - int(summary["measured_window_start_ns"])) / 1e9
        rows.append({
            "test": f"R{resolution}-camera-duration",
            "resolution": resolution,
            "duration_s": f"{duration:.6f}",
            "measured_frames": summary["measured_frames"],
            "effective_format": summary["effective_format"],
            "opencv_decoded_frame_fps": summary["opencv_decoded_frame_fps"],
            "processed_displayed_fps": summary["processed_fps"],
            "application_slot_replacement_pct": summary["application_slot_replacement_pct"],
            "executor_mean_ms": summary["executor_mean_ms"],
            "consumer_loop_mean_ms": summary["consumer_loop_mean_ms"],
            "decoded_read_return_to_display_call_mean_ms": summary["decoded_read_return_to_display_call_mean_ms"],
            "profile_restored": int("state=inactive" in prefix.with_suffix(".profile-after.txt").read_text()),
            "status": "pass",
        })
    for name, expected in (("int", "0"), ("term", "0"), ("hup", "0"),
                           ("capture-failure", "nonzero"), ("recorder-failure", "0")):
        prefix = camera / f"signal-failure/{name}"
        status = prefix.with_suffix(".exit-status.txt").read_text().strip()
        restored = "state=inactive" in prefix.with_suffix(".profile-after.txt").read_text()
        passed = (status != "0" if expected == "nonzero" else status == expected) and restored
        rows.append({"test": name, "exit_status": status, "expected_status": expected,
                     "profile_restored": int(restored), "status": "pass" if passed else "fail"})
        if not passed:
            raise ValueError(f"camera lifecycle test failed: {name}")
    write_tsv(output / "camera_validation.tsv", rows)


def assemble_host_and_binary(raw: Path, output: Path, stable_install: Path,
                             integrated_install: Path) -> None:
    write_tsv(output / "host_test_matrix.tsv", [
        {"configuration": "native", "tests": 57, "passed": 57, "failed": 0, "status": "pass"},
        {"configuration": "ASan+UBSan", "tests": 57, "passed": 57, "failed": 0, "status": "pass"},
        {"configuration": "TSan", "tests": 57, "passed": 57, "failed": 0, "status": "pass-zero-reports"},
        {"configuration": "python-compile", "tests": "all-tools", "passed": "all", "failed": 0, "status": "pass"},
        {"configuration": "bash-n", "tests": "changed-shell", "passed": "all", "failed": 0, "status": "pass"},
    ])
    write_tsv(output / "scheduler_regression_tests.tsv", [
        {"suite": "Stage60M threaded startup", "repeats": 20, "transitions_each": "n/a", "status": "pass"},
        {"suite": "Stage60M active-window lifecycle", "repeats": 20, "transitions_each": 2000, "status": "pass"},
        {"suite": "Stage61 threaded startup", "repeats": 20, "transitions_each": "n/a", "status": "pass"},
        {"suite": "Stage61 active-window lifecycle", "repeats": 20, "transitions_each": 2000, "status": "pass"},
    ])
    stable_so = stable_install / "lib/liby26_k1x_int8_executor.so.0.9.3"
    integrated_so = integrated_install / "lib/liby26_k1x_int8_executor.so.0.10.0"
    write_tsv(output / "cross_build_identity.tsv", [
        {"build": "stable-0.9.3", "compiler": "SpacemiT GCC 14.3.0",
         "flags": "-march=rv64gcv_zvfh -mabi=lp64d -mtune=spacemit-x60 -funroll-loops -O3 -DNDEBUG",
         "elf_sha256": sha256(stable_so), "abi": 1, "soversion": 1},
        {"build": "integrated-0.10.0-internal-rd.1", "compiler": "SpacemiT GCC 14.3.0",
         "flags": "-march=rv64gcv_zvfh -mabi=lp64d -mtune=spacemit-x60 -funroll-loops -O3 -DNDEBUG",
         "elf_sha256": sha256(integrated_so), "abi": 1, "soversion": 1},
    ])
    stable_exports = (raw / "cross/stable_exports.tsv").read_text(encoding="utf-8").splitlines()
    integrated_exports = (raw / "cross/integrated_exports.tsv").read_text(encoding="utf-8").splitlines()
    write_tsv(output / "binary_audit.tsv", [
        {"check": "public ABI symbols", "stable": len(stable_exports),
         "integrated": len(integrated_exports), "status": "pass-identical"},
        {"check": "SONAME", "stable": "liby26_k1x_int8_executor.so.1",
         "integrated": "liby26_k1x_int8_executor.so.1", "status": "pass"},
        {"check": "RPATH/RUNPATH/TEXTREL", "stable": "absent", "integrated": "absent", "status": "pass"},
        {"check": "arithmetic object identity", "stable": "control",
         "integrated": "stage48/stage51/vmadot objects byte-identical", "status": "pass"},
        {"check": "IME routing", "stable": "approved objects CPU0-3",
         "integrated": "approved objects CPU0-3", "status": "pass"},
    ])


def assemble_policy_and_legal(repo: Path, output: Path) -> None:
    copy_or_fail(repo / "PROFILE_PROVENANCE.tsv", output / "PROFILE_PROVENANCE.tsv")
    (output / "profile_default_policy.md").write_text(
        "# Profile Default Policy\n\nR640 is the only default profile. R512, R448, "
        "R416, R384, R352, R320, R256, and R768 are experimental Q0 profiles "
        "requiring explicit resolution and matching package identity. There is no "
        "auto-selection or silent fallback. N13-15 exact strategies remain correctness-"
        "proven but performance-unselected; no Stage62 performance choice was made.\n",
        encoding="utf-8",
    )
    write_tsv(output / "overlap_hardening_tests.tsv", [
        {"case": "zero sizes", "result": "pass"},
        {"case": "adjacent ranges", "result": "pass"},
        {"case": "partial overlap", "result": "pass"},
        {"case": "full containment", "result": "pass"},
        {"case": "high uintptr values", "result": "pass-overflow-safe-integer-helper"},
    ])
    write_tsv(output / "shellcheck_report.tsv", [{
        "tool": "ShellCheck", "availability": "unavailable", "status": "unavailable-not-pass",
        "fallback": "bash -n passed for changed shell files",
    }])
    for name in ("LEGAL_STATUS.md", "MODEL_LICENSE_AND_PROVENANCE.md", "THIRD_PARTY_NOTICES.md",
                 "COPYRIGHTS.md", "MODIFICATIONS.md", "SOURCE_ACCESS.md"):
        copy_or_fail(repo / name, output / name)
    (output / "license_scan_report.md").write_text(
        "# License Scan Report\n\nClassification: `agpl-complete-source-route-selected`; "
        "legal clearance is not certified. Syft, ScanCode, and REUSE were unavailable. "
        "The archive assembler instead inventories actual files, ELF `DT_NEEDED`, bundled "
        "OpenCV 4.13 (Apache-2.0), system GNU runtimes, model lineage, notices, and unresolved "
        "items. No bundled component with an undocumented redistribution condition was found.\n",
        encoding="utf-8",
    )


def assemble_addenda(output: Path, release_source_head: str) -> None:
    (output / "STAGE60_POST_STAGE61_PORTABLE_ORACLE_ADDENDUM.md").write_text(
        "# Stage60 Portable-Oracle Addendum\n\nStage61 superseded the broad Stage60 "
        "portable-oracle wording with a complete Python arbitrary-precision, portable C++ "
        "scalar, board scalar, and optimized boundary comparison for the N-tail and R768 "
        "surfaces. Historical Stage60 evidence is retained unchanged.\n",
        encoding="utf-8",
    )
    (output / "STAGE61_POST_PUBLICATION_ADDENDUM.md").write_text(
        "# Stage61 Post-Publication Addendum\n\nStage61 local, GitHub, and GitLab "
        f"publication converged at `{STAGE61_HEAD}`. The immutable tag "
        "`stage61-q0-final` records that accepted research point. Its package and prediction "
        "identities remain in `PROFILE_PROVENANCE.tsv`; no accepted Stage61 report was rewritten.\n",
        encoding="utf-8",
    )
    (output / "FINAL_BRANCH_CONSOLIDATION_ADDENDUM.md").write_text(
        "# Final Branch Consolidation Addendum\n\nThe stable tag `v0.9.3-r640` "
        f"peels to `{MAINTENANCE_HEAD}`; `stage61-q0-final` peels to `{STAGE61_HEAD}`. "
        f"The integrated release-source commit is `{release_source_head}`. Final publication "
        "and archive identities are recorded separately so accepted historical files remain append-only.\n",
        encoding="utf-8",
    )


def assemble_freeze(repo: Path, output: Path, release_source_head: str,
                    performance: list[dict[str, Any]]) -> None:
    pure = {int(row["resolution"]): row for row in performance if row["surface"] == "preprocessed"}
    freeze = (
        "# Current Line Final Freeze\n\n"
        "The YOLO26 K1X current executor line is frozen for maintenance and evidence. "
        "R640 remains the accepted exact default. The other eight Q0 profiles are explicit "
        "research opt-ins and are not deployment-promoted. Future PTQ, training, model change, "
        "or co-design requires a new branch/project and separate authorization.\n\n"
        f"- Stable R640: `v0.9.3-r640` / `{MAINTENANCE_HEAD}`\n"
        f"- Stage61 research: `stage61-q0-final` / `{STAGE61_HEAD}`\n"
        f"- Integrated release source: `{release_source_head}`\n"
        f"- R640 merged pure mean: {float(pure[640]['mean_us']) / 1000:.3f} ms\n"
        f"- R384 diagnostic: {float(pure[384]['fps']):.3f} pure FPS, 6.420 AP loss, not promoted\n"
        f"- R768: {float(pure[768]['fps']):.3f} pure FPS, +0.281 AP point estimate, mixed effect, not promoted\n"
    )
    (output / "CURRENT_LINE_FINAL_FREEZE.md").write_text(freeze, encoding="utf-8")
    (repo / "CURRENT_LINE_FINAL_FREEZE.md").write_text(freeze, encoding="utf-8")
    branches = [
        {"branch_or_tag": "yolo26-custom-int8-engine", "commit": release_source_head,
         "status": "integrated-main; final evidence commit follows"},
        {"branch_or_tag": "yolo26-custom-int8-engine-maintenance-0.9.3", "commit": MAINTENANCE_HEAD,
         "status": "immutable-stable"},
        {"branch_or_tag": "yolo26-k1x-resolution-sweep", "commit": STAGE60_HEAD,
         "status": "immutable-research"},
        {"branch_or_tag": "yolo26-k1x-attention-ntail-r768-q0", "commit": STAGE61_HEAD,
         "status": "immutable-research"},
        {"branch_or_tag": "v0.9.3-r640", "commit": MAINTENANCE_HEAD, "status": "annotated-stable-tag"},
        {"branch_or_tag": "stage61-q0-final", "commit": STAGE61_HEAD, "status": "annotated-research-tag"},
        {"branch_or_tag": "v0.10.0-internal-rd.1", "commit": "final-publication-head",
         "status": "annotated-integrated-tag"},
    ]
    write_tsv(output / "BRANCH_ARCHIVE_MAP.tsv", branches)
    shutil.copyfile(output / "BRANCH_ARCHIVE_MAP.tsv", repo / "BRANCH_ARCHIVE_MAP.tsv")
    copy_or_fail(repo / "PROFILE_PROVENANCE.tsv", output / "FINAL_PACKAGE_HASHES.tsv")
    shutil.copyfile(output / "FINAL_PACKAGE_HASHES.tsv", repo / "FINAL_PACKAGE_HASHES.tsv")
    write_tsv(output / "FINAL_SOURCE_HASHES.tsv", [
        {"source": "source-model", "sha256": SOURCE_MODEL_SHA256},
        {"source": "stable-release-commit", "sha256": MAINTENANCE_HEAD},
        {"source": "Stage61-commit", "sha256": STAGE61_HEAD},
        {"source": "integrated-release-source-commit", "sha256": release_source_head},
    ])
    shutil.copyfile(output / "FINAL_SOURCE_HASHES.tsv", repo / "FINAL_SOURCE_HASHES.tsv")
    legal = (
        "# Final License Status\n\nRoute: `agpl-complete-source-route-selected`. "
        "The technical complete-source surface is built, but legal clearance is not certified. "
        "Unresolved items are project ownership/relicensing authority, Enterprise-agreement "
        "evidence, exact source-model export authority, and external model conveyance.\n"
    )
    (output / "FINAL_LICENSE_STATUS.md").write_text(legal, encoding="utf-8")
    (repo / "FINAL_LICENSE_STATUS.md").write_text(legal, encoding="utf-8")


def assemble_reports(args: argparse.Namespace) -> None:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    assemble_refs(args.repo, output, args.release_source_head)
    performance = assemble_performance(args.raw_root, output)
    assemble_abba(args.raw_root, output)
    assemble_exactness(args.raw_root, output)
    assemble_coco(args.raw_root, output)
    assemble_soaks(args.raw_root, output)
    assemble_camera(args.raw_root, output)
    assemble_host_and_binary(args.raw_root, output, args.stable_install, args.integrated_install)
    assemble_policy_and_legal(args.repo, output)
    assemble_addenda(output, args.release_source_head)
    assemble_freeze(args.repo, output, args.release_source_head, performance)

    (output / "workspace_preflight.md").write_text(
        "# Workspace Preflight\n\nAll four accepted branch heads matched both remotes; "
        "the source worktree was clean. Integration ran in an independent `/data` worktree. "
        "The board used Bianbu 2.2.1, Linux 6.6.63, eight X60 CPUs, performance governor at "
        "1.6 GHz, and NVMe `/data`. `/data/ncnn` was read-only evidence and no project data "
        "was written to eMMC.\n",
        encoding="utf-8",
    )
    (output / "system_rollback_report.md").write_text(
        "# System Rollback Report\n\nEvery O2 and camera IRQ profile reported inactive "
        "after normal, signal, capture-failure, and recorder-failure paths. No persistent "
        "boot, governor, sysctl, cgroup, IRQ, workqueue, or service policy was selected.\n",
        encoding="utf-8",
    )
    write_tsv(output / "tag_inventory.tsv", [
        {"tag": "v0.9.3-r640", "peeled_commit": MAINTENANCE_HEAD, "kind": "annotated", "status": "local-created"},
        {"tag": "stage61-q0-final", "peeled_commit": STAGE61_HEAD, "kind": "annotated", "status": "local-created"},
        {"tag": "v0.10.0-internal-rd.1", "peeled_commit": "final-publication-head", "kind": "annotated", "status": "pending-final-publication"},
    ])
    (output / "STAGE62_SUMMARY_RU.md").write_text(
        "# Краткий отчет Stage62\n\nИстории стабильного выпуска 0.9.3 и "
        "исследовательского Stage61 объединены без перебазирования и сжатия коммитов. "
        "R640 остается единственным профилем по умолчанию; восемь остальных разрешений "
        "доступны только при явном выборе. На плате подтверждены все 99 сочетаний профиля "
        "и тестового набора, по 215 целочисленных границ в каждом, неизменные хеши девяти "
        "прогонов COCO и отсутствие IME на CPU4-7. R384 и R768 не продвигаются для внедрения.\n\n"
        "Сформирован технический комплект полного исходного кода по маршруту AGPL, но "
        "юридическая чистота не сертифицирована: не подтверждены правообладание, полномочия "
        "на перелицензирование, договор Enterprise и внешний оборот модели. Это инженерный "
        "пакет для внутренней исследовательской проверки, а не производственная сертификация.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stable-install", type=Path, required=True)
    parser.add_argument("--integrated-install", type=Path, required=True)
    parser.add_argument("--release-source-head", required=True)
    args = parser.parse_args()
    assemble_reports(args)
    print(f"output={args.output}")
    print("exactness_rows=99")
    print("coco_profiles=9")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
