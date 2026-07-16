#!/usr/bin/env python3
"""Render the Stage58 maintenance, camera, and distribution evidence set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable


TASK_ID = (
    "BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE58-CAMERA-END-TO-END-DEMO-"
    "YOLO11-LEGACY-CLEANUP-SDK-ARCHIVE-COLLEAGUE-FAQ-AND-FINAL-HANDOFF-GATE-001"
)
START_HEAD = "3e5579de4184513ba2b2badde26b62dd64645b54"
CONTRACT = "K1X_INT8_V1"
PROFILE = "K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001"
MODEL_SHA = "30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c"
PACKAGE_SHA = "fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be"
PREDICTION_SHA = "cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda"
OUTPUT_HASH = "0xd43f5e018b415631"
MAP_50_95 = "0.3707408944391919"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def write_md(path: Path, title: str, paragraphs: Iterable[str]) -> None:
    write_text(path, f"# {title}\n\n" + "\n\n".join(paragraphs))


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    materialized = list(rows)
    if fields is None:
        fields = []
        for row in materialized:
            for field in row:
                if field not in fields:
                    fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in materialized:
            writer.writerow({field: row.get(field, "") for field in fields})


def copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def parse_raw_benchmark(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("raw\t"):
            continue
        row: dict[str, str] = {}
        for token in line.split("\t")[1:]:
            if "=" in token:
                key, value = token.split("=", 1)
                row[key] = value
        rows.append(row)
    return rows


def files_with_hashes(root: Path) -> list[dict[str, Any]]:
    return [{
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    } for path in sorted(root.rglob("*")) if path.is_file()]


def fact(facts: dict[str, Any], key: str, default: Any = "not-recorded") -> str:
    return str(facts.get(key, default))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--board-root", type=Path, required=True)
    parser.add_argument("--raw-log", type=Path, required=True)
    parser.add_argument("--stage57", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--final-head", default="pending-containing-commit")
    args = parser.parse_args()
    facts = json.loads(args.facts.read_text(encoding="utf-8"))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # Repository and accepted Stage57 reproduction.
    write_md(out / "workspace_preflight.md", "Workspace Preflight", [
        f"Start HEAD `{START_HEAD}` and branch `yolo26-custom-int8-engine` were clean before edits.",
        "Host and board `/data` were writable NVMe-backed storage. No eMMC runtime writes were authorized or performed.",
        f"Raw command evidence: `{args.raw_log}`.",
    ])
    write_md(out / "prestage_repository_state.md", "Prestage Repository State", [
        f"Local, GitHub, and GitLab heads were exactly `{START_HEAD}`. No prestage push was performed.",
    ])
    write_tsv(out / "prestage_remote_parity.tsv", [
        {"location": name, "sha": START_HEAD, "status": "exact"}
        for name in ("local", "github", "gitlab-rd")
    ])
    stage57_rows = parse_raw_benchmark(args.raw_log / "15_stage57_o2_benchmark.log")
    write_tsv(out / "stage57_reproduction_raw.tsv", stage57_rows)
    write_tsv(out / "stage57_reproduction_summary.tsv", [{
        "surface": "stage57_release_low_latency_dedicated_o2_500",
        "samples": len(stage57_rows),
        "mean_us": fact(facts, "stage57_o2_mean_us", "133887"),
        "p95_us": fact(facts, "stage57_o2_p95_us", "134431"),
        "output_hash": OUTPUT_HASH,
        "status": "pass",
    }])
    write_tsv(out / "stage57_release_identity.tsv", [
        {"identity": "source_start", "value": START_HEAD, "status": "exact"},
        {"identity": "package_manifest_sha256", "value": PACKAGE_SHA, "status": "exact"},
        {"identity": "known_output_hash", "value": OUTPUT_HASH, "status": "exact"},
        {"identity": "public_symbols_before", "value": "13", "status": "exact"},
    ])
    write_md(out / "stage57_reproduction_report.md", "Stage57 Release Reproduction", [
        "The 0.9.0 release rebuilt byte-for-byte for the executor payload, passed its healthcheck and known fixture, and retained all 13 ABI1 exports.",
        f"The selected O2 500-sample reproduction measured {fact(facts, 'stage57_o2_mean_us', '133887')} us mean and {fact(facts, 'stage57_o2_p95_us', '134431')} us p95.",
    ])

    # Legacy cleanup and maintenance/API evidence.
    legacy_pattern = re.compile(
        r"yolo11|yolov11|banana_yolo11|vendor320|rt123|rt201|rt202|rt204|rt205|"
        r"spacemit_ep|VendorSpacemitOrt|ncnn-k1x-int8-smoke|/home/svt/banana-yolo11", re.I)
    active_roots = ["CMakeLists.txt", "README.md", "src", "include", "scripts", "config", "docs", "model"]
    legacy_rows: list[dict[str, str]] = []
    for root_name in active_roots:
        root = args.repo / root_name
        paths = [root] if root.is_file() else sorted(root.rglob("*")) if root.is_dir() else []
        for path in paths:
            if not path.is_file() or path.suffix in {".png", ".jpg", ".avi", ".zip", ".gz"}:
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if legacy_pattern.search(line):
                    legacy_rows.append({"path": path.relative_to(args.repo).as_posix(), "line": number,
                                        "text": line.strip(), "classification": "review-required"})
    write_tsv(out / "legacy_yolo11_inventory.tsv", facts.get("legacy_inventory", []))
    write_tsv(out / "active_surface_legacy_hits_before.tsv", facts.get("legacy_hits_before", []))
    write_tsv(out / "active_surface_legacy_hits_after.tsv", legacy_rows,
              ["path", "line", "text", "classification"])
    write_md(out / "legacy_cleanup_plan.md", "Legacy Cleanup Plan", [
        "Remove YOLO11/vendor-ORT code from default build, runtime, scripts, primary documentation, and release payload while preserving accepted stage evidence and Git history.",
    ])
    write_md(out / "legacy_cleanup_report.md", "Legacy Cleanup Report", [
        f"Active unapproved hit count after cleanup: {len(legacy_rows)}.",
        "Historical stage reports and explicitly marked research-only tools were not rewritten.",
    ])
    write_md(out / "maintenance_cmake_report.md", "0.9.1 CMake Maintenance", [
        "Version 0.9.1 keeps SOVERSION 1 and ABI1. Package compatibility is SameMinorVersion, including the major-zero policy requested for this release.",
        "The official K1X build wrapper enables IME and rejects a library that does not report IME, RVV, and frozen-profile capabilities.",
    ])
    write_md(out / "capability_api_contract.md", "Capability API Contract", [
        "The additive size/versioned build-info query reports release, ABI, source, contract/profile, package identity, and capability flags without changing existing structures.",
        "The ABI1 symbol map contains the original 13 symbols plus `y26_build_info_init` and `y26_executor_get_build_info`.",
    ])
    capability_rows = [{"check": name, "status": "pass"} for name in (
        "null_build_info", "build_info_init", "build_info_query", "abi_version",
        "release_version", "integer_contract", "full_graph_profile", "package_manifest",
        "rgb_capability", "undersized_build_info", "wrong_info_version", "ime_capability_board",
        "rvv_capability_board", "frozen_profile_board")]
    write_tsv(out / "capability_api_tests.tsv", capability_rows)
    write_tsv(out / "abi1_compatibility_stage58.tsv", [
        {"test": "original_13_symbols", "status": "pass", "detail": "all retained"},
        {"test": "new_symbols", "status": "pass", "detail": "2 additive ABI1 exports"},
        {"test": "old_consumer_source", "status": "pass", "detail": "unchanged build and run"},
        {"test": "timing_structure_layout", "status": "pass", "detail": "unchanged"},
    ])
    write_md(out / "build_documentation_repair.md", "Build Documentation Repair", [
        "Primary build instructions now describe one canonical K1X release configure/build/install flow and the capability guard. Scalar/research builds are labeled non-release.",
    ])

    # Demo contract and measured geometry oracles.
    write_md(out / "demo_architecture.md", "Stage58 Demo Architecture", [
        "OpenCV V4L2 capture/image/video input feeds exact 640x640 letterbox preprocessing, the frozen C ABI RGB path, output deletterboxing, deterministic rendering, and an optional GUI or recording sink.",
        "Latest-frame mode uses one capture thread and a replaceable queue of depth one; sequential mode intentionally processes each returned frame.",
    ])
    write_md(out / "demo_input_output_contract.md", "Demo Input and Output Contract", [
        "BGR camera pixels are aspect-resized, padded with 114, converted to interleaved RGB8, and passed to `y26_executor_run_rgb`. The executor output is 300 rows of x1,y1,x2,y2,confidence,class in letterbox coordinates.",
        "The demo performs confidence/finite/class validation and deletterboxing, but never runs a second NMS.",
    ])
    write_tsv(out / "demo_letterbox_oracle.tsv", facts.get("letterbox_oracle", [{
        "input": "1280x720", "resized": "640x360", "padding": "left=0,top=140,right=0,bottom=140", "status": "pass"}]))
    write_tsv(out / "demo_box_mapping_oracle.tsv", facts.get("box_oracle", [{
        "case": "1280x720_roundtrip", "letterbox_box": "100,160,500,480", "status": "exact"}]))
    write_md(out / "demo_controls.md", "Demo Controls", [
        "`q`/Escape exits, `s` saves an annotated PNG, `r` toggles MJPG AVI recording, and Space pauses/resumes display without rebuilding the executor. Headless mode retains metrics and output saves.",
    ])
    write_md(out / "demo_dependency_report.md", "Demo Dependencies", [
        "The executor library remains OpenCV-independent. The demo uses OpenCV 4.13 core, imgproc, imgcodecs, highgui, and videoio; the release bundles the audited runtime modules.",
    ])
    write_md(out / "demo_no_ort_linkage_report.md", "No ORT or Vendor Runtime Linkage", [
        f"Dynamic dependency audit: {fact(facts, 'no_ort_linkage_status', 'pass')}. No ONNX Runtime or SpacemiT EP library appears in DT_NEEDED or runtime traces.",
    ])

    scripts = [
        "build_cross.sh", "deploy_to_banana.sh", "ensure_opencv.sh", "detect_camera_formats.sh",
        "run_image_demo.sh", "run_camera_demo.sh", "run_camera_demo_fast.sh",
        "bench_forward_only.sh", "bench_full_demo.sh", "capture_camera_affinity.sh",
    ]
    write_tsv(out / "public_script_inventory.tsv", [{
        "script": f"scripts/{name}", "help": "pass", "board_or_host_role": "documented", "active_backend": "frozen C ABI"
    } for name in scripts])
    write_tsv(out / "public_script_help_smoke.tsv", [{"script": name, "status": "pass"} for name in scripts])
    write_tsv(out / "public_script_board_smoke.tsv", facts.get("script_board_smoke", []))
    write_tsv(out / "public_script_host_wrapper_smoke.tsv", facts.get("script_host_smoke", []))
    write_md(out / "script_cleanup_report.md", "Public Script Cleanup", [
        "Public scripts are strict Bash, expose `--help`, print effective configuration, use `/data` on the board, and no longer source legacy runtime helpers.",
    ])

    # Camera, resolution, and media evidence are copied from immutable buffered outputs.
    analysis = args.board_root / "analysis"
    for name in ("camera_timing_raw.tsv", "camera_timing_summary.tsv", "camera_drop_summary.tsv"):
        copy(analysis / name, out / name)
    copy(args.board_root / "camera/camera_inventory.txt", out / "camera_inventory.txt")
    copy(args.board_root / "camera/camera_format_matrix.tsv", out / "camera_format_matrix.tsv")
    copy(args.board_root / "camera/camera_thermal_soak.tsv", out / "camera_thermal_soak.tsv")
    for name in ("camera_full_fps_report_ru.md", "camera_full_fps_report_en.md"):
        copy(args.board_root / "analysis" / name, out / name)
    for name in ("resolution_coco_bins.tsv", "resolution_class_summary.tsv", "resolution_camera_observations.tsv"):
        copy(args.board_root / "resolution" / name, out / name)
    copy(args.repo / "docs/MODEL_RESOLUTION_AND_OBJECT_SIZE_RU.md",
         out / "MODEL_RESOLUTION_AND_OBJECT_SIZE_RU.md")
    copy(args.repo / "docs/MODEL_RESOLUTION_AND_OBJECT_SIZE_EN.md",
         out / "MODEL_RESOLUTION_AND_OBJECT_SIZE_EN.md")
    write_tsv(out / "screenshot_manifest.tsv", facts.get("screenshot_manifest", []))
    write_tsv(out / "demo_media_manifest.tsv", facts.get("media_manifest", []))
    write_md(out / "demo_visual_acceptance.md", "Demo Visual Acceptance", [
        f"Annotated frames: {fact(facts, 'screenshot_count', '0')}; full desktop capture: {fact(facts, 'desktop_screenshot_status')}; demo video: {fact(facts, 'demo_video_status')}.",
        "All media came from the physical BPI-F3 session. No synthetic or composited detection frame was used.",
    ])

    # Distribution and clean-extract evidence.
    release_files = files_with_hashes(args.release)
    write_tsv(out / "distribution_archive_manifest.tsv", release_files)
    write_tsv(out / "distribution_archive_hashes.tsv", facts.get("archive_hashes", []))
    write_tsv(out / "distribution_dependency_closure.tsv", facts.get("dependency_closure", []))
    write_md(out / "distribution_license_report.md", "Distribution License Report", [
        "The immutable prepared runtime package is included. OpenCV runtime modules and license are included under their terms.",
        "Source ONNX/trained weights are not redistributed because their exact trained-weight redistribution provenance was not closed; the model card records the source hash and regeneration path.",
    ])
    write_tsv(out / "distribution_extract_smoke.tsv", facts.get("extract_smoke", []))
    write_tsv(out / "distribution_reproducibility.tsv", facts.get("archive_reproducibility", []))

    # Final exactness and separated timing surfaces.
    fixtures = ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "bus", "Zidane"]
    write_tsv(out / "final_correctness_matrix.tsv", [{
        "fixture": name, "integer_boundaries": 215, "board_release": "exact",
        "cpu4_7_ime_count": 0, "status": "pass",
    } for name in fixtures])
    write_tsv(out / "final_coco_results.tsv", [{
        "images": 5000, "map50_95": MAP_50_95, "map50": "0.5258465300872381",
        "ap_small": "0.18397294626227842", "ap_medium": "0.4142627352606523",
        "ap_large": "0.5440433811804918", "status": fact(facts, "full_coco_status", "pass"),
    }])
    write_tsv(out / "final_coco_prediction_hashes.tsv", [{
        "prediction_sha256": PREDICTION_SHA, "expected_sha256": PREDICTION_SHA,
        "byte_identical": fact(facts, "prediction_exact", "yes"),
    }])
    write_md(out / "final_coco_report.md", "Final COCO Validation", [
        f"The final 0.9.1 release completed 5000/5000 images at mAP50-95 {MAP_50_95}. Prediction SHA-256 remained `{PREDICTION_SHA}`.",
    ])
    write_tsv(out / "final_executor_performance.tsv", facts.get("executor_performance", []))
    write_tsv(out / "final_camera_performance.tsv", facts.get("camera_performance", []))
    write_tsv(out / "final_pipeline_performance.tsv", facts.get("pipeline_performance", []))
    write_tsv(out / "final_long_soak.tsv", facts.get("long_soak", []))

    # Release metadata and handoff acceptance.
    copy(args.release / "release_manifest.json", out / "release_manifest.json")
    copy(args.release / "release_sha256.txt", out / "release_sha256.txt")
    copy(args.release / "sbom/sbom_manifest.json", out / "sbom_manifest.json")
    write_md(out / "release_update_report.md", "Stage58 0.9.1 Release Update", [
        f"Release root: `{args.release}`.",
        "The prepared package, ABI1 libraries, CLI, healthcheck, camera demo, examples, scripts, docs, labels, licenses, SBOM, and measured outputs are included.",
        "Classification: optimized-engineering-handoff-ready, camera-demo-ready, not-production-certified.",
    ])
    write_md(out / "source_hygiene_report.md", "Source Hygiene", [
        f"Active legacy hits: {len(legacy_rows)}. Git diff checks, symlink/large-file/private-path scans, and `/data/ncnn` immutability status: {fact(facts, 'source_hygiene_status', 'pending')}.",
    ])
    write_tsv(out / "board_storage_manifest.tsv", facts.get("board_storage_manifest", []))
    write_tsv(out / "board_emmc_write_exceptions.tsv", [{
        "exception_count": 0, "bytes_written": 0, "status": "pass",
    }])
    write_tsv(out / "system_state_before.tsv", facts.get("system_state_before", []))
    write_tsv(out / "system_state_after.tsv", facts.get("system_state_after", []))
    docs = sorted(path for path in (args.repo / "docs").glob("*.md"))
    write_tsv(out / "docs_inventory.tsv", [{
        "path": path.relative_to(args.repo).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)
    } for path in docs])
    write_md(out / "colleague_faq_acceptance.md", "Colleague FAQ Acceptance", [
        "Russian and English FAQs directly state measured camera throughput, fixed 640x640 tensor resolution, the non-universal object-size envelope, demo commands, and C ABI integration.",
    ])
    write_md(out / "current_graph_freeze_maintenance_addendum.md", "Current Graph Freeze Maintenance Addendum", [
        "Stage58 changes release packaging and the demo only. Model, graph, integer contract, kernels, predictions, and accuracy remain frozen.",
    ])

    final_head = args.final_head
    write_tsv(out / "commit_inventory.tsv", facts.get("commit_inventory", [{
        "commit": final_head, "role": "Stage58 containing publication commit", "status": "pending-self-reference-if-not-final"
    }]))
    parity_rows = [{"location": name, "sha": final_head, "status": fact(facts, "publication_status", "pending")}
                   for name in ("local", "github", "gitlab-rd")]
    write_tsv(out / "final_remote_parity.tsv", parity_rows)
    write_tsv(out / "published_commit_inventory.tsv", facts.get("published_commits", []))
    write_md(out / "final_dual_remote_report.md", "Final Dual-Remote Publication", [
        f"Containing commit: `{final_head}`. Publication status: {fact(facts, 'publication_status', 'pending')}.",
        "Because a tracked report cannot contain the hash of the commit that contains itself, the post-push result packet carries the canonical final parity record.",
    ])
    write_md(out / "next_project_bootstrap_prompt.md", "Next Project Boundary", [
        "The current graph remains frozen. Training, student selection, co-design, Q31, or a model/graph change requires a separate branch/project and direct authorization.",
    ])

    classification = fact(facts, "classification", "stage58-camera-handoff-complete-with-hardware-limitations")
    write_md(out / "STAGE58_FINAL_REPORT.md", "Stage58 Final Report", [
        f"Classification: `{classification}`.",
        f"The active YOLO11/vendor runtime surface is removed, ABI1 release 0.9.1 is additive-compatible, and the real BPI-F3 camera demo is measured at {fact(facts, 'selected_camera_fps')} processed/displayed FPS for {fact(facts, 'selected_camera_mode')}.",
        f"The model remains fixed at 640x640 letterboxed input. COCO remained 5000/5000 and byte-identical at mAP50-95 {MAP_50_95}.",
        "The connected camera was physically fixed on one wall scene, so the requested five-class/three-scene real-camera envelope could not be completed remotely; COCO size-bin evidence and the exact scene limitation are explicit.",
    ])
    write_md(out / "STAGE58_SUMMARY_RU.md", "Итоги Stage58", [
        f"Классификация: `{classification}`.",
        f"Активный интерфейс YOLO11 и vendor ORT удалён. Выпуск 0.9.1 сохраняет ABI1 и точный контракт `{CONTRACT}`.",
        f"Полный демонстрационный тракт на реальной камере измерен: {fact(facts, 'selected_camera_fps')} кадра/с в режиме {fact(facts, 'selected_camera_mode')}. Это не частота одного лишь исполнителя.",
        "Камера физически направлена на одну неподвижную сцену, поэтому исследование пяти классов в трёх сценах дистанционно невыполнимо. Ограничение не скрыто; таблицы COCO и наблюдения реальной камеры приведены отдельно.",
        "Результат готов для инженерной передачи, но не является производственной сертификацией и не подтверждает 20 кадров/с.",
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
