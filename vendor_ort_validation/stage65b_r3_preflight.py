#!/usr/bin/env python3
"""Verify immutable Stage65B-R3 launch identities and write compact evidence."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import onnx
import onnxruntime

from stage65b_r3_common import sha256, write_tsv


STAGE_ID = (
    "BANANA-YOLO26-XSLIM-STAGE65B-R3-HIERARCHICAL-EARLY-SUBGRAPH-"
    "CUT-SPLICE-LOCALIZATION-AND-XSLIM-TARGET-POLICY-CHARTER-001"
)
R2_ID = (
    "BANANA-YOLO26-XSLIM-STAGE65B-R2-HOST-INDEPENDENT-SELECTION-FP32-"
    "SPLIT-BOUNDARY-QDQ-DISAMBIGUATION-AND-B2-VARIANCE-GATE-001"
)
R1_ID = (
    "BANANA-YOLO26-XSLIM-STAGE65B-R1-COCO-TRAIN2017-EVALUATION-"
    "DISJOINT-CORPUS-PTQ-GRAPHWISE-AND-PYRAMID-CAUSAL-LOCALIZATION-001"
)
S64_ID = (
    "BANANA-YOLO26-VENDOR-ORT-STAGE64-XSLIM211-AND-VENDOR-COMMIT-S8-"
    "QDQ-YOLO26-FULLMODEL-COCO-AND-RT206-GATE-001"
)
PUB4_ID = (
    "BANANA-YOLO26-XSLIM-STAGE65A-PUB4-GITHUB-CREDENTIAL-REBIND-DUAL-"
    "REMOTE-RELEASE-ASSET-PARITY-CLOSURE-001"
)


def command(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def packet_manifest(packet: Path) -> tuple[str, int, int, list[str]]:
    paths = sorted(
        path.relative_to(packet).as_posix()
        for path in packet.rglob("*")
        if path.is_file()
    )
    # Reproduce the accepted export manifest's en_US.UTF-8 collation without a
    # login shell, whose startup banner would corrupt the path stream.
    raw = "\n".join(paths) + "\n"
    environment = os.environ.copy()
    environment["LC_ALL"] = "en_US.UTF-8"
    sorted_paths = subprocess.check_output(
        ["sort"], input=raw, text=True, env=environment
    ).splitlines()
    rows = [f"{sha256(packet / relative)}\t{relative}\n" for relative in sorted_paths]
    import hashlib

    digest = hashlib.sha256("".join(rows).encode()).hexdigest()
    return digest, len(sorted_paths), sum((packet / path).stat().st_size for path in sorted_paths), rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", required=True, type=Path)
    parser.add_argument("--protected-repo", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--tracked-root", required=True, type=Path)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--shared-log", required=True, type=Path)
    options = parser.parse_args()
    if options.raw_root.exists() or options.tracked_root.exists():
        raise RuntimeError("refusing to reuse Stage65B-R3 raw or tracked root")
    options.raw_root.mkdir(parents=True)
    options.tracked_root.mkdir(parents=True)
    (options.raw_root / "tmp").mkdir()
    (options.raw_root / "cache").mkdir()
    (options.raw_root / "models").mkdir()
    (options.raw_root / "logs").mkdir()

    expected_refs = {
        "research_head": "84caf893c73f2a33dfbbedacfa56c7ca40843557",
        "research_tree": "7a03d2ad4ba3589171e57227f2f4c6e4b80fcd28",
        "research_parent": "09ae3655e13a2bd22b8ac4cc22f226be212c8313",
        "protected_main": "1fd2e71bb1d5a924e7c0444cada94f681b73aa91",
    }
    observed_refs = {
        "research_head": command("git", "rev-parse", "HEAD", cwd=options.worktree),
        "research_tree": command("git", "rev-parse", "HEAD^{tree}", cwd=options.worktree),
        "research_parent": command("git", "rev-parse", "HEAD^", cwd=options.worktree),
        "protected_main": command(
            "git", "rev-parse", "yolo26-custom-int8-engine", cwd=options.protected_repo
        ),
    }
    if expected_refs != observed_refs:
        raise RuntimeError(f"Git identity mismatch: {observed_refs}")
    remote_rows: list[dict[str, Any]] = []
    for ref, expected in (
        ("research", expected_refs["research_head"]),
        ("protected", expected_refs["protected_main"]),
    ):
        branch = (
            "yolo26-vendor-ort-xslim211-s8-qdq-validation"
            if ref == "research"
            else "yolo26-custom-int8-engine"
        )
        for surface, revision in (
            ("local", branch),
            ("github", f"refs/remotes/github/{branch}"),
            ("gitlab-rd", f"refs/remotes/gitlab-rd/{branch}"),
        ):
            observed = command("git", "rev-parse", revision, cwd=options.worktree)
            remote_rows.append(
                {
                    "ref": ref,
                    "surface": surface,
                    "expected": expected,
                    "observed": observed,
                    "status": "pass" if observed == expected else "fail",
                }
            )
    if any(row["status"] != "pass" for row in remote_rows):
        raise RuntimeError("remote ref parity mismatch")
    write_tsv(options.tracked_root / "remote_parity_before.tsv", remote_rows)

    packet_hash, packet_files, packet_bytes, packet_rows = packet_manifest(options.packet)
    packet_expected = (
        "d80b3362fa0abd99afa3fc1985fe547e774113f690e5ab0140b0b4c7276639db",
        75,
        809627,
    )
    packet_observed = (packet_hash, packet_files, packet_bytes)
    if packet_observed != packet_expected:
        raise RuntimeError(f"R2 packet mismatch: {packet_observed}")
    (options.raw_root / "r2_packet_hashes.tsv").write_text(
        "".join(packet_rows), encoding="utf-8"
    )
    write_tsv(
        options.tracked_root / "r2_packet_verification.tsv",
        [
            {"field": "tree_sha256", "expected": packet_expected[0], "observed": packet_hash, "status": "pass"},
            {"field": "file_count", "expected": packet_expected[1], "observed": packet_files, "status": "pass"},
            {"field": "byte_count", "expected": packet_expected[2], "observed": packet_bytes, "status": "pass"},
            {"field": "symlink_count", "expected": 0, "observed": sum(path.is_symlink() for path in options.packet.rglob("*")), "status": "pass"},
        ],
    )

    stage64 = Path("/data/k1x-stage-runs") / S64_ID
    r1 = Path("/data/k1x-stage-runs") / R1_ID
    r2 = Path("/data/k1x-stage-runs") / R2_ID
    artifacts = (
        ("FP32-source", options.protected_repo / ".deps/models/yolo26/fp32_fp16_xslim_effect_matrix/yolo26n_640_e2e_fp32.onnx", "d71286588abe691ede49faa5ca9a471b7e9e5257669953ee59abbc2e9d115fc2"),
        ("FP32-split-inference", stage64 / "models/fp32-split/yolo26n_640_e2e_fp32.inference.onnx", "72eb6136b41104753c53b8e13aeff50e7961c4cefba79e50b70894cbd169f8d8"),
        ("FP32-native-tail", stage64 / "models/fp32-split/yolo26n_640_e2e_fp32.postprocess.onnx", "cd27158a5b393331305329f51d6194747dc7757b30c34453b61c123bb9568691"),
        ("common-tail", stage64 / "models/R211_PROJECT_EXACT_SPLIT_run1/r211_project_exact_split.postprocess.onnx", "18ffff41e6812fa781baf7b9c1fcd41b41d6118145d785c3e550499070a512a3"),
        ("B2-deployable", r1 / "quantization/B2/run1/output/stage65b_r1_b2_split_s8_qdq.onnx", "0e7040d4e8b1b2d08a4e36cec4c99dcea6d52294e04901d17dfce10725c6d617"),
        ("B2-inference", r1 / "postprocess/B2/candidate-gate/B2/models/stage65b_r1_b2.inference.onnx", "40ba6a7f9aebaa98a1c3abe5fce1f66f1bebcd0b10b7af3d26d30414a331d853"),
        ("B2-full-val-predictions", r1 / "full-matrix/full-coco/B2/predictions.json", "51f8d4b25245a5f3e24feafea8aa49547c0f530f59cabcd18e61a744b4740add"),
        ("D8-diagnostic", r2 / "models/B2-D8-final-six-output-qdq-bypass-diagnostic.onnx", "a77f2efea1dee7578d66859159a01c08ea45b76b44865d40b813732aa84372d4"),
        ("H8-full-val-predictions", r1 / "full-matrix/hybrid-full-coco/H8/predictions.json", "b9ff8fa19cba9682970d8e932f3318cdf5833094ab22256a24062019309b5b2a"),
        ("R1-evaluator", options.worktree / "vendor_ort_validation/stage65b_r1_evaluate.py", "79ad059411bb153f3abcb8d4abd0f1e79e5e04b12863fc121dc227d2fe89bd65"),
        ("R1-full-matrix", options.worktree / "vendor_ort_validation/stage65b_r1_full_matrix.py", "aa462ab0b9f5fa2a5df200d3ee778af0f379f11fde191fe4af5d8eaa1b9f2ab4"),
        ("R2-D8-tool", options.worktree / "vendor_ort_validation/stage65b_r2_d8.py", "157d0d94c513ae7fa2518ab03f2f965a083fe919304ab96b5ed9e04fa61e320d"),
        ("R2-bootstrap-tool", options.worktree / "vendor_ort_validation/stage65b_r2_bootstrap.py", "5d3649908f7f0cf3ff02e133dfbeb58504baa14afac102ea0de9d62af916d245"),
        ("R2-reconcile-tool", options.worktree / "vendor_ort_validation/stage65b_r2_reconcile.py", "818d3dc818d02ac91e83ff3888628248e735ba41b4121df9b40188b3a54e65c9"),
        ("XSlim-wheel", Path("/data/k1x-stage-runs") / PUB4_ID / "downloads/github-public/xslim-2.1.2+riscy.1-py3-none-any.whl", "635441d26458c6754627dd9595132cfd9a16d762d3ed0252f0039be668c01784"),
    )
    artifact_rows: list[dict[str, Any]] = []
    for name, path, expected in artifacts:
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256(path)
        artifact_rows.append(
            {
                "artifact": name,
                "canonical_path": str(path),
                "bytes": path.stat().st_size,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "canonical_match_count": 1,
                "status": "pass" if observed == expected else "fail",
            }
        )
    if any(row["status"] != "pass" for row in artifact_rows):
        raise RuntimeError("frozen artifact identity mismatch")
    write_tsv(options.tracked_root / "frozen_artifact_identity.tsv", artifact_rows)

    paths = [
        ("protected checkout", "/data/banana-yolo26-spacemit-demo", "/mnt/DataSets/XuBanana/bf3-ncnn/data/banana-yolo26-spacemit-demo", "bf3-ncnn/data/banana-yolo26-spacemit-demo"),
        ("research worktree", str(options.worktree), str(options.worktree).replace("/data", "/mnt/DataSets/XuBanana/bf3-ncnn/data", 1), str(options.worktree).replace("/data/", "bf3-ncnn/data/", 1)),
        ("Stage65B-R2 tracked", str(options.worktree / "stages" / R2_ID), str(options.worktree / "stages" / R2_ID).replace("/data", "/mnt/DataSets/XuBanana/bf3-ncnn/data", 1), str(options.worktree / "stages" / R2_ID).replace("/data/", "bf3-ncnn/data/", 1)),
        ("Stage65B-R2 raw", str(r2), str(r2).replace("/data", "/mnt/DataSets/XuBanana/bf3-ncnn/data", 1), str(r2).replace("/data/", "bf3-ncnn/data/", 1)),
        ("Stage65B-R3 tracked", str(options.tracked_root), str(options.tracked_root).replace("/data", "/mnt/DataSets/XuBanana/bf3-ncnn/data", 1), str(options.tracked_root).replace("/data/", "bf3-ncnn/data/", 1)),
        ("Stage65B-R3 raw", str(options.raw_root), str(options.raw_root).replace("/data", "/mnt/DataSets/XuBanana/bf3-ncnn/data", 1), str(options.raw_root).replace("/data/", "bf3-ncnn/data/", 1)),
        ("dataset", "/data/datasets/coco2017-independent-stage65b-r1", "/mnt/DataSets/XuBanana/bf3-ncnn/data/datasets/coco2017-independent-stage65b-r1", "manifest-only-no-image-sync"),
        ("result packet", f"/exchange/results/outbox/{STAGE_ID}", "separate-managed-exchange-mapping", "managed ai-team-exchange result/outbox"),
    ]
    write_tsv(
        options.tracked_root / "path_mapping_revalidated.tsv",
        [
            {"surface": surface, "container": container, "RHEL10": rhel, "Google_Drive": drive, "status": "pass"}
            for surface, container, rhel, drive in paths
        ],
    )
    environment = [
        {"field": "timestamp_utc", "value": command("date", "-u", "+%Y-%m-%dT%H:%M:%SZ")},
        {"field": "boot_id", "value": Path("/proc/sys/kernel/random/boot_id").read_text().strip()},
        {"field": "platform", "value": platform.platform()},
        {"field": "python", "value": sys.version.replace("\n", " ")},
        {"field": "python_executable", "value": sys.executable},
        {"field": "onnx", "value": onnx.__version__},
        {"field": "onnxruntime", "value": onnxruntime.__version__},
        {"field": "cpu_count", "value": os.cpu_count()},
        {"field": "TMPDIR", "value": str(options.raw_root / "tmp")},
        {"field": "XDG_CACHE_HOME", "value": str(options.raw_root / "cache")},
        {"field": "shared_log", "value": str(options.shared_log)},
    ]
    write_tsv(options.tracked_root / "host_environment.tsv", environment)

    launch = {
        "stage_id": STAGE_ID,
        "execution_authority": "direct-user-authorization",
        "research_branch": "yolo26-vendor-ort-xslim211-s8-qdq-validation",
        "research_start": expected_refs["research_head"],
        "protected_main": expected_refs["protected_main"],
        "r2_packet": {"tree_sha256": packet_hash, "files": packet_files, "bytes": packet_bytes},
        "board_execution_authorized": False,
        "targeted_model_generation_authorized": False,
        "new_branch_authorized": False,
        "raw_root": str(options.raw_root),
        "tracked_root": str(options.tracked_root),
    }
    # This is JSON-compatible YAML and avoids a runtime YAML dependency.
    (options.tracked_root / "effective_launch_manifest.yaml").write_text(
        json.dumps(launch, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (options.tracked_root / "workspace_preflight.md").write_text(
        f"# Stage65B-R3 workspace preflight\n\n"
        f"- Research head/tree/parent: `{observed_refs['research_head']}` / "
        f"`{observed_refs['research_tree']}` / `{observed_refs['research_parent']}`.\n"
        f"- Local/GitHub/GitLab-RD parity: exact.\n"
        f"- Protected main: `{observed_refs['protected_main']}`.\n"
        f"- R2 packet: `{packet_hash}`, {packet_files} files, {packet_bytes} bytes.\n"
        f"- Frozen artifact and tooling identities: all exact.\n"
        f"- Commands use container paths; `/exchange` remains a separate managed surface.\n"
        f"- No board execution, targeted model generation, XSlim mutation, or new branch is authorized.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
