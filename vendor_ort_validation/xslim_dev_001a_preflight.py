#!/usr/bin/env python3
"""Verify immutable XSLIM-DEV-001A launch inputs and create preflight evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

STAGE_ID = (
    "BANANA-YOLO26-XSLIM-DEV-001A-SPACEMIT-S8-QDQ-CONSTRAINED-RANGE-"
    "OBSERVER-TERMINAL-DOMAIN-AND-POLICY-A-HOST-CANDIDATE-GATE-001"
)
R3_ID = (
    "BANANA-YOLO26-XSLIM-STAGE65B-R3-HIERARCHICAL-EARLY-SUBGRAPH-"
    "CUT-SPLICE-LOCALIZATION-AND-XSLIM-TARGET-POLICY-CHARTER-001"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def write_tsv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("\t".join(columns) + "\n")
        for row in rows:
            values = [
                str(row.get(column, "")).replace("\t", " ").replace("\n", " ")
                for column in columns
            ]
            stream.write("\t".join(values) + "\n")


def packet_manifest(packet: Path) -> tuple[str, int, int]:
    paths = sorted(
        path.relative_to(packet).as_posix()
        for path in packet.rglob("*")
        if path.is_file()
    )
    environment = os.environ.copy()
    environment["LC_ALL"] = "en_US.UTF-8"
    sorted_paths = subprocess.check_output(
        ["sort"], input="\n".join(paths) + "\n", text=True, env=environment
    ).splitlines()
    rows = [f"{sha256(packet / relative)}\t{relative}\n" for relative in sorted_paths]
    return (
        hashlib.sha256("".join(rows).encode()).hexdigest(),
        len(sorted_paths),
        sum((packet / relative).stat().st_size for relative in sorted_paths),
    )


def verify_artifacts(specs: list[tuple[str, Path, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: dict[str, list[Path]] = {}
    for name, path, expected in specs:
        if not path.is_file():
            raise RuntimeError(f"missing frozen artifact {name}: {path}")
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"frozen artifact mismatch {name}: {observed} != {expected}"
            )
        seen.setdefault(expected, []).append(path.resolve())
        rows.append(
            {
                "artifact": name,
                "canonical_path": path,
                "bytes": path.stat().st_size,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "canonical_match_count": 1,
                "status": "pass",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--banana-worktree", required=True, type=Path)
    parser.add_argument("--banana-protected", required=True, type=Path)
    parser.add_argument("--xslim-worktree", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--tracked-root", required=True, type=Path)
    parser.add_argument("--xslim-evidence-root", required=True, type=Path)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--shared-log", required=True, type=Path)
    options = parser.parse_args()

    expected_refs = {
        "xslim_head": "2bc1be073c84ffd8b4e22e372b8f33de4218f9f8",
        "xslim_tree": "110c73c35a664231a7f6bea1ed0d5fee02d124cb",
        "banana_head": "69b686dbf13ab31aec85ea5ebe11effbe4cc7917",
        "banana_tree": "f52a12d4f5532d80dabb1a3fa4617f406871a1de",
        "banana_closure": "a6ebce4417f9ace858fabbcb047f460635eea684",
        "protected_main": "1fd2e71bb1d5a924e7c0444cada94f681b73aa91",
        "custom_tree": "c2e400de14fb1c88d4aed70a249d9eff19a05d0f",
    }
    observed_refs = {
        "xslim_head": command("git", "rev-parse", "HEAD", cwd=options.xslim_worktree),
        "xslim_tree": command(
            "git", "rev-parse", "HEAD^{tree}", cwd=options.xslim_worktree
        ),
        "banana_head": command("git", "rev-parse", "HEAD", cwd=options.banana_worktree),
        "banana_tree": command(
            "git", "rev-parse", "HEAD^{tree}", cwd=options.banana_worktree
        ),
        "banana_closure": command(
            "git",
            "rev-parse",
            "a6ebce4417f9ace858fabbcb047f460635eea684",
            cwd=options.banana_worktree,
        ),
        "protected_main": command(
            "git",
            "rev-parse",
            "yolo26-custom-int8-engine",
            cwd=options.banana_protected,
        ),
        "custom_tree": command(
            "git",
            "rev-parse",
            "yolo26-custom-int8-engine:custom_int8_engine",
            cwd=options.banana_protected,
        ),
    }
    for name, expected in expected_refs.items():
        if observed_refs[name] != expected:
            raise RuntimeError(
                f"identity mismatch {name}: {observed_refs[name]} != {expected}"
            )

    remote_rows: list[dict[str, object]] = []
    for repo_name, repo, branch, remotes, expected in (
        (
            "xslim",
            options.xslim_worktree,
            "riscy/k1x-yolo26",
            ("github", "gitlab"),
            expected_refs["xslim_head"],
        ),
        (
            "banana",
            options.banana_worktree,
            "yolo26-vendor-ort-xslim211-s8-qdq-validation",
            ("github", "gitlab-rd"),
            expected_refs["banana_head"],
        ),
    ):
        local = command("git", "rev-parse", branch, cwd=repo)
        for remote in remotes:
            remote_head = command("git", "rev-parse", f"{remote}/{branch}", cwd=repo)
            status = "pass" if local == remote_head == expected else "fail"
            remote_rows.append(
                {
                    "repository": repo_name,
                    "branch": branch,
                    "surface": remote,
                    "local": local,
                    "remote": remote_head,
                    "expected": expected,
                    "status": status,
                }
            )
            if status != "pass":
                raise RuntimeError(f"remote parity failed for {repo_name}/{remote}")

    packet_hash, packet_files, packet_bytes = packet_manifest(options.packet)
    packet_expected = (
        "3ec1a304afd09213869728c2db2d5b5d5d4236807d4f1733dfa8945094fb2cd7",
        68,
        416301,
    )
    if (packet_hash, packet_files, packet_bytes) != packet_expected:
        raise RuntimeError("Stage65B-R3 result packet identity mismatch")

    r1 = Path(
        "/data/k1x-stage-runs/BANANA-YOLO26-XSLIM-STAGE65B-R1-COCO-TRAIN2017-"
        "EVALUATION-DISJOINT-CORPUS-PTQ-GRAPHWISE-AND-PYRAMID-CAUSAL-LOCALIZATION-001"
    )
    r2 = Path(
        "/data/k1x-stage-runs/BANANA-YOLO26-XSLIM-STAGE65B-R2-HOST-INDEPENDENT-"
        "SELECTION-FP32-SPLIT-BOUNDARY-QDQ-DISAMBIGUATION-AND-B2-VARIANCE-GATE-001"
    )
    r3 = Path(
        "/data/k1x-stage-runs/BANANA-YOLO26-XSLIM-STAGE65B-R3-HIERARCHICAL-EARLY-"
        "SUBGRAPH-CUT-SPLICE-LOCALIZATION-AND-XSLIM-TARGET-POLICY-CHARTER-001"
    )
    s64 = Path(
        "/data/k1x-stage-runs/BANANA-YOLO26-VENDOR-ORT-STAGE64-XSLIM211-AND-VENDOR-"
        "COMMIT-S8-QDQ-YOLO26-FULLMODEL-COCO-AND-RT206-GATE-001"
    )
    specs = [
        (
            "FP32-source",
            Path(
                "/data/banana-yolo26-spacemit-demo/.deps/models/yolo26/fp32_fp16_xslim_effect_matrix/yolo26n_640_e2e_fp32.onnx"
            ),
            "d71286588abe691ede49faa5ca9a471b7e9e5257669953ee59abbc2e9d115fc2",
        ),
        (
            "FP32-split-inference",
            s64 / "models/fp32-split/yolo26n_640_e2e_fp32.inference.onnx",
            "72eb6136b41104753c53b8e13aeff50e7961c4cefba79e50b70894cbd169f8d8",
        ),
        (
            "common-tail",
            s64
            / "models/R211_PROJECT_EXACT_SPLIT_run1/r211_project_exact_split.postprocess.onnx",
            "18ffff41e6812fa781baf7b9c1fcd41b41d6118145d785c3e550499070a512a3",
        ),
        (
            "B2-deployable",
            r1 / "quantization/B2/run1/output/stage65b_r1_b2_split_s8_qdq.onnx",
            "0e7040d4e8b1b2d08a4e36cec4c99dcea6d52294e04901d17dfce10725c6d617",
        ),
        (
            "B2-inference",
            r1
            / "postprocess/B2/candidate-gate/B2/models/stage65b_r1_b2.inference.onnx",
            "40ba6a7f9aebaa98a1c3abe5fce1f66f1bebcd0b10b7af3d26d30414a331d853",
        ),
        (
            "B2-full-val-predictions",
            r1 / "full-matrix/full-coco/B2/predictions.json",
            "51f8d4b25245a5f3e24feafea8aa49547c0f530f59cabcd18e61a744b4740add",
        ),
        (
            "D8-diagnostic",
            r2 / "models/B2-D8-final-six-output-qdq-bypass-diagnostic.onnx",
            "a77f2efea1dee7578d66859159a01c08ea45b76b44865d40b813732aa84372d4",
        ),
        (
            "H8-full-val-predictions",
            r1 / "full-matrix/hybrid-full-coco/H8/predictions.json",
            "b9ff8fa19cba9682970d8e932f3318cdf5833094ab22256a24062019309b5b2a",
        ),
        (
            "FQ8-L3-full-val-predictions",
            r3 / "full-val/fq8/L3/predictions.json",
            "70079ab14f1f30361e8f728683bb4b5e6c4a6059e8fdd321bfc4aab5323a62b4",
        ),
        (
            "XSlim-release-wheel",
            Path(
                "/data/k1x-stage-runs/BANANA-YOLO26-XSLIM-STAGE65A-PUB4-GITHUB-"
                "CREDENTIAL-REBIND-DUAL-REMOTE-RELEASE-ASSET-PARITY-CLOSURE-001/"
                "downloads/github-public/xslim-2.1.2+riscy.1-py3-none-any.whl"
            ),
            "635441d26458c6754627dd9595132cfd9a16d762d3ed0252f0039be668c01784",
        ),
    ]
    artifact_rows = verify_artifacts(specs)

    ncnn = Path("/data/ncnn")
    ncnn_values = {
        "ncnn_head": command("git", "rev-parse", "HEAD", cwd=ncnn),
        "ncnn_tree": command("git", "rev-parse", "HEAD^{tree}", cwd=ncnn),
        "ncnn_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "--binary"], cwd=ncnn)
        ).hexdigest(),
        "ncnn_dirty_path_count": len(
            command("git", "status", "--porcelain=v1", cwd=ncnn).splitlines()
        ),
    }
    ncnn_expected = {
        "ncnn_head": "a245a70c641a1f20f357c65d103e5f9e50fe84a1",
        "ncnn_tree": "20b96dadbd1fc0a53159cb35749719e967b55906",
        "ncnn_diff_sha256": "2bf1cc38885018a02478aa7542581639786c79bca5ce11a6e827d24bcc5f4eca",
        "ncnn_dirty_path_count": 3,
    }
    if ncnn_values != ncnn_expected:
        raise RuntimeError(f"/data/ncnn identity mismatch: {ncnn_values}")

    write_tsv(
        options.tracked_root / "remote_parity_before.tsv",
        ["repository", "branch", "surface", "local", "remote", "expected", "status"],
        remote_rows,
    )
    write_tsv(
        options.tracked_root / "r3_packet_verification.tsv",
        ["field", "expected", "observed", "status"],
        [
            {
                "field": "tree_sha256",
                "expected": packet_expected[0],
                "observed": packet_hash,
                "status": "pass",
            },
            {
                "field": "file_count",
                "expected": packet_expected[1],
                "observed": packet_files,
                "status": "pass",
            },
            {
                "field": "byte_count",
                "expected": packet_expected[2],
                "observed": packet_bytes,
                "status": "pass",
            },
            {
                "field": "symlink_count",
                "expected": 0,
                "observed": sum(
                    1 for path in options.packet.rglob("*") if path.is_symlink()
                ),
                "status": "pass",
            },
        ],
    )
    write_tsv(
        options.tracked_root / "frozen_artifact_identity.tsv",
        [
            "artifact",
            "canonical_path",
            "bytes",
            "expected_sha256",
            "observed_sha256",
            "canonical_match_count",
            "status",
        ],
        artifact_rows,
    )
    protected_rows = []
    for key, value in {**observed_refs, **ncnn_values}.items():
        expected = expected_refs.get(key, ncnn_expected.get(key))
        protected_rows.append(
            {"surface": key, "observed": value, "expected": expected, "status": "pass"}
        )
    write_tsv(
        options.tracked_root / "protected_state_before.tsv",
        ["surface", "observed", "expected", "status"],
        protected_rows,
    )

    mapping_rows = [
        ("XSlim checkout", str(options.xslim_worktree)),
        ("Banana protected checkout", str(options.banana_protected)),
        ("Banana research worktree", str(options.banana_worktree)),
        ("Stage raw root", str(options.raw_root)),
        ("Banana tracked root", str(options.tracked_root)),
        ("XSlim evidence root", str(options.xslim_evidence_root)),
        ("dataset", "/data/datasets/coco2017-independent-stage65b-r1"),
    ]
    write_tsv(
        options.tracked_root / "path_mapping_revalidated.tsv",
        ["surface", "container", "rhel10", "google_drive"],
        [
            {
                "surface": surface,
                "container": path,
                "rhel10": path.replace(
                    "/data/", "/mnt/DataSets/XuBanana/bf3-ncnn/data/", 1
                ),
                "google_drive": path.replace("/data/", "bf3-ncnn/data/", 1),
            }
            for surface, path in mapping_rows
        ],
    )
    manifest = {
        "stage_id": STAGE_ID,
        "execution_authority": "direct-user-authorization",
        "board_execution_authorized": False,
        "policy_b_implementation_authorized": False,
        "xslim_branch_start": expected_refs["xslim_head"],
        "banana_branch_start": expected_refs["banana_head"],
        "r3_packet": {
            "tree_sha256": packet_hash,
            "files": packet_files,
            "bytes": packet_bytes,
        },
        "raw_root": str(options.raw_root),
        "tracked_root": str(options.tracked_root),
        "xslim_evidence_root": str(options.xslim_evidence_root),
    }
    (options.tracked_root / "effective_launch_manifest.yaml").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (options.tracked_root / "workspace_preflight.md").write_text(
        "# XSLIM-DEV-001A Workspace Preflight\n\n"
        f"- UTC: `{command('date', '-u', '+%Y-%m-%dT%H:%M:%SZ')}`\n"
        f"- XSlim source: `{expected_refs['xslim_head']}` / tree `{expected_refs['xslim_tree']}`.\n"
        f"- Banana research: `{expected_refs['banana_head']}` / tree `{expected_refs['banana_tree']}`.\n"
        f"- R3 packet: `{packet_hash}`, {packet_files} files, {packet_bytes} bytes.\n"
        "- GitHub/GitLab parity, frozen artifacts, protected main, custom executor and /data/ncnn: pass.\n"
        "- Board execution, release publication, Policy B and custom-executor mutation are disabled.\n"
        f"- Host: Python `{sys.version.split()[0]}`, `{platform.platform()}`.\n",
        encoding="utf-8",
    )
    options.shared_log.mkdir(parents=True, exist_ok=True)
    (options.shared_log / "preflight-complete.txt").write_text(
        f"{STAGE_ID}\npreflight=pass\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
