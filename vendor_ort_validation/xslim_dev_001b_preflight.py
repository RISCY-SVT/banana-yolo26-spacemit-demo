#!/usr/bin/env python3
"""Verify immutable DEV-001B inputs and freeze the reconstruction split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

STAGE_ID = (
    "BANANA-YOLO26-XSLIM-DEV-001B-ALL-S8-GENERIC-HARDENING-ADAPTIVE-"
    "ROUNDING-BLOCK-RECONSTRUCTION-AND-DETECTOR-PARETO-HOST-GATE-001"
)
R1_ID = (
    "BANANA-YOLO26-XSLIM-STAGE65B-R1-COCO-TRAIN2017-EVALUATION-DISJOINT-"
    "CORPUS-PTQ-GRAPHWISE-AND-PYRAMID-CAUSAL-LOCALIZATION-001"
)
DEV001A_ID = (
    "BANANA-YOLO26-XSLIM-DEV-001A-SPACEMIT-S8-QDQ-CONSTRAINED-RANGE-"
    "OBSERVER-TERMINAL-DOMAIN-AND-POLICY-A-HOST-CANDIDATE-GATE-001"
)
STAGE65C_R1_ID = (
    "BANANA-YOLO26-XSLIM-STAGE65C-R1-A1-CPU-EP-LARGE-RECALL-DIVERGENCE-"
    "AND-TERMINAL-BOUNDARY-CAUSAL-DIAGNOSTIC-001"
)

EXPECTED_LAUNCH_HEAD = "d3afe14480ec2efbb2df9436deaa9022d631faa0"
ACCEPTED_ERRATUM_HEAD = "16d36c569e267016cecabc6333515d2feecb12aa"
EXPECTED_XSLIM_HEAD = "3e275c6496d603d3f75f363ed00aa633ffc00408"
EXPECTED_XSLIM_TREE = "acdd6d64f35c7554f2559c781c5cbe0806acac1a"
EXPECTED_PACKET = (
    "8398831b147cc890436e968d830b14c0d5347ee5a24946b03156c66aa08b22e6",
    63,
    1_983_169,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def command(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def packet_identity(root: Path) -> tuple[str, int, int]:
    paths = [item.relative_to(root).as_posix() for item in root.rglob("*") if item.is_file()]
    environment = os.environ.copy()
    environment["LC_ALL"] = "en_US.UTF-8"
    ordered = subprocess.check_output(
        ["sort"], input="\n".join(paths) + "\n", text=True, env=environment
    ).splitlines()
    digest = hashlib.sha256()
    total = 0
    for relative in ordered:
        path = root / relative
        digest.update(f"{sha256(path)}\t{relative}\n".encode())
        total += path.stat().st_size
    return digest.hexdigest(), len(ordered), total


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty report: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def require(label: str, observed: object, expected: object) -> dict[str, object]:
    if observed != expected:
        raise RuntimeError(f"{label} mismatch: {observed!r} != {expected!r}")
    return {"surface": label, "observed": observed, "expected": expected, "status": "pass"}


def freeze_partition(c200: Path, h500: Path, val: Path, output: Path) -> list[dict[str, object]]:
    policy = b"xslim-dev-001b-reconstruction-split-v1\0"
    c200_rows = [line.strip() for line in c200.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(c200_rows) != 200 or len(set(c200_rows)) != 200:
        raise RuntimeError("accepted C200 must contain 200 unique paths")
    ranked = sorted(c200_rows, key=lambda value: (hashlib.sha256(policy + value.encode()).hexdigest(), value))
    optimization, validation = ranked[:160], ranked[160:]
    h500_names = {Path(line).name for line in h500.read_text(encoding="utf-8").splitlines() if line.strip()}
    val_names = {Path(line).name for line in val.read_text(encoding="utf-8").splitlines() if line.strip()}
    selected_names = {Path(line).name for line in ranked}
    if selected_names & h500_names or selected_names & val_names:
        raise RuntimeError("reconstruction split overlaps H500 or val2017")
    output.mkdir(parents=True, exist_ok=True)
    opt_path = output / "optimization_C160.txt"
    validation_path = output / "validation_C40.txt"
    rank_path = output / "partition_rank.tsv"
    opt_path.write_text("\n".join(optimization) + "\n", encoding="utf-8")
    validation_path.write_text("\n".join(validation) + "\n", encoding="utf-8")
    write_tsv(
        rank_path,
        [
            {
                "rank": index,
                "path": value,
                "rank_sha256": hashlib.sha256(policy + value.encode()).hexdigest(),
                "partition": "optimization" if index < 160 else "validation",
            }
            for index, value in enumerate(ranked)
        ],
    )
    return [
        {
            "surface": "C200",
            "count": 200,
            "sha256": sha256(c200),
            "overlap_h500": 0,
            "overlap_val2017": 0,
            "policy": policy.rstrip(b"\0").decode(),
            "status": "pass",
        },
        {
            "surface": "optimization_C160",
            "count": 160,
            "sha256": sha256(opt_path),
            "overlap_h500": 0,
            "overlap_val2017": 0,
            "policy": policy.rstrip(b"\0").decode(),
            "status": "pass",
        },
        {
            "surface": "validation_C40",
            "count": 40,
            "sha256": sha256(validation_path),
            "overlap_h500": 0,
            "overlap_val2017": 0,
            "policy": policy.rstrip(b"\0").decode(),
            "status": "pass",
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--banana", required=True, type=Path)
    parser.add_argument("--protected", required=True, type=Path)
    parser.add_argument("--xslim", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--tracked-root", required=True, type=Path)
    parser.add_argument("--shared-log", required=True, type=Path)
    options = parser.parse_args()

    options.raw_root.mkdir(parents=True, exist_ok=True)
    options.tracked_root.mkdir(parents=True, exist_ok=True)
    options.shared_log.mkdir(parents=True, exist_ok=True)

    banana_head = command("git", "rev-parse", "HEAD", cwd=options.banana)
    banana_tree = command("git", "rev-parse", "HEAD^{tree}", cwd=options.banana)
    if banana_head not in {EXPECTED_LAUNCH_HEAD, ACCEPTED_ERRATUM_HEAD}:
        raise RuntimeError(f"unexpected Banana head: {banana_head}")
    reconciliation = "launch-exact"
    if banana_head == ACCEPTED_ERRATUM_HEAD:
        require(
            "banana_erratum_parent",
            command("git", "rev-parse", f"{banana_head}^", cwd=options.banana),
            EXPECTED_LAUNCH_HEAD,
        )
        changed = command("git", "diff", "--name-only", f"{EXPECTED_LAUNCH_HEAD}..{banana_head}", cwd=options.banana).splitlines()
        expected_changed = sorted(
            [
                f"stages/{STAGE65C_R1_ID}/EVIDENCE_INDEX.yaml",
                f"stages/{STAGE65C_R1_ID}/STAGE65C_R1_INTERPRETATION_ERRATUM.md",
            ]
        )
        require("banana_erratum_paths", sorted(changed), expected_changed)
        reconciliation = "accepted-prior-append-only-erratum"

    xslim_head = command("git", "rev-parse", "HEAD", cwd=options.xslim)
    xslim_tree = command("git", "rev-parse", "HEAD^{tree}", cwd=options.xslim)
    identities = [
        require("xslim_head", xslim_head, EXPECTED_XSLIM_HEAD),
        require("xslim_tree", xslim_tree, EXPECTED_XSLIM_TREE),
        require("xslim_version", (options.xslim / "VERSION_NUMBER").read_text().strip(), "2.1.2+riscy.2.dev1"),
        require("upstream_main", command("git", "rev-parse", "upstream/main", cwd=options.xslim), "9a33f2f770d00fd02ff8bc0f1907135e9bf47f8c"),
        require("upstream_tree", command("git", "rev-parse", "upstream/main^{tree}", cwd=options.xslim), "05d2c8425ab8587abf401fa5976a08d008fdd719"),
        require("banana_protected_main", command("git", "rev-parse", "yolo26-custom-int8-engine", cwd=options.protected), "1fd2e71bb1d5a924e7c0444cada94f681b73aa91"),
        require("custom_executor_tree", command("git", "rev-parse", "yolo26-custom-int8-engine:custom_int8_engine", cwd=options.protected), "c2e400de14fb1c88d4aed70a249d9eff19a05d0f"),
    ]

    parity_rows: list[dict[str, object]] = []
    for repository, repo, branch, remotes in (
        ("Banana", options.banana, "yolo26-vendor-ort-xslim211-s8-qdq-validation", ("github", "gitlab-rd")),
        ("XSlim", options.xslim, "riscy/k1x-yolo26", ("github", "gitlab")),
    ):
        local = command("git", "rev-parse", branch, cwd=repo)
        for remote in remotes:
            remote_head = command("git", "rev-parse", f"{remote}/{branch}", cwd=repo)
            require(f"{repository}_{remote}_parity", remote_head, local)
            parity_rows.append(
                {"repository": repository, "branch": branch, "surface": remote, "local": local, "remote": remote_head, "status": "pass"}
            )

    packet = packet_identity(options.packet)
    require("packet_identity", packet, EXPECTED_PACKET)

    data = Path("/data")
    r1 = data / "k1x-stage-runs" / R1_ID
    dev001a = data / "k1x-stage-runs" / DEV001A_ID
    artifacts = [
        ("B2-deployable", r1 / "quantization/B2/run1/output/stage65b_r1_b2_split_s8_qdq.onnx", "0e7040d4e8b1b2d08a4e36cec4c99dcea6d52294e04901d17dfce10725c6d617"),
        ("B2-inference", r1 / "postprocess/B2/candidate-gate/B2/models/stage65b_r1_b2.inference.onnx", "40ba6a7f9aebaa98a1c3abe5fce1f66f1bebcd0b10b7af3d26d30414a331d853"),
        ("A1-deployable", dev001a / "candidates/quantization/A1/run1/output/xslim_dev_001a_a1_split_s8_qdq.onnx", "8fad9fa0e385f58da281d963c5e18b010c80c402dcbeed0b46e3ca3065d010f3"),
        ("A1-inference", dev001a / "candidates/postprocess/A1/models/stage65b_r1_a1.inference.onnx", "f7c5345f68cf79a5c3748274239a14cdaa59f77eac0425f7771694febaa24632"),
        ("A1-range-manifest", dev001a / "candidates/quantization/A1/run1/range-policy-manifest.json", "e9ce9a1e71005d60ad18213d8110fbf51d4ab9ceb8509d9786989685aa0f7e6f"),
        ("common-tail", r1 / "postprocess/B2/candidate-gate/B2/models/stage65b_r1_b2.postprocess.onnx", "18ffff41e6812fa781baf7b9c1fcd41b41d6118145d785c3e550499070a512a3"),
    ]
    artifact_rows = []
    for name, path, expected in artifacts:
        if not path.is_file():
            raise RuntimeError(f"missing frozen artifact: {path}")
        observed = sha256(path)
        require(name, observed, expected)
        artifact_rows.append(
            {"artifact": name, "canonical_path": path, "bytes": path.stat().st_size, "sha256": observed, "expected_sha256": expected, "status": "pass"}
        )

    c200 = options.dataset / "lists/selection_C200.txt"
    h500 = options.dataset / "lists/selection_H500_holdout.txt"
    val = options.dataset / "lists/val2017_all.txt"
    partition_rows = freeze_partition(c200, h500, val, options.raw_root / "dataset-partition")
    partition_rows.extend(
        [
            {"surface": "H500", "count": 500, "sha256": sha256(h500), "overlap_h500": "self", "overlap_val2017": 0, "policy": "accepted-R1", "status": "pass"},
            {"surface": "val2017", "count": 5000, "sha256": sha256(val), "overlap_h500": 0, "overlap_val2017": "self", "policy": "accepted-R1", "status": "pass"},
            {"surface": "train2017-annotations", "count": 118287, "sha256": sha256(options.dataset / "annotations/instances_train2017.json"), "overlap_h500": "n/a", "overlap_val2017": "n/a", "policy": "accepted-R1", "status": "pass"},
            {"surface": "val2017-annotations", "count": 5000, "sha256": sha256(options.dataset / "annotations/instances_val2017.json"), "overlap_h500": "n/a", "overlap_val2017": "n/a", "policy": "accepted-R1", "status": "pass"},
        ]
    )

    ncnn = data / "ncnn"
    ncnn_diff = hashlib.sha256(subprocess.check_output(["git", "diff", "--binary"], cwd=ncnn)).hexdigest()
    ncnn_rows = [
        require("ncnn_head", command("git", "rev-parse", "HEAD", cwd=ncnn), "a245a70c641a1f20f357c65d103e5f9e50fe84a1"),
        require("ncnn_tree", command("git", "rev-parse", "HEAD^{tree}", cwd=ncnn), "20b96dadbd1fc0a53159cb35749719e967b55906"),
        require("ncnn_diff", ncnn_diff, "2bf1cc38885018a02478aa7542581639786c79bca5ce11a6e827d24bcc5f4eca"),
        require("ncnn_dirty_paths", len(command("git", "status", "--porcelain=v1", cwd=ncnn).splitlines()), 3),
    ]

    write_tsv(options.tracked_root / "stage65c_r1_packet_verification.tsv", [
        {"field": "tree_sha256", "observed": packet[0], "expected": EXPECTED_PACKET[0], "status": "pass"},
        {"field": "file_count", "observed": packet[1], "expected": EXPECTED_PACKET[1], "status": "pass"},
        {"field": "byte_count", "observed": packet[2], "expected": EXPECTED_PACKET[2], "status": "pass"},
    ])
    write_tsv(options.tracked_root / "frozen_artifact_identity.tsv", artifact_rows)
    write_tsv(options.tracked_root / "dataset_partition_attestation.tsv", partition_rows)
    write_tsv(options.tracked_root / "protected_state_before.tsv", identities + ncnn_rows)
    write_tsv(options.tracked_root / "remote_parity_before.tsv", parity_rows)

    evidence_index = {
        "stage_id": STAGE_ID,
        "inputs": {
            "stage65c_r1_packet": str(options.packet),
            "dataset_root": str(options.dataset),
            "frozen_artifacts": [str(path) for _, path, _ in artifacts],
        },
        "outputs": {
            "banana_tracked": str(options.tracked_root),
            "raw": str(options.raw_root),
            "shared_log": str(options.shared_log),
        },
        "banana_launch_reconciliation": reconciliation,
    }
    (options.tracked_root / "input_evidence_index.yaml").write_text(
        json.dumps(evidence_index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    launch = {
        "stage_id": STAGE_ID,
        "execution_authority": "direct-user-authorization",
        "banana_declared_start": EXPECTED_LAUNCH_HEAD,
        "banana_effective_start": banana_head,
        "banana_effective_tree": banana_tree,
        "banana_reconciliation": reconciliation,
        "xslim_start": xslim_head,
        "xslim_tree": xslim_tree,
        "packet": {"tree_sha256": packet[0], "files": packet[1], "bytes": packet[2]},
        "board_execution_authorized": False,
        "new_branch_authorized": False,
    }
    (options.tracked_root / "effective_launch_manifest.yaml").write_text(
        json.dumps(launch, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (options.tracked_root / "workspace_preflight.md").write_text(
        "# XSLIM-DEV-001B workspace preflight\n\n"
        f"- Gate 0: `pass`.\n- Banana effective start: `{banana_head}` ({reconciliation}).\n"
        f"- XSlim start/tree: `{xslim_head}` / `{xslim_tree}`.\n"
        f"- Stage65C-R1 packet: `{packet[0]}`, {packet[1]} files, {packet[2]} bytes.\n"
        "- B2, A1, range manifest, common tail, dataset lists/annotations and protected states match.\n"
        "- C200 was deterministically split into 160 optimization and 40 reconstruction-validation images; overlap with H500/val2017 is zero.\n"
        "- Commands use container `/data`; `/exchange` is a separate managed handoff surface.\n",
        encoding="utf-8",
    )
    (options.shared_log / "preflight-complete.txt").write_text(
        f"{STAGE_ID}\npreflight=pass\nbanana_effective_start={banana_head}\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
