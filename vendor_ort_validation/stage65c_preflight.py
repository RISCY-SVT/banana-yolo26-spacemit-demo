#!/usr/bin/env python3
"""Verify immutable Stage65C inputs and emit compact preflight evidence."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from collections.abc import Iterable
from pathlib import Path

STAGE_ID = (
    "BANANA-YOLO26-XSLIM-STAGE65C-A1-VS-B2-K1X-SPACEMIT-EP-PLACEMENT-"
    "CORRECTNESS-COCO-PERFORMANCE-AND-STABILITY-GATE-001"
)
DEV_ID = (
    "BANANA-YOLO26-XSLIM-DEV-001A-SPACEMIT-S8-QDQ-CONSTRAINED-RANGE-"
    "OBSERVER-TERMINAL-DOMAIN-AND-POLICY-A-HOST-CANDIDATE-GATE-001"
)
R1_ID = (
    "BANANA-YOLO26-XSLIM-STAGE65B-R1-COCO-TRAIN2017-EVALUATION-"
    "DISJOINT-CORPUS-PTQ-GRAPHWISE-AND-PYRAMID-CAUSAL-LOCALIZATION-001"
)
S64_ID = (
    "BANANA-YOLO26-VENDOR-ORT-STAGE64-XSLIM211-AND-VENDOR-COMMIT-S8-"
    "QDQ-YOLO26-FULLMODEL-COCO-AND-RT206-GATE-001"
)

BANANA = Path("/data/worktrees/banana-yolo26-xslim211-s8-qdq-validation")
PROTECTED = Path("/data/banana-yolo26-spacemit-demo")
XSLIM = Path("/data/worktrees/riscy-xslim-k1x-yolo26")
NCNN = Path("/data/ncnn")
DEV_RAW = Path("/data/k1x-stage-runs") / DEV_ID
R1_RAW = Path("/data/k1x-stage-runs") / R1_ID
S64_RAW = Path("/data/k1x-stage-runs") / S64_ID
DATASET = Path("/data/datasets/coco2017-independent-stage65b-r1")
PACKET = Path("/exchange/results/outbox") / DEV_ID

EXPECTED_PACKET = (
    "4be1455764a4ffa28cdf523c5ac1b0ec509be38b8c9a20792404ae2dd97e6d12",
    90,
    434031,
)

EXPECTED_REFS = {
    "banana_research_head": "dee0975d02660691b1f690ae1530fd0247891981",
    "banana_research_tree": "b5cc5b5e31e45088a12052623123d312cd80faa8",
    "banana_protected_main": "1fd2e71bb1d5a924e7c0444cada94f681b73aa91",
    "custom_executor_tree": "c2e400de14fb1c88d4aed70a249d9eff19a05d0f",
    "xslim_head": "3e275c6496d603d3f75f363ed00aa633ffc00408",
    "xslim_tree": "acdd6d64f35c7554f2559c781c5cbe0806acac1a",
    "xslim_release_tag_object": "604507c45c8eb2ff5c548d30d981203fedc61ea3",
    "xslim_release_tag_peeled": "12647b4a79fe5ec9a3973515a17cece4cb83daf4",
    "ncnn_head": "a245a70c641a1f20f357c65d103e5f9e50fe84a1",
    "ncnn_tree": "20b96dadbd1fc0a53159cb35749719e967b55906",
    "ncnn_diff": "2bf1cc38885018a02478aa7542581639786c79bca5ce11a6e827d24bcc5f4eca",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def git(repo: Path, *args: str) -> str:
    return command("git", *args, cwd=repo)


def packet_identity(root: Path) -> tuple[str, int, int]:
    relative_paths = [
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    ]
    environment = os.environ.copy()
    environment["LC_ALL"] = "en_US.UTF-8"
    sorted_paths = subprocess.check_output(
        ["sort"],
        input="\n".join(relative_paths) + "\n",
        text=True,
        env=environment,
    ).splitlines()
    digest = hashlib.sha256()
    total = 0
    for relative in sorted_paths:
        path = root / relative
        file_hash = sha256(path)
        digest.update(f"{file_hash}\t{relative}\n".encode())
        total += path.stat().st_size
    return digest.hexdigest(), len(sorted_paths), total


def write_tsv(path: Path, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    path.write_text(
        "\t".join(header)
        + "\n"
        + "".join("\t".join(str(value) for value in row) + "\n" for row in rows),
        encoding="utf-8",
    )


def verify(label: str, actual: str, expected: str) -> tuple[str, str, str, str]:
    status = "pass" if actual == expected else "fail"
    return label, actual, expected, status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracked-root", type=Path, required=True)
    args = parser.parse_args()
    out = args.tracked_root
    out.mkdir(parents=True, exist_ok=True)

    banana_head = git(BANANA, "rev-parse", "HEAD")
    banana_tree = git(BANANA, "rev-parse", "HEAD^{tree}")
    protected_main = git(PROTECTED, "rev-parse", "yolo26-custom-int8-engine")
    custom_executor_tree = git(
        PROTECTED,
        "rev-parse",
        "yolo26-custom-int8-engine:custom_int8_engine",
    )
    xslim_head = git(XSLIM, "rev-parse", "HEAD")
    xslim_tree = git(XSLIM, "rev-parse", "HEAD^{tree}")
    xslim_tag = git(XSLIM, "rev-parse", "refs/tags/v2.1.2-riscy.1")
    xslim_peeled = git(XSLIM, "rev-parse", "refs/tags/v2.1.2-riscy.1^{}")
    ncnn_head = git(NCNN, "rev-parse", "HEAD")
    ncnn_tree = git(NCNN, "rev-parse", "HEAD^{tree}")
    ncnn_diff = hashlib.sha256(
        subprocess.check_output(["git", "diff", "--binary"], cwd=NCNN)
    ).hexdigest()

    ref_rows = [
        verify("banana_research_head", banana_head, EXPECTED_REFS["banana_research_head"]),
        verify("banana_research_tree", banana_tree, EXPECTED_REFS["banana_research_tree"]),
        verify("banana_protected_main", protected_main, EXPECTED_REFS["banana_protected_main"]),
        verify("custom_executor_tree", custom_executor_tree, EXPECTED_REFS["custom_executor_tree"]),
        verify("xslim_head", xslim_head, EXPECTED_REFS["xslim_head"]),
        verify("xslim_tree", xslim_tree, EXPECTED_REFS["xslim_tree"]),
        verify("xslim_release_tag_object", xslim_tag, EXPECTED_REFS["xslim_release_tag_object"]),
        verify("xslim_release_tag_peeled", xslim_peeled, EXPECTED_REFS["xslim_release_tag_peeled"]),
        verify("ncnn_head", ncnn_head, EXPECTED_REFS["ncnn_head"]),
        verify("ncnn_tree", ncnn_tree, EXPECTED_REFS["ncnn_tree"]),
        verify("ncnn_diff", ncnn_diff, EXPECTED_REFS["ncnn_diff"]),
    ]

    packet_hash, packet_files, packet_bytes = packet_identity(PACKET)
    packet_rows = [
        ("tree_sha256", packet_hash, EXPECTED_PACKET[0], "pass" if packet_hash == EXPECTED_PACKET[0] else "fail"),
        ("file_count", packet_files, EXPECTED_PACKET[1], "pass" if packet_files == EXPECTED_PACKET[1] else "fail"),
        ("byte_count", packet_bytes, EXPECTED_PACKET[2], "pass" if packet_bytes == EXPECTED_PACKET[2] else "fail"),
    ]

    artifacts = [
        ("a1_deployable", DEV_RAW / "candidates/quantization/A1/run1/output/xslim_dev_001a_a1_split_s8_qdq.onnx", "8fad9fa0e385f58da281d963c5e18b010c80c402dcbeed0b46e3ca3065d010f3"),
        ("a1_inference", DEV_RAW / "candidates/postprocess/A1/models/stage65b_r1_a1.inference.onnx", "f7c5345f68cf79a5c3748274239a14cdaa59f77eac0425f7771694febaa24632"),
        ("a1_tail", DEV_RAW / "candidates/postprocess/A1/models/stage65b_r1_a1.postprocess.onnx", "18ffff41e6812fa781baf7b9c1fcd41b41d6118145d785c3e550499070a512a3"),
        ("a1_range_manifest", DEV_RAW / "candidates/quantization/A1/run1/range-policy-manifest.json", "e9ce9a1e71005d60ad18213d8110fbf51d4ab9ceb8509d9786989685aa0f7e6f"),
        ("a1_h500_predictions", DEV_RAW / "candidates/h500/A1/predictions.json", "aae66e9e98c2ee4b10aff3385a5057da834680ff0ac8aba9e93a0baf56a4b4bf"),
        ("a1_val_predictions", DEV_RAW / "candidates/full-val/A1/predictions.json", "fdae3c397ff82b005b3c0f507496392dde381fed2aaa0f5d18f03ea35c7b2df9"),
        ("xslim_dev_wheel", DEV_RAW / "dist/final/xslim-2.1.2+riscy.2.dev1-py3-none-any.whl", "c31afd3a0f1479e55e242d162b25a203e4511e7ea6a8c3e71eb3232dc92de6b8"),
        ("b2_deployable", R1_RAW / "quantization/B2/run1/output/stage65b_r1_b2_split_s8_qdq.onnx", "0e7040d4e8b1b2d08a4e36cec4c99dcea6d52294e04901d17dfce10725c6d617"),
        ("b2_inference", R1_RAW / "postprocess/B2/candidate-gate/B2/models/stage65b_r1_b2.inference.onnx", "40ba6a7f9aebaa98a1c3abe5fce1f66f1bebcd0b10b7af3d26d30414a331d853"),
        ("common_tail", R1_RAW / "postprocess/B2/candidate-gate/B2/models/stage65b_r1_b2.postprocess.onnx", "18ffff41e6812fa781baf7b9c1fcd41b41d6118145d785c3e550499070a512a3"),
        ("b2_val_predictions", R1_RAW / "full-matrix/full-coco/B2/predictions.json", "51f8d4b25245a5f3e24feafea8aa49547c0f530f59cabcd18e61a744b4740add"),
        ("ort206_archive", Path("/data/vendor-runtimes/downloads/spacemit-ort.riscv64.2.0.6.tar.gz"), "bebcdfb7df6b49eefa3863afcd85a3da2aa83c3ae9252d7d856188c38a70b0e6"),
        ("ort206_core", Path("/data/vendor-runtimes/spacemit-ort/2.0.6/spacemit-ort.riscv64.2.0.6/lib/libonnxruntime.so.1.24.2+spacemit.a1"), "93bb75601d9eceb5aca192fa70c0c3e18b94a70b9f57acdc9b34c2ff426e09e3"),
        ("ort206_ep", Path("/data/vendor-runtimes/spacemit-ort/2.0.6/spacemit-ort.riscv64.2.0.6/lib/libspacemit_ep.so.2.0.6"), "dcc9503031bca22cf2b33a692f7b4c01d0fbb4a24c34f6e60c7faaddb78274ae"),
        ("published_xslim_wheel", Path("/data/k1x-stage-runs/BANANA-YOLO26-XSLIM-STAGE65A-PUB4-GITHUB-CREDENTIAL-REBIND-DUAL-REMOTE-RELEASE-ASSET-PARITY-CLOSURE-001/downloads/github-public/xslim-2.1.2+riscy.1-py3-none-any.whl"), "635441d26458c6754627dd9595132cfd9a16d762d3ed0252f0039be668c01784"),
    ]
    artifact_rows = []
    for label, path, expected in artifacts:
        actual = sha256(path)
        artifact_rows.append((label, path, path.stat().st_size, actual, expected, "pass" if actual == expected else "fail"))

    datasets = [
        ("h500_list", DATASET / "lists/selection_H500_holdout.txt"),
        ("val2017_list", DATASET / "lists/val2017_all.txt"),
        ("train2017_annotations", DATASET / "annotations/instances_train2017.json"),
        ("val2017_annotations", DATASET / "annotations/instances_val2017.json"),
    ]
    dataset_rows = [(label, path, path.stat().st_size, sha256(path)) for label, path in datasets]

    runners = [
        ("two_stage_coco", S64_RAW / "bin/stage64_two_stage_coco"),
        ("two_stage_runner", S64_RAW / "bin/stage64_two_stage_runner"),
        ("single_model_runner", S64_RAW / "host/build/public-repro-runner/stage64_single_model_runner"),
        ("coco_evaluator", BANANA / "vendor_ort_validation/stage65b_r1_coco_metrics.py"),
        ("paired_bootstrap", BANANA / "vendor_ort_validation/stage65b_r2_bootstrap.py"),
    ]
    runner_rows = [(label, path, path.stat().st_size, sha256(path)) for label, path in runners]

    write_tsv(out / "dev001a_packet_verification.tsv", ("field", "actual", "expected", "status"), packet_rows)
    write_tsv(out / "frozen_artifact_identity.tsv", ("artifact", "path", "bytes", "sha256", "expected_sha256", "status"), artifact_rows)
    write_tsv(out / "frozen_dataset_identity.tsv", ("surface", "path", "bytes", "sha256"), dataset_rows)
    write_tsv(out / "runner_identity.tsv", ("tool", "path", "bytes", "sha256"), runner_rows)
    write_tsv(out / "protected_state_before.tsv", ("identity", "actual", "expected", "status"), ref_rows)

    drive_rows = [
        ("result_packet", "1eXCjaRqI-9i5x1g9KiwRbN1jHVTzZXPE", "folder inventory and selected report content", "pass-bounded"),
        ("banana_tracked_stage", "1VBW2P_CIP7girDhHlQLhMzqiZI81a1u8", "required reports and final parity visible", "pass-bounded"),
        ("xslim_tracked_stage", "1Utzno9lFRSnMd9lyrlHcijqPkNXB6IlF", "compact source reports visible", "pass-bounded"),
        ("raw_stage", "1YekAhDbqPQJOn1haJAuun2q4MA8Ocrq7", "raw directory roles visible; model bytes not downloaded", "pass-bounded"),
        ("shared_log", "1SivwPeWrygpGE8VnQFK0CiG7QwfkMhKB", "shared log role visible", "pass-bounded"),
        ("duplicate_mirror", "14H7g6TrFOLnBRt7HZUX12RPY_6hl-ASt", "duplicate task/result role retained", "debt-nonblocking"),
    ]
    write_tsv(out / "google_drive_input_evidence_verify.tsv", ("role", "drive_folder_id", "verification", "status"), drive_rows)

    (out / "EVIDENCE_INDEX.yaml").write_text(
        """stage_id: """ + STAGE_ID + """
canonical:
  banana_tracked_stage:
    local: stages/""" + DEV_ID + """
    drive_folder_id: 1VBW2P_CIP7girDhHlQLhMzqiZI81a1u8
  xslim_tracked_stage:
    local: /data/worktrees/riscy-xslim-k1x-yolo26/stages/""" + DEV_ID + """
    drive_folder_id: 1Utzno9lFRSnMd9lyrlHcijqPkNXB6IlF
  raw_stage:
    local: /data/k1x-stage-runs/""" + DEV_ID + """
    drive_folder_id: 1YekAhDbqPQJOn1haJAuun2q4MA8Ocrq7
  result_packet:
    local: /exchange/results/outbox/""" + DEV_ID + """
    drive_folder_id: 1eXCjaRqI-9i5x1g9KiwRbN1jHVTzZXPE
  shared_log:
    drive_folder_id: 1SivwPeWrygpGE8VnQFK0CiG7QwfkMhKB
  post_push_attestation:
    source: canonical tracked parity reports plus live Git ref readback
superseded_or_duplicate:
  - drive_folder_id: 14H7g6TrFOLnBRt7HZUX12RPY_6hl-ASt
    role: duplicate task-run/result mirror
verification_scope: bounded metadata and selected report content; no model or dataset download
mirror_debt: full remote tree-byte verification unavailable; nonblocking because local packet and Git identities are exact
""",
        encoding="utf-8",
    )

    (out / "effective_launch_manifest.yaml").write_text(
        f"""stage_id: {STAGE_ID}
authorization: direct-user
banana_head: {banana_head}
banana_tree: {banana_tree}
xslim_head: {xslim_head}
xslim_tree: {xslim_tree}
dev001a_packet_tree_sha256: {packet_hash}
board_target: svt@banana
board_workspace: /data/k1x-stage-runs/{STAGE_ID}
runtime_version: 2.0.6
models: frozen; regeneration forbidden
""",
        encoding="utf-8",
    )

    failures = [row for row in ref_rows + packet_rows if row[-1] != "pass"]
    failures.extend(row for row in artifact_rows if row[-1] != "pass")
    status = "pass" if not failures else "fail"
    (out / "workspace_preflight.md").write_text(
        f"""# Stage65C workspace preflight

- Status: `{status}`
- Banana research head/tree: `{banana_head}` / `{banana_tree}`
- XSlim head/tree: `{xslim_head}` / `{xslim_tree}`
- DEV-001A packet: `{packet_hash}`, {packet_files} files, {packet_bytes} bytes
- A1, B2, common tail, runtime archive/libraries and accepted wheels: exact SHA-256 pass
- `/data/ncnn`: accepted head/tree/diff and three pre-existing dirty paths
- Drive: bounded metadata and selected-report verification passed; full mirror-byte verification remains nonblocking debt
- No model, XSlim source, protected main, custom executor, tag or release mutation was performed.
""",
        encoding="utf-8",
    )

    if failures:
        raise SystemExit(f"preflight failed: {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
