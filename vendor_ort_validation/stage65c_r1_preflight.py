#!/usr/bin/env python3
"""Verify immutable Stage65C-R1 inputs and emit compact Gate 0 evidence."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from collections.abc import Iterable
from pathlib import Path

STAGE_ID = (
    "BANANA-YOLO26-XSLIM-STAGE65C-R1-A1-CPU-EP-LARGE-RECALL-DIVERGENCE-"
    "AND-TERMINAL-BOUNDARY-CAUSAL-DIAGNOSTIC-001"
)
STAGE65C_ID = (
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

DATA = Path("/data")
BANANA = DATA / "worktrees/banana-yolo26-xslim211-s8-qdq-validation"
PROTECTED = DATA / "banana-yolo26-spacemit-demo"
XSLIM = DATA / "worktrees/riscy-xslim-k1x-yolo26"
NCNN = DATA / "ncnn"
RAW = DATA / "k1x-stage-runs" / STAGE_ID
STAGE65C_RAW = DATA / "k1x-stage-runs" / STAGE65C_ID
DEV_RAW = DATA / "k1x-stage-runs" / DEV_ID
R1_RAW = DATA / "k1x-stage-runs" / R1_ID
S64_RAW = DATA / "k1x-stage-runs" / S64_ID
PACKET = Path("/exchange/results/outbox") / STAGE65C_ID

EXPECTED_PACKET = (
    "27bfec346a38cf365754478ca386f4985303eb3d910f71726c7ec09f5432ebcd",
    69,
    116456,
)
EXPECTED = {
    "banana_head": "4868cbfbe0d1de41437b3568026069a12f806585",
    "banana_tree": "eef2dd835834e3163e721dd79b5fc1af11a226ff",
    "protected_main": "1fd2e71bb1d5a924e7c0444cada94f681b73aa91",
    "custom_executor_tree": "c2e400de14fb1c88d4aed70a249d9eff19a05d0f",
    "xslim_head": "3e275c6496d603d3f75f363ed00aa633ffc00408",
    "xslim_tree": "acdd6d64f35c7554f2559c781c5cbe0806acac1a",
    "xslim_version": "2.1.2+riscy.2.dev1",
    "ncnn_head": "a245a70c641a1f20f357c65d103e5f9e50fe84a1",
    "ncnn_tree": "20b96dadbd1fc0a53159cb35749719e967b55906",
    "ncnn_diff": "2bf1cc38885018a02478aa7542581639786c79bca5ce11a6e827d24bcc5f4eca",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(*args: str, cwd: Path | None = None, binary: bool = False):
    return subprocess.check_output(args, cwd=cwd, text=not binary).strip()


def git(repo: Path, *args: str) -> str:
    return str(run("git", *args, cwd=repo))


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


def write_tsv(path: Path, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    path.write_text(
        "\t".join(header)
        + "\n"
        + "".join("\t".join(str(value) for value in row) + "\n" for row in rows),
        encoding="utf-8",
    )


def checked(label: str, actual: object, expected: object) -> tuple[object, ...]:
    return label, actual, expected, "pass" if actual == expected else "fail"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracked-root", required=True, type=Path)
    options = parser.parse_args()
    out = options.tracked_root
    out.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    actual = {
        "banana_head": git(BANANA, "rev-parse", "HEAD"),
        "banana_tree": git(BANANA, "rev-parse", "HEAD^{tree}"),
        "protected_main": git(PROTECTED, "rev-parse", "yolo26-custom-int8-engine"),
        "custom_executor_tree": git(
            PROTECTED, "rev-parse", "yolo26-custom-int8-engine:custom_int8_engine"
        ),
        "xslim_head": git(XSLIM, "rev-parse", "HEAD"),
        "xslim_tree": git(XSLIM, "rev-parse", "HEAD^{tree}"),
        "xslim_version": (XSLIM / "VERSION_NUMBER").read_text().strip(),
        "ncnn_head": git(NCNN, "rev-parse", "HEAD"),
        "ncnn_tree": git(NCNN, "rev-parse", "HEAD^{tree}"),
        "ncnn_diff": hashlib.sha256(
            subprocess.check_output(["git", "diff", "--binary"], cwd=NCNN)
        ).hexdigest(),
    }
    identity_rows = [checked(key, actual[key], expected) for key, expected in EXPECTED.items()]
    ncnn_dirty = int(run("git", "status", "--porcelain", cwd=NCNN).count("\n")) + bool(
        run("git", "status", "--porcelain", cwd=NCNN)
    )
    identity_rows.append(checked("ncnn_dirty_path_count", int(ncnn_dirty), 3))

    remote_rows = []
    for repo, remote, ref, expected in (
        (BANANA, "github", "refs/heads/yolo26-vendor-ort-xslim211-s8-qdq-validation", EXPECTED["banana_head"]),
        (BANANA, "gitlab-rd", "refs/heads/yolo26-vendor-ort-xslim211-s8-qdq-validation", EXPECTED["banana_head"]),
        (XSLIM, "github", "refs/heads/riscy/k1x-yolo26", EXPECTED["xslim_head"]),
        (XSLIM, "gitlab", "refs/heads/riscy/k1x-yolo26", EXPECTED["xslim_head"]),
    ):
        value = str(run("git", "ls-remote", remote, ref, cwd=repo)).split()[0]
        remote_rows.append((repo.name, remote, ref, value, expected, "pass" if value == expected else "fail"))

    packet_hash, packet_files, packet_bytes = packet_identity(PACKET)
    packet_rows = [
        checked("tree_sha256", packet_hash, EXPECTED_PACKET[0]),
        checked("file_count", packet_files, EXPECTED_PACKET[1]),
        checked("byte_count", packet_bytes, EXPECTED_PACKET[2]),
    ]

    artifacts = [
        ("a1_deployable", DEV_RAW / "candidates/quantization/A1/run1/output/xslim_dev_001a_a1_split_s8_qdq.onnx", "8fad9fa0e385f58da281d963c5e18b010c80c402dcbeed0b46e3ca3065d010f3"),
        ("a1_inference", DEV_RAW / "candidates/postprocess/A1/models/stage65b_r1_a1.inference.onnx", "f7c5345f68cf79a5c3748274239a14cdaa59f77eac0425f7771694febaa24632"),
        ("a1_range_manifest", DEV_RAW / "candidates/quantization/A1/run1/range-policy-manifest.json", "e9ce9a1e71005d60ad18213d8110fbf51d4ab9ceb8509d9786989685aa0f7e6f"),
        ("b2_deployable", R1_RAW / "quantization/B2/run1/output/stage65b_r1_b2_split_s8_qdq.onnx", "0e7040d4e8b1b2d08a4e36cec4c99dcea6d52294e04901d17dfce10725c6d617"),
        ("b2_inference", R1_RAW / "postprocess/B2/candidate-gate/B2/models/stage65b_r1_b2.inference.onnx", "40ba6a7f9aebaa98a1c3abe5fce1f66f1bebcd0b10b7af3d26d30414a331d853"),
        ("common_tail", R1_RAW / "postprocess/B2/candidate-gate/B2/models/stage65b_r1_b2.postprocess.onnx", "18ffff41e6812fa781baf7b9c1fcd41b41d6118145d785c3e550499070a512a3"),
        ("ort206_asset", DATA / "vendor-runtimes/downloads/spacemit-ort.riscv64.2.0.6.tar.gz", "bebcdfb7df6b49eefa3863afcd85a3da2aa83c3ae9252d7d856188c38a70b0e6"),
        ("ort206_core", DATA / "vendor-runtimes/spacemit-ort/2.0.6/spacemit-ort.riscv64.2.0.6/lib/libonnxruntime.so.1.24.2+spacemit.a1", "93bb75601d9eceb5aca192fa70c0c3e18b94a70b9f57acdc9b34c2ff426e09e3"),
        ("ort206_ep", DATA / "vendor-runtimes/spacemit-ort/2.0.6/spacemit-ort.riscv64.2.0.6/lib/libspacemit_ep.so.2.0.6", "dcc9503031bca22cf2b33a692f7b4c01d0fbb4a24c34f6e60c7faaddb78274ae"),
        ("two_stage_runner", S64_RAW / "bin/stage64_two_stage_runner", "b39da697c4c0f309a9955475946ce5ce6b53b264bd3bae7cb50283fb49ce2188"),
        ("two_stage_coco", S64_RAW / "bin/stage64_two_stage_coco", "d39f5280a46b0bd342de933d6b25005a5858390430e8ce84377374f4a8225ad0"),
    ]
    artifact_rows = []
    for name, path, expected in artifacts:
        value = sha256(path) if path.is_file() else "missing"
        artifact_rows.append((name, path, path.stat().st_size if path.is_file() else 0, value, expected, "pass" if value == expected else "fail"))

    predictions = [
        ("B2_CPU", STAGE65C_RAW / "board/coco/h500/B2-cpu/predictions.json", "8d4ff1740a318a7749259e8b18058cda7b62dc198a73e2fb459e992f28e8f681"),
        ("B2_EP", STAGE65C_RAW / "board/coco/h500/B2-spacemit/predictions.json", "c0854d85ba119701548bd7050ec503f006718824f95c19fb8bb0c675b7f64f99"),
        ("A1_CPU", STAGE65C_RAW / "board/coco/h500/A1-cpu/predictions.json", "bddd8c6687c724012e98e5a63beb4ac0764103696b2f9e063123d72b49fc8cc4"),
        ("A1_EP", STAGE65C_RAW / "board/coco/h500/A1-spacemit/predictions.json", "15be531efa93b28a3f7b84b233b9f7c2401722af21b9fcc5b42e415e4b1ebf34"),
    ]
    prediction_rows = []
    for name, path, expected in predictions:
        value = sha256(path) if path.is_file() else "missing"
        prediction_rows.append((name, path, path.stat().st_size if path.is_file() else 0, value, expected, "pass" if value == expected else "fail"))

    all_statuses = [row[-1] for row in identity_rows + packet_rows + artifact_rows + prediction_rows + remote_rows]
    status = "pass" if all(value == "pass" for value in all_statuses) else "fail"

    write_tsv(out / "stage65c_packet_verification.tsv", ("field", "actual", "expected", "status"), packet_rows)
    write_tsv(out / "frozen_artifact_identity.tsv", ("artifact", "path", "bytes", "sha256", "expected_sha256", "status"), artifact_rows)
    write_tsv(out / "stage65c_prediction_identity.tsv", ("surface", "path", "bytes", "sha256", "expected_sha256", "status"), prediction_rows)
    write_tsv(out / "protected_state_before.tsv", ("field", "actual", "expected", "status"), identity_rows)
    write_tsv(out / "remote_parity_before.tsv", ("repository", "remote", "ref", "actual", "expected", "status"), remote_rows)
    write_tsv(
        out / "runtime_binding.tsv",
        ("field", "value", "status"),
        (
            ("declared_runtime", "2.0.6", "pass"),
            ("ort_asset_sha256", artifacts[6][2], artifact_rows[6][-1]),
            ("ort_core_sha256", artifacts[7][2], artifact_rows[7][-1]),
            ("spacemit_ep_sha256", artifacts[8][2], artifact_rows[8][-1]),
            ("runtime_root", DATA / "vendor-runtimes/spacemit-ort/2.0.6", "pass"),
        ),
    )
    (out / "effective_launch_manifest.yaml").write_text(
        f"""stage_id: {STAGE_ID}
authorization: direct-user
banana_head: {actual['banana_head']}
banana_tree: {actual['banana_tree']}
xslim_head: {actual['xslim_head']}
xslim_tree: {actual['xslim_tree']}
stage65c_packet_tree_sha256: {packet_hash}
board_target: svt@banana
board_workspace: /data/k1x-stage-runs/{STAGE_ID}
runtime_version: 2.0.6
models: frozen; regeneration forbidden
h500_bootstrap_seed: 65008
full_val_bootstrap_seed: 65009
""",
        encoding="utf-8",
    )
    (out / "workspace_preflight.md").write_text(
        f"""# Stage65C-R1 workspace preflight

- Status: `{status}`.
- Banana research branch and both remotes: `{actual['banana_head']}`.
- XSlim and both remotes: `{actual['xslim_head']}`; source is read-only.
- Stage65C packet: `{packet_hash}`, {packet_files} files, {packet_bytes} bytes.
- Frozen A1/B2/tail, runtime asset/libraries, runners and H500 predictions match.
- Protected Banana main/custom executor and accepted ncnn state match.
- Commands use container `/data`; `/exchange` remains a separate managed surface.
""",
        encoding="utf-8",
    )
    if status != "pass":
        raise RuntimeError("immutable Stage65C-R1 preflight failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
