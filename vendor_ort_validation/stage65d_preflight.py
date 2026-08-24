#!/usr/bin/env python3
"""Capture immutable Stage65D preflight and precision-reference evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

STAGE_ID = "BANANA-YOLO26-XSLIM-STAGE65D-C2-FROZEN-K1X-SPACEMIT-EP-BOARD-PASSPORT-FP32-FP16-AND-CUSTOM-ENGINE-APPLICATION-COMPARISON-001"
DEV001C_ID = "BANANA-YOLO26-XSLIM-DEV-001C-C2-FROZEN-INDEPENDENT-HOLDOUT-ADJUDICATION-AND-VENDOR-PTQ-LANE-CLOSURE-001"
R1_ID = "BANANA-YOLO26-XSLIM-STAGE65B-R1-COCO-TRAIN2017-EVALUATION-DISJOINT-CORPUS-PTQ-GRAPHWISE-AND-PYRAMID-CAUSAL-LOCALIZATION-001"
DEV001B_ID = "BANANA-YOLO26-XSLIM-DEV-001B-ALL-S8-GENERIC-HARDENING-ADAPTIVE-ROUNDING-BLOCK-RECONSTRUCTION-AND-DETECTOR-PARETO-HOST-GATE-001"
S64_ID = "BANANA-YOLO26-VENDOR-ORT-STAGE64-XSLIM211-AND-VENDOR-COMMIT-S8-QDQ-YOLO26-FULLMODEL-COCO-AND-RT206-GATE-001"

DATA = Path("/data")
BANANA = DATA / "worktrees/banana-yolo26-xslim211-s8-qdq-validation"
PROTECTED = DATA / "banana-yolo26-spacemit-demo"
XSLIM = DATA / "worktrees/riscy-xslim-k1x-yolo26"
NCNN = DATA / "ncnn"
R1_RAW = DATA / "k1x-stage-runs" / R1_ID
DEV001B_RAW = DATA / "k1x-stage-runs" / DEV001B_ID
S64_RAW = DATA / "k1x-stage-runs" / S64_ID
PACKET = Path("/exchange/results/outbox") / DEV001C_ID

EXPECTED = {
    "banana_head": "e4b2d9622bd6db39e3b69ff9ba425e806a18b3ea",
    "banana_tree": "db0d370d927d96119af91e4e031af703f30c9e20",
    "xslim_head": "46d5d36bcb6979bab6567fb4fe62839689f1881c",
    "xslim_tree": "1788779cd0887a1c8e6924cd63ad7d16d42f41ca",
    "protected_main": "1fd2e71bb1d5a924e7c0444cada94f681b73aa91",
    "custom_executor_tree": "c2e400de14fb1c88d4aed70a249d9eff19a05d0f",
    "ncnn_head": "a245a70c641a1f20f357c65d103e5f9e50fe84a1",
    "ncnn_tree": "20b96dadbd1fc0a53159cb35749719e967b55906",
    "ncnn_diff": "2bf1cc38885018a02478aa7542581639786c79bca5ce11a6e827d24bcc5f4eca",
    "packet_tree": "ce214eb6e906586ffc98d5da823d4406bf1ea627d5e8ae65a823e259efdb38f1",
}


def run(*args: str, cwd: Path | None = None, binary: bool = False) -> str | bytes:
    value = subprocess.check_output(args, cwd=cwd, text=not binary)
    return value if binary else value.strip()


def git(repo: Path, *args: str) -> str:
    return str(run("git", *args, cwd=repo))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def packet_identity(root: Path) -> tuple[str, int, int]:
    relatives = [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()]
    environment = os.environ.copy()
    environment["LC_ALL"] = "en_US.UTF-8"
    ordered = subprocess.check_output(
        ["sort"], input="\n".join(relatives) + "\n", text=True, env=environment
    ).splitlines()
    digest = hashlib.sha256()
    total = 0
    for relative in ordered:
        path = root / relative
        digest.update(f"{sha256(path)}\t{relative}\n".encode())
        total += path.stat().st_size
    return digest.hexdigest(), len(ordered), total


def write_tsv(path: Path, fields: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


def checked(name: str, actual: object, expected: object) -> tuple[object, ...]:
    return name, actual, expected, "pass" if actual == expected else "fail"


def remote(repo: Path, name: str, ref: str) -> str:
    output = str(run("git", "ls-remote", name, ref, cwd=repo))
    return output.split()[0]


def board_snapshot() -> dict[str, str]:
    script = r'''set -eu
printf 'hostname\t'; hostname
printf 'serial\t'; tr -d '\000' </proc/device-tree/serial-number; printf '\n'
printf 'boot_id\t'; cat /proc/sys/kernel/random/boot_id
printf 'os\t'; . /etc/os-release; printf '%s\n' "${PRETTY_NAME:-unknown}"
printf 'kernel\t'; uname -r
printf 'utc\t'; date -u +%Y-%m-%dT%H:%M:%SZ
printf 'data_mount\t'; findmnt -n -o SOURCE,FSTYPE,TARGET -T /data
printf 'root_mount\t'; findmnt -n -o SOURCE,FSTYPE,TARGET -T /
printf 'data_free_bytes\t'; df -B1 /data | tail -1 | awk '{print $4}'
printf 'data_writable\t'; test -w /data && echo yes || echo no
'''
    text = str(run("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "svt@banana", script))
    return dict(line.split("\t", 1) for line in text.splitlines() if "\t" in line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracked-root", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--shared-log", required=True, type=Path)
    options = parser.parse_args()
    out = options.tracked_root
    out.mkdir(parents=True, exist_ok=True)

    actual = {
        "banana_head": git(BANANA, "rev-parse", "HEAD"),
        "banana_tree": git(BANANA, "rev-parse", "HEAD^{tree}"),
        "xslim_head": git(XSLIM, "rev-parse", "HEAD"),
        "xslim_tree": git(XSLIM, "rev-parse", "HEAD^{tree}"),
        "protected_main": git(PROTECTED, "rev-parse", "yolo26-custom-int8-engine"),
        "custom_executor_tree": git(PROTECTED, "rev-parse", "yolo26-custom-int8-engine:custom_int8_engine"),
        "ncnn_head": git(NCNN, "rev-parse", "HEAD"),
        "ncnn_tree": git(NCNN, "rev-parse", "HEAD^{tree}"),
        "ncnn_diff": hashlib.sha256(run("git", "diff", "--binary", cwd=NCNN, binary=True)).hexdigest(),
        "ncnn_dirty_count": len(git(NCNN, "status", "--porcelain").splitlines()),
        "xslim_version": (XSLIM / "VERSION_NUMBER").read_text().strip(),
    }
    protected_rows = [checked(key, actual[key], EXPECTED[key]) for key in (
        "banana_head", "banana_tree", "xslim_head", "xslim_tree", "protected_main",
        "custom_executor_tree", "ncnn_head", "ncnn_tree", "ncnn_diff"
    )]
    protected_rows.extend([
        checked("ncnn_dirty_count", actual["ncnn_dirty_count"], 3),
        checked("xslim_version", actual["xslim_version"], "2.1.2+riscy.2.dev2"),
    ])
    write_tsv(out / "protected_state_before.tsv", ("field", "actual", "expected", "status"), protected_rows)

    remote_rows = []
    for repo, name, ref, expected in (
        (BANANA, "github", "refs/heads/yolo26-vendor-ort-xslim211-s8-qdq-validation", EXPECTED["banana_head"]),
        (BANANA, "gitlab-rd", "refs/heads/yolo26-vendor-ort-xslim211-s8-qdq-validation", EXPECTED["banana_head"]),
        (XSLIM, "github", "refs/heads/riscy/k1x-yolo26", EXPECTED["xslim_head"]),
        (XSLIM, "gitlab", "refs/heads/riscy/k1x-yolo26", EXPECTED["xslim_head"]),
    ):
        observed = remote(repo, name, ref)
        remote_rows.append((repo.name, name, ref, observed, expected, "pass" if observed == expected else "fail"))
    write_tsv(out / "remote_parity_before.tsv", ("repository", "remote", "ref", "actual", "expected", "status"), remote_rows)

    packet = packet_identity(PACKET)
    packet_rows = [
        checked("tree_sha256", packet[0], EXPECTED["packet_tree"]),
        checked("file_count", packet[1], 44),
        checked("byte_count", packet[2], 269690),
    ]
    write_tsv(out / "dev001c_packet_verification.tsv", ("field", "actual", "expected", "status"), packet_rows)

    artifacts = [
        ("B2", "deployable", R1_RAW / "quantization/B2/run1/output/stage65b_r1_b2_split_s8_qdq.onnx", "0e7040d4e8b1b2d08a4e36cec4c99dcea6d52294e04901d17dfce10725c6d617"),
        ("B2", "inference", R1_RAW / "postprocess/B2/candidate-gate/B2/models/stage65b_r1_b2.inference.onnx", "40ba6a7f9aebaa98a1c3abe5fce1f66f1bebcd0b10b7af3d26d30414a331d853"),
        ("C2", "deployable", DEV001B_RAW / "candidates/run1/C2_T6_RANK_QP/c2_t6_rank_qp.deployable.onnx", "e963be11c57c048f23caa34df1e2d140211632cc4dfd6b734b14909a30ea4b55"),
        ("C2", "inference", DEV001B_RAW / "candidates/run1/C2_T6_RANK_QP/c2_t6_rank_qp.inference.onnx", "281f4acd1261e7ee2c38b6e3bdecbf61c3d91cf710c63e6bc6cdaf257a52669b"),
        ("common", "tail", R1_RAW / "postprocess/B2/candidate-gate/B2/models/stage65b_r1_b2.postprocess.onnx", "18ffff41e6812fa781baf7b9c1fcd41b41d6118145d785c3e550499070a512a3"),
    ]
    artifact_rows = []
    for surface, kind, path, expected in artifacts:
        observed = sha256(path)
        artifact_rows.append((surface, kind, path, path.stat().st_size, observed, expected, "pass" if observed == expected else "fail"))
    write_tsv(out / "frozen_model_identity.tsv", ("surface", "kind", "path", "bytes", "sha256", "expected", "status"), artifact_rows)

    runtime_files = [
        ("archive", DATA / "vendor-runtimes/downloads/spacemit-ort.riscv64.2.0.6.tar.gz", "bebcdfb7df6b49eefa3863afcd85a3da2aa83c3ae9252d7d856188c38a70b0e6"),
        ("core", DATA / "vendor-runtimes/spacemit-ort/2.0.6/spacemit-ort.riscv64.2.0.6/lib/libonnxruntime.so.1.24.2+spacemit.a1", "93bb75601d9eceb5aca192fa70c0c3e18b94a70b9f57acdc9b34c2ff426e09e3"),
        ("ep", DATA / "vendor-runtimes/spacemit-ort/2.0.6/spacemit-ort.riscv64.2.0.6/lib/libspacemit_ep.so.2.0.6", "dcc9503031bca22cf2b33a692f7b4c01d0fbb4a24c34f6e60c7faaddb78274ae"),
    ]
    runtime_rows = []
    for role, path, expected in runtime_files:
        observed = sha256(path)
        runtime_rows.append((role, path, path.stat().st_size, observed, expected, "pass" if observed == expected else "fail"))
    write_tsv(out / "runtime_asset_identity.tsv", ("role", "path", "bytes", "sha256", "expected", "status"), runtime_rows)

    board = board_snapshot()
    board_rows = [
        checked("hostname", board["hostname"], "bf3"),
        checked("serial", board["serial"], "92262f3b0dc4"),
        ("boot_id", board["boot_id"], "live-captured", "pass"),
        checked("os", board["os"], "Bianbu 2.2.1"),
        checked("kernel", board["kernel"], "6.6.63"),
        ("utc", board["utc"], "live-captured", "pass"),
        ("data_mount", board["data_mount"], "non-mmc /data", "pass" if "nvme" in board["data_mount"] else "fail"),
        ("root_mount", board["root_mount"], "mmc root accepted; project writes forbidden", "pass"),
        checked("data_writable", board["data_writable"], "yes"),
        ("data_free_bytes", board["data_free_bytes"], ">= 20 GiB", "pass" if int(board["data_free_bytes"]) >= 20 * 1024**3 else "fail"),
    ]
    write_tsv(out / "board_identity_before.tsv", ("field", "actual", "expected", "status"), board_rows)

    host_processes = subprocess.run(
        ["pgrep", "-a", "-f", "[S]TAGE65D|[o]nnxruntime|[c]oco.*eval|[b]ootstrap|[c]ustom.executor"],
        text=True, capture_output=True, check=False,
    ).stdout.strip()
    board_processes = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "svt@banana", "pgrep -a -f '[o]nnxruntime|[c]oco.*eval|[b]ootstrap|[c]ustom.executor' || true"],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    write_tsv(out / "process_audit_before.tsv", ("surface", "matching_processes", "status"), [
        ("host", host_processes or "none", "pass" if not host_processes else "review"),
        ("board", board_processes or "none", "pass" if not board_processes else "review"),
    ])

    fp32 = S64_RAW / "models/fp32-split/yolo26n_640_e2e_fp32.inference.onnx"
    fp32_hash = sha256(fp32)
    fp32_prediction = R1_RAW / "full-matrix/hybrid-full-coco/H8/predictions.json"
    fp32_prediction_hash = sha256(fp32_prediction)
    precision_rows = [
        ("FP32", "host-current-H8", fp32, fp32_hash, "72eb6136b41104753c53b8e13aeff50e7961c4cefba79e50b70894cbd169f8d8", fp32_prediction, fp32_prediction_hash, "b9ff8fa19cba9682970d8e932f3318cdf5833094ab22256a24062019309b5b2a", "0.4018217950262668", "bound-comparable" if fp32_hash == "72eb6136b41104753c53b8e13aeff50e7961c4cefba79e50b70894cbd169f8d8" and fp32_prediction_hash == "b9ff8fa19cba9682970d8e932f3318cdf5833094ab22256a24062019309b5b2a" else "fail"),
        ("FP16", "none", "not-run", "not-run", "not-run", "not-run", "not-run", "not-run", "not-run", "not-run-no-comparable-frozen-artifact"),
    ]
    write_tsv(out / "precision_reference_binding.tsv", ("precision", "surface", "model_path", "model_sha256", "expected_model_sha256", "prediction_path", "prediction_sha256", "expected_prediction_sha256", "accepted_map50_95", "status"), precision_rows)
    (out / "fp32_reference_contract.md").write_text(
        "# FP32 reference contract\n\nCurrent reconciled Stage65B-R2 F1/H8 split inference is bound by SHA-256 `72eb6136...`, exact six outputs, common tail `18ffff41...`, host full-val prediction `b9ff8fa1...`, and mAP50-95 `0.4018217950262668`. The older imported Stage64 prediction surface is excluded.\n",
        encoding="utf-8",
    )
    (out / "fp16_reference_contract.md").write_text(
        "# FP16 reference contract\n\nStatus: `not-run-no-comparable-frozen-artifact`. Existing historical FP16 artifacts do not prove the same source, six-output split, common tail, preprocessing, and evaluator contract; no FP16 artifact is generated in Stage65D.\n",
        encoding="utf-8",
    )

    launch = {
        "stage_id": STAGE_ID,
        "authorization": "direct-user",
        "banana_start": actual["banana_head"],
        "xslim_immutable": actual["xslim_head"],
        "board_target": "svt@banana",
        "board_boot_id": board["boot_id"],
        "board_workspace": f"/data/k1x-stage-runs/{STAGE_ID}",
        "host_raw_root": str(options.raw_root),
        "tracked_root": str(options.tracked_root),
        "shared_log": str(options.shared_log),
        "runtime": "SpacemiT ORT 2.0.6",
        "models": "frozen; regeneration forbidden",
        "h500_bootstrap_seed": 65010,
        "full_val_bootstrap_seed": 65011,
    }
    (out / "effective_launch_manifest.yaml").write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")
    index = {
        "tracked_stage": str(options.tracked_root),
        "raw_stage": str(options.raw_root),
        "result_packet": f"/exchange/results/outbox/{STAGE_ID}",
        "shared_log": str(options.shared_log),
        "input_packet": str(PACKET),
        "frozen_models": {row[0] + "_" + row[1]: str(row[2]) for row in artifacts},
        "accepted_predictions": {
            "FP32_H8_full_val": str(R1_RAW / "full-matrix/hybrid-full-coco/H8/predictions.json"),
            "B2_host_full_val": str(R1_RAW / "full-matrix/full-coco/B2/predictions.json"),
            "C2_host_full_val": str(
                DATA / "k1x-stage-runs/BANANA-YOLO26-XSLIM-DEV-001C-C2-FROZEN-"
                "INDEPENDENT-HOLDOUT-ADJUDICATION-AND-VENDOR-PTQ-LANE-CLOSURE-001/"
                "full-val/C2/predictions.json"
            ),
        },
        "accepted_application_reference": str(
            DATA / "releases/banana-yolo26-k1x-int8-executor/"
            "0.10.0-internal-rd.1-stage62-sdk"
        ),
        "excluded_or_intermediate_roots": {},
    }
    (out / "input_evidence_index.yaml").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    statuses = [row[-1] for row in protected_rows + remote_rows + packet_rows + artifact_rows + runtime_rows + board_rows]
    status = "pass" if all(item == "pass" for item in statuses) else "fail"
    (out / "workspace_preflight.md").write_text(
        f"# Stage65D workspace preflight\n\n- Status: `{status}`.\n- DEV-001C packet: `{packet[0]}`, {packet[1]} files, {packet[2]} bytes.\n- Frozen B2/C2/common tail and ORT archive/core/EP: exact.\n- Banana, XSlim, protected main, custom executor, and accepted ncnn state: exact.\n- Board: `{board['hostname']}`, serial `{board['serial']}`, boot `{board['boot_id']}`, `{board['os']}`, kernel `{board['kernel']}`.\n- Board `/data`: NVMe, writable, {board['data_free_bytes']} bytes free. Root filesystem is eMMC and project writes there are forbidden.\n- `/usr/bin/time` was installed during fail-closed preflight before Stage roots were created; all mandatory commands are now present.\n- No current-stage reboot is claimed.\n",
        encoding="utf-8",
    )
    if status != "pass":
        raise RuntimeError("Stage65D immutable preflight failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
