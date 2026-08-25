#!/usr/bin/env python3
"""Fail-closed immutable preflight for Stage65D-R1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

STAGE_ID = (
    "BANANA-YOLO26-XSLIM-STAGE65D-R1-C2-FROZEN-FULL-VAL-PROVIDER-"
    "INTERACTION-CONDITIONAL-PERFORMANCE-STABILITY-AND-CROSS-SURFACE-"
    "PASSPORT-001"
)
STAGE65D_ID = (
    "BANANA-YOLO26-XSLIM-STAGE65D-C2-FROZEN-K1X-SPACEMIT-EP-BOARD-"
    "PASSPORT-FP32-FP16-AND-CUSTOM-ENGINE-APPLICATION-COMPARISON-001"
)
STAGE65C_R1_ID = (
    "BANANA-YOLO26-XSLIM-STAGE65C-R1-A1-CPU-EP-LARGE-RECALL-DIVERGENCE-"
    "AND-TERMINAL-BOUNDARY-CAUSAL-DIAGNOSTIC-001"
)
R1_ID = (
    "BANANA-YOLO26-XSLIM-STAGE65B-R1-COCO-TRAIN2017-EVALUATION-"
    "DISJOINT-CORPUS-PTQ-GRAPHWISE-AND-PYRAMID-CAUSAL-LOCALIZATION-001"
)
DEV001B_ID = (
    "BANANA-YOLO26-XSLIM-DEV-001B-ALL-S8-GENERIC-HARDENING-ADAPTIVE-"
    "ROUNDING-BLOCK-RECONSTRUCTION-AND-DETECTOR-PARETO-HOST-GATE-001"
)
DEV001C_ID = (
    "BANANA-YOLO26-XSLIM-DEV-001C-C2-FROZEN-INDEPENDENT-HOLDOUT-"
    "ADJUDICATION-AND-VENDOR-PTQ-LANE-CLOSURE-001"
)
S64_ID = (
    "BANANA-YOLO26-VENDOR-ORT-STAGE64-XSLIM211-AND-VENDOR-COMMIT-S8-QDQ-"
    "YOLO26-FULLMODEL-COCO-AND-RT206-GATE-001"
)

DATA = Path("/data")
BANANA = DATA / "worktrees/banana-yolo26-xslim211-s8-qdq-validation"
PROTECTED = DATA / "banana-yolo26-spacemit-demo"
XSLIM = DATA / "worktrees/riscy-xslim-k1x-yolo26"
NCNN = DATA / "ncnn"
R1_RAW = DATA / "k1x-stage-runs" / R1_ID
DEV001B_RAW = DATA / "k1x-stage-runs" / DEV001B_ID
DEV001C_RAW = DATA / "k1x-stage-runs" / DEV001C_ID
S64_RAW = DATA / "k1x-stage-runs" / S64_ID
DATASET = DATA / "datasets/coco2017-independent-stage65b-r1"
STAGE65D_PACKET = Path("/exchange/results/outbox") / STAGE65D_ID
STAGE65C_R1_PACKET = Path("/exchange/results/outbox") / STAGE65C_R1_ID
OLD_BOARD_ROOT = f"/data/k1x-stage-runs/{STAGE65D_ID}"
BOARD_VAL = (
    "/data/k1x-stage-runs/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE46-"
    "RT205-SPACEMIT-EP-INT8-FULL-REVALIDATION-PLUGIN-COCO-GATE-001/"
    "datasets/coco-val2017/val2017"
)

EXPECTED = {
    "banana_head": "43a523a4df7a6168fba9dcda4d57198998ee8b89",
    "banana_tree": "d7e29dd88968173d0a5669fc75a932239855cc48",
    "xslim_head": "46d5d36bcb6979bab6567fb4fe62839689f1881c",
    "xslim_tree": "1788779cd0887a1c8e6924cd63ad7d16d42f41ca",
    "xslim_version": "2.1.2+riscy.2.dev2",
    "upstream_main": "9a33f2f770d00fd02ff8bc0f1907135e9bf47f8c",
    "protected_main": "1fd2e71bb1d5a924e7c0444cada94f681b73aa91",
    "custom_executor_tree": "c2e400de14fb1c88d4aed70a249d9eff19a05d0f",
    "ncnn_head": "a245a70c641a1f20f357c65d103e5f9e50fe84a1",
    "ncnn_tree": "20b96dadbd1fc0a53159cb35749719e967b55906",
    "ncnn_diff": "2bf1cc38885018a02478aa7542581639786c79bca5ce11a6e827d24bcc5f4eca",
    "stage65d_packet": ("adcc6e12b0760f4d65725e34de2306bd4c42b64f7f202bcf21e2828a4a7face5", 108, 319268),
    "stage65c_r1_packet": ("8398831b147cc890436e968d830b14c0d5347ee5a24946b03156c66aa08b22e6", 63, 1983169),
    "board_val_files": 5000,
    "board_val_tree": "2e210112a7ea2a0a7687ad7f3ef25ea4488522f47f985680cd792e6c43f95e8f",
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
    relatives = [
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    ]
    environment = os.environ.copy()
    environment["LC_ALL"] = "en_US.UTF-8"
    relatives = subprocess.check_output(
        ["sort"], input="\n".join(relatives) + "\n", text=True, env=environment
    ).splitlines()
    digest = hashlib.sha256()
    total = 0
    for relative in relatives:
        path = root / relative
        digest.update(f"{sha256(path)}\t{relative}\n".encode())
        total += path.stat().st_size
    return digest.hexdigest(), len(relatives), total


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


def ssh_script(script: str) -> str:
    process = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "svt@banana", "bash", "-s"],
        input=script,
        text=True,
        capture_output=True,
        check=True,
    )
    return process.stdout.strip()


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
printf 'commands\t'; for tool in /usr/bin/time taskset sha256sum python3; do command -v "$tool"; done | paste -sd, -
printf 'active_processes\t'; pgrep -a -f '[o]nnxruntime|[c]oco.*eval|[b]ootstrap|[c]ustom.executor|[s]tage65d' || true
'''
    return dict(line.split("\t", 1) for line in ssh_script(script).splitlines() if "\t" in line)


def board_val_identity() -> tuple[int, str]:
    script = f'''set -eu
cd {BOARD_VAL}
count=$(find . -maxdepth 1 -type f -name '*.jpg' | wc -l)
tree=$(find . -maxdepth 1 -type f -name '*.jpg' -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{{print $1}}')
printf '%s\t%s\n' "$count" "$tree"
'''
    count, tree = ssh_script(script).split("\t")
    return int(count), tree


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracked-root", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--shared-log", required=True, type=Path)
    options = parser.parse_args()

    for root in (options.tracked_root, options.raw_root, options.result_root):
        if root.exists():
            raise RuntimeError(f"refusing existing accepted root: {root}")
    board_root = f"/data/k1x-stage-runs/{STAGE_ID}"
    if ssh_script(f"test ! -e {board_root} && echo absent") != "absent":
        raise RuntimeError(f"board accepted root already exists: {board_root}")

    actual = {
        "banana_head": git(BANANA, "rev-parse", "HEAD"),
        "banana_tree": git(BANANA, "rev-parse", "HEAD^{tree}"),
        "xslim_head": git(XSLIM, "rev-parse", "HEAD"),
        "xslim_tree": git(XSLIM, "rev-parse", "HEAD^{tree}"),
        "xslim_version": (XSLIM / "VERSION_NUMBER").read_text().strip(),
        "upstream_main": remote(XSLIM, "upstream", "refs/heads/main"),
        "protected_main": git(PROTECTED, "rev-parse", "yolo26-custom-int8-engine"),
        "custom_executor_tree": git(PROTECTED, "rev-parse", "yolo26-custom-int8-engine:custom_int8_engine"),
        "ncnn_head": git(NCNN, "rev-parse", "HEAD"),
        "ncnn_tree": git(NCNN, "rev-parse", "HEAD^{tree}"),
        "ncnn_diff": hashlib.sha256(run("git", "diff", "--binary", cwd=NCNN, binary=True)).hexdigest(),
        "ncnn_dirty_count": len(git(NCNN, "status", "--porcelain").splitlines()),
    }
    protected_rows = [checked(key, actual[key], EXPECTED[key]) for key in (
        "banana_head", "banana_tree", "xslim_head", "xslim_tree", "xslim_version",
        "upstream_main", "protected_main", "custom_executor_tree", "ncnn_head",
        "ncnn_tree", "ncnn_diff",
    )]
    protected_rows.append(checked("ncnn_dirty_count", actual["ncnn_dirty_count"], 3))

    remote_rows = []
    for repo, name, ref, expected in (
        (BANANA, "github", "refs/heads/yolo26-vendor-ort-xslim211-s8-qdq-validation", EXPECTED["banana_head"]),
        (BANANA, "gitlab-rd", "refs/heads/yolo26-vendor-ort-xslim211-s8-qdq-validation", EXPECTED["banana_head"]),
        (XSLIM, "github", "refs/heads/riscy/k1x-yolo26", EXPECTED["xslim_head"]),
        (XSLIM, "gitlab", "refs/heads/riscy/k1x-yolo26", EXPECTED["xslim_head"]),
    ):
        observed = remote(repo, name, ref)
        remote_rows.append((repo.name, name, ref, observed, expected, "pass" if observed == expected else "fail"))

    packets = {
        "stage65d": packet_identity(STAGE65D_PACKET),
        "stage65c_r1": packet_identity(STAGE65C_R1_PACKET),
    }
    packet_rows = []
    for name, expected in (
        ("stage65d", EXPECTED["stage65d_packet"]),
        ("stage65c_r1", EXPECTED["stage65c_r1_packet"]),
    ):
        actual_packet = packets[name]
        packet_rows.extend([
            (name, "tree_sha256", actual_packet[0], expected[0], "pass" if actual_packet[0] == expected[0] else "fail"),
            (name, "file_count", actual_packet[1], expected[1], "pass" if actual_packet[1] == expected[1] else "fail"),
            (name, "byte_count", actual_packet[2], expected[2], "pass" if actual_packet[2] == expected[2] else "fail"),
        ])
    stage65d_artifacts = len([p for p in (STAGE65D_PACKET / "artifacts").rglob("*") if p.is_file()])
    packet_rows.append(("stage65d", "artifact_file_count", stage65d_artifacts, 103, "pass" if stage65d_artifacts == 103 else "fail"))

    artifacts = [
        ("B2", "deployable", R1_RAW / "quantization/B2/run1/output/stage65b_r1_b2_split_s8_qdq.onnx", "0e7040d4e8b1b2d08a4e36cec4c99dcea6d52294e04901d17dfce10725c6d617"),
        ("B2", "inference", R1_RAW / "postprocess/B2/candidate-gate/B2/models/stage65b_r1_b2.inference.onnx", "40ba6a7f9aebaa98a1c3abe5fce1f66f1bebcd0b10b7af3d26d30414a331d853"),
        ("C2", "deployable", DEV001B_RAW / "candidates/run1/C2_T6_RANK_QP/c2_t6_rank_qp.deployable.onnx", "e963be11c57c048f23caa34df1e2d140211632cc4dfd6b734b14909a30ea4b55"),
        ("C2", "inference", DEV001B_RAW / "candidates/run1/C2_T6_RANK_QP/c2_t6_rank_qp.inference.onnx", "281f4acd1261e7ee2c38b6e3bdecbf61c3d91cf710c63e6bc6cdaf257a52669b"),
        ("common", "tail", R1_RAW / "postprocess/B2/candidate-gate/B2/models/stage65b_r1_b2.postprocess.onnx", "18ffff41e6812fa781baf7b9c1fcd41b41d6118145d785c3e550499070a512a3"),
        ("FP32", "inference", S64_RAW / "models/fp32-split/yolo26n_640_e2e_fp32.inference.onnx", "72eb6136b41104753c53b8e13aeff50e7961c4cefba79e50b70894cbd169f8d8"),
        ("FP32", "host_prediction", R1_RAW / "full-matrix/hybrid-full-coco/H8/predictions.json", "b9ff8fa19cba9682970d8e932f3318cdf5833094ab22256a24062019309b5b2a"),
        ("B2", "host_prediction", R1_RAW / "full-matrix/full-coco/B2/predictions.json", "51f8d4b25245a5f3e24feafea8aa49547c0f530f59cabcd18e61a744b4740add"),
        ("C2", "host_prediction", DEV001C_RAW / "full-val/C2/predictions.json", "0f040fe848e9fe5306f0d8974e410a3471d2bff9cbf1f3c53dc5feb1c47fa345"),
    ]
    artifact_rows = []
    for surface, kind, path, expected in artifacts:
        observed = sha256(path)
        artifact_rows.append((surface, kind, path, path.stat().st_size, observed, expected, "pass" if observed == expected else "fail"))

    runtime_files = [
        ("archive", DATA / "vendor-runtimes/downloads/spacemit-ort.riscv64.2.0.6.tar.gz", "bebcdfb7df6b49eefa3863afcd85a3da2aa83c3ae9252d7d856188c38a70b0e6"),
        ("core", DATA / "vendor-runtimes/spacemit-ort/2.0.6/spacemit-ort.riscv64.2.0.6/lib/libonnxruntime.so.1.24.2+spacemit.a1", "93bb75601d9eceb5aca192fa70c0c3e18b94a70b9f57acdc9b34c2ff426e09e3"),
        ("ep", DATA / "vendor-runtimes/spacemit-ort/2.0.6/spacemit-ort.riscv64.2.0.6/lib/libspacemit_ep.so.2.0.6", "dcc9503031bca22cf2b33a692f7b4c01d0fbb4a24c34f6e60c7faaddb78274ae"),
    ]
    runtime_rows = []
    for role, path, expected in runtime_files:
        observed = sha256(path)
        runtime_rows.append((role, path, path.stat().st_size, observed, expected, "pass" if observed == expected else "fail"))

    dataset_files = [
        ("val2017_list", DATASET / "lists/val2017_all.txt", "d4b401d6be0446f1cea0aa2ea99fc4d367c498c02b18ebac75b77c2e2fe21bae"),
        ("h500_list", DATASET / "lists/selection_H500_holdout.txt", "35b0afc2c4d9dfd20634fd7cb5b883e6eef8d67a34d5a5c832259b36ea8ce097"),
        ("val_annotations", DATASET / "annotations/instances_val2017.json", "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"),
        ("train_annotations", DATASET / "annotations/instances_train2017.json", "610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d"),
        ("host_runner", BANANA / "vendor_ort_validation/stage65b_r1_evaluate.py", "79ad059411bb153f3abcb8d4abd0f1e79e5e04b12863fc121dc227d2fe89bd65"),
        ("host_metrics", BANANA / "vendor_ort_validation/stage65b_r1_coco_metrics.py", "ebd252d34473b7645d679fcdb209d32a86ab45dc5d0e75279ac1a0533c68eec9"),
        ("preprocess", BANANA / "vendor_ort_validation/stage64_preprocess.py", "d9d149adf9e1c5242f63722d0cd700a2f204c74edbafd045a5ad6bdf04d1d0b8"),
        ("board_coco_runner", S64_RAW / "bin/stage64_two_stage_coco", "d39f5280a46b0bd342de933d6b25005a5858390430e8ce84377374f4a8225ad0"),
    ]
    dataset_rows = []
    for role, path, expected in dataset_files:
        observed = sha256(path)
        dataset_rows.append((role, path, path.stat().st_size, observed, expected, "pass" if observed == expected else "fail"))
    board_val_count, board_val_tree = board_val_identity()
    dataset_rows.extend([
        ("board_val_image_count", BOARD_VAL, board_val_count, str(board_val_count), str(EXPECTED["board_val_files"]), "pass" if board_val_count == EXPECTED["board_val_files"] else "fail"),
        ("board_val_file_tree", BOARD_VAL, "n/a", board_val_tree, EXPECTED["board_val_tree"], "pass" if board_val_tree == EXPECTED["board_val_tree"] else "fail"),
    ])

    board = board_snapshot()
    board_utc = datetime.fromisoformat(board["utc"].replace("Z", "+00:00"))
    clock_delta = abs((datetime.now(UTC) - board_utc).total_seconds())
    board_rows = [
        checked("hostname", board["hostname"], "bf3"),
        checked("serial", board["serial"], "92262f3b0dc4"),
        ("boot_id", board["boot_id"], "live-captured", "pass"),
        checked("os", board["os"], "Bianbu 2.2.1"),
        checked("kernel", board["kernel"], "6.6.63"),
        ("clock_delta_seconds", f"{clock_delta:.3f}", "<=10", "pass" if clock_delta <= 10 else "fail"),
        ("data_mount", board["data_mount"], "non-mmc /data", "pass" if "nvme" in board["data_mount"] else "fail"),
        ("root_mount", board["root_mount"], "mmc root accepted; project writes forbidden", "pass"),
        checked("data_writable", board["data_writable"], "yes"),
        ("data_free_bytes", board["data_free_bytes"], ">=20GiB", "pass" if int(board["data_free_bytes"]) >= 20 * 1024**3 else "fail"),
        ("required_commands", board["commands"], "four commands present", "pass" if len(board["commands"].split(",")) == 4 else "fail"),
    ]

    host_processes = subprocess.run(
        ["pgrep", "-a", "-f", "[o]nnxruntime|[c]oco.*eval|[b]ootstrap|[c]ustom.executor"],
        text=True, capture_output=True, check=False,
    ).stdout.strip()
    process_rows = [
        ("host", host_processes or "none", "pass" if not host_processes else "fail"),
        ("board", board.get("active_processes", "") or "none", "pass" if not board.get("active_processes", "") else "fail"),
    ]

    all_statuses = [row[-1] for row in protected_rows + remote_rows + packet_rows + artifact_rows + runtime_rows + dataset_rows + board_rows + process_rows]
    if any(status != "pass" for status in all_statuses):
        failed_rows = [
            row
            for row in protected_rows + remote_rows + packet_rows + artifact_rows
            + runtime_rows + dataset_rows + board_rows + process_rows
            if row[-1] != "pass"
        ]
        raise RuntimeError(f"immutable preflight mismatches: {failed_rows}")

    options.tracked_root.mkdir(parents=True)
    options.raw_root.mkdir(parents=True)
    write_tsv(options.tracked_root / "protected_state_before.tsv", ("field", "actual", "expected", "status"), protected_rows)
    write_tsv(options.tracked_root / "remote_parity_before.tsv", ("repository", "remote", "ref", "actual", "expected", "status"), remote_rows)
    write_tsv(options.tracked_root / "stage65d_packet_verification.tsv", ("packet", "field", "actual", "expected", "status"), [row for row in packet_rows if row[0] == "stage65d"])
    write_tsv(options.tracked_root / "stage65c_r1_packet_verification.tsv", ("packet", "field", "actual", "expected", "status"), [row for row in packet_rows if row[0] == "stage65c_r1"])
    write_tsv(options.tracked_root / "frozen_model_identity.tsv", ("surface", "kind", "path", "bytes", "sha256", "expected", "status"), artifact_rows)
    write_tsv(options.tracked_root / "runtime_asset_identity.tsv", ("role", "path", "bytes", "sha256", "expected", "status"), runtime_rows)
    write_tsv(options.tracked_root / "dataset_evaluator_contract.tsv", ("role", "path", "bytes", "sha256", "expected", "status"), dataset_rows)
    write_tsv(options.tracked_root / "board_identity_before.tsv", ("field", "actual", "expected", "status"), board_rows)
    write_tsv(options.tracked_root / "process_audit_before.tsv", ("surface", "matching_processes", "status"), process_rows)

    launch = {
        "stage_id": STAGE_ID,
        "authorization": "direct-user",
        "banana_start": actual["banana_head"],
        "xslim_immutable": actual["xslim_head"],
        "board_target": "svt@banana",
        "board_boot_id_before": board["boot_id"],
        "board_workspace": board_root,
        "host_raw_root": str(options.raw_root),
        "tracked_root": str(options.tracked_root),
        "result_root": str(options.result_root),
        "shared_log": str(options.shared_log),
        "runtime": "SpacemiT ORT 2.0.6",
        "models": "frozen; regeneration forbidden",
        "full_val_bootstrap_seed": 65012,
        "performance_bootstrap_seed": 65013,
        "camera": "forbidden-not-run",
    }
    (options.tracked_root / "effective_launch_manifest.yaml").write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")
    index = {
        "tracked_stage": str(options.tracked_root),
        "raw_stage": str(options.raw_root),
        "result_packet": str(options.result_root),
        "shared_log": str(options.shared_log),
        "input_stage65d_packet": str(STAGE65D_PACKET),
        "input_stage65c_r1_packet": str(STAGE65C_R1_PACKET),
        "frozen_models": {f"{row[0]}_{row[1]}": str(row[2]) for row in artifacts},
        "accepted_b2_board_predictions": {
            "cpu": str(DATA / "k1x-stage-runs" / STAGE65C_R1_ID / "board/coco/val/B2-cpu/predictions.json"),
            "ep": str(DATA / "k1x-stage-runs" / STAGE65C_R1_ID / "board/coco/val/B2-spacemit/predictions.json"),
        },
        "excluded_or_intermediate_roots": {},
    }
    (options.tracked_root / "input_evidence_index.yaml").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    (options.tracked_root / "workspace_preflight.md").write_text(
        "# Stage65D-R1 workspace preflight\n\n"
        "Status: `pass`. Both input packets, repositories/remotes, protected projects, frozen models, FP32 reference, runtime bytes, dataset/evaluator contract, board identity, board NVMe and process audit passed. The accepted Stage roots were absent before this report was created. No current-stage reboot is claimed.\n",
        encoding="utf-8",
    )
    print(f"preflight status=pass board_boot_id={board['boot_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
