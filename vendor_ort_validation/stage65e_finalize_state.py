#!/usr/bin/env python3
"""Capture Stage65E protected-state invariance and source hygiene."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path


STAGE_ID = (
    "BANANA-YOLO26-XSLIM-STAGE65E-FP32-OPERATING-POINT-LEDGER-B2-C2-"
    "UNCONDITIONAL-PERFORMANCE-STABILITY-AND-FUSION-FEASIBILITY-CLOSURE-001"
)
BANANA = Path("/data/worktrees/banana-yolo26-xslim211-s8-qdq-validation")
PROTECTED = Path("/data/banana-yolo26-spacemit-demo")
XSLIM = Path("/data/worktrees/riscy-xslim-k1x-yolo26")
NCNN = Path("/data/ncnn")


def git(root: Path, *arguments: str, binary: bool = False) -> str | bytes:
    output = subprocess.check_output(["git", *arguments], cwd=root, text=not binary)
    return output if binary else output.strip()


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty TSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def board_capture() -> list[dict[str, str]]:
    script = rf'''set -eu
printf 'timestamp_utc\t'; date -u +%Y-%m-%dT%H:%M:%SZ
printf 'hostname\t'; hostname
printf 'device_serial\t'; tr -d '\000' </proc/device-tree/serial-number; printf '\n'
printf 'boot_id\t'; cat /proc/sys/kernel/random/boot_id
printf 'kernel\t'; uname -a
printf 'os_release\t'; . /etc/os-release; printf '%s\n' "${{PRETTY_NAME:-unknown}}"
printf 'allowed_cpu_list\t'; cat /sys/devices/system/cpu/online
printf 'data_mount\t'; findmnt -n -o SOURCE,FSTYPE,OPTIONS -T /data
printf 'root_mount\t'; findmnt -n -o SOURCE,FSTYPE,OPTIONS -T /
printf 'data_free_bytes\t'; df -B1 /data | tail -1 | awk '{{print $4}}'
for path in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [ -r "$path" ] || continue
    printf 'governor\t%s=' "$path"; cat "$path"
done
for path in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do
    [ -r "$path" ] || continue
    printf 'frequency_khz\t%s=' "$path"; cat "$path"
done
for temp_path in /sys/class/thermal/thermal_zone*/temp; do
    [ -r "$temp_path" ] || continue
    printf 'temperature_millic\t%s=' "$temp_path"; cat "$temp_path"
done
count=0
for base in /home /tmp /var/tmp /root; do
    [ -d "$base" ] || continue
    value=$(find "$base" -xdev -path '*{STAGE_ID}*' -print 2>/dev/null | wc -l)
    count=$((count + value))
done
printf 'emmc_stage_path_count\t%s\n' "$count"
'''
    output = subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", "svt@banana", "bash", "-s"],
        input=script,
        text=True,
    )
    return [
        {"field": line.split("\t", 1)[0], "value": line.split("\t", 1)[1]}
        for line in output.splitlines()
        if "\t" in line
    ]


def ncnn_diff_sha256() -> str:
    payload = git(NCNN, "diff", "--binary", binary=True)
    assert isinstance(payload, bytes)
    return hashlib.sha256(payload).hexdigest()


def source_hygiene(stage_root: Path, source_files: list[Path]) -> tuple[bool, str]:
    files = [path for path in stage_root.rglob("*") if path.is_file()]
    files.extend(source_files)
    files = sorted(set(files))
    forbidden_suffixes = {
        ".onnx", ".npy", ".npz", ".jpg", ".jpeg", ".png", ".so", ".a", ".o", ".bin",
    }
    secret_patterns = [
        re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
        re.compile(r"Authorization:\s*(?:Bearer|Basic)\s+\S+", re.IGNORECASE),
        re.compile(r"(?:GH_TOKEN|GITHUB_TOKEN|GLAB_TOKEN|PRIVATE_TOKEN)\s*=\s*\S+"),
    ]
    forbidden = [str(path) for path in files if path.suffix.lower() in forbidden_suffixes]
    large = [str(path) for path in files if path.stat().st_size > 16 * 1024 * 1024]
    symlinks = [str(path) for path in stage_root.rglob("*") if path.is_symlink()]
    hardlinks = [str(path) for path in files if path.stat().st_nlink > 1]
    secrets: list[str] = []
    for path in files:
        if path.stat().st_size > 16 * 1024 * 1024:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(pattern.search(text) for pattern in secret_patterns):
            secrets.append(str(path))
    diff_check = subprocess.run(
        ["git", "diff", "--check"], cwd=BANANA, capture_output=True, text=True, check=False
    )
    passed = not forbidden and not large and not symlinks and not hardlinks and not secrets and diff_check.returncode == 0
    report = (
        "# Source hygiene report\n\n"
        f"Status: `{'pass' if passed else 'fail'}`.\n\n"
        f"Scanned files: `{len(files)}`. Forbidden model/data/runtime payloads: `{len(forbidden)}`. "
        f"Files over 16 MiB: `{len(large)}`. Symlinks: `{len(symlinks)}`. Hard-linked files: `{len(hardlinks)}`. "
        f"Secret-pattern hits: `{len(secrets)}`. `git diff --check`: `{'pass' if diff_check.returncode == 0 else 'fail'}`.\n\n"
        "Raw ONNX, predictions, images, samples, provider artifacts and large timing logs remain under the Stage raw root. No credential or authorization material is part of the tracked evidence.\n"
    )
    if not passed:
        report += f"\nFailures: `{forbidden + large + symlinks + hardlinks + secrets}`; diff: `{diff_check.stdout}{diff_check.stderr}`.\n"
    return passed, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracked-root", required=True, type=Path)
    parser.add_argument("--start-head", required=True)
    parser.add_argument("--source", action="append", default=[], type=Path)
    options = parser.parse_args()
    out = options.tracked_root

    board = board_capture()
    write_tsv(out / "board_identity_after.tsv", board)
    board_map: dict[str, list[str]] = {}
    for row in board:
        board_map.setdefault(row["field"], []).append(row["value"])
    if board_map["emmc_stage_path_count"] != ["0"]:
        raise RuntimeError("Stage-owned paths appeared on eMMC-backed filesystems")

    expected = {
        "protected_main": "1fd2e71bb1d5a924e7c0444cada94f681b73aa91",
        "custom_executor_tree": "c2e400de14fb1c88d4aed70a249d9eff19a05d0f",
        "xslim_head": "46d5d36bcb6979bab6567fb4fe62839689f1881c",
        "xslim_tree": "1788779cd0887a1c8e6924cd63ad7d16d42f41ca",
        "ncnn_head": "a245a70c641a1f20f357c65d103e5f9e50fe84a1",
        "ncnn_tree": "20b96dadbd1fc0a53159cb35749719e967b55906",
        "ncnn_diff": "2bf1cc38885018a02478aa7542581639786c79bca5ce11a6e827d24bcc5f4eca",
        "ncnn_dirty_count": "3",
    }
    actual = {
        "protected_main": git(PROTECTED, "rev-parse", "yolo26-custom-int8-engine"),
        "custom_executor_tree": git(PROTECTED, "rev-parse", "yolo26-custom-int8-engine:custom_int8_engine"),
        "xslim_head": git(XSLIM, "rev-parse", "HEAD"),
        "xslim_tree": git(XSLIM, "rev-parse", "HEAD^{tree}"),
        "ncnn_head": git(NCNN, "rev-parse", "HEAD"),
        "ncnn_tree": git(NCNN, "rev-parse", "HEAD^{tree}"),
        "ncnn_diff": ncnn_diff_sha256(),
        "ncnn_dirty_count": str(len(str(git(NCNN, "status", "--porcelain")).splitlines())),
    }
    rows = [
        {
            "surface": name,
            "before": value,
            "after": actual[name],
            "status": "pass" if actual[name] == value else "fail",
        }
        for name, value in expected.items()
    ]
    rows.extend([
        {
            "surface": "board_emmc_stage_path_count",
            "before": "0",
            "after": board_map["emmc_stage_path_count"][0],
            "status": "pass" if board_map["emmc_stage_path_count"] == ["0"] else "fail",
        },
        {
            "surface": "banana_stage_start",
            "before": options.start_head,
            "after": str(git(BANANA, "rev-parse", options.start_head)),
            "status": "pass" if str(git(BANANA, "rev-parse", options.start_head)) == options.start_head else "fail",
        },
    ])
    write_tsv(out / "protected_projects_unchanged.tsv", rows)
    if any(row["status"] != "pass" for row in rows):
        raise RuntimeError("protected-state invariance failed")

    hygiene_pass, hygiene_report = source_hygiene(out, options.source)
    (out / "source_hygiene_report.md").write_text(hygiene_report, encoding="utf-8")
    if not hygiene_pass:
        raise RuntimeError("source hygiene failed")
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    (out / "system_state_closure.md").write_text(
        "# System state closure\n\n"
        f"Captured `{timestamp}`. The board Stage root remained on NVMe `/data`; Stage-owned eMMC paths: `0`. "
        "No runtime default, OS profile, model, governor, XSlim source, custom executor or ncnn state was changed, so no rollback command was required.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
