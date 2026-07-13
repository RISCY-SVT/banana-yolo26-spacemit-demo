#!/usr/bin/env python3
"""Create the deterministic Stage 52 handoff manifest and checksum inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--package-manifest-sha256", required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise ValueError(f"release root is not a directory: {root}")
    manifest_path = root / "release_manifest.json"
    checksums_path = root / "release_sha256.txt"
    excluded = {manifest_path, checksums_path}
    files = sorted(path for path in root.rglob("*") if path.is_file() and path not in excluded)
    if not files:
        raise ValueError("release bundle is empty")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("release bundle must not contain symlinks")

    entries = [{
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    } for path in files]
    manifest = {
        "release_id": "banana-yolo26-k1x-int8-executor-stage52",
        "integer_contract_id": "K1X_INT8_V1",
        "profile_id": "K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001",
        "model_sha256": "30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c",
        "package_manifest_sha256": args.package_manifest_sha256,
        "source_commit": args.source_commit,
        "target": "Banana-Pi BPI-F3 / SpacemiT K1X",
        "safe_scheduler": "SCHED_OTHER",
        "ime_cpu_set": "0-3",
        "controller_cpu": 4,
        "production_claim": False,
        "files": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    checksum_files = files + [manifest_path]
    checksums_path.write_text("".join(
        f"{sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in checksum_files
    ), encoding="utf-8")
    print(json.dumps({
        "files": len(checksum_files),
        "payload_bytes": sum(path.stat().st_size for path in checksum_files),
        "release_manifest_sha256": sha256(manifest_path),
        "release_sha256_sha256": sha256(checksums_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
