#!/usr/bin/env python3
"""Create deterministic Stage58 release manifests and checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


GENERATED = {
    "release_manifest.json",
    "release_sha256.txt",
    "release_tree_manifest.tsv",
    "SHA256SUMS",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--package-manifest-sha256", required=True)
    parser.add_argument("--prediction-sha256", required=True)
    parser.add_argument("--known-output-hash", required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise ValueError(f"release root is not a directory: {root}")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("release bundle must not contain symlinks")

    sbom_path = root / "sbom/sbom_manifest.json"
    sbom_path.parent.mkdir(parents=True, exist_ok=True)
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": "urn:y26:banana-yolo26-k1x-int8-executor:0.9.1",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "name": "banana-yolo26-k1x-int8-executor",
                "version": "0.9.1",
                "properties": [
                    {"name": "source.commit", "value": args.source_commit},
                    {"name": "integer.contract", "value": "K1X_INT8_V1"},
                    {"name": "target", "value": "Banana-Pi BPI-F3 / SpacemiT K1X"},
                ],
            }
        },
        "components": [
            {"type": "library", "name": "y26-k1x-int8-executor", "version": "0.9.1"},
            {"type": "application", "name": "y26-k1x-demo", "version": "0.9.1"},
            {"type": "library", "name": "OpenCV", "version": "4.13.0"},
            {
                "type": "data",
                "name": "K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001",
                "version": args.package_manifest_sha256,
            },
        ],
    }
    sbom_path.write_text(json.dumps(sbom, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    payload = [
        path for path in regular_files(root)
        if path.relative_to(root).as_posix() not in GENERATED
    ]
    if not payload:
        raise ValueError("release bundle is empty")
    tree_path = root / "release_tree_manifest.tsv"
    tree_path.write_text(
        "path\tbytes\tsha256\n" + "".join(
            f"{path.relative_to(root).as_posix()}\t{path.stat().st_size}\t{sha256(path)}\n"
            for path in payload
        ),
        encoding="utf-8",
    )
    tree_sha = sha256(tree_path)

    manifest = {
        "release_id": "banana-yolo26-k1x-int8-executor-0.9.1-stage58-camera-handoff",
        "release_version": "0.9.1",
        "c_abi": 1,
        "soversion": 1,
        "integer_contract_id": "K1X_INT8_V1",
        "profile_id": "K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001",
        "model_sha256": "30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c",
        "package_manifest_sha256": args.package_manifest_sha256,
        "prediction_sha256": args.prediction_sha256,
        "known_fixture_output_hash": args.known_output_hash,
        "source_commit": args.source_commit,
        "release_tree_manifest_sha256": tree_sha,
        "prepared_model_included": True,
        "source_onnx_included": False,
        "demo_backend": "frozen executor C ABI",
        "opencv_version": "4.13.0",
        "ime_cpu_set": "0-3",
        "controller_cpu": 4,
        "production_claim": False,
        "handoff_classification": [
            "optimized-engineering-handoff-ready",
            "camera-demo-ready",
            "not-production-certified",
        ],
        "payload_file_count": len(payload),
        "payload_bytes": sum(path.stat().st_size for path in payload),
    }
    manifest_path = root / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    checksum_files = payload + [tree_path, manifest_path]
    checksums = "".join(
        f"{sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in checksum_files
    )
    (root / "release_sha256.txt").write_text(checksums, encoding="utf-8")
    (root / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    print(json.dumps({
        "payload_file_count": len(payload),
        "release_manifest_sha256": sha256(manifest_path),
        "release_tree_manifest_sha256": tree_sha,
        "release_sha256_file_sha256": sha256(root / "release_sha256.txt"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
