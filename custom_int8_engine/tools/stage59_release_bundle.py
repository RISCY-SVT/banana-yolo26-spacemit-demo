#!/usr/bin/env python3
"""Create deterministic Stage59 release metadata for one release tree."""

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


def files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bundle-kind", choices=("runtime", "internal-rd"), required=True)
    parser.add_argument("--release-version", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--integer-contract-id", required=True)
    parser.add_argument("--full-graph-profile-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--package-manifest-sha256", required=True)
    parser.add_argument("--prediction-sha256", required=True)
    parser.add_argument("--known-output-hash", required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise ValueError(f"release root is not a directory: {root}")
    symlinks = [path for path in root.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ValueError(f"release tree contains symlink: {symlinks[0]}")

    onnx = root / "model/manual_e2e_rep_conv_matmul_qdq.onnx"
    expected_onnx = args.bundle_kind == "internal-rd"
    if onnx.exists() != expected_onnx:
        raise ValueError("source ONNX presence does not match bundle kind")
    if expected_onnx and sha256(onnx) != args.model_sha256:
        raise ValueError("internal-R&D ONNX SHA-256 mismatch")

    sbom_path = root / "sbom/sbom_manifest.json"
    sbom_path.parent.mkdir(parents=True, exist_ok=True)
    components: list[dict[str, object]] = [
        {"type": "library", "name": "y26-k1x-int8-executor", "version": args.release_version},
        {"type": "application", "name": "y26-k1x-demo", "version": args.release_version},
        {"type": "library", "name": "OpenCV", "version": "4.13.0"},
        {
            "type": "data",
            "name": args.full_graph_profile_id,
            "version": args.package_manifest_sha256,
        },
    ]
    if expected_onnx:
        components.append({
            "type": "machine-learning-model",
            "name": "manual_e2e_rep_conv_matmul_qdq.onnx",
            "hashes": [{"alg": "SHA-256", "content": args.model_sha256}],
            "properties": [
                {"name": "validated.purpose", "value": "internal-rd"},
                {"name": "external.legal-clearance", "value": "not-certified"},
            ],
        })
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": (
            f"urn:y26:banana-yolo26-k1x-int8-executor:"
            f"{args.release_version}:{args.bundle_kind}"
        ),
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "name": "banana-yolo26-k1x-int8-executor",
                "version": args.release_version,
                "properties": [
                    {"name": "source.commit", "value": args.source_commit},
                    {"name": "integer.contract", "value": args.integer_contract_id},
                    {"name": "target", "value": "Banana-Pi BPI-F3 / SpacemiT K1X"},
                    {"name": "bundle.kind", "value": args.bundle_kind},
                ],
            }
        },
        "components": components,
    }
    sbom_path.write_text(json.dumps(sbom, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    payload = [path for path in files(root)
               if path.relative_to(root).as_posix() not in GENERATED]
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
        "release_id": (
            f"banana-yolo26-k1x-int8-executor-{args.release_version}"
            f"-{args.bundle_kind}"
        ),
        "release_version": args.release_version,
        "bundle_kind": args.bundle_kind,
        "c_abi": 1,
        "soversion": 1,
        "integer_contract_id": args.integer_contract_id,
        "profile_id": args.full_graph_profile_id,
        "model_sha256": args.model_sha256,
        "package_manifest_sha256": args.package_manifest_sha256,
        "prediction_sha256": args.prediction_sha256,
        "known_fixture_output_hash": args.known_output_hash,
        "source_commit": args.source_commit,
        "release_tree_manifest_sha256": tree_sha,
        "prepared_model_included": True,
        "source_onnx_included": expected_onnx,
        "source_onnx_external_legal_clearance": "not-certified",
        "project_license_status": (
            "agpl-complete-source-route-selected-legal-clearance-not-certified"
        ),
        "supported_platform": "BPI-F3 Bianbu 2.2.1 / Linux 6.6.63",
        "rootfs_independent": False,
        "offline_on_supported_platform": True,
        "demo_backend": "frozen executor C ABI",
        "ime_cpu_set": "0-3",
        "controller_cpu": 4,
        "production_claim": False,
        "payload_file_count": len(payload),
        "payload_bytes": sum(path.stat().st_size for path in payload),
    }
    manifest_path = root / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    checksum_files = payload + [tree_path, manifest_path]
    checksums = "".join(
        f"{sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in checksum_files
    )
    (root / "release_sha256.txt").write_text(checksums, encoding="utf-8")
    (root / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    print(json.dumps({
        "bundle_kind": args.bundle_kind,
        "payload_file_count": len(payload),
        "release_manifest_sha256": sha256(manifest_path),
        "release_tree_manifest_sha256": tree_sha,
        "release_sha256_file_sha256": sha256(root / "release_sha256.txt"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
