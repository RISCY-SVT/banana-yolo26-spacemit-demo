#!/usr/bin/env python3
"""Assemble and archive deterministic Stage62 stable and multiprofile deliveries."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tarfile
import tempfile
import time
import zipfile


GENERATED = {
    "SHA256SUMS",
    "release_manifest.json",
    "release_sha256.txt",
    "release_tree_manifest.tsv",
}
RESOLUTIONS = (640, 512, 448, 416, 384, 352, 320, 256, 768)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def remove_tree(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or path.is_symlink():
            raise ValueError(f"refusing to replace non-directory: {path}")
        shutil.rmtree(path)


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise ValueError(f"missing source tree: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file() or path.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path.resolve() if path.is_symlink() else path, target)


def copy_file(source: Path, destination: Path, executable: bool = False) -> None:
    if not source.is_file():
        raise ValueError(f"missing source file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    destination.chmod(0o755 if executable else 0o644)


def read_profiles(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = {int(row["resolution"]): row for row in csv.DictReader(stream, delimiter="\t")}
    if tuple(sorted(rows)) != tuple(sorted(RESOLUTIONS)):
        raise ValueError("PROFILE_PROVENANCE.tsv does not contain exactly nine profiles")
    return rows


def verify_profile_assets(args: argparse.Namespace, profiles: dict[int, dict[str, str]]) -> None:
    if sha256(args.source_model) != profiles[640]["source_model_sha256"]:
        raise ValueError("source model SHA-256 mismatch")
    for resolution in RESOLUTIONS:
        package = package_path(args, resolution)
        model = model_path(args, resolution)
        if sha256(package / "asset_hashes.tsv") != profiles[resolution]["package_manifest_sha256"]:
            raise ValueError(f"R{resolution} package-manifest SHA-256 mismatch")
        if sha256(model) != profiles[resolution]["static_model_sha256"]:
            raise ValueError(f"R{resolution} static-model SHA-256 mismatch")


def package_path(args: argparse.Namespace, resolution: int) -> Path:
    return args.r768_package if resolution == 768 else args.stage60_packages / f"r{resolution}"


def model_path(args: argparse.Namespace, resolution: int) -> Path:
    if resolution == 640:
        return args.source_model
    return args.r768_model if resolution == 768 else args.stage60_models / f"r{resolution}.onnx"


def overlay_install(install: Path, root: Path) -> None:
    for directory in ("bin", "lib", "include", "share"):
        source = install / directory
        if source.exists():
            copy_tree(source, root / directory)


def overlay_project_material(repo: Path, root: Path, integrated: bool) -> None:
    for name in (
        "LICENSE",
        "COPYRIGHTS.md",
        "INTENDED_USE_AND_CERTIFICATION_STATUS.md",
        "LEGAL_STATUS.md",
        "MODEL_LICENSE_AND_PROVENANCE.md",
        "MODIFICATIONS.md",
        "NO_WARRANTY.md",
        "PROFILE_PROVENANCE.tsv",
        "SOURCE_ACCESS.md",
        "THIRD_PARTY_NOTICES.md",
    ):
        copy_file(repo / name, root / name)
    copy_tree(repo / "LICENSES", root / "LICENSES")
    copy_tree(repo / "docs", root / "docs")
    copy_tree(repo / "model", root / "model-evidence")
    if integrated:
        remove_tree(root / "scripts")
        copy_tree(repo / "scripts", root / "scripts")
        copy_file(repo / "config/release.env", root / "config/release.env")


def extract_git_source(repo: Path, commit: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stage62-source-") as temporary:
        archive = Path(temporary) / "source.tar"
        with archive.open("wb") as stream:
            subprocess.run(
                ["git", "-C", str(repo), "archive", "--format=tar", commit],
                check=True,
                stdout=stream,
            )
        with tarfile.open(archive, "r") as bundle:
            bundle.extractall(destination, filter="data")


def elf_needed(path: Path) -> list[str]:
    try:
        with path.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                return []
    except OSError:
        return []
    process = subprocess.run(
        ["readelf", "-d", str(path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    result = []
    for line in process.stdout.splitlines():
        marker = "Shared library: ["
        if marker in line:
            result.append(line.split(marker, 1)[1].split("]", 1)[0])
    return result


def write_dependency_inventory(root: Path) -> list[str]:
    rows: list[tuple[str, str]] = []
    sonames: set[str] = set()
    for path in files(root):
        needed = elf_needed(path)
        for soname in needed:
            rows.append((path.relative_to(root).as_posix(), soname))
            sonames.add(soname)
    (root / "binary_dependency_inventory.tsv").write_text(
        "binary\tneeded_soname\n" + "".join(f"{binary}\t{soname}\n" for binary, soname in rows),
        encoding="utf-8",
    )
    bundled = {path.name for path in files(root)}
    (root / "required-system-sonames.tsv").write_text(
        "soname\tresolution\n" + "".join(
            f"{soname}\t{'bundled' if soname in bundled else 'supported-board-system'}\n"
            for soname in sorted(sonames)
        ),
        encoding="utf-8",
    )
    return sorted(sonames)


def write_licenses(root: Path, version: str) -> None:
    (root / "third_party_license_inventory.tsv").write_text(
        "component\tversion\tlicense\tbundled\tevidence\n"
        f"y26-k1x-int8-executor\t{version}\tAGPL-3.0-or-later\tyes\tLICENSE\n"
        "Ultralytics lineage\t8.4.82\tAGPL-3.0 / Enterprise terms\tmodel-data\tMODEL_LICENSE_AND_PROVENANCE.md\n"
        "OpenCV\t4.13.0\tApache-2.0\tyes\tLICENSES/Apache-2.0.txt\n"
        "GNU runtime\t14.3.0\truntime/library exceptions\tno\tDT_NEEDED and supported sysroot\n",
        encoding="utf-8",
    )
    (root / "unresolved_license_items.tsv").write_text(
        "item\tstatus\trequired_human_evidence\n"
        "project ownership\tunresolved\tauthoritative ownership/relicensing record\n"
        "Ultralytics Enterprise agreement\tnot-found\tagreement or AGPL legal review\n"
        "source-model export authority\tunresolved\tcreator/export/provenance record\n"
        "external model conveyance\tnot-certified\tresponsible legal approval\n",
        encoding="utf-8",
    )


def write_sboms(root: Path, version: str, kind: str, commit: str,
                profiles: dict[int, dict[str, str]], sonames: list[str]) -> None:
    components = [{
        "type": "application",
        "name": "banana-yolo26-k1x-int8-executor",
        "version": version,
        "licenses": [{"license": {"id": "AGPL-3.0-or-later"}}],
    }, {
        "type": "library", "name": "OpenCV", "version": "4.13.0",
        "licenses": [{"license": {"id": "Apache-2.0"}}],
    }]
    for resolution in RESOLUTIONS if "integrated" in kind else (640,):
        row = profiles[resolution]
        components.append({
            "type": "machine-learning-model",
            "name": row["profile_id"],
            "version": row["package_manifest_sha256"],
            "hashes": [{"alg": "SHA-256", "content": row["static_model_sha256"]}],
            "properties": [{"name": "profile.status", "value": row["status"]}],
        })
    cyclone = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{hashlib.sha256((version + kind + commit).encode()).hexdigest()[:32]}",
        "version": 1,
        "metadata": {"component": components[0]},
        "components": components[1:],
    }
    (root / "SBOM.cyclonedx.json").write_text(
        json.dumps(cyclone, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    spdx_packages = [{
        "SPDXID": "SPDXRef-Package-Executor",
        "name": "banana-yolo26-k1x-int8-executor",
        "versionInfo": version,
        "downloadLocation": "NOASSERTION",
        "licenseConcluded": "AGPL-3.0-or-later",
        "licenseDeclared": "AGPL-3.0-or-later",
        "copyrightText": "NOASSERTION",
    }]
    spdx = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"banana-yolo26-k1x-{version}-{kind}",
        "documentNamespace": f"https://example.invalid/spdx/{commit}/{kind}",
        "creationInfo": {
            "created": "1970-01-01T00:00:00Z",
            "creators": ["Tool: stage62_release.py"],
        },
        "packages": spdx_packages,
        "annotations": [{
            "annotationDate": "1970-01-01T00:00:00Z",
            "annotationType": "OTHER",
            "annotator": "Tool: stage62_release.py",
            "comment": "Required system SONAMEs: " + ", ".join(sonames),
        }],
    }
    (root / "SBOM.spdx.json").write_text(
        json.dumps(spdx, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_reproduce(root: Path, version: str, commit: str, kind: str) -> None:
    (root / "BUILD_REPRODUCE.md").write_text(
        "# Reproduce This Delivery\n\n"
        f"Version: `{version}`  \nSource commit: `{commit}`  \nBundle: `{kind}`\n\n"
        "Compiler flags: `-march=rv64gcv_zvfh -mabi=lp64d "
        "-mtune=spacemit-x60 -funroll-loops -O3 -DNDEBUG`.\n\n"
        "Use `source/project/scripts/build_cross.sh` from a complete-source tree, "
        "or `docs/BUILDING_K1X_INT8_EXECUTOR.md` in an SDK tree. The base sysroot "
        "is read-only and the accepted K1X overlay is required.\n",
        encoding="utf-8",
    )


def write_manifests(root: Path, version: str, kind: str, commit: str,
                    profiles: dict[int, dict[str, str]]) -> None:
    symlinks = [path for path in root.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ValueError(f"release contains symlink: {symlinks[0]}")
    initial_files = files(root)
    payload = [path for path in initial_files if path.name not in GENERATED]
    digests = {path: sha256(path) for path in payload}
    tree = root / "release_tree_manifest.tsv"
    tree.write_text(
        "path\tbytes\tsha256\n" + "".join(
            f"{path.relative_to(root).as_posix()}\t{path.stat().st_size}\t{digests[path]}\n"
            for path in payload
        ),
        encoding="utf-8",
    )
    manifest = {
        "release_version": version,
        "bundle_kind": kind,
        "source_commit": commit,
        "abi": 1,
        "soversion": 1,
        "integer_contract": "K1X_INT8_V1",
        "default_profile": profiles[640]["profile_id"],
        "available_profiles": [profiles[r]["profile_id"] for r in
                               (RESOLUTIONS if "integrated" in kind else (640,))],
        "legal_route": "agpl-complete-source-route-selected",
        "legal_clearance": "not-certified",
        "production_certified": False,
        "release_tree_manifest_sha256": sha256(tree),
        "payload_file_count": len(payload),
        "payload_bytes": sum(path.stat().st_size for path in payload),
    }
    (root / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksummed = payload + [tree, root / "release_manifest.json"]
    sums = "".join(
        f"{digests[path] if path in digests else sha256(path)}  "
        f"{path.relative_to(root).as_posix()}\n"
        for path in checksummed)
    (root / "SHA256SUMS").write_text(sums, encoding="utf-8")
    (root / "release_sha256.txt").write_text(sums, encoding="utf-8")


def assemble(args: argparse.Namespace) -> None:
    profiles = read_profiles(args.repo / "PROFILE_PROVENANCE.tsv")
    verify_profile_assets(args, profiles)
    root = args.root.resolve()
    remove_tree(root)
    copy_tree(args.template, root)
    for generated in GENERATED:
        path = root / generated
        if path.exists():
            path.unlink()
    for stale in (root / "sbom", root / "SBOM.spdx.json", root / "SBOM.cyclonedx.json"):
        if stale.is_dir():
            shutil.rmtree(stale)
        elif stale.exists():
            stale.unlink()
    integrated = args.kind.startswith("integrated-")
    source_bundle = args.kind.endswith("source") or args.kind == "stable-internal"
    for directory in ("bin", "lib", "include", "share"):
        remove_tree(root / directory)
    overlay_install(args.install, root)
    overlay_project_material(args.repo, root, integrated)
    package_root = root / "package"
    remove_tree(package_root)
    copy_tree(package_path(args, 640), package_root)
    if integrated:
        profiles_root = root / "profiles"
        remove_tree(profiles_root)
        for resolution in RESOLUTIONS:
            copy_tree(package_path(args, resolution), profiles_root / f"r{resolution}" / "package")
    source_onnx = root / "model" / "manual_e2e_rep_conv_matmul_qdq.onnx"
    if source_onnx.exists():
        source_onnx.unlink()
    if source_bundle:
        copy_file(args.source_model, source_onnx)
        source_root = root / "source" / "project"
        remove_tree(source_root)
        extract_git_source(args.repo, args.source_commit, source_root)
        (source_root / "SOURCE_COMMIT").write_text(
            args.source_commit + "\n", encoding="utf-8"
        )
        if integrated:
            static_root = root / "model" / "static"
            for resolution in RESOLUTIONS:
                copy_file(model_path(args, resolution), static_root / f"r{resolution}.onnx")

    evidence = root / "outputs" / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    for name in (
        "resolution_performance_summary_v2.tsv",
        "resolution_coco_results_v2.tsv",
        "resolution_prediction_hashes_v2.tsv",
        "resolution_pareto_q0_v2.tsv",
        "r768_camera_matrix.tsv",
    ):
        copy_file(args.stage61_reports / name, evidence / name)

    version = "0.9.3" if args.kind.startswith("stable-") else "0.10.0-internal-rd.1"
    write_reproduce(root, version, args.source_commit, args.kind)
    sonames = write_dependency_inventory(root)
    write_licenses(root, version)
    write_sboms(root, version, args.kind, args.source_commit, profiles, sonames)
    write_manifests(root, version, args.kind, args.source_commit, profiles)


def normalized_zip_time(epoch: int) -> tuple[int, int, int, int, int, int]:
    value = time.gmtime(max(epoch, 315532800))
    return value.tm_year, value.tm_mon, value.tm_mday, value.tm_hour, value.tm_min, value.tm_sec


def archive(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    if not root.is_dir():
        raise ValueError(f"missing release root: {root}")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("release tree contains symlinks")
    args.output.mkdir(parents=True, exist_ok=True)
    tar_path = args.output / f"{args.base}.tar.gz"
    zip_path = args.output / f"{args.base}.zip"
    with tar_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as bundle:
                for path in [root, *sorted(root.rglob("*"))]:
                    relative = Path(args.base) / path.relative_to(root)
                    info = bundle.gettarinfo(str(path), arcname=relative.as_posix())
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = args.epoch
                    if path.is_file():
                        with path.open("rb") as stream:
                            bundle.addfile(info, stream)
                    else:
                        bundle.addfile(info)
    timestamp = normalized_zip_time(args.epoch)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as bundle:
        directory = zipfile.ZipInfo(f"{args.base}/", timestamp)
        directory.external_attr = (stat.S_IFDIR | 0o755) << 16
        bundle.writestr(directory, b"")
        for path in sorted(root.rglob("*")):
            relative = (Path(args.base) / path.relative_to(root)).as_posix()
            if path.is_dir():
                info = zipfile.ZipInfo(relative + "/", timestamp)
                info.external_attr = (stat.S_IFDIR | 0o755) << 16
                bundle.writestr(info, b"")
            else:
                info = zipfile.ZipInfo(relative, timestamp)
                mode = 0o755 if os.access(path, os.X_OK) else 0o644
                info.external_attr = (stat.S_IFREG | mode) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                bundle.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED,
                                compresslevel=9)
    print(f"{sha256(tar_path)}  {tar_path}")
    print(f"{sha256(zip_path)}  {zip_path}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)
    build = commands.add_parser("assemble")
    build.add_argument("--kind", choices=(
        "stable-runtime", "stable-internal", "integrated-sdk", "integrated-source"),
        required=True)
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--repo", type=Path, required=True)
    build.add_argument("--template", type=Path, required=True)
    build.add_argument("--install", type=Path, required=True)
    build.add_argument("--source-commit", required=True)
    build.add_argument("--source-model", type=Path, required=True)
    build.add_argument("--stage60-packages", type=Path, required=True)
    build.add_argument("--stage60-models", type=Path, required=True)
    build.add_argument("--r768-package", type=Path, required=True)
    build.add_argument("--r768-model", type=Path, required=True)
    build.add_argument("--stage61-reports", type=Path, required=True)
    build.set_defaults(handler=assemble)
    pack = commands.add_parser("archive")
    pack.add_argument("--root", type=Path, required=True)
    pack.add_argument("--base", required=True)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--epoch", type=int, required=True)
    pack.set_defaults(handler=archive)
    return result


def main() -> int:
    args = parser().parse_args()
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
