#!/usr/bin/env python3
"""Freeze the DEV-001C inputs and fresh train2017 H5000 surface."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import struct
import subprocess
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import PIL
from PIL import Image, ImageOps


STAGE_ID = (
    "BANANA-YOLO26-XSLIM-DEV-001C-C2-FROZEN-INDEPENDENT-HOLDOUT-"
    "ADJUDICATION-AND-VENDOR-PTQ-LANE-CLOSURE-001"
)
DEV001B_ID = (
    "BANANA-YOLO26-XSLIM-DEV-001B-ALL-S8-GENERIC-HARDENING-ADAPTIVE-"
    "ROUNDING-BLOCK-RECONSTRUCTION-AND-DETECTOR-PARETO-HOST-GATE-001"
)
R1_ID = (
    "BANANA-YOLO26-XSLIM-STAGE65B-R1-COCO-TRAIN2017-EVALUATION-"
    "DISJOINT-CORPUS-PTQ-GRAPHWISE-AND-PYRAMID-CAUSAL-LOCALIZATION-001"
)
R2_ID = (
    "BANANA-YOLO26-XSLIM-STAGE65B-R2-HOST-INDEPENDENT-SELECTION-FP32-"
    "SPLIT-BOUNDARY-QDQ-DISAMBIGUATION-AND-B2-VARIANCE-GATE-001"
)
DEV001A_ID = (
    "BANANA-YOLO26-XSLIM-DEV-001A-SPACEMIT-S8-QDQ-CONSTRAINED-RANGE-"
    "OBSERVER-TERMINAL-DOMAIN-AND-POLICY-A-HOST-CANDIDATE-GATE-001"
)

POLICY = "xslim-dev-001c-fresh-h5000-v1"
PIXEL_HASH_PREFIX = b"stage65b-r1-rgb8-v1\0"
BOOTSTRAP_SEED = 65007
TARGET_COUNT = 5000
EXTRACTION_BATCH = 256

EXPECTED = {
    "banana_head": "6e7a2baf0d4b8dc2922c72ddfdcb8c83b85356f1",
    "banana_tree": "c2774c6beb88f558f71202c2a08f86a9fbd5929c",
    "xslim_head": "46d5d36bcb6979bab6567fb4fe62839689f1881c",
    "xslim_tree": "1788779cd0887a1c8e6924cd63ad7d16d42f41ca",
    "upstream_main": "9a33f2f770d00fd02ff8bc0f1907135e9bf47f8c",
    "protected_main": "1fd2e71bb1d5a924e7c0444cada94f681b73aa91",
    "custom_executor_tree": "c2e400de14fb1c88d4aed70a249d9eff19a05d0f",
    "packet_tree": "139092b15cede35760edf5d6fdfef98503a9bab788974cd4f7409d5bdfb997f9",
    "packet_files": 59,
    "packet_bytes": 181055,
    "train_annotations": "610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d",
    "val_annotations": "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f",
    "runner": "79ad059411bb153f3abcb8d4abd0f1e79e5e04b12863fc121dc227d2fe89bd65",
    "b2_deployable": "0e7040d4e8b1b2d08a4e36cec4c99dcea6d52294e04901d17dfce10725c6d617",
    "b2_inference": "40ba6a7f9aebaa98a1c3abe5fce1f66f1bebcd0b10b7af3d26d30414a331d853",
    "a1_deployable": "8fad9fa0e385f58da281d963c5e18b010c80c402dcbeed0b46e3ca3065d010f3",
    "a1_inference": "f7c5345f68cf79a5c3748274239a14cdaa59f77eac0425f7771694febaa24632",
    "c2_deployable": "e963be11c57c048f23caa34df1e2d140211632cc4dfd6b734b14909a30ea4b55",
    "c2_inference": "281f4acd1261e7ee2c38b6e3bdecbf61c3d91cf710c63e6bc6cdaf257a52669b",
    "tail": "18ffff41e6812fa781baf7b9c1fcd41b41d6118145d785c3e550499070a512a3",
    "ncnn_head": "a245a70c641a1f20f357c65d103e5f9e50fe84a1",
    "ncnn_tree": "20b96dadbd1fc0a53159cb35749719e967b55906",
    "ncnn_diff": "2bf1cc38885018a02478aa7542581639786c79bca5ce11a6e827d24bcc5f4eca",
    "ncnn_dirty_paths": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--banana", required=True, type=Path)
    parser.add_argument("--protected", required=True, type=Path)
    parser.add_argument("--xslim", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--tracked-root", required=True, type=Path)
    parser.add_argument("--shared-log", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    return parser.parse_args()


def command(*args: str, cwd: Path | None = None, binary: bool = False) -> Any:
    return subprocess.check_output(args, cwd=cwd, text=not binary).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(label: str, observed: object, expected: object) -> dict[str, Any]:
    if observed != expected:
        raise RuntimeError(f"{label} mismatch: {observed!r} != {expected!r}")
    return {
        "surface": label,
        "observed": observed,
        "expected": expected,
        "status": "pass",
    }


def write_tsv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise ValueError(f"refusing empty TSV: {path}")
    fields: list[str] = []
    for row in values:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in values)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


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


def canonical_digest(path: Path) -> dict[str, Any]:
    file_hash = sha256(path)
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        digest = hashlib.sha256()
        digest.update(PIXEL_HASH_PREFIX)
        digest.update(struct.pack("<Q", width))
        digest.update(struct.pack("<Q", height))
        digest.update(image.tobytes())
        pixel_hash = digest.hexdigest()
    return {
        "file_sha256": file_hash,
        "pixel_sha256": pixel_hash,
        "decoded_width": width,
        "decoded_height": height,
    }


def path_ids(path: Path) -> list[int]:
    values = [int(Path(line).stem) for line in path.read_text().splitlines() if line]
    if len(values) != len(set(values)) or not values:
        raise RuntimeError(f"list is empty or contains duplicate IDs: {path}")
    return values


def rank_digest(annotation_sha: str, image_id: int) -> str:
    payload = (
        POLICY.encode()
        + b"\0"
        + annotation_sha.encode("ascii")
        + b"\0"
        + str(image_id).encode("ascii")
    )
    return hashlib.sha256(payload).hexdigest()


def safe_member(name: str, expected: str) -> None:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != expected:
        raise RuntimeError(f"unsafe or unexpected ZIP member: {name}")


def extract_batch(
    archive: zipfile.ZipFile,
    rows: list[dict[str, Any]],
    destination: Path,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for row in rows:
        member = f"train2017/{row['file_name']}"
        info = archive.getinfo(member)
        safe_member(info.filename, member)
        mode = (info.external_attr >> 16) & 0xFFFF
        if mode & 0o170000 not in {0, 0o100000}:
            raise RuntimeError(f"unsafe ZIP file type for {member}: {oct(mode)}")
        target = destination / str(row["file_name"])
        temporary = target.with_suffix(".jpg.part")
        if target.exists() or temporary.exists():
            raise RuntimeError(f"refusing existing extracted member: {target}")
        with archive.open(info) as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output, length=1 << 20)
        temporary.replace(target)


def verify_reference_images(
    rows: list[tuple[Path, dict[str, str]]], workers: int
) -> None:
    def verify(item: tuple[Path, dict[str, str]]) -> None:
        path, expected = item
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = canonical_digest(path)
        for field in ("file_sha256", "pixel_sha256"):
            if observed[field] != expected[field]:
                raise RuntimeError(f"reference {field} mismatch: {path}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(verify, rows))


def size_bin(area: float) -> str:
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def artifact_paths() -> list[tuple[str, Path, str, str]]:
    data = Path("/data/k1x-stage-runs")
    r1 = data / R1_ID
    dev1a = data / DEV001A_ID
    dev1b = data / DEV001B_ID
    return [
        (
            "B2",
            r1 / "quantization/B2/run1/output/stage65b_r1_b2_split_s8_qdq.onnx",
            "deployable",
            EXPECTED["b2_deployable"],
        ),
        (
            "B2",
            r1 / "postprocess/B2/candidate-gate/B2/models/stage65b_r1_b2.inference.onnx",
            "inference",
            EXPECTED["b2_inference"],
        ),
        (
            "A1",
            dev1a / "candidates/quantization/A1/run1/output/xslim_dev_001a_a1_split_s8_qdq.onnx",
            "deployable",
            EXPECTED["a1_deployable"],
        ),
        (
            "A1",
            dev1a / "candidates/postprocess/A1/models/stage65b_r1_a1.inference.onnx",
            "inference",
            EXPECTED["a1_inference"],
        ),
        (
            "C2",
            dev1b / "candidates/run1/C2_T6_RANK_QP/c2_t6_rank_qp.deployable.onnx",
            "deployable",
            EXPECTED["c2_deployable"],
        ),
        (
            "C2",
            dev1b / "candidates/run1/C2_T6_RANK_QP/c2_t6_rank_qp.inference.onnx",
            "inference",
            EXPECTED["c2_inference"],
        ),
        (
            "common",
            r1 / "postprocess/B2/candidate-gate/B2/models/stage65b_r1_b2.postprocess.onnx",
            "tail",
            EXPECTED["tail"],
        ),
    ]


def main() -> int:
    options = parse_args()
    if options.workers < 1:
        raise ValueError("--workers must be positive")
    if options.raw_root.exists() or options.tracked_root.exists():
        raise RuntimeError("refusing pre-existing DEV-001C raw or tracked root")

    preflight: list[dict[str, Any]] = []
    preflight.extend(
        [
            require("banana_head", command("git", "rev-parse", "HEAD", cwd=options.banana), EXPECTED["banana_head"]),
            require("banana_tree", command("git", "rev-parse", "HEAD^{tree}", cwd=options.banana), EXPECTED["banana_tree"]),
            require("xslim_head", command("git", "rev-parse", "HEAD", cwd=options.xslim), EXPECTED["xslim_head"]),
            require("xslim_tree", command("git", "rev-parse", "HEAD^{tree}", cwd=options.xslim), EXPECTED["xslim_tree"]),
            require("xslim_version", (options.xslim / "VERSION_NUMBER").read_text().strip(), "2.1.2+riscy.2.dev2"),
            require("upstream_main", command("git", "rev-parse", "upstream/main", cwd=options.xslim), EXPECTED["upstream_main"]),
            require("protected_main", command("git", "rev-parse", "yolo26-custom-int8-engine", cwd=options.protected), EXPECTED["protected_main"]),
            require("custom_executor_tree", command("git", "rev-parse", "yolo26-custom-int8-engine:custom_int8_engine", cwd=options.protected), EXPECTED["custom_executor_tree"]),
        ]
    )
    parity_rows: list[dict[str, Any]] = []
    for repository, repo, branch, remotes in (
        ("Banana", options.banana, "yolo26-vendor-ort-xslim211-s8-qdq-validation", ("github", "gitlab-rd")),
        ("XSlim", options.xslim, "riscy/k1x-yolo26", ("github", "gitlab")),
    ):
        local = command("git", "rev-parse", branch, cwd=repo)
        for remote in remotes:
            observed = command("git", "rev-parse", f"{remote}/{branch}", cwd=repo)
            require(f"{repository}_{remote}_parity", observed, local)
            parity_rows.append(
                {
                    "repository": repository,
                    "branch": branch,
                    "surface": remote,
                    "local": local,
                    "remote": observed,
                    "status": "pass",
                }
            )

    packet = packet_identity(options.packet)
    require("dev001b_packet", packet, (EXPECTED["packet_tree"], EXPECTED["packet_files"], EXPECTED["packet_bytes"]))

    artifact_rows: list[dict[str, Any]] = []
    for surface, path, kind, expected_hash in artifact_paths():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256(path)
        require(f"{surface}_{kind}", observed, expected_hash)
        artifact_rows.append(
            {
                "surface": surface,
                "kind": kind,
                "canonical_path": path,
                "bytes": path.stat().st_size,
                "sha256": observed,
                "expected_sha256": expected_hash,
                "status": "pass",
            }
        )

    train_annotations = options.dataset / "annotations/instances_train2017.json"
    val_annotations = options.dataset / "annotations/instances_val2017.json"
    train_archive = options.dataset / "archives/train2017.zip"
    require("train_annotations", sha256(train_annotations), EXPECTED["train_annotations"])
    require("val_annotations", sha256(val_annotations), EXPECTED["val_annotations"])
    runner = options.banana / "vendor_ort_validation/stage65b_r1_evaluate.py"
    require("runner", sha256(runner), EXPECTED["runner"])
    if not train_archive.is_file():
        raise FileNotFoundError(train_archive)

    ncnn = Path("/data/ncnn")
    ncnn_diff = hashlib.sha256(
        subprocess.check_output(["git", "diff", "--binary"], cwd=ncnn)
    ).hexdigest()
    protected_rows = [
        require("banana_protected_main", command("git", "rev-parse", "yolo26-custom-int8-engine", cwd=options.protected), EXPECTED["protected_main"]),
        require("custom_executor_tree", command("git", "rev-parse", "yolo26-custom-int8-engine:custom_int8_engine", cwd=options.protected), EXPECTED["custom_executor_tree"]),
        require("ncnn_head", command("git", "rev-parse", "HEAD", cwd=ncnn), EXPECTED["ncnn_head"]),
        require("ncnn_tree", command("git", "rev-parse", "HEAD^{tree}", cwd=ncnn), EXPECTED["ncnn_tree"]),
        require("ncnn_diff", ncnn_diff, EXPECTED["ncnn_diff"]),
        require("ncnn_dirty_paths", len(command("git", "status", "--porcelain=v1", cwd=ncnn).splitlines()), EXPECTED["ncnn_dirty_paths"]),
    ]

    r1_stage = options.banana / "stages" / R1_ID
    r2_stage = options.banana / "stages" / R2_ID
    membership_path = r1_stage / "selection_membership.tsv"
    val_hash_path = r1_stage / "val2017_file_hashes.tsv"
    prior_rows_all = read_tsv(membership_path)
    prior_by_id = {int(row["image_id"]): row for row in prior_rows_all}
    preprocess_union = options.dataset / "lists/preprocess_union.txt"
    excluded_ids = set(path_ids(preprocess_union))
    if len(excluded_ids) != 1515 or set(prior_by_id) != excluded_ids:
        raise RuntimeError("accepted train selection union does not equal the 1515-row membership manifest")
    vdraw = r2_stage / "b2_variance_effective_configs/lists/Vdraw.txt"
    if not set(path_ids(vdraw)) <= excluded_ids:
        raise RuntimeError("R2 Vdraw is not covered by the frozen prior-use union")

    val_rows = read_tsv(val_hash_path)
    if len(val_rows) != 5000:
        raise RuntimeError("accepted val2017 hash manifest must contain 5000 rows")
    val_by_id = {int(row["image_id"]): row for row in val_rows}
    if len(val_by_id) != 5000:
        raise RuntimeError("duplicate image IDs in val2017 hash manifest")
    prior_verify = [
        (options.dataset / "selected/images" / row["file_name"], row)
        for row in prior_rows_all
    ]
    val_root = Path(options.dataset / "lists/val2017_all.txt").read_text().splitlines()
    val_paths = {int(Path(item).stem): Path(item) for item in val_root if item}
    if set(val_paths) != set(val_by_id):
        raise RuntimeError("val2017 list and hash manifest IDs differ")
    verify_reference_images(prior_verify, options.workers)
    verify_reference_images([(val_paths[key], val_by_id[key]) for key in sorted(val_by_id)], options.workers)

    payload = json.loads(train_annotations.read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in payload["images"]}
    licenses = {int(row["id"]): row for row in payload["licenses"]}
    if len(images) != 118287 or len(payload["categories"]) != 80 or not licenses:
        raise RuntimeError("unexpected train2017 annotation surface")
    val_ids = set(val_by_id)
    if set(images) & val_ids:
        raise RuntimeError("train2017 and val2017 image IDs overlap")
    ranked = sorted(
        (row for key, row in images.items() if key not in excluded_ids),
        key=lambda row: (rank_digest(EXPECTED["train_annotations"], int(row["id"])), int(row["id"])),
    )
    if len(ranked) < TARGET_COUNT:
        raise RuntimeError("insufficient unused train2017 candidate pool")

    options.raw_root.mkdir(parents=True)
    options.tracked_root.mkdir(parents=True)
    images_root = options.raw_root / "fresh-h5000/images"
    val_file_hashes = {row["file_sha256"] for row in val_rows}
    val_pixel_hashes = {row["pixel_sha256"] for row in val_rows}
    prior_file_hashes = {row["file_sha256"] for row in prior_rows_all}
    prior_pixel_hashes = {row["pixel_sha256"] for row in prior_rows_all}
    accepted: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    selected_file_hashes: set[str] = set()
    selected_pixel_hashes: set[str] = set()
    cursor = 0
    with zipfile.ZipFile(train_archive) as archive:
        while len(accepted) < TARGET_COUNT:
            batch = ranked[cursor : cursor + EXTRACTION_BATCH]
            if not batch:
                raise RuntimeError("ranked reserve exhausted before 5000 qualified images")
            extract_batch(archive, batch, images_root)
            inputs = [(images_root / str(row["file_name"])) for row in batch]
            with ThreadPoolExecutor(max_workers=options.workers) as pool:
                digests = list(pool.map(canonical_digest, inputs))
            for row, digest in zip(batch, digests):
                image_id = int(row["id"])
                reasons: list[str] = []
                if digest["decoded_width"] != int(row["width"]) or digest["decoded_height"] != int(row["height"]):
                    reasons.append("annotation-dimension-mismatch")
                if digest["file_sha256"] in val_file_hashes:
                    reasons.append("val-file-overlap")
                if digest["pixel_sha256"] in val_pixel_hashes:
                    reasons.append("val-pixel-overlap")
                if digest["file_sha256"] in prior_file_hashes:
                    reasons.append("prior-file-overlap")
                if digest["pixel_sha256"] in prior_pixel_hashes:
                    reasons.append("prior-pixel-overlap")
                if digest["file_sha256"] in selected_file_hashes:
                    reasons.append("internal-file-duplicate")
                if digest["pixel_sha256"] in selected_pixel_hashes:
                    reasons.append("internal-pixel-duplicate")
                selected = not reasons and len(accepted) < TARGET_COUNT
                if not reasons and not selected:
                    reasons.append("rank-after-target-count")
                decision = {
                    "rank": cursor + batch.index(row) + 1,
                    "image_id": image_id,
                    "file_name": row["file_name"],
                    "rank_sha256": rank_digest(EXPECTED["train_annotations"], image_id),
                    **digest,
                    "status": (
                        "accepted"
                        if selected
                        else "reserve-not-selected"
                        if reasons == ["rank-after-target-count"]
                        else "rejected"
                    ),
                    "reason": ",".join(reasons),
                }
                decisions.append(decision)
                if selected:
                    selected_file_hashes.add(str(digest["file_sha256"]))
                    selected_pixel_hashes.add(str(digest["pixel_sha256"]))
                    accepted.append({**row, **decision})
            cursor += len(batch)

    if len({int(row["id"]) for row in accepted}) != TARGET_COUNT:
        raise RuntimeError("internal image-ID duplicates in selected H5000")
    selected_ids = {int(row["id"]) for row in accepted}
    if selected_ids & excluded_ids or selected_ids & val_ids:
        raise RuntimeError("selected H5000 has an excluded image-ID overlap")

    annotation_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        annotation_by_image[int(annotation["image_id"])].append(annotation)
    category_counts: Counter[int] = Counter()
    size_counts: Counter[str] = Counter()
    license_counts: Counter[int] = Counter()
    images_with_size: Counter[str] = Counter()
    for row in accepted:
        image_id = int(row["id"])
        license_counts[int(row["license"])] += 1
        observed_sizes: set[str] = set()
        for annotation in annotation_by_image.get(image_id, []):
            if int(annotation.get("iscrowd", 0)):
                continue
            category_counts[int(annotation["category_id"])] += 1
            size = size_bin(float(annotation.get("area", 0.0)))
            size_counts[size] += 1
            observed_sizes.add(size)
        for size in observed_sizes:
            images_with_size[size] += 1

    raw_list = options.raw_root / "fresh-h5000/H5000_C2_ADJUDICATION.txt"
    raw_list.write_text(
        "".join(f"{images_root / row['file_name']}\n" for row in accepted),
        encoding="utf-8",
    )
    tracked_list = options.tracked_root / "fresh_h5000_list.txt"
    tracked_list.write_text(
        "".join(f"{row['file_name']}\n" for row in accepted), encoding="utf-8"
    )
    write_tsv(options.raw_root / "fresh-h5000/selection_rank_and_decisions.tsv", decisions)
    write_tsv(
        options.raw_root / "fresh-h5000/file_pixel_manifest.tsv",
        [
            {
                "rank": index,
                "image_id": row["id"],
                "file_name": row["file_name"],
                "file_sha256": row["file_sha256"],
                "pixel_sha256": row["pixel_sha256"],
                "width": row["decoded_width"],
                "height": row["decoded_height"],
                "license_id": row["license"],
            }
            for index, row in enumerate(accepted, 1)
        ],
    )
    write_tsv(
        options.raw_root / "fresh-h5000/excluded_union_manifest.tsv",
        [
            {
                "image_id": row["image_id"],
                "file_name": row["file_name"],
                "file_sha256": row["file_sha256"],
                "pixel_sha256": row["pixel_sha256"],
                "sources": "Stage65B calibration/H500; DEV-001A; DEV-001B; R2-Vdraw-subset",
            }
            for row in prior_rows_all
        ],
    )

    distribution: list[dict[str, Any]] = []
    categories = {int(row["id"]): row["name"] for row in payload["categories"]}
    for category_id in sorted(categories):
        distribution.append(
            {
                "dimension": "category-instance",
                "key": category_id,
                "label": categories[category_id],
                "count": category_counts[category_id],
                "license_url": "not-applicable",
            }
        )
    for size in ("small", "medium", "large"):
        distribution.extend(
            [
                {"dimension": "object-size-instance", "key": size, "label": size, "count": size_counts[size], "license_url": "not-applicable"},
                {"dimension": "image-contains-size", "key": size, "label": size, "count": images_with_size[size], "license_url": "not-applicable"},
            ]
        )
    for license_id in sorted(licenses):
        distribution.append(
            {
                "dimension": "license-image",
                "key": license_id,
                "label": licenses[license_id]["name"],
                "count": license_counts[license_id],
                "license_url": licenses[license_id]["url"],
            }
        )
    write_tsv(options.tracked_root / "fresh_h5000_distribution.tsv", distribution)

    rejected = [row for row in decisions if row["status"] == "rejected"]
    reserve = [row for row in decisions if row["status"] == "reserve-not-selected"]
    attestation = [
        {"gate": "selected_count", "observed": len(accepted), "required": TARGET_COUNT, "status": "pass"},
        {"gate": "excluded_union_count", "observed": len(excluded_ids), "required": 1515, "status": "pass"},
        {"gate": "image_id_overlap", "observed": len(selected_ids & (excluded_ids | val_ids)), "required": 0, "status": "pass"},
        {"gate": "exact_file_overlap", "observed": len(selected_file_hashes & (prior_file_hashes | val_file_hashes)), "required": 0, "status": "pass"},
        {"gate": "decoded_pixel_overlap", "observed": len(selected_pixel_hashes & (prior_pixel_hashes | val_pixel_hashes)), "required": 0, "status": "pass"},
        {"gate": "internal_duplicate_image_ids", "observed": TARGET_COUNT - len(selected_ids), "required": 0, "status": "pass"},
        {"gate": "internal_exact_file_duplicates", "observed": TARGET_COUNT - len(selected_file_hashes), "required": 0, "status": "pass"},
        {"gate": "internal_exact_pixel_duplicates", "observed": TARGET_COUNT - len(selected_pixel_hashes), "required": 0, "status": "pass"},
        {"gate": "ranked_rows_examined", "observed": cursor, "required": f">={TARGET_COUNT}", "status": "pass"},
        {"gate": "rejected_ranked_rows", "observed": len(rejected), "required": "recorded", "status": "pass"},
        {"gate": "reserve_not_selected_rows", "observed": len(reserve), "required": "recorded", "status": "pass"},
        {"gate": "quantization_selection_independent", "observed": "yes", "required": "yes", "status": "pass"},
        {"gate": "model_training_independent", "observed": "no-COCO-train2017-lineage", "required": "reported", "status": "pass"},
        {"gate": "final_generalization_authority", "observed": "no-val2017-remains-authority", "required": "reported", "status": "pass"},
        {"gate": "selection_list_sha256", "observed": sha256(tracked_list), "required": "frozen-before-metrics", "status": "pass"},
    ]
    write_tsv(options.tracked_root / "fresh_h5000_partition_attestation.tsv", attestation)
    write_tsv(options.tracked_root / "frozen_model_identity.tsv", artifact_rows)
    write_tsv(options.tracked_root / "protected_state_before.tsv", protected_rows)
    write_tsv(options.tracked_root / "remote_parity_before.tsv", parity_rows)
    write_tsv(
        options.tracked_root / "dev001b_packet_verification.tsv",
        [
            {"field": "tree_sha256", "observed": packet[0], "expected": EXPECTED["packet_tree"], "status": "pass"},
            {"field": "file_count", "observed": packet[1], "expected": EXPECTED["packet_files"], "status": "pass"},
            {"field": "byte_count", "observed": packet[2], "expected": EXPECTED["packet_bytes"], "status": "pass"},
        ],
    )

    policy = {
        "surface": "H5000_C2_ADJUDICATION",
        "count": TARGET_COUNT,
        "policy_version": POLICY,
        "rank_byte_encoding": "utf8(policy)+NUL+ascii(train_annotation_sha256)+NUL+ascii(decimal_image_id)",
        "rank_tie_break": "numeric image_id ascending",
        "selection": "natural rank only; no category, size, prediction, or result optimization",
        "replacement": "reject exact ID/file/pixel overlaps and internal duplicates, then take next rank",
        "pixel_hash": "stage65b-r1-rgb8-v1 NUL + uint64le(width) + uint64le(height) + RGB8 bytes",
        "decoder": f"Pillow {PIL.__version__}",
        "orientation": "ImageOps.exif_transpose",
        "train_annotation_sha256": EXPECTED["train_annotations"],
        "val_annotation_sha256": EXPECTED["val_annotations"],
        "excluded_union_count": len(excluded_ids),
        "selection_list_sha256": sha256(tracked_list),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": 10000,
        "classification": {
            "quantization_selection_independent": True,
            "model_training_independent": False,
            "final_generalization_authority": False,
        },
    }
    (options.tracked_root / "fresh_h5000_selection_policy.json").write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    runtime_python = Path(
        f"/data/k1x-stage-runs/{R1_ID}/host/venv/bin/python"
    )
    runtime = {
        "python": str(runtime_python),
        "python_sha256": sha256(runtime_python.resolve()),
        "runner": str(runner),
        "runner_sha256": sha256(runner),
        "metrics": str(options.banana / "vendor_ort_validation/stage65b_r1_coco_metrics.py"),
        "metrics_sha256": sha256(options.banana / "vendor_ort_validation/stage65b_r1_coco_metrics.py"),
        "preprocess": str(options.banana / "vendor_ort_validation/stage64_preprocess.py"),
        "preprocess_sha256": sha256(options.banana / "vendor_ort_validation/stage64_preprocess.py"),
        "threads": 4,
        "threshold": 0.001,
        "provider": "CPUExecutionProvider",
    }
    (options.tracked_root / "runtime_binding.tsv").write_text(
        "field\tvalue\n" + "".join(f"{key}\t{value}\n" for key, value in runtime.items()),
        encoding="utf-8",
    )

    evidence_index = {
        "tracked_stage": str(options.tracked_root),
        "raw_stage": str(options.raw_root),
        "result_packet": f"/exchange/results/outbox/{STAGE_ID}",
        "shared_log": str(options.shared_log),
        "frozen_models": [str(path) for _, path, _, _ in artifact_paths()],
        "accepted_predictions": {
            "B2_full_val": f"/data/k1x-stage-runs/{R1_ID}/full-matrix/full-coco/B2/predictions.json",
            "A1_full_val": f"/data/k1x-stage-runs/{DEV001A_ID}/candidates/full-val/A1/predictions.json",
        },
        "excluded_or_intermediate_roots": [
            str(options.raw_root / "fresh-h5000/images"),
            str(options.raw_root / "h5000"),
            str(options.raw_root / "bootstrap"),
        ],
    }
    (options.tracked_root / "input_evidence_index.yaml").write_text(
        json.dumps(evidence_index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    launch = {
        "stage_id": STAGE_ID,
        "execution_authority": "direct-user-authorization",
        "host_only": True,
        "banana_start": EXPECTED["banana_head"],
        "banana_tree": EXPECTED["banana_tree"],
        "xslim_start": EXPECTED["xslim_head"],
        "xslim_tree": EXPECTED["xslim_tree"],
        "dev001b_packet": {"tree_sha256": packet[0], "files": packet[1], "bytes": packet[2]},
        "selection_policy": POLICY,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "conditional_full_val": True,
        "model_generation_authorized": False,
        "board_execution_authorized": False,
        "new_branch_authorized": False,
    }
    (options.tracked_root / "effective_launch_manifest.yaml").write_text(
        json.dumps(launch, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (options.tracked_root / "workspace_preflight.md").write_text(
        "# XSLIM-DEV-001C workspace preflight\n\n"
        "- Gate 0: `pass`.\n"
        f"- Banana start/tree: `{EXPECTED['banana_head']}` / `{EXPECTED['banana_tree']}`.\n"
        f"- XSlim unchanged start/tree/version: `{EXPECTED['xslim_head']}` / `{EXPECTED['xslim_tree']}` / `2.1.2+riscy.2.dev2`.\n"
        f"- DEV-001B packet: `{packet[0]}`, {packet[1]} files, {packet[2]} bytes.\n"
        "- B2, A1, C2 and common-tail bytes match the frozen identities.\n"
        "- The accepted 1515-image prior-use train union and all 5000 val2017 image bytes/pixels were revalidated.\n"
        f"- H5000 was frozen before task metrics with policy `{POLICY}` and zero ID/file/pixel overlap.\n"
        "- Protected Banana main, custom executor and `/data/ncnn` match the accepted state.\n",
        encoding="utf-8",
    )
    (options.shared_log / "selection-complete.txt").write_text(
        f"stage_id={STAGE_ID}\npreflight=pass\nselection=pass\nselection_sha256={sha256(tracked_list)}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "selected": len(accepted),
                "examined": cursor,
                "rejected": len(rejected),
                "selection_sha256": sha256(tracked_list),
                "packet": packet,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
