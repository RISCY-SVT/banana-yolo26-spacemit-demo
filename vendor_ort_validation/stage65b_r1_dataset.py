#!/usr/bin/env python3
"""Audit COCO archives and create deterministic Stage65B-R1 corpus splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import struct
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import PIL
from PIL import Image, ImageOps


POLICY_VERSION = "stage65b-r1-coco-train2017-selection-v1"
VAL_SCOUT_POLICY_VERSION = "stage65b-r1-coco-val2017-scout-v1"
PIXEL_HASH_VERSION = b"stage65b-r1-rgb8-v1\0"
NATURAL_POOL_SIZE = 3000
BALANCED_POOL_SIZE = 1000
COCO_SMALL_MAX = 32 * 32
COCO_MEDIUM_MAX = 96 * 96


@dataclass(frozen=True)
class ImageDigest:
    image_id: int
    file_name: str
    file_sha256: str
    pixel_sha256: str
    dhash64: int
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-zip", required=True, type=Path)
    parser.add_argument("--train-zip", required=True, type=Path)
    parser.add_argument("--val-images", required=True, type=Path)
    parser.add_argument("--val-annotations", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha512_file(path: Path) -> str:
    digest = hashlib.sha512()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_list(path: Path, names: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{name}\n" for name in names), encoding="utf-8")


def safe_archive_members(path: Path) -> tuple[list[zipfile.ZipInfo], list[str]]:
    issues: list[str] = []
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        for member in members:
            pure = PurePosixPath(member.filename)
            mode = (member.external_attr >> 16) & 0xFFFF
            file_type = mode & 0o170000
            if pure.is_absolute() or ".." in pure.parts:
                issues.append(f"unsafe-path:{member.filename}")
            if file_type in {0o120000, 0o060000, 0o020000, 0o010000}:
                issues.append(f"unsafe-file-type:{oct(file_type)}:{member.filename}")
    return members, issues


def extract_annotation(
    archive_path: Path, member_name: str, destination: Path
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo(member_name)
        with archive.open(info) as source, destination.open("wb") as output:
            while chunk := source.read(1024 * 1024):
                output.write(chunk)


def area_bin(area: float) -> str:
    if area < COCO_SMALL_MAX:
        return "small"
    if area < COCO_MEDIUM_MAX:
        return "medium"
    return "large"


def aspect_bin(width: int, height: int) -> str:
    ratio = width / height
    if ratio < 0.75:
        return "portrait"
    if ratio > 4.0 / 3.0:
        return "landscape"
    return "squareish"


def object_count_bin(count: int) -> str:
    if count <= 1:
        return "0-1"
    if count <= 4:
        return "2-4"
    if count <= 10:
        return "5-10"
    return "11+"


def rank_digest(annotation_sha: str, image_id: int) -> str:
    payload = (
        POLICY_VERSION.encode("utf-8")
        + b"\0"
        + annotation_sha.encode("ascii")
        + b"\0"
        + str(image_id).encode("ascii")
    )
    return hashlib.sha256(payload).hexdigest()


def val_rank_digest(annotation_sha: str, image_id: int) -> str:
    payload = (
        VAL_SCOUT_POLICY_VERSION.encode("utf-8")
        + b"\0"
        + annotation_sha.encode("ascii")
        + b"\0"
        + str(image_id).encode("ascii")
    )
    return hashlib.sha256(payload).hexdigest()


def canonical_pixel_digest(path: Path) -> tuple[str, int, int, int]:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        pixels = image.tobytes()
        digest = hashlib.sha256()
        digest.update(PIXEL_HASH_VERSION)
        digest.update(struct.pack("<Q", width))
        digest.update(struct.pack("<Q", height))
        digest.update(pixels)
        grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        values = list(grayscale.getdata())
        dhash = 0
        for row in range(8):
            offset = row * 9
            for column in range(8):
                dhash = (dhash << 1) | int(
                    values[offset + column] > values[offset + column + 1]
                )
    return digest.hexdigest(), dhash, width, height


def digest_image(image_id: int, file_name: str, path: Path) -> ImageDigest:
    file_hash = sha256_file(path)
    pixel_hash, dhash, width, height = canonical_pixel_digest(path)
    return ImageDigest(
        image_id=image_id,
        file_name=file_name,
        file_sha256=file_hash,
        pixel_sha256=pixel_hash,
        dhash64=dhash,
        width=width,
        height=height,
    )


def digest_images(
    images: list[tuple[int, str, Path]], workers: int
) -> list[ImageDigest]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda item: digest_image(*item), images))


def annotation_index(payload: dict[str, Any]) -> tuple[
    dict[int, list[dict[str, Any]]], dict[int, dict[str, Any]], dict[int, dict[str, Any]]
]:
    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        if annotation.get("iscrowd", 0) == 0 and annotation.get("bbox"):
            annotations[int(annotation["image_id"])].append(annotation)
    images = {int(item["id"]): item for item in payload["images"]}
    licenses = {int(item["id"]): item for item in payload["licenses"]}
    return annotations, images, licenses


def enriched_image(
    image: dict[str, Any], annotations: list[dict[str, Any]], annotation_sha: str
) -> dict[str, Any]:
    scale = min(640.0 / float(image["width"]), 640.0 / float(image["height"]))
    original_bins: Counter[str] = Counter()
    input_bins: Counter[str] = Counter()
    categories: set[int] = set()
    for annotation in annotations:
        area = float(annotation.get("area") or annotation["bbox"][2] * annotation["bbox"][3])
        original_bins[area_bin(area)] += 1
        input_bins[area_bin(float(annotation["bbox"][2] * annotation["bbox"][3]) * scale * scale)] += 1
        categories.add(int(annotation["category_id"]))
    return {
        "image_id": int(image["id"]),
        "file_name": image["file_name"],
        "width": int(image["width"]),
        "height": int(image["height"]),
        "coco_url": image.get("coco_url", ""),
        "flickr_url": image.get("flickr_url", ""),
        "license_id": int(image["license"]),
        "annotation_count": len(annotations),
        "category_ids": ",".join(map(str, sorted(categories))),
        "small_original": original_bins["small"],
        "medium_original": original_bins["medium"],
        "large_original": original_bins["large"],
        "small_letterboxed": input_bins["small"],
        "medium_letterboxed": input_bins["medium"],
        "large_letterboxed": input_bins["large"],
        "aspect_bin": aspect_bin(int(image["width"]), int(image["height"])),
        "object_count_bin": object_count_bin(len(annotations)),
        "rank_sha256": rank_digest(annotation_sha, int(image["id"])),
    }


def balanced_pool(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    remaining = list(rows)
    selected: list[dict[str, Any]] = []
    bin_counts = Counter()
    categories: set[int] = set()
    aspects: set[str] = set()
    object_bins: set[str] = set()
    licenses: set[int] = set()
    quota = 167
    while remaining and len(selected) < count:
        def score(row: dict[str, Any]) -> tuple[Any, ...]:
            present = {
                name
                for name in ("small", "medium", "large")
                if int(row[f"{name}_original"]) > 0
            }
            deficit_gain = sum(bin_counts[name] < quota for name in present)
            row_categories = {int(value) for value in row["category_ids"].split(",") if value}
            return (
                deficit_gain,
                len(row_categories - categories),
                int(row["aspect_bin"] not in aspects),
                int(row["object_count_bin"] not in object_bins),
                int(int(row["license_id"]) not in licenses),
                -int(row["rank_sha256"], 16),
            )

        winner = max(remaining, key=score)
        remaining.remove(winner)
        selected.append(winner)
        for name in ("small", "medium", "large"):
            if int(winner[f"{name}_original"]) > 0:
                bin_counts[name] += 1
        categories.update(int(value) for value in winner["category_ids"].split(",") if value)
        aspects.add(str(winner["aspect_bin"]))
        object_bins.add(str(winner["object_count_bin"]))
        licenses.add(int(winner["license_id"]))
        if (
            min(bin_counts[name] for name in ("small", "medium", "large")) >= quota
            and len(categories) == 80
            and len(aspects) == len({str(row["aspect_bin"]) for row in rows})
            and len(object_bins) == len({str(row["object_count_bin"]) for row in rows})
            and len(licenses) == len({int(row["license_id"]) for row in rows})
        ):
            selected.extend(remaining[: count - len(selected)])
            break
    return selected


def extract_train_images(
    archive_path: Path, names: set[str], destination: Path
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        lookup = {PurePosixPath(info.filename).name: info for info in archive.infolist()}
        missing = sorted(names - lookup.keys())
        if missing:
            raise RuntimeError(f"train archive misses {len(missing)} selected files")
        for index, name in enumerate(sorted(names), 1):
            target = destination / name
            if target.is_file():
                continue
            with archive.open(lookup[name]) as source, target.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
            if index % 250 == 0:
                print(f"extracted {index}/{len(names)}", flush=True)


def main() -> int:
    options = parse_args()
    evidence = options.evidence_root
    dataset_root = options.dataset_root
    annotations_dir = dataset_root / "annotations"
    selected_dir = dataset_root / "selected" / "images"
    evidence.mkdir(parents=True, exist_ok=True)

    annotation_members, annotation_issues = safe_archive_members(options.annotations_zip)
    train_members, train_issues = safe_archive_members(options.train_zip)
    if annotation_issues or train_issues:
        raise RuntimeError(f"archive safety failure: {annotation_issues + train_issues}")
    train_jpegs = [item for item in train_members if item.filename.lower().endswith(".jpg")]
    if len(train_jpegs) != 118287:
        raise RuntimeError(f"unexpected train2017 JPEG count: {len(train_jpegs)}")

    train_annotation_path = annotations_dir / "instances_train2017.json"
    val_annotation_copy = annotations_dir / "instances_val2017.json"
    extract_annotation(
        options.annotations_zip,
        "annotations/instances_train2017.json",
        train_annotation_path,
    )
    extract_annotation(
        options.annotations_zip,
        "annotations/instances_val2017.json",
        val_annotation_copy,
    )
    annotation_sha = sha256_file(train_annotation_path)
    train_payload = json.loads(train_annotation_path.read_text(encoding="utf-8"))
    annotations, images, licenses = annotation_index(train_payload)
    if len(images) != 118287 or len(train_payload["categories"]) != 80:
        raise RuntimeError(
            f"unexpected COCO train identity images={len(images)} categories={len(train_payload['categories'])}"
        )
    if not train_payload["annotations"] or not licenses:
        raise RuntimeError("COCO annotations or licenses are empty")
    orphan_annotation_ids = {
        int(item["image_id"])
        for item in train_payload["annotations"]
        if int(item["image_id"]) not in images
    }
    if orphan_annotation_ids:
        raise RuntimeError(
            f"{len(orphan_annotation_ids)} annotation image IDs are unresolved"
        )

    all_rows = [
        enriched_image(image, annotations.get(image_id, []), annotation_sha)
        for image_id, image in images.items()
    ]
    rows = [row for row in all_rows if int(row["annotation_count"]) > 0]
    rows.sort(key=lambda row: row["rank_sha256"])
    all_rows.sort(key=lambda row: row["rank_sha256"])
    metadata_fields = list(all_rows[0])
    write_tsv(evidence / "coco_image_metadata_all.tsv", all_rows, metadata_fields)
    write_tsv(
        evidence / "selection_rank.tsv",
        (
            {
                "rank": rank,
                "image_id": row["image_id"],
                "file_name": row["file_name"],
                "rank_sha256": row["rank_sha256"],
            }
            for rank, row in enumerate(rows, 1)
        ),
        ["rank", "image_id", "file_name", "rank_sha256"],
    )

    natural_pool = rows[:NATURAL_POOL_SIZE]
    balanced_candidates = balanced_pool(rows, BALANCED_POOL_SIZE)
    extract_names = {
        row["file_name"] for row in [*natural_pool, *balanced_candidates]
    }
    extract_train_images(options.train_zip, extract_names, selected_dir)

    val_payload = json.loads(options.val_annotations.read_text(encoding="utf-8"))
    val_annotation_sha = sha256_file(options.val_annotations)
    if sha256_file(val_annotation_copy) != val_annotation_sha:
        raise RuntimeError("local val2017 annotations differ from official archive")
    val_images = {
        int(item["id"]): item for item in val_payload["images"]
    }
    train_val_id_overlap = sorted(set(images) & set(val_images))
    if train_val_id_overlap:
        raise RuntimeError(
            f"COCO train/val image ID overlap: {len(train_val_id_overlap)}"
        )
    val_inputs = [
        (image_id, item["file_name"], options.val_images / item["file_name"])
        for image_id, item in sorted(val_images.items())
    ]
    train_rows_by_id = {int(row["image_id"]): row for row in rows}
    train_inputs = [
        (int(row["image_id"]), row["file_name"], selected_dir / row["file_name"])
        for row in sorted(
            {int(item["image_id"]): item for item in [*natural_pool, *balanced_candidates]}.values(),
            key=lambda item: item["rank_sha256"],
        )
    ]
    val_digests = digest_images(val_inputs, options.workers)
    train_digests = digest_images(train_inputs, options.workers)
    val_file_hashes = {item.file_sha256 for item in val_digests}
    val_pixel_hashes = {item.pixel_sha256 for item in val_digests}
    valid_train: list[ImageDigest] = []
    seen_file: dict[str, int] = {}
    seen_pixel: dict[str, int] = {}
    exact_file_overlap: list[dict[str, Any]] = []
    exact_pixel_overlap: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for item in train_digests:
        if item.file_sha256 in val_file_hashes:
            exact_file_overlap.append({"image_id": item.image_id, "file_name": item.file_name, "sha256": item.file_sha256})
            continue
        if item.pixel_sha256 in val_pixel_hashes:
            exact_pixel_overlap.append({"image_id": item.image_id, "file_name": item.file_name, "sha256": item.pixel_sha256})
            continue
        prior_file = seen_file.get(item.file_sha256)
        prior_pixel = seen_pixel.get(item.pixel_sha256)
        if prior_file is not None or prior_pixel is not None:
            duplicates.append(
                {
                    "image_id": item.image_id,
                    "file_name": item.file_name,
                    "file_duplicate_of": prior_file or "",
                    "pixel_duplicate_of": prior_pixel or "",
                }
            )
            continue
        seen_file[item.file_sha256] = item.image_id
        seen_pixel[item.pixel_sha256] = item.image_id
        valid_train.append(item)

    valid_by_id = {item.image_id: item for item in valid_train}
    natural_valid = [
        row for row in natural_pool if int(row["image_id"]) in valid_by_id
    ]
    if len(natural_valid) < 1500:
        raise RuntimeError(f"only {len(natural_valid)} qualified natural images")
    selections = {
        "C50": natural_valid[:50],
        "C200": natural_valid[:200],
        "C500": natural_valid[:500],
        "C1000": natural_valid[:1000],
        "H500": natural_valid[1000:1500],
    }
    holdout_ids = {int(row["image_id"]) for row in selections["H500"]}
    balanced_valid = [
        row
        for row in balanced_candidates
        if int(row["image_id"]) in valid_by_id
        and int(row["image_id"]) not in holdout_ids
    ]
    if len(balanced_valid) < 500:
        raise RuntimeError(f"only {len(balanced_valid)} qualified balanced images")
    selections["BALANCED"] = balanced_valid[:500]
    balanced_bin_counts = {
        name: sum(int(row[f"{name}_original"]) > 0 for row in selections["BALANCED"])
        for name in ("small", "medium", "large")
    }
    if min(balanced_bin_counts.values()) < 167:
        raise RuntimeError(f"balanced quota failure: {balanced_bin_counts}")
    holdout_file_hashes = {
        valid_by_id[int(row["image_id"])].file_sha256
        for row in selections["H500"]
    }
    holdout_pixel_hashes = {
        valid_by_id[int(row["image_id"])].pixel_sha256
        for row in selections["H500"]
    }
    holdout_overlap_rows: list[dict[str, Any]] = []
    for key, values in selections.items():
        if key == "H500":
            continue
        for row in values:
            digest = valid_by_id[int(row["image_id"])]
            if (
                int(row["image_id"]) in holdout_ids
                or digest.file_sha256 in holdout_file_hashes
                or digest.pixel_sha256 in holdout_pixel_hashes
            ):
                holdout_overlap_rows.append(
                    {
                        "selection": key,
                        "image_id": row["image_id"],
                        "file_name": row["file_name"],
                    }
                )
    if holdout_overlap_rows:
        raise RuntimeError(
            f"calibration/holdout overlap: {len(holdout_overlap_rows)}"
        )
    write_tsv(
        evidence / "calibration_holdout_overlap.tsv",
        holdout_overlap_rows,
        ["selection", "image_id", "file_name"],
    )

    digest_fields = [
        "image_id", "file_name", "file_sha256", "pixel_sha256", "dhash64", "width", "height"
    ]
    digest_rows = lambda values: [
        {
            "image_id": item.image_id,
            "file_name": item.file_name,
            "file_sha256": item.file_sha256,
            "pixel_sha256": item.pixel_sha256,
            "dhash64": f"{item.dhash64:016x}",
            "width": item.width,
            "height": item.height,
        }
        for item in values
    ]
    write_tsv(evidence / "val2017_file_hashes.tsv", digest_rows(val_digests), digest_fields)
    write_tsv(evidence / "val2017_pixel_hashes.tsv", digest_rows(val_digests), digest_fields)
    write_tsv(evidence / "train_candidate_file_hashes.tsv", digest_rows(train_digests), digest_fields)
    write_tsv(evidence / "train_candidate_pixel_hashes.tsv", digest_rows(train_digests), digest_fields)
    write_tsv(evidence / "overlap_exact_file.tsv", exact_file_overlap, ["image_id", "file_name", "sha256"])
    write_tsv(evidence / "overlap_exact_pixels.tsv", exact_pixel_overlap, ["image_id", "file_name", "sha256"])
    write_tsv(evidence / "duplicate_report.tsv", duplicates, ["image_id", "file_name", "file_duplicate_of", "pixel_duplicate_of"])

    val_ranked = sorted(
        val_images.values(),
        key=lambda item: val_rank_digest(val_annotation_sha, int(item["id"])),
    )
    write_list(
        evidence / "scout500_list.txt",
        (item["file_name"] for item in val_ranked[:500]),
    )
    write_list(
        dataset_root / "lists" / "scout500_list.txt",
        (str((options.val_images / item["file_name"]).resolve()) for item in val_ranked[:500]),
    )
    write_list(
        dataset_root / "lists" / "val2017_all.txt",
        (
            str((options.val_images / item["file_name"]).resolve())
            for item in sorted(val_images.values(), key=lambda item: int(item["id"]))
        ),
    )

    near_rows: list[dict[str, Any]] = []
    for train in valid_train:
        for val in val_digests:
            distance = (train.dhash64 ^ val.dhash64).bit_count()
            if distance <= 1:
                near_rows.append(
                    {
                        "train_image_id": train.image_id,
                        "train_file": train.file_name,
                        "val_image_id": val.image_id,
                        "val_file": val.file_name,
                        "dhash_hamming": distance,
                        "classification": "warning-review-only",
                    }
                )
    write_tsv(
        evidence / "near_duplicate_review.tsv",
        near_rows,
        ["train_image_id", "train_file", "val_image_id", "val_file", "dhash_hamming", "classification"],
    )

    names = {
        "C50": "selection_C50.txt",
        "C200": "selection_C200.txt",
        "C500": "selection_C500.txt",
        "C1000": "selection_C1000.txt",
        "BALANCED": "selection_C500_size_balanced.txt",
        "H500": "selection_H500_holdout.txt",
    }
    for key, filename in names.items():
        write_list(evidence / filename, (row["file_name"] for row in selections[key]))
        write_list(
            dataset_root / "lists" / filename,
            (str((selected_dir / row["file_name"]).resolve()) for row in selections[key]),
        )

    membership: list[dict[str, Any]] = []
    selected_ids = sorted({int(row["image_id"]) for values in selections.values() for row in values})
    for image_id in selected_ids:
        row = train_rows_by_id[image_id]
        digest = valid_by_id[image_id]
        membership.append(
            {
                **row,
                "license_name": licenses[int(row["license_id"])].get("name", ""),
                "license_url": licenses[int(row["license_id"])].get("url", ""),
                "file_sha256": digest.file_sha256,
                "pixel_sha256": digest.pixel_sha256,
                **{key: int(any(int(item["image_id"]) == image_id for item in values)) for key, values in selections.items()},
            }
        )
    membership_fields = list(membership[0])
    write_tsv(evidence / "selection_membership.tsv", membership, membership_fields)

    selection_hash_rows: list[dict[str, Any]] = []
    for key, filename in names.items():
        portable = evidence / filename
        effective = dataset_root / "lists" / filename
        selection_hash_rows.append(
            {
                "selection": key,
                "count": len(selections[key]),
                "portable_list_sha256": sha256_file(portable),
                "effective_list_sha256": sha256_file(effective),
                "membership_digest": hashlib.sha256(
                    b"".join(
                        f"{int(row['image_id'])}\n".encode("ascii")
                        for row in selections[key]
                    )
                ).hexdigest(),
            }
        )
    write_tsv(
        evidence / "selection_hashes.tsv",
        selection_hash_rows,
        ["selection", "count", "portable_list_sha256", "effective_list_sha256", "membership_digest"],
    )

    def distribution(field: str, output: str) -> None:
        out_rows: list[dict[str, Any]] = []
        for key, values in selections.items():
            counter = Counter(str(row[field]) for row in values)
            for value, count in sorted(counter.items()):
                out_rows.append({"selection": key, field: value, "image_count": count})
        write_tsv(evidence / output, out_rows, ["selection", field, "image_count"])

    size_original: list[dict[str, Any]] = []
    size_letterboxed: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    for key, values in selections.items():
        for name in ("small", "medium", "large"):
            size_original.append(
                {"selection": key, "size_bin": name, "image_count": sum(int(row[f"{name}_original"]) > 0 for row in values), "object_count": sum(int(row[f"{name}_original"]) for row in values)}
            )
            size_letterboxed.append(
                {"selection": key, "size_bin": name, "image_count": sum(int(row[f"{name}_letterboxed"]) > 0 for row in values), "object_count": sum(int(row[f"{name}_letterboxed"]) for row in values)}
            )
        counter = Counter(
            category
            for row in values
            for category in (int(value) for value in row["category_ids"].split(",") if value)
        )
        for category, count in sorted(counter.items()):
            category_rows.append({"selection": key, "category_id": category, "image_count": count})
    write_tsv(evidence / "size_bin_distribution_original.tsv", size_original, ["selection", "size_bin", "image_count", "object_count"])
    write_tsv(evidence / "size_bin_distribution_letterboxed.tsv", size_letterboxed, ["selection", "size_bin", "image_count", "object_count"])
    write_tsv(evidence / "category_distribution.tsv", category_rows, ["selection", "category_id", "image_count"])
    distribution("aspect_bin", "aspect_ratio_distribution.tsv")
    distribution("license_id", "license_distribution.tsv")

    license_rows = [
        {"license_id": key, "name": value.get("name", ""), "url": value.get("url", "")}
        for key, value in sorted(licenses.items())
    ]
    write_tsv(evidence / "coco_license_table.tsv", license_rows, ["license_id", "name", "url"])
    archive_rows = []
    for role, path, members, jpegs in (
        ("annotations", options.annotations_zip, annotation_members, 0),
        ("train2017", options.train_zip, train_members, len(train_jpegs)),
    ):
        archive_rows.append(
            {
                "role": role,
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "sha512": sha512_file(path),
                "member_count": len(members),
                "jpeg_count": jpegs,
                "safety": "pass",
            }
        )
    write_tsv(evidence / "coco_archive_identity.tsv", archive_rows, list(archive_rows[0]))
    annotation_rows = [
        {
            "file": train_annotation_path.name,
            "sha256": annotation_sha,
            "images": len(train_payload["images"]),
            "annotations": len(train_payload["annotations"]),
            "categories": len(train_payload["categories"]),
            "licenses": len(train_payload["licenses"]),
            "qualified_images": len(rows),
        }
    ]
    write_tsv(evidence / "coco_annotation_identity.tsv", annotation_rows, list(annotation_rows[0]))

    policy = {
        "policy_version": POLICY_VERSION,
        "val_scout_policy_version": VAL_SCOUT_POLICY_VERSION,
        "rank_byte_encoding": "utf8(policy_version)+NUL+ascii(annotation_sha256)+NUL+ascii(decimal_image_id)",
        "pixel_hash_version": PIXEL_HASH_VERSION.decode("ascii", errors="replace").rstrip("\0"),
        "pixel_hash_encoding": "prefix+uint64le(width)+uint64le(height)+contiguous_RGB8_bytes",
        "decoder": f"Pillow {PIL.__version__}",
        "orientation": "ImageOps.exif_transpose",
        "mode": "RGB8",
        "near_duplicate": "64-bit dHash, Hamming <= 1, warning only",
        "natural_pool_size": NATURAL_POOL_SIZE,
        "balanced_pool_size": BALANCED_POOL_SIZE,
        "balanced_min_images_per_original_coco_size_bin": 167,
        "balanced_observed": balanced_bin_counts,
        "classification": "internal-rd-only; no image redistribution; no commercial-clearance claim",
    }
    (evidence / "selection_policy.json").write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (evidence / "hash_contract.md").write_text(
        "# Hash contract\n\n"
        f"Decoder: Pillow `{PIL.__version__}`. Orientation is normalized with "
        "`ImageOps.exif_transpose`, then converted to contiguous row-major RGB8. "
        "The canonical digest hashes `stage65b-r1-rgb8-v1\\0`, little-endian "
        "uint64 width and height, and the RGB bytes. Exact JPEG SHA-256 is kept "
        "separately. A 64-bit 9x8 grayscale dHash with Hamming distance <= 1 is "
        "a review warning only.\n",
        encoding="utf-8",
    )
    (evidence / "coco_archive_safety.md").write_text(
        "# COCO archive safety\n\n"
        f"Annotations members: {len(annotation_members)}. Train members: {len(train_members)}; "
        f"JPEG entries: {len(train_jpegs)}. Absolute paths, traversal, symlinks, "
        "hard links, and device nodes: 0. Full CRC integrity is recorded separately.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "qualified_rows": len(rows),
        "extracted_images": len(extract_names),
        "valid_digests": len(valid_train),
        "file_overlap": len(exact_file_overlap),
        "pixel_overlap": len(exact_pixel_overlap),
        "duplicates": len(duplicates),
        "near_duplicate_warnings": len(near_rows),
        "balanced_bins": balanced_bin_counts,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
