#!/usr/bin/env python3
"""Prepare the bounded Stage65B-R2 B2 variance configuration set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from stage65b_r2_common import sha256, write_tsv


ORDER_POLICY = "stage65b-r2-vorder-sha256-permutation-v1"
DRAW_POLICY = "stage65b-r2-vdraw-natural-ranks-51-through-100-v1"


def read_paths(path: Path) -> list[Path]:
    rows = [Path(line.strip()).resolve() for line in path.read_text().splitlines() if line.strip()]
    if not rows or len(rows) != len(set(rows)):
        raise ValueError(f"empty or duplicate image list: {path}")
    for row in rows:
        if not row.is_file():
            raise FileNotFoundError(row)
    return rows


def image_ids(paths: list[Path]) -> set[int]:
    try:
        return {int(path.stem) for path in paths}
    except ValueError as exc:
        raise ValueError("all variance images must have numeric COCO stems") from exc


def list_sha(paths: list[Path]) -> str:
    payload = "".join(f"{path}\n" for path in paths).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_list(path: Path, rows: list[Path]) -> None:
    path.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")


def ranked_draw(rank_path: Path, image_root: Path) -> list[Path]:
    with rank_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) < 100:
        raise ValueError("selection rank has fewer than 100 entries")
    selected = rows[50:100]
    if [int(row["rank"]) for row in selected] != list(range(51, 101)):
        raise ValueError("selection rank is not contiguous at ranks 51-100")
    result = [(image_root / row["file_name"]).resolve() for row in selected]
    for path in result:
        if not path.is_file():
            raise FileNotFoundError(path)
    return result


def variant_config(base: dict[str, Any], lane: str, image_list: Path) -> dict[str, Any]:
    result = json.loads(json.dumps(base))
    result["model_parameters"]["output_prefix"] = (
        f"stage65b_r2_{lane.lower()}_split_s8_qdq"
    )
    result["model_parameters"]["working_dir"] = "STAGE65B_R2_DRIVER_REPLACES_THIS"
    parameters = result["calibration_parameters"]["input_parameters"]
    if len(parameters) != 1:
        raise ValueError("B2 must have exactly one calibration input")
    parameters[0]["data_list_path"] = str(image_list.resolve())
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--selection-rank", required=True, type=Path)
    parser.add_argument("--c50-list", required=True, type=Path)
    parser.add_argument("--h500-list", required=True, type=Path)
    parser.add_argument("--val-list", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    options = parser.parse_args()
    if options.output_dir.exists():
        raise RuntimeError(f"refusing to reuse output directory: {options.output_dir}")
    options.output_dir.mkdir(parents=True)

    base = json.loads(options.base_config.read_text(encoding="utf-8"))
    c50 = read_paths(options.c50_list)
    h500 = read_paths(options.h500_list)
    val = read_paths(options.val_list)
    if len(c50) != 50 or len(h500) != 500 or len(val) != 5000:
        raise ValueError("unexpected C50, H500, or val2017 list size")

    order_seed = 65003
    order_key = lambda path: hashlib.sha256(  # noqa: E731
        f"{ORDER_POLICY}\0{order_seed}\0{int(path.stem)}".encode("ascii")
    ).digest()
    vorder = sorted(c50, key=order_key)
    if set(vorder) != set(c50) or vorder == c50:
        raise RuntimeError("Vorder is not a nontrivial exact C50 permutation")
    vdraw = ranked_draw(options.selection_rank, options.image_root)

    h500_ids = image_ids(h500)
    val_ids = image_ids(val)
    variants = {
        "Vseed": (c50, 65002, "same-membership-same-order-seed-only"),
        "Vorder": (vorder, 65001, ORDER_POLICY),
        "Vdraw": (vdraw, 65001, DRAW_POLICY),
    }
    manifest: list[dict[str, Any]] = []
    configs = options.output_dir / "configs"
    lists = options.output_dir / "lists"
    configs.mkdir()
    lists.mkdir()
    for lane, (members, xslim_seed, policy) in variants.items():
        ids = image_ids(members)
        overlap_h500 = ids & h500_ids
        overlap_val = ids & val_ids
        if overlap_h500 or overlap_val or len(ids) != 50:
            raise RuntimeError(
                f"{lane} overlap/uniqueness failure: "
                f"H500={len(overlap_h500)} val={len(overlap_val)} unique={len(ids)}"
            )
        list_path = lists / f"{lane}.txt"
        write_list(list_path, members)
        config_path = configs / f"{lane}.json"
        config_path.write_text(
            json.dumps(variant_config(base, lane, list_path), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        manifest.append(
            {
                "lane": lane,
                "changed_factor": {
                    "Vseed": "xslim-seed",
                    "Vorder": "calibration-order",
                    "Vdraw": "calibration-membership",
                }[lane],
                "policy": policy,
                "xslim_seed": xslim_seed,
                "permutation_seed": order_seed if lane == "Vorder" else "NA",
                "count": len(members),
                "list_path": str(list_path.resolve()),
                "list_sha256": sha256(list_path),
                "list_payload_sha256": list_sha(members),
                "membership_ids_sha256": hashlib.sha256(
                    "\n".join(map(str, sorted(ids))).encode("ascii") + b"\n"
                ).hexdigest(),
                "same_c50_membership": int(ids == image_ids(c50)),
                "same_c50_order": int(members == c50),
                "h500_id_overlap": len(overlap_h500),
                "val2017_id_overlap": len(overlap_val),
                "config_path": str(config_path.resolve()),
                "config_sha256": sha256(config_path),
                "python_version": sys.version.split()[0],
                "status": "pass",
            }
        )
    write_tsv(options.output_dir / "b2_variance_matrix.tsv", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
