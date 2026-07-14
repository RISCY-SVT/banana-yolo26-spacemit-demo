#!/usr/bin/env python3
"""Build the Stage54 dense-Conv census from a package and measured profile."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_shape(value: str) -> tuple[int, int, int, int]:
    dims = tuple(int(part) for part in value.split("x"))
    if len(dims) != 4:
        raise ValueError(f"expected rank-four shape, got {value!r}")
    return dims  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operations", type=Path, required=True)
    parser.add_argument("--tensors", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    operations = read_tsv(args.operations)
    tensors = {int(row["id"]): row for row in read_tsv(args.tensors)}
    profile = {
        int(row["operation_index"]): row
        for row in read_tsv(args.profile)
        if row["operation_index"].lstrip("-").isdigit()
    }

    fields = [
        "operation_index", "name", "kernel", "stride", "m", "n", "k", "input_c",
        "output_c", "input_h", "input_w", "output_h", "output_w", "macs",
        "current_route", "current_mean_us", "current_p95_us", "packed_weight_bytes",
        "a_panel_bytes_per_worker", "output_bytes", "n_tail", "m_tail",
        "worker_partition", "profile_category",
    ]
    rows: list[dict[str, object]] = []
    for operation in operations:
        if operation["kind"] != "conv_dense":
            continue
        index = int(operation["index"])
        input_id = int(operation["inputs"].split(",")[0])
        output_id = int(operation["output"])
        input_tensor = tensors[input_id]
        output_tensor = tensors[output_id]
        _, input_c, input_h, input_w = parse_shape(input_tensor["shape"])
        _, output_c, output_h, output_w = parse_shape(output_tensor["shape"])
        kernel_h = int(operation["kernel_h"])
        kernel_w = int(operation["kernel_w"])
        stride_h = int(operation["stride_h"])
        stride_w = int(operation["stride_w"])
        m = output_h * output_w
        n = output_c
        k = input_c * kernel_h * kernel_w
        measured = profile.get(index, {})
        packed_name = operation.get("packed_weight_file", "")
        packed_path = args.operations.parent / packed_name if packed_name else None
        packed_bytes = packed_path.stat().st_size if packed_path and packed_path.is_file() else 0
        macs = m * n * k
        rows.append({
            "operation_index": index,
            "name": operation["name"],
            "kernel": f"{kernel_h}x{kernel_w}",
            "stride": f"{stride_h}x{stride_w}",
            "m": m,
            "n": n,
            "k": k,
            "input_c": input_c,
            "output_c": output_c,
            "input_h": input_h,
            "input_w": input_w,
            "output_h": output_h,
            "output_w": output_w,
            "macs": macs,
            "current_route": "rgb_stem_rvv" if input_c == 3 else "packed_m12_n4_n8_n16",
            "current_mean_us": measured.get("mean_us", "unmeasured-resident"),
            "current_p95_us": measured.get("p95_us", "unmeasured-resident"),
            "packed_weight_bytes": packed_bytes,
            "a_panel_bytes_per_worker": ((k + 7) // 8) * 12 * 8,
            "output_bytes": int(output_tensor["storage_bytes"]),
            "n_tail": n % 16,
            "m_tail": m % 12,
            "worker_partition": "spatial_m_tiles",
            "profile_category": measured.get("category", "resident_model4_final_model9"),
        })

    write_tsv(args.out_dir / "dense_shape_census.tsv", fields, rows)
    ranked = sorted(
        rows,
        key=lambda row: float(row["current_mean_us"])
        if str(row["current_mean_us"]).replace(".", "", 1).isdigit() else -1.0,
        reverse=True,
    )
    write_tsv(args.out_dir / "dense_shape_ranked.tsv", fields, ranked)

    categories: dict[tuple[str, str], dict[str, float | int]] = {}
    for row in rows:
        if row["profile_category"] != "dense_conv_outside_resident_region":
            continue
        mean = row["current_mean_us"]
        if not str(mean).replace(".", "", 1).isdigit():
            continue
        key = (str(row["kernel"]), str(row["stride"]))
        aggregate = categories.setdefault(key, {"operations": 0, "macs": 0, "mean_us": 0.0})
        aggregate["operations"] = int(aggregate["operations"]) + 1
        aggregate["macs"] = int(aggregate["macs"]) + int(row["macs"])
        aggregate["mean_us"] = float(aggregate["mean_us"]) + float(mean)
    reconciliation = [
        {
            "kernel": key[0],
            "stride": key[1],
            "operations": value["operations"],
            "macs": value["macs"],
            "mean_us": f"{float(value['mean_us']):.6f}",
        }
        for key, value in sorted(categories.items())
    ]
    reconciliation.append({
        "kernel": "all-profiled-dense-outside",
        "stride": "all",
        "operations": sum(int(row["operations"]) for row in reconciliation),
        "macs": sum(int(row["macs"]) for row in reconciliation),
        "mean_us": f"{sum(float(row['mean_us']) for row in reconciliation):.6f}",
    })
    write_tsv(
        args.out_dir / "dense_category_reconciliation.tsv",
        ["kernel", "stride", "operations", "macs", "mean_us"],
        reconciliation,
    )


if __name__ == "__main__":
    main()
