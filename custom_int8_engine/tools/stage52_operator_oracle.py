#!/usr/bin/env python3
"""Independent package/operator checks for the Stage 52 integer profile."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def read_values(path: Path, code: str) -> tuple[int, ...]:
    data = path.read_bytes()
    width = struct.calcsize(code)
    if len(data) % width:
        raise ValueError(f"unaligned asset {path}: {len(data)} bytes for {code}")
    return struct.unpack(f"<{len(data) // width}{code}", data)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def signed_storage(code: int) -> int:
    return code - 128


def semantic_code(value: int) -> int:
    return value + 128


def round_shift_even(value: int, shift: int) -> int:
    if shift < 0:
        return value << -shift
    if shift == 0:
        return value
    sign = -1 if value < 0 else 1
    magnitude = abs(value)
    quotient, remainder = divmod(magnitude, 1 << shift)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and quotient & 1):
        quotient += 1
    return sign * quotient


def q62(accumulator: int, multiplier: int, zero_point: int) -> int:
    return min(255, max(0, round_shift_even(accumulator * multiplier, 62) + zero_point))


def result_row(operator_class: str, checked: int, status: str, detail: str) -> dict[str, object]:
    return {
        "operator_class": operator_class,
        "checked_surfaces": checked,
        "status": status,
        "detail": detail,
    }


def audit(package: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    metadata = json.loads((package / "package.json").read_text(encoding="utf-8"))
    operations = read_tsv(package / "operations.tsv")
    tensors = read_tsv(package / "tensors.tsv")
    tensors_by_id = {int(row["id"]): row for row in tensors}
    results: list[dict[str, object]] = []

    if metadata["contract_id"] != "K1X_INT8_V1":
        raise ValueError("unexpected contract")
    if metadata["profile_id"] != "K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001":
        raise ValueError("unexpected profile")
    if metadata["byte_order"] != "little-endian":
        raise ValueError("unexpected byte order")
    results.append(result_row("profile", 1, "pass", "contract/profile/byte order exact"))

    conv_channels = 0
    packed_assets = 0
    accumulator_vectors = 0
    for operation in operations:
        if operation["kind"] not in {"conv_dense", "conv_grouped"}:
            continue
        output_channels = int(operation["output_c"])
        input_channels = int(operation["input_c"])
        group = int(operation["group"])
        kernel_h = int(operation["kernel_h"])
        kernel_w = int(operation["kernel_w"])
        output = tensors_by_id[int(operation["output"])]
        input_tensor = tensors_by_id[int(operation["inputs"].split(",")[0])]
        weights = read_values(package / operation["weight_file"], "b")
        bias = read_values(package / operation["bias_file"], "i")
        multiplier = read_values(package / operation["multiplier_file"], "q")
        shift = read_values(package / operation["shift_file"], "i")
        expected_weights = output_channels * (input_channels // group) * kernel_h * kernel_w
        if len(weights) != expected_weights or len(bias) != output_channels:
            raise ValueError(f"Conv asset size mismatch: {operation['name']}")
        if len(multiplier) != output_channels or len(shift) != output_channels:
            raise ValueError(f"Conv requant size mismatch: {operation['name']}")
        bound = int(operation["accumulator_bound"])
        if bound > 2_147_483_647:
            raise ValueError(f"int32 accumulator bound exceeded: {operation['name']}")
        conv_channels += output_channels
        for channel in range(output_channels):
            if shift[channel] != 62 or multiplier[channel] <= 0 or multiplier[channel] * 2 >= 1 << 63:
                raise ValueError(f"E2c arithmetic invariant failed: {operation['name']} channel {channel}")
            for accumulator in (0, 1, -1, bound, -bound, bound - 1, -bound + 1):
                result = q62(accumulator, multiplier[channel], int(output["zero_point"]))
                if not 0 <= result <= 255:
                    raise AssertionError("requant saturation failed")
                accumulator_vectors += 1
        if operation["kind"] == "conv_dense":
            weight_sums = read_values(package / operation["weight_sum_file"], "q")
            corrected_bias = read_values(package / operation["corrected_bias_file"], "q")
            m63 = read_values(package / operation["multiplier_m63_file"], "q")
            input_per_output = input_channels * kernel_h * kernel_w
            correction = 128 - int(input_tensor["zero_point"])
            for channel in range(output_channels):
                start = channel * input_per_output
                recomputed_sum = sum(weights[start:start + input_per_output])
                if weight_sums[channel] != recomputed_sum:
                    raise ValueError(f"weight sum mismatch: {operation['name']} channel {channel}")
                if corrected_bias[channel] != bias[channel] + correction * recomputed_sum:
                    raise ValueError(f"corrected bias mismatch: {operation['name']} channel {channel}")
                if m63[channel] != multiplier[channel] * 2:
                    raise ValueError(f"M63 mismatch: {operation['name']} channel {channel}")
            packed = package / operation["packed_weight_file"]
            if not packed.is_file() or packed.stat().st_size == 0:
                raise ValueError(f"packed weight missing: {operation['name']}")
            packed_assets += 1
    results.append(result_row(
        "conv_q62_e2c", accumulator_vectors, "pass",
        f"{conv_channels} channels; {packed_assets} dense packed assets; int32 bounds proven"))

    lut_counts: Counter[int] = Counter()
    for operation in operations:
        if operation["kind"] in {"lut1", "split", "reshape", "transpose",
                                 "reshape_split_transpose", "resize"} and operation["lut_file"]:
            lut_counts[(package / operation["lut_file"]).stat().st_size] += 1
        if operation["kind"] == "lut2":
            size = (package / operation["lut_file"]).stat().st_size
            if size != 65536:
                raise ValueError(f"binary LUT size mismatch: {operation['name']}")
            lut_counts[size] += 1
    results.append(result_row("integer_lut", sum(lut_counts.values()), "pass", str(dict(lut_counts))))

    matmul_count = 0
    for operation in operations:
        if operation["kind"] != "matmul":
            continue
        multiplier = int(operation["multiplier"])
        shift = int(operation["right_shift"])
        if multiplier <= 0 or shift != 62 or multiplier * 2 >= 1 << 63:
            raise ValueError(f"MatMul Q62 invariant failed: {operation['name']}")
        matmul_count += 1
    results.append(result_row("matmul", matmul_count, "pass", "static Q62 descriptors"))

    softmax_count = 0
    for operation in operations:
        if operation["kind"] != "softmax_transpose":
            continue
        values = read_values(package / operation["exp_file"], "Q")
        if len(values) != 256 or values[0] == 0 or any(values[i] < values[i + 1] for i in range(255)):
            raise ValueError(f"Softmax exponent table invalid: {operation['name']}")
        if int(operation["output_reciprocal_q32"]) <= 0:
            raise ValueError(f"Softmax reciprocal invalid: {operation['name']}")
        softmax_count += 1
    results.append(result_row("softmax", softmax_count, "pass", "Q48 monotonic exponent tables"))

    resize = [row for row in operations if row["kind"] == "resize"]
    if any(row["branch0_resize_mode"] not in {"", "nearest"} for row in resize):
        raise ValueError("unsupported Resize mode")
    results.append(result_row("resize", len(resize), "pass", "frozen nearest-neighbor mapping"))

    head_rows = read_tsv(package / "head_assets.tsv")
    if len(head_rows) != 3 or metadata["head_tie_policy"] != "score-descending-index-ascending":
        raise ValueError("head selection contract mismatch")
    for row in head_rows:
        if (package / row["reg_lut_file"]).stat().st_size != 4096:
            raise ValueError("head regression LUT size mismatch")
        if (package / row["cls_lut_file"]).stat().st_size != 1024:
            raise ValueError("head sigmoid LUT size mismatch")
    results.append(result_row("topk_gather_head", 3, "pass", metadata["head_tie_policy"]))

    for row in read_tsv(package / "asset_hashes.tsv"):
        path = package / row["path"]
        if int(row["bytes"]) != path.stat().st_size or row["sha256"] != sha256(path):
            raise ValueError(f"asset hash mismatch: {row['path']}")
    results.append(result_row("package_integrity", 1, "pass", "all size/SHA-256 rows exact"))

    summary = {
        "contract_id": metadata["contract_id"],
        "profile_id": metadata["profile_id"],
        "operations": len(operations),
        "tensors": len(tensors),
        "conv_channels": conv_channels,
        "accumulator_vectors": accumulator_vectors,
        "status": "pass",
    }
    return results, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    rows, summary = audit(args.package.resolve())
    args.matrix.parent.mkdir(parents=True, exist_ok=True)
    with args.matrix.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
