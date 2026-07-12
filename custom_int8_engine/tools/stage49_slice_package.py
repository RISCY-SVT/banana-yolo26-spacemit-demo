#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import onnx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage47_executor_assets import (  # noqa: E402
    GraphIndex,
    MODEL4_POSTACT,
    MODEL4_PREACT,
    MODEL5_OUTPUT,
    MODEL6_OUTPUT,
    ScheduleBuilder,
)


CONTRACT_ID = "K1X_INT8_V1"
PROFILE_ID = "K1X_INT8_V1_GENERAL"
LAYOUT_ID = "NCHWc8_SPATIAL_INNER_V1"
SCHEMA_VERSION = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def f32_bits(value: np.float32 | float) -> int:
    return int(np.asarray(value, dtype="<f4").view("<u4"))


def fraction_from_f32_bits(bits: int) -> Fraction:
    sign = -1 if bits >> 31 else 1
    exponent = (bits >> 23) & 0xFF
    fraction = bits & 0x7FFFFF
    if exponent == 0xFF:
        raise ValueError("non-finite scale")
    if exponent == 0:
        significand = fraction
        power = -149
    else:
        significand = (1 << 23) | fraction
        power = exponent - 127 - 23
    value = Fraction(sign * significand, 1)
    return value * (1 << power) if power >= 0 else value / (1 << -power)


def round_fraction_even(value: Fraction) -> int:
    sign = -1 if value < 0 else 1
    absolute = abs(value)
    quotient, remainder = divmod(absolute.numerator, absolute.denominator)
    doubled = remainder * 2
    if doubled > absolute.denominator or (doubled == absolute.denominator and quotient & 1):
        quotient += 1
    return sign * quotient


def encode_multiplier(value: Fraction) -> tuple[int, int]:
    if value < 0:
        raise ValueError("negative requant multiplier")
    if value == 0:
        return 0, 0
    right_shift = 62
    multiplier = round_fraction_even(value * (1 << right_shift))
    while multiplier > (1 << 63) - 1 and right_shift > 0:
        right_shift -= 1
        multiplier = round_fraction_even(value * (1 << right_shift))
    if multiplier < 0 or multiplier > (1 << 63) - 1:
        raise ValueError("requant multiplier cannot be encoded")
    return multiplier, right_shift


def round_product_right_even(value: int, multiplier: int, right_shift: int) -> int:
    product = value * multiplier
    negative = product < 0
    absolute = abs(product)
    quotient = absolute >> right_shift if right_shift else absolute
    if right_shift:
        remainder = absolute & ((1 << right_shift) - 1)
        half = 1 << (right_shift - 1)
        if remainder > half or (remainder == half and quotient & 1):
            quotient += 1
    return -quotient if negative else quotient


def requantize(value: int, multiplier: int, right_shift: int, zero_point: int) -> int:
    return min(255, max(0, round_product_right_even(value, multiplier, right_shift) + zero_point))


def rne_float(value: float) -> int:
    floor = math.floor(value)
    fraction = value - floor
    if fraction > 0.5 or (fraction == 0.5 and floor & 1):
        floor += 1
    return floor


def silu_f32(value: np.float32) -> np.float32:
    return np.float32(value / np.float32(np.float32(1.0) + np.exp(np.float32(-value))))


def quantize_f32(value: np.float32, scale: np.float32, zero_point: int) -> int:
    return min(255, max(0, rne_float(float(value) / float(scale)) + zero_point))


def build_unary_lut(input_scale: np.float32, input_zp: int,
                    output_scale: np.float32, output_zp: int,
                    activation: str) -> np.ndarray:
    result = np.empty(256, dtype=np.int8)
    for code in range(256):
        value = np.float32(np.float32(code - input_zp) * input_scale)
        if activation == "silu":
            value = silu_f32(value)
        elif activation != "none":
            raise ValueError(f"unsupported activation: {activation}")
        result[code] = np.int8(quantize_f32(value, output_scale, output_zp) - 128)
    return result


def build_add_silu_lut(left: dict[str, Any], right: dict[str, Any], output: dict[str, Any]) -> np.ndarray:
    result = np.empty((256, 256), dtype=np.int8)
    for left_code in range(256):
        left_value = np.float32(np.float32(left_code - int(left["zero_point"])) * np.float32(left["scale"]))
        for right_code in range(256):
            right_value = np.float32(np.float32(right_code - int(right["zero_point"])) * np.float32(right["scale"]))
            value = np.float32(left_value + silu_f32(right_value))
            result[left_code, right_code] = np.int8(
                quantize_f32(value, np.float32(output["scale"]), int(output["zero_point"])) - 128
            )
    return result


def pack_weights(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    output_c, kernel_h, kernel_w, input_c = map(int, weights.shape)
    kernel_k = kernel_h * kernel_w * input_c
    if kernel_k % 8 or output_c % 16:
        raise ValueError("Stage49 Conv requires K divisible by 8 and N divisible by 16")
    flat = weights.reshape(output_c, kernel_k)
    packed = np.zeros((output_c // 16, kernel_k // 8, 16, 8), dtype=np.int8)
    for block in range(output_c // 16):
        for tile in range(kernel_k // 8):
            for lane in range(16):
                packed[block, tile, lane] = flat[block * 16 + lane, tile * 8 : tile * 8 + 8]
    return packed, flat.astype(np.int32).sum(axis=1, dtype=np.int32)


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = []
        seen: set[str] = set()
        for row in rows:
            for field in row:
                if field not in seen:
                    fields.append(field)
                    seen.add(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def nchw_to_nchwc8_s8(values: np.ndarray) -> np.ndarray:
    if values.dtype != np.uint8 or values.ndim != 4 or values.shape[1] % 8:
        raise ValueError("NCHWc8 conversion requires uint8 NCHW and C divisible by 8")
    n, c, h, w = values.shape
    blocked = values.reshape(n, c // 8, 8, h, w).transpose(0, 1, 3, 4, 2)
    return np.ascontiguousarray((blocked.astype(np.int16) - 128).astype(np.int8))


def im2col_accumulators(input_value: np.ndarray, weights: np.ndarray,
                        bias: np.ndarray, input_zp: int,
                        stride_h: int, stride_w: int,
                        pad_h: int, pad_w: int) -> np.ndarray:
    _, input_c, input_h, input_w = input_value.shape
    output_c, kernel_h, kernel_w, _ = weights.shape
    output_h = (input_h + 2 * pad_h - kernel_h) // stride_h + 1
    output_w = (input_w + 2 * pad_w - kernel_w) // stride_w + 1
    centered = input_value.astype(np.int16) - input_zp
    padded = np.pad(centered, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), constant_values=0)
    matrix = np.empty((output_h * output_w, kernel_h * kernel_w * input_c), dtype=np.int16)
    column = 0
    for ky in range(kernel_h):
        for kx in range(kernel_w):
            patch = padded[0, :, ky : ky + output_h * stride_h : stride_h,
                           kx : kx + output_w * stride_w : stride_w]
            matrix[:, column : column + input_c] = patch.transpose(1, 2, 0).reshape(-1, input_c)
            column += input_c
    accumulators = matrix.astype(np.int64) @ weights.reshape(output_c, -1).astype(np.int64).T
    accumulators += bias.astype(np.int64)[None, :]
    return accumulators.reshape(output_h, output_w, output_c)


def exact_conv_codes(accumulators: np.ndarray, multipliers: np.ndarray,
                     shifts: np.ndarray, zero_point: int) -> np.ndarray:
    output = np.empty(accumulators.shape, dtype=np.uint8)
    flat_in = accumulators.reshape(-1, accumulators.shape[-1])
    flat_out = output.reshape(-1, output.shape[-1])
    for pixel in range(flat_in.shape[0]):
        for channel in range(flat_in.shape[1]):
            flat_out[pixel, channel] = requantize(
                int(flat_in[pixel, channel]), int(multipliers[channel]), int(shifts[channel]), zero_point
            )
    return output


def build_schedule(index: GraphIndex, package: Path) -> ScheduleBuilder:
    builder = ScheduleBuilder(index, package)
    input_id = builder.tensor("model4.preact", index.qspec(MODEL4_PREACT))
    postact_id = builder.tensor("model4.postact", index.qspec(MODEL4_POSTACT))
    builder.add_lut("/model.4/cv2/final_silu", input_id, postact_id, "silu")
    model5_id = builder.simple_conv_block("model.5", postact_id, MODEL5_OUTPUT)
    builder.c2f_cib_block("model.6", model5_id, MODEL6_OUTPUT)
    builder.finalize()
    return builder


def derive_integer_assets(package: Path, builder: ScheduleBuilder) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tensors = builder.tensors
    operations = builder.ops
    for tensor in tensors:
        tensor["physical_layout"] = LAYOUT_ID
    write_tsv(package / "tensors.tsv", tensors)

    scale_rows: list[dict[str, Any]] = []
    bound_rows: list[dict[str, Any]] = []
    enriched: list[dict[str, Any]] = []
    extra_fields = [
        "output_c", "k", "k_tiles", "n_blocks", "input_zero_point", "conv_output_zero_point",
        "accumulator_absolute_bound", "packed_weights_file", "weight_sums_file",
        "requant_multiplier_file", "requant_shift_file", "segment0_begin", "segment0_count",
        "segment0_lut_file", "segment1_begin", "segment1_count", "segment1_lut_file",
        "lut_file", "add_lut_file", "concat0_lut_file", "concat1_lut_file", "concat2_lut_file",
    ]
    for source in operations:
        row = dict(source)
        for key in extra_fields:
            row.setdefault(key, "-" if key.endswith("_file") else 0)
        operation_dir = package / "integer_assets" / f"op_{int(row['index']):03d}"
        operation_dir.mkdir(parents=True, exist_ok=True)
        if row["kind"] == "conv":
            input_tensor = tensors[int(row["input0"])]
            output_ids = [int(row["output0"]), int(row["output1"])]
            output_c = sum(int(row[f"segment{index}_channel_count"]) for index in range(2))
            input_c = int(input_tensor["c"])
            kernel_h = int(row["kernel_h"])
            kernel_w = int(row["kernel_w"])
            weights_path = package / str(row["weights_file"])
            scales_path = package / str(row["weight_scales_file"])
            bias_path = package / str(row["bias_file"])
            weights = np.fromfile(weights_path, dtype=np.int8).reshape(output_c, kernel_h, kernel_w, input_c)
            scales = np.fromfile(scales_path, dtype="<f4")
            bias = np.fromfile(bias_path, dtype="<i4")
            packed, sums = pack_weights(weights)
            packed_path = operation_dir / "weights_packed_n16k8_s8.bin"
            sums_path = operation_dir / "weight_sums_i32.bin"
            multiplier_path = operation_dir / "requant_multiplier_i64.bin"
            shift_path = operation_dir / "requant_right_shift_i32.bin"
            packed.tofile(packed_path)
            sums.astype("<i4").tofile(sums_path)
            input_fraction = fraction_from_f32_bits(f32_bits(np.float32(input_tensor["scale"])))
            conv_scale = np.float32(row["conv_output_scale"])
            conv_fraction = fraction_from_f32_bits(f32_bits(conv_scale))
            multipliers = np.empty(output_c, dtype="<i8")
            shifts = np.empty(output_c, dtype="<i4")
            for channel, weight_scale in enumerate(scales):
                ratio = input_fraction * fraction_from_f32_bits(f32_bits(weight_scale)) / conv_fraction
                multiplier, shift = encode_multiplier(ratio)
                multipliers[channel] = multiplier
                shifts[channel] = shift
                scale_rows.append({
                    "operation_index": row["index"], "channel": channel,
                    "input_scale_bits": f"0x{f32_bits(np.float32(input_tensor['scale'])):08x}",
                    "weight_scale_bits": f"0x{f32_bits(weight_scale):08x}",
                    "conv_scale_bits": f"0x{f32_bits(conv_scale):08x}",
                    "ratio_numerator": ratio.numerator, "ratio_denominator": ratio.denominator,
                    "multiplier": multiplier, "right_shift": shift,
                })
            multipliers.tofile(multiplier_path)
            shifts.tofile(shift_path)
            maximum_weight = int(np.max(np.abs(weights.astype(np.int16))))
            maximum_bias = int(np.max(np.abs(bias.astype(np.int64))))
            input_zp = int(input_tensor["zero_point"])
            maximum_activation = max(input_zp, 255 - input_zp)
            k = kernel_h * kernel_w * input_c
            bound = maximum_bias + k * maximum_activation * maximum_weight
            if bound > (1 << 31) - 1:
                raise ValueError(f"unsafe int32 accumulator: {row['name']}")
            row.update({
                "output_c": output_c, "k": k, "k_tiles": k // 8, "n_blocks": output_c // 16,
                "input_zero_point": input_zp,
                "conv_output_zero_point": int(row["conv_output_zero_point"]),
                "accumulator_absolute_bound": bound,
                "packed_weights_file": str(packed_path.relative_to(package)),
                "weight_sums_file": str(sums_path.relative_to(package)),
                "requant_multiplier_file": str(multiplier_path.relative_to(package)),
                "requant_shift_file": str(shift_path.relative_to(package)),
            })
            bound_rows.append({
                "operation_index": row["index"], "name": row["name"], "k": k,
                "input_zero_point": input_zp, "maximum_weight": maximum_weight,
                "maximum_bias": maximum_bias, "absolute_bound": bound, "int32_safe": 1,
            })
            for segment_index in range(2):
                count = int(row[f"segment{segment_index}_channel_count"])
                row[f"segment{segment_index}_begin"] = int(row[f"segment{segment_index}_channel_begin"])
                row[f"segment{segment_index}_count"] = count
                if count <= 0:
                    continue
                target = tensors[output_ids[segment_index]]
                lut = build_unary_lut(
                    conv_scale, int(row["conv_output_zero_point"]), np.float32(target["scale"]),
                    int(target["zero_point"]), str(row[f"segment{segment_index}_activation"]),
                )
                lut_path = operation_dir / f"segment{segment_index}_lut_s8.bin"
                lut.tofile(lut_path)
                row[f"segment{segment_index}_lut_file"] = str(lut_path.relative_to(package))
        elif row["kind"] == "lut":
            source = tensors[int(row["input0"])]
            target = tensors[int(row["output0"])]
            lut = build_unary_lut(np.float32(source["scale"]), int(source["zero_point"]),
                                  np.float32(target["scale"]), int(target["zero_point"]), "silu")
            path = operation_dir / "lut_s8.bin"
            lut.tofile(path)
            row["lut_file"] = str(path.relative_to(package))
        elif row["kind"] == "add_silu":
            left = tensors[int(row["input0"])]
            right = tensors[int(row["input1"])]
            target = tensors[int(row["output0"])]
            lut = build_add_silu_lut(left, right, target)
            path = operation_dir / "add_silu_lut_256x256_s8.bin"
            lut.tofile(path)
            row["add_lut_file"] = str(path.relative_to(package))
        elif row["kind"] == "concat":
            target = tensors[int(row["output0"])]
            for input_index in range(3):
                tensor_id = int(row[f"input{input_index}"])
                if tensor_id < 0:
                    continue
                source = tensors[tensor_id]
                lut = build_unary_lut(np.float32(source["scale"]), int(source["zero_point"]),
                                      np.float32(target["scale"]), int(target["zero_point"]), "none")
                path = operation_dir / f"concat{input_index}_lut_s8.bin"
                lut.tofile(path)
                row[f"concat{input_index}_lut_file"] = str(path.relative_to(package))
        enriched.append(row)
    write_tsv(package / "operations.tsv", enriched)
    write_tsv(package / "scale_encoding.tsv", scale_rows)
    write_tsv(package / "accumulator_bounds.tsv", bound_rows)
    return tensors, enriched


def execute_integer_fixture(package: Path, tensors: list[dict[str, Any]], operations: list[dict[str, Any]],
                            input_value: np.ndarray) -> dict[int, np.ndarray]:
    values: dict[int, np.ndarray] = {0: np.ascontiguousarray(input_value, dtype=np.uint8)}
    for row in operations:
        kind = str(row["kind"])
        if kind == "lut":
            source = values[int(row["input0"])]
            lut = np.fromfile(package / str(row["lut_file"]), dtype=np.int8).astype(np.int16) + 128
            values[int(row["output0"])] = np.ascontiguousarray(lut[source], dtype=np.uint8)
        elif kind == "conv":
            input_id = int(row["input0"])
            source = values[input_id]
            output_c = int(row["output_c"])
            kernel_h = int(row["kernel_h"])
            kernel_w = int(row["kernel_w"])
            input_c = source.shape[1]
            weights = np.fromfile(package / str(row["weights_file"]), dtype=np.int8).reshape(
                output_c, kernel_h, kernel_w, input_c
            )
            bias = np.fromfile(package / str(row["bias_file"]), dtype="<i4")
            multiplier = np.fromfile(package / str(row["requant_multiplier_file"]), dtype="<i8")
            shifts = np.fromfile(package / str(row["requant_shift_file"]), dtype="<i4")
            accumulators = im2col_accumulators(
                source, weights, bias, int(row["input_zero_point"]),
                int(row["stride_h"]), int(row["stride_w"]), int(row["pad_h"]), int(row["pad_w"]),
            )
            conv_codes = exact_conv_codes(accumulators, multiplier, shifts, int(row["conv_output_zero_point"]))
            for segment_index in range(2):
                count = int(row[f"segment{segment_index}_count"])
                if count <= 0:
                    continue
                begin = int(row[f"segment{segment_index}_begin"])
                lut = np.fromfile(package / str(row[f"segment{segment_index}_lut_file"]), dtype=np.int8).astype(np.int16) + 128
                segment = lut[conv_codes[:, :, begin : begin + count]]
                values[int(row[f"output{segment_index}"])] = np.ascontiguousarray(
                    segment.transpose(2, 0, 1)[None, ...], dtype=np.uint8
                )
        elif kind == "add_silu":
            left = values[int(row["input0"])]
            right = values[int(row["input1"])]
            lut = np.fromfile(package / str(row["add_lut_file"]), dtype=np.int8).reshape(256, 256).astype(np.int16) + 128
            values[int(row["output0"])] = np.ascontiguousarray(lut[left, right], dtype=np.uint8)
        elif kind == "concat":
            target_parts: list[np.ndarray] = []
            for input_index in range(3):
                tensor_id = int(row[f"input{input_index}"])
                if tensor_id < 0:
                    continue
                lut = np.fromfile(package / str(row[f"concat{input_index}_lut_file"]), dtype=np.int8).astype(np.int16) + 128
                target_parts.append(np.ascontiguousarray(lut[values[tensor_id]], dtype=np.uint8))
            values[int(row["output0"])] = np.ascontiguousarray(np.concatenate(target_parts, axis=1), dtype=np.uint8)
        else:
            raise ValueError(f"unsupported integer operation: {kind}")
    return values


def generate_fixtures(args: argparse.Namespace, package: Path,
                      tensors: list[dict[str, Any]], operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    with (args.stage43_oracle_root / "model5_8_oracle_manifest.tsv").open(newline="", encoding="utf-8") as stream:
        source_manifest = list(csv.DictReader(stream, delimiter="\t"))
    rows: list[dict[str, Any]] = []
    for fixture_index in range(8):
        fixture_id = f"F{fixture_index}"
        source_row = next(
            row for row in source_manifest
            if row["fixture_id"] == fixture_id and row["tensor_name"] == MODEL4_PREACT
        )
        input_value = np.fromfile(source_row["raw_path"], dtype=np.uint8).reshape(1, 128, 80, 80)
        values = execute_integer_fixture(package, tensors, operations, input_value)
        fixture_dir = package / "oracles" / fixture_id
        fixture_dir.mkdir(parents=True, exist_ok=True)
        for tensor in tensors:
            tensor_id = int(tensor["id"])
            value = values[tensor_id]
            nchw_path = fixture_dir / f"tensor_{tensor_id:03d}_nchw_u8.bin"
            blocked_path = fixture_dir / f"tensor_{tensor_id:03d}_nchwc8_s8.bin"
            value.tofile(nchw_path)
            nchw_to_nchwc8_s8(value).tofile(blocked_path)
            rows.append({
                "fixture_id": fixture_id, "tensor_id": tensor_id, "tensor_key": tensor["key"],
                "tensor_name": tensor["logical_name"], "shape": f"1x{tensor['c']}x{tensor['h']}x{tensor['w']}",
                "nchw_file": str(nchw_path.relative_to(package)), "nchw_sha256": sha256_file(nchw_path),
                "nchwc8_file": str(blocked_path.relative_to(package)), "nchwc8_sha256": sha256_file(blocked_path),
                "sum": int(value.astype(np.uint64).sum()), "oracle": "K1X_INT8_V1_python_arbitrary_precision",
            })
    write_tsv(package / "fixture_manifest.tsv", rows)
    return rows


def generate(args: argparse.Namespace) -> None:
    model_path = args.model.resolve()
    output = args.out_dir.resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    model = onnx.load(model_path)
    index = GraphIndex(model)
    builder = build_schedule(index, output)
    tensors, operations = derive_integer_assets(output, builder)
    fixture_rows = generate_fixtures(args, output, tensors, operations)
    package_meta = {
        "arena_bytes": max(int(row["arena_offset"]) + int(row["bytes"]) for row in tensors),
        "byte_order": "little-endian",
        "contract_id": CONTRACT_ID,
        "input_tensor_id": 0,
        "layout_id": LAYOUT_ID,
        "model5_input_tensor_id": 1,
        "model5_output_tensor_id": 2,
        "model6_output_tensor_id": 16,
        "model_sha256": sha256_file(model_path),
        "operation_count": len(operations),
        "profile_id": PROFILE_ID,
        "schema_version": SCHEMA_VERSION,
        "source_lineage_id": f"accepted-yolo26-qdq:{sha256_file(model_path)}:stage49-model4final-model6",
        "tensor_count": len(tensors),
    }
    (output / "package.json").write_text(json.dumps(package_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = [path for path in sorted(output.rglob("*")) if path.is_file() and path.name != "asset_hashes.tsv"]
    hash_rows = [{"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
                  "sha256": sha256_file(path)} for path in files]
    write_tsv(output / "asset_hashes.tsv", hash_rows, ["path", "bytes", "sha256"])
    manifest_sha = sha256_file(output / "asset_hashes.tsv")
    summary = {
        "package": str(output), "manifest_sha256": manifest_sha, "model_sha256": sha256_file(model_path),
        "tensors": len(tensors), "operations": len(operations), "fixtures": len(fixture_rows),
        "contract_id": CONTRACT_ID, "schema_version": SCHEMA_VERSION,
    }
    print(json.dumps(summary, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--stage43-oracle-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    generate(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
