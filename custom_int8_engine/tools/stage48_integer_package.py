#!/usr/bin/env python3
"""Export the K1X_INT8_V1 model5 integer package and independent oracles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper, shape_inference


CONTRACT_ID = "K1X_INT8_V1"
PROFILE_ID = "K1X_INT8_V1_GENERAL"
LAYOUT_ID = "NCHWc8_SPATIAL_INNER_V1"
MODEL5_NODE = "/model.5/conv/Conv"
MODEL5_INPUT = "/model.4/cv2/act/Mul_output_0_QuantizeLinear_Output"
MODEL5_CONV_OUTPUT = "/model.5/conv/Conv_output_0_QuantizeLinear_Output"
MODEL5_OUTPUT = "/model.5/act/Mul_output_0_QuantizeLinear_Output"
MULTIPLIER_BITS = 62


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def f32_bits(value: float | np.floating[object]) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def fraction_from_f32_bits(bits: int) -> Fraction:
    sign = -1 if bits >> 31 else 1
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    if exponent == 0xFF:
        raise ValueError("non-finite float32 is not a contract scale")
    if exponent == 0:
        if mantissa == 0:
            return Fraction(0)
        significand = mantissa
        binary_exponent = -149
    else:
        significand = (1 << 23) | mantissa
        binary_exponent = exponent - 127 - 23
    value = Fraction(sign * significand)
    return value * (1 << binary_exponent) if binary_exponent >= 0 else value / (1 << -binary_exponent)


def round_fraction_even(value: Fraction) -> int:
    if value < 0:
        return -round_fraction_even(-value)
    quotient, remainder = divmod(value.numerator, value.denominator)
    doubled = remainder * 2
    if doubled > value.denominator or (doubled == value.denominator and quotient & 1):
        quotient += 1
    return quotient


def floor_log2_fraction(value: Fraction) -> int:
    if value <= 0:
        raise ValueError("log2 requires a positive fraction")
    guess = value.numerator.bit_length() - value.denominator.bit_length()
    if guess >= 0:
        if value.numerator < value.denominator << guess:
            guess -= 1
    elif value.numerator << -guess < value.denominator:
        guess -= 1
    return guess


def encode_multiplier(ratio: Fraction) -> tuple[int, int]:
    if ratio < 0:
        raise ValueError("negative requant multiplier")
    if ratio == 0:
        return 0, 0
    exponent = floor_log2_fraction(ratio) + 1
    right_shift = MULTIPLIER_BITS - exponent
    if right_shift < 0 or right_shift > 126:
        raise ValueError(f"unsupported right shift {right_shift}")
    scaled = ratio * (1 << right_shift)
    multiplier = round_fraction_even(scaled)
    if multiplier == 1 << MULTIPLIER_BITS:
        multiplier >>= 1
        right_shift -= 1
    if multiplier <= 0 or multiplier >= 1 << MULTIPLIER_BITS or right_shift < 0:
        raise ValueError("encoded multiplier is outside K1X_INT8_V1")
    return multiplier, right_shift


def round_product_right_even(value: int, multiplier: int, right_shift: int) -> int:
    product = value * multiplier
    negative = product < 0
    magnitude = abs(product)
    if right_shift == 0:
        rounded = magnitude
    else:
        quotient, remainder = divmod(magnitude, 1 << right_shift)
        half = 1 << (right_shift - 1)
        if remainder > half or (remainder == half and quotient & 1):
            quotient += 1
        rounded = quotient
    return -rounded if negative else rounded


def requantize(value: int, multiplier: int, right_shift: int, zero_point: int) -> int:
    return min(255, max(0, round_product_right_even(value, multiplier, right_shift) + zero_point))


def rne_float64(value: float) -> int:
    floor = math.floor(value)
    fraction = value - floor
    if fraction > 0.5 or (fraction == 0.5 and floor & 1):
        floor += 1
    return floor


def build_silu_lut(conv_scale: np.float32,
                   conv_zero_point: int,
                   output_scale: np.float32,
                   output_zero_point: int) -> np.ndarray:
    result = np.empty(256, dtype=np.int8)
    for code in range(256):
        dequantized = np.float32(np.float32(code - conv_zero_point) * conv_scale)
        activated = np.float32(dequantized / np.float32(np.float32(1.0) + np.exp(np.float32(-dequantized))))
        quantized = rne_float64(float(activated) / float(output_scale)) + output_zero_point
        semantic = min(255, max(0, quantized))
        result[code] = np.int8(semantic - 128)
    return result


class GraphIndex:
    def __init__(self, model: onnx.ModelProto) -> None:
        inferred = shape_inference.infer_shapes(model)
        self.initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
        self.producer = {name: node for node in model.graph.node for name in node.output}
        self.nodes = {node.name: node for node in model.graph.node}
        self.metadata: dict[str, tuple[int, tuple[int, ...]]] = {}
        for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
            tensor = value.type.tensor_type
            if all(dim.HasField("dim_value") for dim in tensor.shape.dim):
                self.metadata[value.name] = (
                    int(tensor.elem_type),
                    tuple(int(dim.dim_value) for dim in tensor.shape.dim),
                )

    def scalar(self, name: str) -> np.generic:
        value = np.asarray(self.initializers[name]).reshape(-1)
        if value.size != 1:
            raise ValueError(f"expected scalar initializer: {name}")
        return value[0]

    def qspec(self, name: str) -> tuple[np.float32, int, tuple[int, ...]]:
        node = self.producer[name]
        if node.op_type != "QuantizeLinear" or len(node.input) < 3:
            raise ValueError(f"not a QuantizeLinear output: {name}")
        dtype, shape = self.metadata[name]
        if dtype != TensorProto.UINT8:
            raise ValueError(f"K1X_INT8_V1_GENERAL expects uint8 semantics: {name}")
        return np.float32(self.scalar(node.input[1])), int(self.scalar(node.input[2])), shape


def node_attributes(node: onnx.NodeProto) -> dict[str, object]:
    return {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}


def pack_weights(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    output_c, kernel_h, kernel_w, input_c = map(int, weights.shape)
    kernel_k = kernel_h * kernel_w * input_c
    k_tiles = (kernel_k + 7) // 8
    n_blocks = (output_c + 15) // 16
    flat = weights.reshape(output_c, kernel_k)
    packed = np.zeros((n_blocks, k_tiles, 16, 8), dtype=np.int8)
    for block in range(n_blocks):
        for tile in range(k_tiles):
            for lane in range(16):
                output_channel = block * 16 + lane
                if output_channel >= output_c:
                    continue
                begin = tile * 8
                end = min(begin + 8, kernel_k)
                packed[block, tile, lane, : end - begin] = flat[output_channel, begin:end]
    sums = flat.astype(np.int32).sum(axis=1, dtype=np.int32)
    return packed, sums


def nchw_to_nchwc8_s8(values: np.ndarray) -> np.ndarray:
    if values.dtype != np.uint8 or values.ndim != 4 or values.shape[1] % 8:
        raise ValueError("NCHWc8 conversion requires uint8 NCHW with C divisible by 8")
    n, c, h, w = values.shape
    blocked = values.reshape(n, c // 8, 8, h, w).transpose(0, 1, 3, 4, 2)
    signed = blocked.astype(np.int16) - 128
    return np.ascontiguousarray(signed.astype(np.int8))


def full_model5_oracle(input_nchw_u8: np.ndarray,
                       weights: np.ndarray,
                       bias: np.ndarray,
                       multipliers: np.ndarray,
                       shifts: np.ndarray,
                       input_zero_point: int,
                       conv_zero_point: int,
                       lut: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, input_c, input_h, input_w = input_nchw_u8.shape
    output_c, kernel_h, kernel_w, weight_input_c = weights.shape
    if n != 1 or input_c != weight_input_c:
        raise ValueError("unexpected model5 tensor shape")
    output_h = (input_h + 2 - kernel_h) // 2 + 1
    output_w = (input_w + 2 - kernel_w) // 2 + 1
    centered = input_nchw_u8.astype(np.int16) - input_zero_point
    padded = np.pad(centered, ((0, 0), (0, 0), (1, 1), (1, 1)), constant_values=0)
    matrix = np.empty((output_h * output_w, kernel_h * kernel_w * input_c), dtype=np.int16)
    column = 0
    for ky in range(kernel_h):
        for kx in range(kernel_w):
            patch = padded[0, :, ky : ky + output_h * 2 : 2, kx : kx + output_w * 2 : 2]
            matrix[:, column : column + input_c] = patch.transpose(1, 2, 0).reshape(-1, input_c)
            column += input_c
    accumulators = matrix.astype(np.int64) @ weights.reshape(output_c, -1).astype(np.int64).T
    accumulators += bias.astype(np.int64)[None, :]
    output = np.empty((output_h * output_w, output_c), dtype=np.uint8)
    for pixel in range(output_h * output_w):
        for channel in range(output_c):
            conv_code = requantize(
                int(accumulators[pixel, channel]),
                int(multipliers[channel]),
                int(shifts[channel]),
                conv_zero_point,
            )
            output[pixel, channel] = np.uint8(int(lut[conv_code]) + 128)
    nchw = output.reshape(output_h, output_w, output_c).transpose(2, 0, 1)[None, ...]
    return np.ascontiguousarray(nchw), nchw_to_nchwc8_s8(np.ascontiguousarray(nchw))


def build_adversarial_vectors(multipliers: np.ndarray, shifts: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cases = [
        ("positive_tie_even", 1, 1, 1, 0),
        ("positive_tie_odd", 3, 1, 1, 0),
        ("negative_tie_even", -1, 1, 1, 0),
        ("negative_tie_odd", -3, 1, 1, 0),
        ("low_saturation", -(1 << 31), 1 << 30, 30, 0),
        ("high_saturation", (1 << 31) - 1, 1 << 30, 30, 255),
        ("zero", 0, 1 << 61, 62, 127),
    ]
    threshold = 2
    for delta in (-2, -1, 0, 1, 2):
        cases.append(
            (
                f"generic_tie_threshold_{delta:+d}",
                threshold + delta,
                1,
                2,
                127,
            )
        )
    for case_id, (name, value, multiplier, shift, zero_point) in enumerate(cases):
        rounded = round_product_right_even(value, multiplier, shift)
        output = min(255, max(0, rounded + zero_point))
        rows.append(
            {
                "case_id": case_id,
                "name": name,
                "accumulator": value,
                "multiplier": multiplier,
                "right_shift": shift,
                "output_zero_point": zero_point,
                "rounded": rounded,
                "output_code": output,
                "diagnostic_threshold_denominator": threshold if case_id >= 7 else 0,
            }
        )
    return rows


def generate(args: argparse.Namespace) -> None:
    model_path = args.model.resolve()
    source_root = args.stage43_oracle_root.resolve()
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = onnx.load(model_path)
    index = GraphIndex(model)
    node = index.nodes[MODEL5_NODE]
    attributes = node_attributes(node)
    input_scale, input_zero_point, input_shape = index.qspec(MODEL5_INPUT)
    conv_scale, conv_zero_point, conv_shape = index.qspec(MODEL5_CONV_OUTPUT)
    output_scale, output_zero_point, output_shape = index.qspec(MODEL5_OUTPUT)
    source_assets = source_root / "model5_runtime_assets"
    weights_path = source_assets / "weights_ohwi_s8.bin"
    scales_path = source_assets / "weight_scales_f32.bin"
    bias_path = source_assets / "bias_i32.bin"
    weights = np.fromfile(weights_path, dtype=np.int8).reshape(128, 3, 3, 128)
    weight_scales = np.fromfile(scales_path, dtype="<f4")
    bias = np.fromfile(bias_path, dtype="<i4")
    if weight_scales.size != 128 or bias.size != 128:
        raise ValueError("model5 channel asset count mismatch")
    packed, weight_sums = pack_weights(weights)

    input_fraction = fraction_from_f32_bits(f32_bits(input_scale))
    conv_fraction = fraction_from_f32_bits(f32_bits(conv_scale))
    multipliers = np.empty(128, dtype="<i8")
    shifts = np.empty(128, dtype="<i4")
    scale_rows: list[dict[str, object]] = []
    for channel, scale in enumerate(weight_scales):
        weight_bits = f32_bits(scale)
        ratio = input_fraction * fraction_from_f32_bits(weight_bits) / conv_fraction
        multiplier, right_shift = encode_multiplier(ratio)
        multipliers[channel] = multiplier
        shifts[channel] = right_shift
        scale_rows.append(
            {
                "channel": channel,
                "input_scale_bits": f"0x{f32_bits(input_scale):08x}",
                "weight_scale_bits": f"0x{weight_bits:08x}",
                "conv_output_scale_bits": f"0x{f32_bits(conv_scale):08x}",
                "ratio_numerator": ratio.numerator,
                "ratio_denominator": ratio.denominator,
                "multiplier_i64": multiplier,
                "right_shift_i32": right_shift,
            }
        )
    lut = build_silu_lut(conv_scale, conv_zero_point, output_scale, output_zero_point)

    packed.tofile(output / "weights_packed_n16k8_s8.bin")
    weights.tofile(output / "weights_ohwi_s8.bin")
    weight_sums.astype("<i4").tofile(output / "weight_sums_i32.bin")
    bias.astype("<i4").tofile(output / "bias_i32.bin")
    weight_scales.astype("<f4").tofile(output / "weight_scales_f32_bits.bin")
    multipliers.tofile(output / "requant_multiplier_i64.bin")
    shifts.tofile(output / "requant_right_shift_i32.bin")
    lut.tofile(output / "silu_lut_s8.bin")

    maximum_weight = int(np.max(np.abs(weights.astype(np.int16))))
    maximum_bias = int(np.max(np.abs(bias.astype(np.int64))))
    maximum_activation = max(input_zero_point, 255 - input_zero_point)
    accumulator_bound = maximum_bias + 3 * 3 * 128 * maximum_activation * maximum_weight
    int32_safe = accumulator_bound <= (1 << 31) - 1
    if not int32_safe:
        raise ValueError("model5 accumulator is not int32 safe")
    meta = {
        "contract_id": CONTRACT_ID,
        "profile_id": PROFILE_ID,
        "layout_id": LAYOUT_ID,
        "input_h": input_shape[2],
        "input_w": input_shape[3],
        "input_c": input_shape[1],
        "output_h": output_shape[2],
        "output_w": output_shape[3],
        "output_c": output_shape[1],
        "kernel_h": 3,
        "kernel_w": 3,
        "stride_h": int(attributes.get("strides", [2, 2])[0]),
        "stride_w": int(attributes.get("strides", [2, 2])[1]),
        "pad_h": int(attributes.get("pads", [1, 1, 1, 1])[0]),
        "pad_w": int(attributes.get("pads", [1, 1, 1, 1])[1]),
        "input_zero_point": input_zero_point,
        "conv_output_zero_point": conv_zero_point,
        "output_zero_point": output_zero_point,
        "k": 3 * 3 * 128,
        "k_tiles": 3 * 3 * 128 // 8,
        "n_blocks": 128 // 16,
        "accumulator_absolute_bound": accumulator_bound,
        "int32_safe": int(int32_safe),
        "model_sha256": sha256_file(model_path),
        "source_weights_sha256": sha256_file(weights_path),
        "source_weight_scales_sha256": sha256_file(scales_path),
        "source_bias_sha256": sha256_file(bias_path),
        "input_scale_bits": f"0x{f32_bits(input_scale):08x}",
        "conv_output_scale_bits": f"0x{f32_bits(conv_scale):08x}",
        "output_scale_bits": f"0x{f32_bits(output_scale):08x}",
    }
    write_tsv(output / "model5_meta.tsv", [meta], list(meta))
    write_tsv(output / "scale_encoding.tsv", scale_rows, list(scale_rows[0]))

    fixture_rows: list[dict[str, object]] = []
    fixtures_root = output / "fixtures"
    for fixture_id in [f"F{index}" for index in range(8)]:
        source = source_root / "fixtures" / fixture_id / "model.4__cv2__act__Mul_output_0_QuantizeLinear_Output.bin"
        values = np.fromfile(source, dtype=np.uint8).reshape(input_shape)
        expected_nchw, expected_nchwc8 = full_model5_oracle(
            values,
            weights,
            bias,
            multipliers,
            shifts,
            input_zero_point,
            conv_zero_point,
            lut,
        )
        fixture_dir = fixtures_root / fixture_id
        fixture_dir.mkdir(parents=True, exist_ok=True)
        input_copy = fixture_dir / "input_nchw_u8.bin"
        input_blocked = fixture_dir / "input_nchwc8_s8.bin"
        expected_copy = fixture_dir / "expected_nchw_u8.bin"
        expected_blocked = fixture_dir / "expected_nchwc8_s8.bin"
        values.tofile(input_copy)
        nchw_to_nchwc8_s8(values).tofile(input_blocked)
        expected_nchw.tofile(expected_copy)
        expected_nchwc8.tofile(expected_blocked)
        fixture_rows.append(
            {
                "fixture_id": fixture_id,
                "source_input_path": str(source),
                "source_input_sha256": sha256_file(source),
                "input_nchw_sha256": sha256_file(input_copy),
                "input_nchwc8_sha256": sha256_file(input_blocked),
                "expected_nchw_sha256": sha256_file(expected_copy),
                "expected_nchwc8_sha256": sha256_file(expected_blocked),
                "expected_sum": int(expected_nchw.astype(np.uint64).sum()),
                "oracle": "K1X_INT8_V1_python_exact_integer_with_proven_int64_conv_bound",
            }
        )
    write_tsv(output / "fixture_manifest.tsv", fixture_rows, list(fixture_rows[0]))

    adversarial = build_adversarial_vectors(multipliers, shifts)
    write_tsv(output / "adversarial_requant.tsv", adversarial, list(adversarial[0]))
    package_json = {
        "contract_id": CONTRACT_ID,
        "profile_id": PROFILE_ID,
        "layout_id": LAYOUT_ID,
        "byte_order": "little-endian",
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "node_name": MODEL5_NODE,
        "input_tensor": MODEL5_INPUT,
        "conv_output_tensor": MODEL5_CONV_OUTPUT,
        "output_tensor": MODEL5_OUTPUT,
        "multiplier_encoding": "signed_i64_multiplier_plus_nonnegative_right_shift_rne",
        "activation": "authoritative_256_entry_s8_LUT",
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "numpy_version": np.__version__,
        "onnx_version": onnx.__version__,
    }
    (output / "package.json").write_text(json.dumps(package_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = [path for path in sorted(output.rglob("*")) if path.is_file() and path.name != "asset_hashes.tsv"]
    rows = [{"path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files]
    write_tsv(output / "asset_hashes.tsv", rows, ["path", "bytes", "sha256"])
    print(json.dumps({"package": str(output), "fixtures": len(fixture_rows), "assets": len(rows), **meta}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--stage43-oracle-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    generate(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
