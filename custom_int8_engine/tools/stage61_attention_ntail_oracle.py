#!/usr/bin/env python3
"""Independent arbitrary-precision oracle for the Stage61 N-tail probe."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

MASK64 = (1 << 64) - 1


def mix(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def code_for(tag: int, first: int, second: int, mode: int) -> int:
    if mode == 1:
        return 0 if ((first + second) & 1) == 0 else 255
    if mode == 2:
        return 0 if first % 3 == 0 else (255 if second % 3 == 0 else 128)
    return mix(tag ^ (first << 32) ^ second) & 0xFF


def round_product_right_even(value: int, multiplier: int, shift: int) -> int:
    product = value * multiplier
    negative = product < 0
    absolute = abs(product)
    quotient, remainder = divmod(absolute, 1 << shift)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and quotient & 1):
        quotient += 1
    return -quotient if negative else quotient


def fnv1a(values: list[int]) -> int:
    result = 1469598103934665603
    for value in values:
        result ^= value
        result = (result * 1099511628211) & MASK64
    return result


def expected_hash(n: int, k: int, m: int, mode: int) -> int:
    left_zero_point = (n * 37 + k * 11 + mode * 19) & 255
    right_zero_point = (n * 13 + k * 29 + mode * 7) & 255
    output_zero_point = (n * 17 + k * 5 + mode * 31) & 255
    multiplier = (1 << 54) + ((n * 131 + k * 17 + mode) & 0xFFFF)
    output: list[int] = []
    for row in range(m):
        for column in range(n):
            accumulator = sum(
                (code_for(0x413131 + n, row, inner, mode) - left_zero_point)
                * (code_for(0x423232 + n, inner, column, mode) - right_zero_point)
                for inner in range(k)
            )
            rounded = round_product_right_even(accumulator, multiplier, 62)
            output.append(min(255, max(0, rounded + output_zero_point)))
    return fnv1a(output)


def route(live: int, strategy: str) -> tuple[int, int]:
    if live <= 4:
        return 1, 4 - live
    if live <= 8:
        return 1, 8 - live
    if live <= 12:
        return 2, 12 - live
    if live < 16 and strategy == "n8n8":
        return 2, 16 - live
    return 1, 16 - live


def expected_routes(n: int, strategy: str) -> tuple[int, int]:
    calls = 0
    padded = 0
    for begin in range(0, n, 16):
        block_calls, block_padded = route(min(16, n - begin), strategy)
        calls += block_calls
        padded += block_padded
    return calls, padded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("probe_tsv", type=Path)
    args = parser.parse_args()
    checked = 0
    with args.probe_tsv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if not row.get("n") or row["n"].startswith("stage61_"):
                continue
            n = int(row["n"])
            k = int(row["k"])
            m = int(row["m"])
            mode = int(row["mode"])
            strategy = row["strategy"]
            observed_hash = int(row["output_hash"], 16)
            wanted_hash = expected_hash(n, k, m, mode)
            calls, padded = expected_routes(n, strategy)
            if observed_hash != wanted_hash:
                raise SystemExit(
                    f"output hash mismatch n={n} k={k} m={m} mode={mode} "
                    f"strategy={strategy}: {observed_hash:016x} != {wanted_hash:016x}"
                )
            if int(row["kernel_calls"]) != calls:
                raise SystemExit(f"kernel route mismatch for n={n} strategy={strategy}")
            if int(row["padded_k_lanes"]) != (-k) % 8:
                raise SystemExit(f"K padding mismatch for n={n} k={k}")
            if int(row["padded_n_columns"]) != padded:
                raise SystemExit(f"N padding mismatch for n={n} strategy={strategy}")
            if row["status"] != "pass":
                raise SystemExit(f"probe failure for n={n} strategy={strategy}")
            checked += 1
    if checked == 0:
        raise SystemExit("probe TSV contains no cases")
    print(f"stage61_python_oracle_cases={checked}")
    print("stage61_python_oracle_status=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
