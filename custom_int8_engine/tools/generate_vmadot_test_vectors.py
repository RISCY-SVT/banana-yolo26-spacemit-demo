#!/usr/bin/env python3
"""Generate deterministic Stage 1 smt.vmadot 4x4x8 test vectors."""

from __future__ import annotations

import json
from typing import Dict, List


def i8(value: int) -> int:
    value &= 0xFF
    return value - 256 if value >= 128 else value


def lcg_next(state: int) -> int:
    return (state * 1664525 + 1013904223) & 0xFFFFFFFF


def random_panel(seed: int) -> List[int]:
    values: List[int] = []
    state = seed
    for _ in range(32):
        state = lcg_next(state)
        values.append(i8((state >> 24) - 128))
    return values


def scalar(a: List[int], b_transposed: List[int], init: List[int], accumulate: bool) -> List[int]:
    out = list(init)
    for m in range(4):
        for n in range(4):
            acc = out[m * 4 + n] if accumulate else 0
            for k in range(8):
                acc += int(a[m * 8 + k]) * int(b_transposed[n * 8 + k])
            out[m * 4 + n] = acc
    return out


def make_case(name: str, a: List[int], b: List[int], init: List[int] | None = None, accumulate: bool = False) -> Dict[str, object]:
    init_values = init or [0] * 16
    return {
        "name": name,
        "a_4x8_row_major": a,
        "b_4x8_transposed_nk": b,
        "init_c": init_values,
        "accumulate": accumulate,
        "expected_c": scalar(a, b, init_values, accumulate),
    }


def main() -> None:
    zeros = [0] * 32
    ones = [1] * 32
    ramp_a = [i - 16 for i in range(32)]
    ramp_b = [15 - i for i in range(32)]
    edges_a = [-128 if i % 2 == 0 else 127 for i in range(32)]
    edges_b = [127 if i % 2 == 0 else -128 for i in range(32)]
    seed_0_a = random_panel(0)
    seed_0_b = random_panel(1)
    seed_1_a = random_panel(1)
    seed_1_b = random_panel(2)
    seed_12345_a = random_panel(12345)
    seed_12345_b = random_panel(12346)
    init = [i * 19 - 143 for i in range(16)]
    vectors = [
        make_case("all_zeros", zeros, zeros),
        make_case("all_ones", ones, ones),
        make_case("ramp", ramp_a, ramp_b),
        make_case("alternating_edges", edges_a, edges_b),
        make_case("random_seed_0", seed_0_a, seed_0_b),
        make_case("random_seed_1", seed_1_a, seed_1_b),
        make_case("random_seed_12345", seed_12345_a, seed_12345_b),
        make_case("accumulate_true", seed_12345_a, seed_12345_b, init=init, accumulate=True),
    ]
    print(json.dumps({"schema": "y26_vmadot_4x4x8_vectors_v1", "vectors": vectors}, indent=2))


if __name__ == "__main__":
    main()
