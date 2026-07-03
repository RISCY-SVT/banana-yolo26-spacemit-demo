# Conv1x1 Kernel Report

## Contract

- input layout: `NHWC int8`
- weights layout: `OC x IC int8`
- output layout: `NHWC int32`
- bias: optional `int32[OC]`
- operation: raw signed dot product, `sum(int8 * int8) -> int32`
- zero-point correction: not integrated
- requantization: not integrated

The IME path lowers Conv1x1 to MMT4D/GEMM tiles:

- `M = output_h * output_w`
- `N = output_c`
- `K = input_c`
- A tile: `4x8`, row-major, K-contiguous
- B tile: `4x8`, output-channel-major, K-contiguous
- C tile: `4x4 int32`

Tails are handled by zero-padding A/B panels and storing only valid output rows/channels.

## Correctness

Board command scope: `taskset -c 0-3 ./test_conv1x1_ime_fixture`.

| case | shape | layout | kernel path | scalar status | IME status | mismatches | max_abs_diff | board affinity |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| tail Cin/Cout | `3x4x5 -> OC6`, output `3x4` | NHWC/OCxIC/NHWC | scalar vs IME | 0 | 0 | 0 | 0 | CPU0-3 |
| tile-aligned | `4x4x8 -> OC8`, output `4x4` | NHWC/OCxIC/NHWC | scalar vs IME | 0 | 0 | 0 | 0 | CPU0-3 |
| stride2 tail | `5x5x9 -> OC7`, output `3x3` | NHWC/OCxIC/NHWC | scalar vs IME | 0 | 0 | 0 | 0 | CPU0-3 |

## Microbench

Board command:

```bash
taskset -c 0-3 ./bench_conv_kernels 200 5
```

| case | shape | packing included | scalar mean | IME mean | speedup vs scalar | mismatches |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Conv1x1 MMT4D tile core | `4x4x8` | no | 253.361 ns | 38.759 ns | 6.537 | 0 |
| Conv1x1 full kernel wrapper | `8x8x16 -> OC16` | yes | 85.702 us | 94.621 us | 0.906 | 0 |

The tile core is faster than scalar. The current packing-included Conv1x1 wrapper is slower than scalar, so Stage 3 should optimize packing, B prepacking, and block reuse before any larger graph claim.
