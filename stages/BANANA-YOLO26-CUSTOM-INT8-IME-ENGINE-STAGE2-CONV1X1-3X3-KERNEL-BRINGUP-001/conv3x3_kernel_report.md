# Conv3x3 Kernel Report

## Contract

- input layout: `NHWC int8`
- weights layout: `OC x KH x KW x IC int8`
- output layout: `NHWC int32`
- bias: optional `int32[OC]`
- supported synthetic fixtures: `3x3 stride1 padding1`, `3x3 stride2 padding1`
- operation: raw signed dot product, `sum(int8 * int8) -> int32`
- zero-point correction: not integrated
- requantization: not integrated

The Stage 2 implementation uses on-the-fly im2col/MMT4D lowering, not direct sliding-window `vmadot1/2/3`.

## Correctness

Board command scope: `taskset -c 0-3 ./test_conv3x3_ime_fixture`.

| case | shape | layout | kernel path | scalar status | IME status | mismatches | max_abs_diff | board affinity |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| tile-aligned stride1 | `5x5x8 -> OC8`, output `5x5` | NHWC/OCxKHxKWxIC/NHWC | scalar vs IME | 0 | 0 | 0 | 0 | CPU0-3 |
| stride2 tail | `6x6x5 -> OC6`, output `3x3` | NHWC/OCxKHxKWxIC/NHWC | scalar vs IME | 0 | 0 | 0 | 0 | CPU0-3 |
| tail Cin/Cout | `4x4x9 -> OC7`, output `4x4` | NHWC/OCxKHxKWxIC/NHWC | scalar vs IME | 0 | 0 | 0 | 0 | CPU0-3 |

## Microbench

Board command:

```bash
taskset -c 0-3 ./bench_conv_kernels 200 5
```

| case | shape | packing included | scalar mean | IME mean | speedup vs scalar | mismatches |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Conv3x3 MMT4D tile core | `4x4x8` | no | 251.471 ns | 39.268 ns | 6.404 | 0 |
| Conv3x3 full kernel wrapper | `8x8x16 -> OC16`, stride1/pad1 | yes | 669.155 us | 1834.478 us | 0.365 | 0 |

The tile core is faster than scalar. The current on-the-fly im2col packing is too expensive and must be optimized before block-level performance gates.
