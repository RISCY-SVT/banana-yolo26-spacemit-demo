# Block Microbench Report

This is selected-subset microbenchmark evidence only. It is not a YOLO26 inference benchmark and is not a model FPS claim.

Board command:

```text
taskset -c 0 ./bench_stage6_multiblock 3
```

Selected subset:

```text
640x640x3 -> Conv0 -> 320x320x16 -> SiLU/requant -> Conv1 -> 160x160x32
```

## CPU0 Repeated Run

| metric | value |
| --- | ---: |
| iterations | `3` |
| scalar total mean | `1009980 us` |
| IME total mean | `419769 us` |
| speedup IME vs scalar | `2.41x` |
| scalar checksum | `5095626339` |
| IME checksum | `5095626339` |
| Stage 5 Conv0 replay IME mean | `70203.2 us` |
| prepacked bytes | `5312` |
| workspace bytes | `18023104` |

Status:

- scalar status: `0`
- IME status: `0`
- Stage 5 replay status: `0`
- checksums match between scalar and IME selected-subset outputs.

## Caveat

The activation/requant fallback is still scalar float and dominates the IME total. This stage proves correctness and useful Conv IME composition across the selected subset; it does not prove final full-engine throughput.

