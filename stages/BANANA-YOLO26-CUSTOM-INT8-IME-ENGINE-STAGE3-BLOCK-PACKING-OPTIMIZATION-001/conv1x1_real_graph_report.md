# Conv1x1 Real Graph Report

## Node

- `/model.2/cv1/conv/Conv`
- Input shape: `[1,32,160,160]`
- Output shape: `[1,32,160,160]`
- Kernel: `1x1`
- Stride: `1`
- Pads: `0`
- Activation: `uint8`, zero-point `0`
- Weight: `int8`, per-output-channel, zero-point `0`

## Correctness

- Host scalar fixture: pass, mismatches `0`
- Board IME fixture: pass, `ime_status=0`, mismatches `0`
- Oracle: ONNX Runtime CPU intermediate output, deterministic seed `20260703`

## Board Microbenchmark

Command: `taskset -c 0-3 ./bench_stage3_packing 3`

| path | mean us | status |
|---|---:|---:|
| scalar | `109660` | `0` |
| old on-the-fly IME wrapper | `133531` | `0` |
| prepacked IME | `46716.2` | `0` |
| weight prepack only | `35.0143` | `0` |
| activation pack included | `33132.8` | `0` |
| correction pass | `2683.75` | `0` |

## Decision

Conv1x1 real-node path is correct and improved. It is suitable for a later graph-block integration attempt after Conv3x3 packing is repaired.
