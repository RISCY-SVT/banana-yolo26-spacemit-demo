# Conv3x3 Real Graph Report

## Node

- `/model.2/m.0/cv1/conv/Conv`
- Input shape: `[1,16,160,160]`
- Output shape: `[1,8,160,160]`
- Kernel: `3x3`
- Stride: `1`
- Pads: `[1,1,1,1]`
- Activation: `uint8`, zero-point `2`
- Weight: `int8`, per-output-channel, zero-point `0`

## Correctness

- Host scalar fixture: pass, mismatches `0`
- Board IME fixture: pass, `ime_status=0`, mismatches `0`
- Oracle: ONNX Runtime CPU intermediate output, deterministic seed `20260703`

## Board Microbenchmark

Command: `taskset -c 0-3 ./bench_stage3_packing 3`

| path | mean us | status |
|---|---:|---:|
| scalar | `143646` | `0` |
| old on-the-fly IME wrapper | `390396` | `0` |
| prepacked IME | `147974` | `0` |
| weight prepack only | `39.334` | `0` |
| im2col/A-pack only | `157547` | `0` |
| correction pass | `661.68` | `0` |

## Decision

Conv3x3 real-node correctness passes, but packing/im2col cost still dominates. Stage 4 should repair packing/dataflow before first graph-block integration.
