# Graph Inventory

Primary inspected candidate:

```text
/data/banana-yolo26-spacemit-demo/.deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx
```

Alternative implementation-contract candidate:

```text
/data/banana-yolo26-spacemit-demo/.deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_trad_rep_conv_matmul_qdq.onnx
```

## Digest

See `onnx_stage0_digest.tsv` for the generated inventory.

| model | input | output | nodes | Q | DQ | notable unsupported/control ops |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `manual_e2e_rep_conv_matmul_qdq.onnx` | `[1,3,640,640]` | `[1,300,6]` | 1069 | 206 | 410 | `Gather`, `ReduceMax`, `TopK` |
| `manual_trad_rep_conv_matmul_qdq.onnx` | `[1,3,640,640]` | `[1,84,8400]` | 1021 | 206 | 410 | `Gather` |

## Operator Counts, Chosen E2E Candidate

Key counts:

- `Conv`: 102
- `MatMul`: 4
- `QuantizeLinear`: 206
- `DequantizeLinear`: 410
- `Add`: 23
- `Concat`: 26
- `Resize`: 2
- `Mul`: 93
- `Sigmoid`: 88
- `TopK`: 2
- `ReduceMax`: 1
- `Gather`: 4

The graph is mostly quantized Conv trunk plus attention MatMul. The e2e export
adds tail selection/postprocess logic: `Transpose`, `Split`, `ReduceMax`,
`TopK`, `GatherElements`, `Cast`, and final `Concat` to `[1,300,6]`.

## Q/DQ Placement

- Activations are Q/DQ around Conv, MatMul, Add/Mul, and tail ops.
- Weights for Conv are int8 per-output-channel with zero-point 0 in the sampled rows.
- Bias tensors are int32 with per-channel scales and zero-point 0.
- Attention `MatMul` inputs are activation tensors, often asymmetric `uint8` with nonzero zero-points.

## Potential First Implementation Subset

Stage 1 should not implement full graph execution. The first kernel should be:

```text
smt.vmadot 4x4x8 s8xs8->s32 microkernel with scalar oracle
```

The first real graph-node target for later integration is:

```text
/model.10/m/m.0/attn/MatMul
```

but that graph node requires zero-point correction or future symmetric
requantization before it can map to the accepted `smt.vmadot` primitive.
