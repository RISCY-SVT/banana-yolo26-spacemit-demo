# First Layer Shape Tile Plan

## First Real Graph Target

```text
node name: /model.10/m/m.0/attn/MatMul
op type: MatMul
model: manual_e2e_rep_conv_matmul_qdq.onnx
```

Shapes:

```text
A: [1,2,400,32]
B: [1,2,32,400]
Y: [1,2,400,400]
```

Interpreted batched GEMM:

```text
batches = 2
M = 400
N = 400
K = 32
MACs = 2 * 400 * 400 * 32 = 10,240,000
```

Quantization:

```text
A scale = 0.04627726227045059
A zero_point = 128
B scale = 0.059319499880075455
B zero_point = 126
Y scale = 0.6259151697158813
Y zero_point = 122
```

## Tile Plan

Primitive:

```text
smt.vmadot 4x4x8 s8xs8->s32
```

Stage 1 microkernel fixture:

```text
M tile = 4
N tile = 4
K tile = 8
A layout = row-major, K-contiguous
B layout = transposed, K-contiguous
C layout = int32 accumulator tile
```

Future candidate data tiles:

```text
8x8x8
8x16x8
```

These are planning candidates only. Stage 0 does not implement IME asm.

## Caveat

The selected graph node has asymmetric activation quantization, so it is not a
direct signed `smt.vmadot` feed. Graph integration must either subtract
zero-points with row/column sums or use a separately approved symmetric
requantization path.
