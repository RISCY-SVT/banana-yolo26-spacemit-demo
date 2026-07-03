# Quantization And Scales Plan

## Current Audit Status

Audit source: `signedness_zero_point_audit.tsv`

Findings:

- Conv weights sampled from Q/DQ are signed int8 with zero-point 0 and per-output-channel scales.
- Bias tensors are int32 with per-channel scales and zero-point 0.
- Activations are frequently asymmetric `uint8` with nonzero zero-points.
- Attention `MatMul` inputs are not directly usable by `smt.vmadot` without correction.

## Direct IME Feed Rule

`smt.vmadot` is accepted only for:

```text
s8 x s8 -> s32
```

Allowed direct path:

```text
input dtype int8, input zero_point 0
weight dtype int8, weight zero_point 0
```

If either side is asymmetric, the implementation must not silently treat the
raw bytes as signed symmetric values.

## Asymmetric Correction

For asymmetric inputs, use:

```text
sum((a - za) * (w - zw))
= sum(a*w) - zw*sum(a) - za*sum(w) + K*za*zw
```

If future work chooses symmetric re-quantization, it must be approved and
validated against the ONNX CPU oracle. Stage 0 does not regenerate or rewrite
the model.

## First MatMul Caveat

`/model.10/m/m.0/attn/MatMul`:

- A shape: `[1,2,400,32]`, scale `0.04627726227045059`, zero-point `128`
- B shape: `[1,2,32,400]`, scale `0.059319499880075455`, zero-point `126`
- output shape: `[1,2,400,400]`, scale `0.6259151697158813`, zero-point `122`

This is a real graph target but not a direct Stage 1 `smt.vmadot` feed. Stage 1
should first prove a pure signed 4x4x8 microkernel and scalar oracle, then a
later stage can add correction/requant plumbing.
