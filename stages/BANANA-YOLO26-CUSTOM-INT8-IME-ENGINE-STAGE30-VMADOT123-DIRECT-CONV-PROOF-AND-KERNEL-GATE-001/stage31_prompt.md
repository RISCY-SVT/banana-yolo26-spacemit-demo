# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE31-VMADOT123-DIRECT-CONV-INTEGRATION-001

## Mission

Use the Stage30-proven `smt.vmadot1/2/3` shifted-tile semantics to build one bounded real 3x3 Conv sidecar candidate for the selected `/model.4` ONNX-cut path.

Primary target:

```text
/model.4/m.0/cv1/conv/Conv  3x3 80x80x32 -> 80x80x16
```

Secondary target only if the primary succeeds:

```text
/model.4/m.0/cv2/conv/Conv  3x3 80x80x16 -> 80x80x32
```

## Hard Boundaries

Do not implement a full YOLO26 engine. Do not create a graph-wide scheduler. Do not expand beyond the selected `/model.4` cut. Do not use `vmadotn`, FP/vfmadot, XSlim, CPU4-7 IME, OpenMP/all-core dispatch, camera/full-image/COCO/mAP, production/default backend, or `/data/ncnn` mutation.

## Required Gates

1. Re-run Stage30 micro-oracle for `vmadot1/2/3` on CPU0-3.
2. Build one expanded-A-panel direct/sliding 3x3 Conv sidecar for `/model.4/m.0/cv1/conv/Conv`.
3. Compare against current Stage28 threaded MMT4D output for the same input and weights.
4. Require mismatches=0 and same-input ONNX-cut output unchanged if used in a temporary selected-cut bench path.
5. Benchmark with warmup=10, runs=100, repeats=5, CPU0 single-thread and CPU0-3 threaded if single-thread correctness passes.

## Acceptance

Minimum: real 3x3 candidate correctness pass and >=1.15x per-node speedup vs current best threaded MMT4D for the same node.

If speedup is below threshold, do not integrate. Return to MMT4D/tile/prepack tuning or stop custom-kernel escalation.
