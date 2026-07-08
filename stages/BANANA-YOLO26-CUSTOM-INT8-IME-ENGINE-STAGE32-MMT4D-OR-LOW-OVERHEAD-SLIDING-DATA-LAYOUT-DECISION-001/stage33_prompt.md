# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE33-MMT4D-MIXED-SIGNEDNESS-CORRECTION-PROOF-001

## Mission

Run a narrow proof stage for MMT4D mixed signedness and correction reduction on the existing `/model.4` same-input ONNX-cut path.

Stage32 rejected the low-overhead direct/sliding `smt.vmadot1/2/3` lane because no attachable layout candidate reached the `<=7800 us` panel-build gate. Stage32 also proved the matrix dot signedness family:

```text
smt.vmadot:   s8 x s8
smt.vmadotu:  u8 x u8
smt.vmadotsu: s8 x u8
smt.vmadotus: u8 x s8
```

Stage33 must determine whether `smt.vmadotus` or related mixed signedness can reduce conversion/correction work in the current MMT4D mainline without changing ONNX-cut bytes.

## Hard Boundaries

Do not implement a full engine, graph-wide scheduler, graph expansion, camera/full-image path, COCO/mAP, production/default backend, `vmadotn`, FP/vfmadot, XSlim, CPU4-7 IME, or `/data/ncnn` mutation.

## Required Gates

1. Replay Stage32 selected cut:

```text
output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
mismatches: 0
frm_sweep: pass
```

2. Build scalar MMT4D tile oracle for one Conv node:

```text
preferred first target: /model.4/cv2/conv/Conv
reason: largest correction bucket in Stage32, 1758.66 us
```

3. Test mixed signedness sidecar:

```text
baseline: plain smt.vmadot s8xs8
candidate: smt.vmadotus u8xs8 if activation is A and weight is B
```

4. Preserve same-input ONNX-cut correctness:

```text
mismatches=0
max_abs_diff=0
output SHA unchanged
CPU0-3 only
```

5. Stable benchmark:

```text
warmup=10
runs=100
repeats=5
taskset -c 0-3
```

## Acceptance

Accept only if the candidate reduces correction/conversion or selected-cut total measurably without correctness loss.

If mixed signedness does not help, recommend MMT4D tile/thread-overhead work or pause Conv microkernel work.
