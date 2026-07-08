# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE35-OUTPUT-QUANTIZE-OR-THREAD-OVERHEAD-LOCAL-REPAIR-001

## Mission

Continue in `/data/banana-yolo26-spacemit-demo` on branch `yolo26-custom-int8-engine` after Stage34.

Stage34 stopped the raw `smt.vmadot` cv2 software-pipeline lane because direct inline/register-blocked shapes trapped on board, while the accepted wrapper path remains executable. Do not continue raw `smt.vmadot` pipelining in Stage35.

Primary goal:

```text
Rebuild the selected /model.4 same-input ONNX-cut bucket map and choose exactly one local repair lane:
  A. output QuantizeLinear repair;
  B. thread/barrier overhead repair;
  C. no local repair, prepare next graph/block planning.
```

## Required starting point

Expected previous stage: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE34-VMADOT-CV2-PIPELINED-MICROKERNEL-001`

Use the actual Stage34 end head recorded in the Stage34 result packet.

## Hard boundaries

Do not implement a full YOLO26 engine, graph-wide scheduler, graph expansion, camera/full-image path, COCO/mAP, model FPS claim, production/default backend, `/data/ncnn` mutation, XSlim, vmadot1/2/3 integration, vmadotn, FP/vfmadot, CPU4-7 IME, or all-core OpenMP/default dispatch.

## Mandatory replay

Replay selected path:

```text
merge_repair: branch1_add_lut
mode: ime_threaded
output_quantize: rvv
thread_branch0=4 thread_branch1=4 thread_model4_cv2=4
taskset -c 0-3
warmup=10 runs=100 repeats=5
FRM sweep RNE/RTZ/RDN/RUP/RMM
expected output SHA: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
```

## Lane selection

Choose one:

```text
Lane A output QuantizeLinear:
  Select if output_quantize_us remains >=15% or absolute >5 ms.
  Candidate must be same-input ONNX-cut exact and FRM robust.

Lane B thread/barrier overhead:
  Select if thread_overhead_us remains material and locally reducible.
  Candidate must preserve CPU0-3-only IME and selected runner semantics.

Lane C no local repair:
  Select if no low-risk exact local repair can plausibly improve selected-cut total by >=5%.
```

Do not run broad optimization search. Do not compare as model FPS.
