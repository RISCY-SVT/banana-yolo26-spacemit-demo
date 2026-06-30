# YOLO26 R&D Status

This repository is an isolated YOLO26 R&D workspace seeded from the frozen
YOLO11 production tag. It has no production claims and no active production
remote.

## Current Status

- Frozen YOLO11 production repo remains read-only and unchanged.
- Current accepted YOLO26 CPU oracle is Ultralytics `8.4.82`.
- Correct default/latest YOLO26 ONNX contract is end-to-end `[1,300,6]`.
- Traditional export is available with latest `end2end=False` and has shape
  `[1,84,8400]`.
- RT204 FP32 SpaceMIT EP smoke passes on BPI-F3/K1X for latest 640 end-to-end
  and traditional ONNX exports.
- Ultralytics `quantize=8` remains unsuitable: task-local small and
  representative calibration runs exported Q/DQ ONNX models, but the CPU oracle
  produced zero detections.
- Manual ONNX Runtime static Q/DQ over `Conv`/`MatMul` creates CPU-good INT8
  candidates for both YOLO26 end-to-end `[1,300,6]` and traditional
  `[1,84,8400]` contracts.
- RT204 SpaceMIT EP currently blocks full accelerated board Q/DQ INT8:
  CPU-good Q/DQ candidates fail EP compilation with
  `output_type not implemented for clip minmax`.
- The current smallest repro is `15_conv_qdq_attr_kernel_shape.onnx`, a tiny
  synthetic Q/DQ Conv graph with explicit `kernel_shape=[3,3]`. The extracted
  `yolo26_first_conv_qdq_output_block.onnx` remains a supplemental real-graph
  repro.
- Provider pass filters do not avoid the compile failure. The smallest correct
  Q/DQ fallback is
  `SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=QuantizeLinear;DequantizeLinear;Conv`,
  which is CPU-heavy and not an accelerated INT8 path.
- QOperator e2e/traditional candidates are not ready for a performance gate:
  raw CPU/EP parity is loose, the bus oracle changes top semantics under EP,
  dumped subgraphs do not include `QLinearConv`/`QLinearMatMul`, and bounded
  timing smoke is slower than FP32.
- Stripping optional Conv `kernel_shape` attributes from the full Q/DQ model is
  CPU-exact and avoids the first Conv blocker, but exposes an attention MatMul
  issue. `SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=MatMul;Add` restores smoke
  semantics for that stripped model and is only a partial fallback candidate.
- RT204 YOLO11 direct tensor-probe results are promising for a future separate
  adoption gate, but this R&D repository does not change the frozen YOLO11
  production policy.
- YOLO26 FP32 now has a reproducible public sanity suite and baseline package.
  The current frozen rt204 FP32 end-to-end 640 baseline uses cluster0 with
  4 threads: `572.153613 ms / 1.74774 FPS` in `perf_test forward`,
  `578.041776 ms / 1.729979 FPS` in app forward-only, and
  `522.079210 ms / 1.915418 FPS` in app full-image benchmark.
- YOLO26 FP16 is available as a native body-FP16/head-FP32 keep-IO R&D model.
  It keeps float32 app input, outputs float16, passes the public sanity suite,
  and runs on rt204 at `383.332266 ms / 2.608703 FPS` app forward-only.
- YOLO26 INT8 ONNX board acceleration is formally closed as blocked by rt204
  Q/DQ Conv compiler support until vendor/runtime changes or a separately
  proven partial fallback becomes useful.
- XSlim `2.1.0` was evaluated as the SpacemiT chip-aware PTQ path. The deeper
  static PTQ follow-up narrowed the e2e blocker to a generic XSlim/PPQ
  two-input `ReduceMax` executor failure that reproduces on tiny ONNX models.
  Traditional `[1,84,8400]` `minmax` and `percentile` configs now emit ONNX,
  but both are CPU-bad because all class scores collapse to zero. XSlim
  `--dynq` remains CPU-good and rt204-runnable diagnostic only; it is not
  accepted as accelerated static INT8.
- Direct full-model FP16 conversions and XSlim FP16 are rejected for the current
  YOLO26 e2e graph because the head contains mixed dtype hazards. XSlim FP32
  simplify-only does not improve YOLO26 app-level performance.
- R&D-copy YOLO11 rt204 checks did not reveal a missed production opportunity:
  dynamic640 INT8 runs on rt204 but is slower than frozen rt201 production, and
  YOLO11 XSlim FP32/FP16 fail or time out after rt204 `YoloDecode` dispatch
  errors.
- Legacy-runtime Q/DQ sanity testing did not find an accepted YOLO26 INT8
  acceleration path. `rt201`, `rt202b1`, and `rt204` reproduce the Q/DQ Conv
  `clip minmax` blocker; stable `rt202` fails with TCM allocation; and `rt123`
  full-model rows do not preserve same-runtime CPU hash parity. Filter rows are
  diagnostic fallback only.

## Raw Evidence

```text
/data/ncnn-logs/ort-logs/2026-06-29_16-03-40/
/data/ncnn-logs/ort-logs/2026-06-29_16-56-34/
/data/ncnn-logs/ort-logs/2026-06-29_21-43-36/
/data/ncnn-logs/ort-logs/2026-06-30_06-12-26/
/data/ncnn-logs/ort-logs/2026-06-30_07-56-51/
/data/ncnn-logs/ort-logs/2026-06-30_08-45-33/
/data/ncnn-logs/ort-logs/2026-06-30_09-38-36/
/data/ncnn-logs/ort-logs/2026-06-30_14-21-27/
/data/ncnn-logs/ort-logs/2026-06-30_17-29-25/
```

## Next Gate

Use the FP32 package as the baseline for future work. Send the tiny Q/DQ Conv
`kernel_shape` repro to the runtime vendor, and send the XSlim ReduceMax plus
traditional zero-score reports to the XSlim maintainers. Treat XSlim dynamic
quantization as a separate compressed/weight-dequantized diagnostic lane unless
future XSlim releases produce a CPU-good static PTQ model that rt204
accelerates. Use the FP16 body/head keep-IO artifact as the current best YOLO26
precision baseline. The next practical stage is FP16 full-I/O compatibility or
vendor/upstream issue submission; another INT8 runtime search is not justified
until one of those upstream blockers changes. Keep YOLO11 rt204 reevaluation as
a separate future adoption gate.
