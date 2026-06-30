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
  The current rt204 FP32 end-to-end 640 baseline is approximately
  `568.943 ms / 1.758 FPS` in `perf_test forward`, and
  `564.531 ms / 1.771 FPS` in app forward-only on the Ultralytics bus image.
- YOLO26 INT8 ONNX board acceleration is formally closed as blocked by rt204
  Q/DQ Conv compiler support until vendor/runtime changes or a separately
  proven partial fallback becomes useful.
- XSlim `2.1.0` was evaluated as the SpacemiT chip-aware PTQ path. Static
  XSlim PTQ did not produce an accepted YOLO26 INT8 model in the bounded gate:
  e2e `[1,300,6]` export fails inside XSlim/PPQ `ReduceMax`, and traditional
  `[1,84,8400]` export enters long block-wise calibration without producing a
  practical task-local candidate. XSlim `--dynq` produces CPU-good and rt204 EP
  runnable diagnostic models, but their graphs contain `Conv` plus
  `DequantizeLinear` weights and no `QLinearConv`/`QLinearMatMul` or static
  activation Q/DQ offload. This is a partial fallback/diagnostic only, not an
  accepted accelerated INT8 path.

## Raw Evidence

```text
/data/ncnn-logs/ort-logs/2026-06-29_16-03-40/
/data/ncnn-logs/ort-logs/2026-06-29_16-56-34/
/data/ncnn-logs/ort-logs/2026-06-29_21-43-36/
/data/ncnn-logs/ort-logs/2026-06-30_06-12-26/
/data/ncnn-logs/ort-logs/2026-06-30_07-56-51/
/data/ncnn-logs/ort-logs/2026-06-30_08-45-33/
```

## Next Gate

Use the FP32 package as the baseline for future work. Send the tiny Q/DQ Conv
`kernel_shape` repro to the runtime vendor or run a narrow partial-fallback
placement/performance gate for the stripped-kernel model. Treat XSlim dynamic
quantization as a separate compressed/weight-dequantized diagnostic lane unless
future XSlim releases produce a CPU-good static PTQ model that rt204 accelerates.
Keep YOLO11 rt204 reevaluation as a separate future adoption gate.
