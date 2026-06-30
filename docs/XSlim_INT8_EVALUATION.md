# XSlim INT8 Evaluation

Raw evidence:

```text
/data/ncnn-logs/ort-logs/2026-06-30_08-45-33/
```

## Scope

This pass evaluated SpacemiT XSlim as the chip-aware quantization path for
YOLO26n INT8 on SpacemiT ORT `2.0.4` (`rt204`). It did not change the frozen
YOLO11 production repository or production runtime/model policy.

## Sources Checked

- XSlim GitHub repository and docs: `https://github.com/spacemit-com/xslim`
- XSlim docs and samples under `doc/` and `samples/`
- Ultralytics YOLO26, end-to-end detection, and export documentation
- ONNX Runtime quantization documentation
- SpacemiT ORT `2.0.4` release notes

## XSlim Version

| Component | Version / commit | Use |
| --- | --- | --- |
| XSlim selected baseline | `2.1.0`, git `bea43d6ae4eafcb8575e87ddf8f18f5c33016b20` | Main evaluation version. |
| XSlim main sanity | `2.1.1`, git `c246694a1eba8d7689c43ba7b5f469bb0cb29c95` | Checked whether the e2e static export failure changed on main. |

XSlim was installed in task-local virtual environments under `.deps/`; these are
not tracked repository artifacts.

## Static PTQ Result

Static XSlim PTQ did not produce an accepted YOLO26 INT8 candidate in this pass.

| Candidate family | Result | Evidence |
| --- | --- | --- |
| YOLO26 e2e `[1,300,6]` static PTQ | blocked before output model | XSlim/PPQ executor fails at `/model.23/ReduceMax` with `ValueError: too many values to unpack (expected 1)`. This reproduced on XSlim `2.1.0` and main `2.1.1`. |
| YOLO26 traditional `[1,84,8400]` static PTQ | not accepted | It passes the e2e `ReduceMax` point, but bounded attempts entered long block-wise calibration with no practical task-local output. No CPU-good static XSlim ONNX was produced. |

## Dynamic Quantization Diagnostic

XSlim `--dynq` produced runnable diagnostic models for both YOLO26 output
contracts:

| Model | Contract | SHA256 | Graph summary | CPU oracle | rt204 EP |
| --- | --- | --- | --- | --- | --- |
| `e2e_xslim_dynq.onnx` | `[1,300,6]` | `ae220ace468dcae3555a4f60fa051ae2ea2d560780bd3f2dd959d4670225cb3c` | `Conv=102`, `DequantizeLinear=102`, `QuantizeLinear=0`, `QLinearConv=0` | pass on public sanity set | pass on bounded board smoke |
| `traditional_xslim_dynq.onnx` | `[1,84,8400]` | `fda140f340acc0ee1ee5517ca63e80410497f5e80e760f890b83642d4c7822ef` | `Conv=102`, `DequantizeLinear=102`, `QuantizeLinear=0`, `QLinearConv=0` | pass on public sanity set | pass on bounded board smoke |

This is useful evidence that XSlim can emit a rt204-runnable compressed graph,
but it is not the desired static INT8/QDQ/QOperator acceleration path. The
dynamic graphs contain dequantized weights and normal `Conv` nodes, not
offloaded `QLinearConv`/`QLinearMatMul` or static Q/DQ activation patterns.

## Board Smoke

The bounded board smoke used `rt204_tensor_probe` on BPI-F3/K1X. Both XSlim
dynamic models executed with CPU and SpaceMIT EP on the selected public images:
Ultralytics bus, a COCO-like bear image, and a blank negative image.

| Model | Provider | Process/session smoke mean |
| --- | --- | ---: |
| e2e dynamic | CPU | `5.407 s` |
| e2e dynamic | SpaceMIT EP | `1.257 s` |
| traditional dynamic | CPU | `5.363 s` |
| traditional dynamic | SpaceMIT EP | `1.240 s` |

This timing includes process start, session/model load, possible compilation,
and one inference. It is not a headline FPS benchmark. It only proves that the
dynamic XSlim graph can use the EP path and is faster than CPU in this bounded
diagnostic.

## Known Blocker Comparison

XSlim dynamic quantization avoids the previous Q/DQ Conv `clip minmax` compiler
failure because it does not generate the same static Q/DQ Conv pattern. That
does not fix the blocker for CPU-good manual ONNX Runtime Q/DQ INT8 models:

```text
output_type not implemented for clip minmax
```

The existing vendor repros remain valid:

- tiny repro: `15_conv_qdq_attr_kernel_shape.onnx`;
- real YOLO26-derived repro: `07_yolo26_first_conv_qdq_output_block.onnx`.

## Decision

```text
XSLIM_YOLO26_INT8_PARTIAL_FALLBACK_ONLY
```

XSlim did not change the current R&D decision. YOLO26 static INT8 ONNX board
acceleration remains blocked by rt204 Q/DQ Conv compiler support. XSlim dynamic
quantization is a useful diagnostic and possible future compressed-float/weight
dequantized lane, but it is not accepted as accelerated YOLO26 INT8.

## YOLO11 Retrospective

This pass found no production-impacting miss in the frozen YOLO11 release:

- YOLO11 already has validated production INT8 paths on `rt201`/`rt123`.
- XSlim static did not produce an accepted YOLO26 INT8 candidate here.
- XSlim dynamic is not the same as the production static INT8 target and was
  not proven to improve YOLO11 production app/camera behavior.

YOLO11 + XSlim remains a possible post-release R&D lane, not a retroactive
production blocker.

## Static PTQ Follow-Up

A follow-up pass in:

```text
/data/ncnn-logs/ort-logs/2026-06-30_09-38-36/
```

refined the static PTQ status:

- the e2e `[1,300,6]` failure is a generic XSlim/PPQ ReduceMax executor issue,
  reproduced with tiny ONNX models containing two-input ReduceMax after XSlim's
  internal opset conversion;
- traditional `[1,84,8400]` `minmax` and `percentile` static configs finish and
  emit Q/DQ ONNX models;
- those static traditional models are CPU-bad because class score channels are
  all zero on public sanity images;
- a diagnostic rt204 run of the CPU-bad `minmax` model executes, but that is not
  a usable INT8 path.

Updated static decision:

```text
XSLIM_STATIC_YOLO26_INT8_NEEDS_UPSTREAM_FIX
```
