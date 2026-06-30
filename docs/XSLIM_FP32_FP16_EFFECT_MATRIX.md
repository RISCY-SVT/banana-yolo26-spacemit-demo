# XSlim FP32/FP16 Effect Matrix

Raw evidence:

```text
/data/ncnn-logs/ort-logs/2026-06-30_14-21-27/
```

## Decision

```text
XSlim does not improve the accepted YOLO26 FP32/FP16 paths in this stage.
```

XSlim also does not change the YOLO26 INT8 closure decision.

## YOLO26 Results

| Artifact | Runtime | Metric class | Mean ms | FPS | Semantic status | Decision |
| --- | --- | --- | ---: | ---: | --- | --- |
| FP32 vanilla e2e | rt204 | `app forward-only`, 50x5 | 578.041776 | 1.729979 | pass | baseline |
| FP32 XSlim simplify-only | rt204 | `app forward-only`, 20x3 | 582.029471 | 1.718126 | pass | no app-level improvement |
| FP32 XSlim simplify-only | rt204 | `app full image single` | 705.793 | 1.417 | pass, 5 objects | no useful full-image gain |
| FP16 native body/head keep-IO | rt204 | `app forward-only`, 50x5 | 383.332266 | 2.608703 | pass | best FP16 path |
| FP16 XSlim | rt204 | load/app smoke | N/A | N/A | fail at `/model.23/Concat_6` | rejected |

The XSlim FP32 perf-test smoke was slightly faster than the vanilla short
perf-test row, but the app forward-only and full-image rows did not improve.
The accepted decision is therefore based on the app-level path.

## YOLO11 R&D-Copy Retrospective

These checks used copied YOLO11 artifacts in the YOLO26 R&D workspace. The
frozen production YOLO11 repository was not modified.

| Artifact | Runtime | Metric class | Mean ms | FPS | Status | Production implication |
| --- | --- | --- | ---: | ---: | --- | --- |
| YOLO11 dynamic640 INT8 | rt204 | `app forward-only`, 10x2 | 201.454112 | 4.963910 | pass | slower than frozen rt201 production; no policy change |
| YOLO11 FP32 vanilla | rt204 | `app forward-only`, 10x2 | 539.131433 | 1.854835 | pass | diagnostic only |
| YOLO11 FP32 XSlim simplify-only | rt204 | app forward | N/A | N/A | fail/hang after `YoloDecode` dispatch error | rejected |
| YOLO11 FP16 keep-IO | rt204 | `app forward-only`, 10x2 | 341.651391 | 2.926960 | pass | R&D signal only |
| YOLO11 FP16 XSlim | rt204 | app forward/full | N/A | N/A | fail/timeout after `YoloDecode` dispatch error | rejected |

## INT8 Closure Impact

None. The accepted INT8 status remains:

```text
YOLO26 INT8 ONNX board acceleration is blocked pending rt204/vendor EP compiler support.
```

XSlim static PTQ still has separate upstream/tooling blockers, and XSlim dynamic
quantization remains diagnostic only.
