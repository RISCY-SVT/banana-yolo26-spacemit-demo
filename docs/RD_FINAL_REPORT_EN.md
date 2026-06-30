# YOLO26 R&D Final Report

Evidence root for the closeout gate:

```text
/data/ncnn-logs/ort-logs/2026-06-30_18-30-54/
```

This report summarizes the isolated YOLO26 R&D line. It does not change the
frozen YOLO11 production release.

## Executive Summary

YOLO26n is technically usable on Banana-Pi BPI-F3 / SpacemiT K1X with
SpacemiT ONNX Runtime `2.0.4` in FP32 and in the native
body-FP16/head-FP32 keep-I/O form. The best local YOLO26 path is:

```text
YOLO26n 640 e2e, native body-FP16/head-FP32 keep-IO, rt204 SpaceMIT EP
```

The full-I/O FP16 candidate is also accepted for correctness and acceleration,
but it is not faster than keep-I/O FP16. YOLO26 INT8 remains blocked pending
vendor/runtime/tooling fixes. Frozen YOLO11 production remains the deployable
release.

## Frozen YOLO11 Baseline

Production repository:

```text
/data/banana-yolo11-spacemit-demo
production-2026-07-02 -> 9c0933be58ee122389d1a43f45f81e80655d6904
```

Frozen policy:

| Path | Runtime | Status |
| --- | --- | --- |
| Primary image visual | dynamic640 INT8 on `rt201` | production |
| Normal camera | dynamic640 INT8 on `rt201` | production |
| Fast-live camera | vendor320 INT8 on `rt123`, 320 letterbox | production |
| Vendor320 trusted visual | `rt123` | production |
| Vendor320 low-latency perf | raw `rt201` | perf-only |
| Vendor320 rt201 workaround | SHA256-guarded | non-default |
| FP16 | keep_io 640 on `rt201`/`rt202b1` | experimental |
| YOLO26n | P2 only in YOLO11 repo | not production |
| Stable rt202 | tested, not adopted | not production |

## Why YOLO26 Was Isolated

YOLO11 was already accepted, tagged, and mirrored. YOLO26 required new
Ultralytics export behavior, SpacemiT ORT `2.0.4`, FP16 and INT8 runtime
forensics, XSlim PTQ investigation, and runtime compatibility gates. Those
experiments were intentionally moved to:

```text
/data/banana-yolo26-spacemit-demo
```

The YOLO11 production repository was used only for read-only comparison.

## rt204 Runtime and API Findings

SpacemiT ORT `2.0.4` runs on K1X/X60 without SIGILL in default mode. It exposes
new diagnostic/runtime surfaces including `SPACEMIT_EP_DISABLE_PASSES_FILTER`
and operator strings such as `YoloDecode`, `GridSample`, `RotaryEmbedding`, and
`ArgMax`.

The `SPACEMIT_EP_PERFER_CORE_ARCH` spelling exists, but no useful documented
K1/K3/X60 override value was found. Default mode is the accepted rt204 mode for
this R&D line.

## YOLO26 Export/API Mismatch and Fix

The old Ultralytics `8.3.233` path exported `yolo26n.pt` as traditional
`[1,84,N]`, rejected the documented `end2end` argument, and reproduced the
large false `refrigerator` result.

Current Ultralytics fixed the oracle:

| Export | Output contract | Status |
| --- | --- | --- |
| default/e2e | `[1,300,6]` xyxy/conf/class_id | accepted |
| `end2end=False` | `[1,84,8400]` traditional output | accepted with decoder/NMS |

PyTorch, ONNX Runtime CPU, and rt204 SpaceMIT EP agree semantically for the
fixed FP32 oracle.

## FP32 Baseline

YOLO26 FP32 e2e 640 is the frozen R&D baseline. It is correct but slow on K1X.

| Metric class | Runtime | Mean ms | FPS |
| --- | --- | ---: | ---: |
| `perf_test forward` | rt204 SpaceMIT EP | 562.934799 | 1.776405 |
| `app forward-only` | rt204 SpaceMIT EP | 573.281930 | 1.744342 |
| `app full image benchmark` | rt204 SpaceMIT EP | 523.119274 | 1.911610 |

These are R&D baseline metrics, not YOLO11 production claims.

## FP16 Findings

The current closeout gate tested FP32, body/head keep-I/O FP16, full-I/O FP16,
direct full-model FP16, and XSlim FP16.

| Candidate | Input dtype | Output dtype | rt204 status | Decision |
| --- | --- | --- | --- | --- |
| FP32 e2e baseline | float32 | float32 | pass | baseline |
| Native body-FP16/head-FP32 keep-I/O | float32 | float16 | pass | best local path |
| Native body-FP16/head-FP32 full-I/O | float16 | float16 | pass | accepted, not faster |
| Native full-model FP16 | float16/float32 variants | mixed invalid head | load fail | rejected |
| XSlim FP16 | float32 | mixed invalid head | load fail | rejected |

Benchmark rows from the closeout gate:

| Variant | Metric class | Mean ms | FPS |
| --- | --- | ---: | ---: |
| FP32 e2e | `perf_test forward` | 562.934799 | 1.776405 |
| FP32 e2e | `app forward-only` | 573.281930 | 1.744342 |
| FP32 e2e | `app full image benchmark` | 523.119274 | 1.911610 |
| FP16 keep-I/O | `perf_test forward` | 379.777774 | 2.633119 |
| FP16 keep-I/O | `app forward-only` | 383.229967 | 2.609399 |
| FP16 keep-I/O | `app full image benchmark` | 398.091562 | 2.511985 |
| FP16 full-I/O | `perf_test forward` | 380.472087 | 2.628314 |
| FP16 full-I/O | `app forward-only` | 388.609808 | 2.573275 |
| FP16 full-I/O | `app full image benchmark` | 399.345008 | 2.504100 |

Decision: full-I/O FP16 is accepted for correctness and rt204 execution, but
keep-I/O FP16 remains the best local YOLO26 path because it is slightly faster
and avoids requiring FP16 application input.

## INT8 Attempts and Blockers

YOLO26 INT8 acceleration is not accepted.

| Path | Result |
| --- | --- |
| Ultralytics `quantize=8` | Q/DQ ONNX emitted, CPU oracle collapsed to zero detections. |
| Manual ORT Q/DQ | CPU-good, but rt204/legacy runtimes block Q/DQ Conv offload. |
| QOperator | Not accepted; no useful QLinear offload proof and weak parity/timing. |
| XSlim dynamic | CPU-good diagnostic only, not static activation INT8. |
| XSlim static e2e | Needs upstream fix for two-input `ReduceMax` handling. |
| XSlim static traditional | Emits ONNX in some configs, but CPU scores collapse to zero. |
| Legacy runtimes | No accepted accelerated Q/DQ path. |

The actionable rt204 blocker is:

```text
output_type not implemented for clip minmax
```

Minimal repros:

| Repro | Purpose |
| --- | --- |
| `15_conv_qdq_attr_kernel_shape.onnx` | Tiny Q/DQ Conv with explicit `kernel_shape=[3,3]`. |
| `07_yolo26_first_conv_qdq_output_block.onnx` | Real YOLO26 first-Conv extracted repro. |

## Vendor and Upstream Packets

Vendor/upstream reports are available in the R&D docs and raw log roots:

| Packet | Status |
| --- | --- |
| rt204 Q/DQ Conv `clip minmax` bug report | ready for vendor submission |
| XSlim ReduceMax static PTQ report | ready for upstream submission |
| XSlim traditional zero-score report | ready for upstream submission |

No issues were opened automatically in this task.

## YOLO11 rt204 and XSlim Retrospective

R&D-only copies of YOLO11 artifacts were tested under rt204. The frozen YOLO11
repository was not modified.

| Variant | Result |
| --- | --- |
| YOLO11 dynamic640 INT8 on rt204 | passes, but slower than production rt201 |
| YOLO11 FP16 keep-I/O on rt204 | passes as R&D signal |
| YOLO11 XSlim FP32/FP16 on rt204 | fails or times out after `YoloDecode` dispatch errors |

No missed YOLO11 production opportunity was proven.

## Final Recommended YOLO26 Path

Recommended local YOLO26 R&D artifact:

```text
YOLO26n 640 e2e native body-FP16/head-FP32 keep-I/O on rt204
```

It is correct on the public sanity suite and private canonical reference, and it
is the fastest locally accepted YOLO26 precision path. It is still an R&D path,
not a production replacement.

## Why YOLO26 Does Not Replace YOLO11 Production

YOLO26 FP16 improves over YOLO26 FP32, but remains slower than frozen YOLO11
INT8 production paths on K1X. YOLO26 INT8 acceleration is blocked. YOLO11
production remains the accepted deployable release.

## Open P2 and Future Work

- Submit rt204 Q/DQ Conv repro to the runtime vendor.
- Submit XSlim ReduceMax and traditional zero-score reports upstream.
- Re-run INT8 only after vendor/runtime/tooling changes.
- Optional future YOLO26 FP16 polish if app-level FP16 I/O becomes useful.
- Keep YOLO11 rt204 reevaluation as a separate adoption gate.

## Repositories, Commits, and Runtime Evidence

| Item | Value |
| --- | --- |
| Frozen YOLO11 production commit/tag | `9c0933be58ee122389d1a43f45f81e80655d6904`, `production-2026-07-02` |
| YOLO26 R&D branch | `yolo26-rd-bootstrap` |
| YOLO26 closeout start HEAD | `02ae1a0c760598e6fd7e396944a4395e9db941b9` |
| Runtime | `spacemit-ort.riscv64.2.0.4` (`rt204`) |
| Closeout log root | `/data/ncnn-logs/ort-logs/2026-06-30_18-30-54/` |

The exact final R&D commit containing this report is recorded in the closeout
run summary and in git history.

