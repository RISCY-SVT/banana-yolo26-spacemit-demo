# YOLO26 FP32 Baseline

Raw evidence:

```text
/data/ncnn-logs/ort-logs/2026-06-30_07-56-51/
/data/ncnn-logs/ort-logs/2026-06-30_14-21-27/
```

## Scope

This is an isolated YOLO26 R&D baseline. It does not modify the frozen
`banana-yolo11-spacemit-demo` production repository and it does not make
YOLO26 production claims.

The current working YOLO26 path is:

| Item | Value |
| --- | --- |
| Model | `yolo26n.pt` public Ultralytics checkpoint |
| Export | YOLO26 end-to-end FP32 ONNX |
| ONNX output contract | `[1,300,6]` |
| Runtime | `spacemit-ort.riscv64.2.0.4` (`rt204`) |
| Board | Banana-Pi BPI-F3 / SpacemiT K1X / X60 |
| Provider | SpaceMITExecutionProvider |

## Standard Sanity Suite

The reproducible public sanity suite is task-local under:

```text
.deps/datasets/yolo26_standard_sanity/
```

It includes Ultralytics package assets, a small COCO-derived image subset,
a synthetic blank image, and one private canonical photo only as an additional
non-public reference. The public suite is the default baseline; the private
canonical photo is not the primary benchmark.

Manifest copies are stored in the raw run directory:

```text
tables/standard_image_suite_manifest.md
tables/standard_image_suite_manifest.tsv
```

## FP32 Oracle Result

PyTorch, ONNX Runtime CPU, and rt204 SpaceMIT EP agree semantically on the
standard suite. Small score/count differences between PyTorch and ONNX are
treated as exporter/runtime numeric tolerance when the object classes and
overall scene interpretation remain the same.

| Image class | PyTorch | ONNX CPU | rt204 EP | Verdict |
| --- | --- | --- | --- | --- |
| Ultralytics bus | bus/person detections | bus/person detections | bus/person detections | pass |
| Ultralytics zidane | person/tie detections | person/tie detections | person/tie detections | pass |
| COCO-derived images | expected COCO objects | matching COCO objects | matching COCO objects | pass |
| Blank white | 0 detections | 0 detections | 0 detections | pass |
| Private canonical reference | laptop/person/bottle | laptop/person/bottle | laptop/person/bottle | pass |

Detailed oracle tables:

```text
tables/yolo26_fp32_oracle_matrix.md
tables/yolo26_rt204_ep_decode_matrix.md
```

## FP32 Performance Baseline

These are R&D baseline numbers, not production FPS claims. Metric classes must
not be mixed.

### Frozen Baseline, 2026-06-30 Effect Matrix Pass

The current frozen FP32 configuration is `cluster0`, 4 threads, rt204
SpaceMITExecutionProvider. Threads 1 and 2 are slower; 8 threads is not valid on
this board/runtime path because rt204 reports four available AI cores.

| Metric class | Runtime | Model | Mean latency ms | FPS | Notes |
| --- | --- | --- | ---: | ---: | --- |
| `perf_test forward` | rt204 SpaceMIT EP | YOLO26n FP32 e2e 640 | 572.153613 | 1.74774 | 100 runs, `taskset` cluster0. |
| `app forward-only` | rt204 SpaceMIT EP | YOLO26n FP32 e2e 640 | 578.041776 | 1.729979 | 50x5 runs, cluster0, 4 threads. |
| `app full image benchmark` | rt204 SpaceMIT EP | YOLO26n FP32 e2e 640 | 522.079210 | 1.915418 | Bus image, 20x3 runs. |
| `app full image single` | rt204 SpaceMIT EP | YOLO26n FP32 e2e 640 | 692-716 | 1.397-1.444 | Blank, COCO-like, bus, and private canonical single-image smoke. |

Thread sweep:

| Threads | `perf_test` mean ms | `perf_test` FPS | App forward mean ms | App forward FPS | Verdict |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 966.161291 | 1.03501 | 1031.074570 | 0.969862 | slower |
| 2 | 728.455689 | 1.37274 | 714.754028 | 1.399083 | slower |
| 4 | 564.233550 | 1.77227 | 569.225401 | 1.756773 | best valid config |
| 8 | N/A | N/A | N/A | N/A | invalid: only four AI cores available |

### Earlier FP32 Baseline Package

| Metric class | Runtime | Model | Mean latency ms | FPS | Notes |
| --- | --- | --- | ---: | ---: | --- |
| `perf_test forward` | rt204 SpaceMIT EP | YOLO26n FP32 e2e 640 | 568.943339 | 1.75761 | Pure ORT forward ceiling, 100 runs. |
| `app forward-only` | rt204 SpaceMIT EP | YOLO26n FP32 e2e 640 | 564.531070 | 1.771382 | App session path, 30x3 runs. |
| `app full image benchmark` | rt204 SpaceMIT EP | YOLO26n FP32 e2e 640 | 521.868004 | 1.916193 | Bus image, 20x3 runs. |
| `app full image single` | rt204 SpaceMIT EP | YOLO26n FP32 e2e 640 | 696-715 | 1.40-1.44 | Single-image smoke on bus, COCO-like, and private canonical. |

Full timing table:

```text
tables/yolo26_fp32_perf_matrix.md
tables/yolo26_fp32_perf_matrix.tsv
```

## Reproducible Helpers

The repo includes:

```text
tools/yolo26_fp32_baseline.py
```

The helper prepares the standard image suite, runs PyTorch and ONNX CPU
oracles, creates deterministic `640x640` NCHW tensor inputs, and decodes
board-side rt204 tensor dumps.

Example:

```bash
.deps/venvs/ultralytics_latest/bin/python tools/yolo26_fp32_baseline.py prepare-suite
.deps/venvs/ultralytics_latest/bin/python tools/yolo26_fp32_baseline.py host-oracle \
  --suite .deps/datasets/yolo26_standard_sanity \
  --model-onnx .deps/probes/models_forensics/yolo26n_latest_e2e640.onnx \
  --model-pt .deps/models/yolo26/yolo26n.pt \
  --out /tmp/yolo26_fp32_oracle
```

## Decision

YOLO26 FP32 is a working R&D baseline on K1X with rt204. It is not currently
competitive with the frozen YOLO11 production INT8 branch on latency, but it is
the correct baseline for future YOLO26 runtime/operator work.

For precision comparison, see `docs/YOLO26_FP16_STATUS.md`. The accepted FP16
body/head keep-IO artifact is faster than this FP32 baseline, but it remains
R&D-only.
