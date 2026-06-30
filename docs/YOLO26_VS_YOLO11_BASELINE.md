# YOLO26 FP32 vs Frozen YOLO11 Production Baseline

YOLO11 production source of truth:

```text
/data/banana-yolo11-spacemit-demo/docs/FPS_SUMMARY.md
production-2026-07-02 -> 9c0933be58ee122389d1a43f45f81e80655d6904
```

YOLO26 R&D evidence:

```text
/data/ncnn-logs/ort-logs/2026-06-30_07-56-51/
```

## Comparison Table

Do not compare unlike metric classes without this caveat: `perf_test forward`,
`app forward-only`, `app full image`, and camera FPS measure different
workloads.

| Variant | Runtime | Metric class | Mean latency ms | FPS | Status |
| --- | --- | --- | ---: | ---: | --- |
| YOLO11 dynamic640 INT8 primary | rt201 | `perf_test forward` | 190.024 | 5.2623 | production-supported |
| YOLO11 dynamic640 INT8 primary | rt201 | `app forward-only` | 190.567794 | 5.247476 | production-supported |
| YOLO11 dynamic640 INT8 primary | rt201 | `app full image` | 233.480423 | 4.283014 | production-supported |
| YOLO11 fast-live vendor320 | rt123 | `app full image` | 57.540777 | 17.378980 | production-supported fast-live branch |
| YOLO26n FP32 e2e | rt204 | `perf_test forward` | 568.943339 | 1.75761 | R&D working baseline |
| YOLO26n FP32 e2e | rt204 | `app forward-only` | 564.531070 | 1.771382 | R&D working baseline |
| YOLO26n FP32 e2e | rt204 | `app full image benchmark` | 521.868004 | 1.916193 | R&D working baseline |

## Semantic Comparison

YOLO26 FP32 now gives sane detections on public COCO-derived images and on the
private canonical reference, including bus/person, person/tie, bear, teddy
bear, baseball glove/person, tennis, and laptop/person/bottle scenes. The
earlier giant false refrigerator result was fixed by moving from the old
Ultralytics export path to the current YOLO26 end-to-end export/decoder chain.

## Decision

YOLO26 FP32 is not currently competitive with YOLO11 production INT8 on K1X.
It is a correct R&D baseline for future operator/runtime work, while frozen
YOLO11 remains the production deliverable.
