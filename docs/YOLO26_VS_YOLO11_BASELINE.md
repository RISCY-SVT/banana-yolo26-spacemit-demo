# YOLO26 FP32 vs Frozen YOLO11 Production Baseline

YOLO11 production source of truth:

```text
/data/banana-yolo11-spacemit-demo/docs/FPS_SUMMARY.md
production-2026-07-02 -> 9c0933be58ee122389d1a43f45f81e80655d6904
```

YOLO26 R&D evidence:

```text
/data/ncnn-logs/ort-logs/2026-06-30_07-56-51/
/data/ncnn-logs/ort-logs/2026-06-30_14-21-27/
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
| YOLO26n FP32 e2e | rt204 | `perf_test forward` | 572.153613 | 1.74774 | R&D frozen FP32 baseline |
| YOLO26n FP32 e2e | rt204 | `app forward-only` | 578.041776 | 1.729979 | R&D frozen FP32 baseline |
| YOLO26n FP32 e2e | rt204 | `app full image benchmark` | 522.079210 | 1.915418 | R&D frozen FP32 baseline |
| YOLO26n FP16 body/head keep-IO | rt204 | `perf_test forward` | 398.610405 | 2.50864 | R&D FP16 baseline |
| YOLO26n FP16 body/head keep-IO | rt204 | `app forward-only` | 383.332266 | 2.608703 | R&D FP16 baseline |
| YOLO26n FP16 body/head keep-IO | rt204 | `app full image benchmark` | 399.035502 | 2.506043 | R&D FP16 baseline |

## R&D-Only YOLO11 rt204 Retrospective

The 2026-06-30 effect-matrix pass also copied selected YOLO11 artifacts into
the YOLO26 R&D workspace and tested rt204. This did not modify the frozen
production YOLO11 repository.

| Variant | Runtime | Metric class | Mean latency ms | FPS | Status |
| --- | --- | --- | ---: | ---: | --- |
| YOLO11 dynamic640 INT8 copy | rt204 | `app forward-only` | 201.454112 | 4.963910 | pass, but slower than production rt201 |
| YOLO11 FP32 vanilla copy | rt204 | `app forward-only` | 539.131433 | 1.854835 | pass, diagnostic only |
| YOLO11 FP32 XSlim simplify copy | rt204 | app forward | N/A | N/A | fails/hangs after `YoloDecode` dispatch error |
| YOLO11 FP16 keep-IO copy | rt204 | `app forward-only` | 341.651391 | 2.926960 | pass, R&D signal only |
| YOLO11 FP16 XSlim copy | rt204 | app forward/full | N/A | N/A | fails/timeouts after `YoloDecode` dispatch error |

## Semantic Comparison

YOLO26 FP32 now gives sane detections on public COCO-derived images and on the
private canonical reference, including bus/person, person/tie, bear, teddy
bear, baseball glove/person, tennis, and laptop/person/bottle scenes. The
earlier giant false refrigerator result was fixed by moving from the old
Ultralytics export path to the current YOLO26 end-to-end export/decoder chain.

## Decision

YOLO26 FP32 is not currently competitive with YOLO11 production INT8 on K1X.
YOLO26 FP16 is materially faster than YOLO26 FP32, but it is still slower than
the frozen YOLO11 production INT8 branch. Frozen YOLO11 remains the production
deliverable.
