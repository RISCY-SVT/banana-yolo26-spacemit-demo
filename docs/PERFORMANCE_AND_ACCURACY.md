# Performance and Accuracy

## Accuracy

The final exact contract remains `K1X_INT8_V1`.

| Surface | Result |
|---|---:|
| COCO val2017 images | 5000/5000 |
| mAP50-95 | 0.3707408944391919 |
| Accepted prediction SHA-256 | `cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda` |
| Integer boundaries | 215/215 exact |
| F0-F7, bus, Zidane | exact |
| Known F0 output hash | `0xd43f5e018b415631` |

## Performance Interpretation

Canonical Stage57 rows are generated in the Stage57 final report. The selected
source bundle (E2c5 plus attention C8 epilogue) cleared the randomized full-model
selection gate with a paired mean improvement of 3.790254897% against the
reproduced Stage56 source control. Final release/O2 statistics are reported after
the release-only 500-run and 10,000-run measurements.

Do not combine columns from different sample surfaces. In particular:

- fixed preprocessed-input executor latency is pure executor timing;
- RGB includes compact input copy/quantization but not JPEG decode or resize;
- serial pipeline depends on preprocessor CPU placement and OpenCV thread count;
- double-buffer interval is steady-state throughput, not one-frame model latency;
- 500-sample and 10,000-run tail statistics are separate rows.

## Accepted Stage56 Reference

| Surface | Mean (us) | p95 (us) |
|---|---:|---:|
| compatibility, 500 | 156620.000 | 160377.150 |
| dedicated O2 low latency, 500 | 142412.512 | 142893.100 |
| dedicated O2 low latency, 10000 | 142444.857 | 142984.050 |
| real 100-image corpus | 141733.800 | separately reported |
| matched B120 ORT | 459954.787608 | separately reported |

Stage57 final values supersede Stage56 only for the installed 0.9.0 release
route. Neither stage demonstrates 20 FPS, camera-service readiness, or production
certification.

## HPM Language

Stage56 HPM values are event counts per cycle. They prove a measured surface
dominated by backend/structural/dependency-or-latency events, not by frontend,
I-cache, or branch events. They do not prove that every backend event is a Q62
dependency. The L1D value is not a miss/access ratio, and a DTLB miss ratio is
unknown because the matching access event returned zero.
