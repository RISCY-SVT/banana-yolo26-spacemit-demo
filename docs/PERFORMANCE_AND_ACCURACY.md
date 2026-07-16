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

The selected source bundle (E2c5 plus attention C8 epilogue) cleared the
randomized full-model selection gate with a paired mean improvement of
3.790254897% against the reproduced Stage56 source control. The installed release
then passed independent 500-sample and long-soak measurements.

### Fixed Preprocessed Input

| Surface | Samples | Mean (us) | Median (us) | p95 (us) | p99 (us) | p99.9 (us) | Max (us) |
|---|---:|---:|---:|---:|---:|---:|---:|
| compatibility | 500 | 147156.530 | 146493.000 | 151343.100 | 157373.510 | 162881.304 | 164031.000 |
| low-latency, original OS | 500 | 133674.926 | 133479.500 | 135944.150 | 136817.000 | 138291.851 | 139465.000 |
| low-latency-dedicated, O2 | 500 | 133305.232 | 133307.500 | 133825.050 | 134031.490 | 134956.415 | 135413.000 |
| compatibility soak | 10000 | 147746.663 | 147181.500 | 151137.950 | 157663.060 | 163959.195 | 181280.000 |
| low-latency-dedicated O2 soak | 13500 | 135040.533 | 134995.000 | 135637.000 | 136675.070 | 138660.577 | 140242.000 |

The 500-sample O2 mean is 6.395163% lower than the accepted Stage56 O2 mean and
71.222984% lower than matched B120 ORT. It corresponds to 7.501594 pure-model
inferences per second. These figures do not constitute a 20 FPS claim.

### Other Surfaces

| Surface | Samples | Mean/interval (us) | Notes |
|---|---:|---:|---|
| 100-image in-memory corpus | 100 | 132913.617 | pure executor, data-dependent inputs |
| RGB8 input | 500 | 131318.676 | compact RGB copy plus executor; no JPEG/resize |
| serial preloaded-image pipeline | 500 | 188654.187 | preprocessing on CPU4; OpenCV inherited default |
| double-buffer frame interval | 500 | 140555.108 | CPU5-7 preprocessing, OpenCV 3 threads; 7.114647 FPS throughput |
| matched B120 ORT | 500 | 463234.271 | per-inference comparison distribution |

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
