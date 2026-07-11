# Accuracy audit

Directional 500-image COCO val2017 result:

| Surface | mAP50-95 | mAP50 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|
| FP32 operational | 0.446714 | 0.613333 | 0.225559 | 0.505159 | 0.606809 |
| INT8 semantic | 0.410683 | 0.570051 | 0.206204 | 0.476896 | 0.569516 |
| INT8 operational | 0.373479 | 0.532193 | 0.182847 | 0.448533 | 0.530056 |

Semantic INT8 loses `-0.036031` absolute AP (`-3.603` AP
points) versus FP32; operational INT8 loses `-0.073235` (`-7.324`
points). The current manual QDQ surface misses both the within-1-AP and within-2-AP
scenarios on this subset. Full val2017 remains unknown; no accuracy result is
transferred to the incomplete custom board engine.
