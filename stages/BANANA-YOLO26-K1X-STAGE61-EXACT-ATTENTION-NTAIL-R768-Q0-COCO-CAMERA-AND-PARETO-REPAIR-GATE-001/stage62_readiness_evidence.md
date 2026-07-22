# Later Calibration-Decision Evidence

This file is an evidence handoff only. It is not a Stage62 prompt, task packet,
branch, commit instruction, or launch authorization.

Stage61 provides nine immutable Q0 profile/model/package identities, exact
route counts, complete-model timings, full COCO metrics, per-class and size-bin
tables, and R768 memory/camera measurements. These rows can bound a later human
decision on whether deterministic per-resolution calibration is worth
authorizing.

## Q0 Bounds

| R | Mean (ms) | mAP50-95 | AP loss vs R640 | Attention route |
| ---: | ---: | ---: | ---: | --- |
| 640 | 131.155 | 0.370741 | 0.000 | aligned N16 control |
| 512 | 94.117 | 0.347630 | 2.311 | aligned N16 |
| 448 | 64.266 | 0.332627 | 3.811 | N16 + N4 tail |
| 416 | 55.808 | 0.317789 | 5.295 | N16 + N8 + N4 tail |
| 384 | 47.380 | 0.306537 | 6.420 | aligned N16 |
| 352 | 40.797 | 0.289709 | 8.103 | N16 + N8 + N4 tail |
| 320 | 34.209 | 0.276269 | 9.447 | N16 + N4 tail |
| 256 | 24.350 | 0.231262 | 13.948 | aligned N16 |
| 768 | 197.530 | 0.373550 | -0.281 | aligned N16 |

The exact tail repair removes runtime alignment as the explanation for the
remaining smaller-profile accuracy/latency tradeoff. R512 is the highest
accuracy smaller bound. R448 is the first repaired tail profile below 70 ms,
and R416 is the first repaired tail profile below 60 ms; both still miss the
fixed accuracy limits. If calibration is later authorized, these are the most
useful initial priorities. R768 is a quality bound with mixed AP-small,
AP-medium, AP-large, and per-class changes, not a preselected target.

R768 identity is static model
`3bb1695a5506b9e0c15ce4c511c30d3006db212c7c0c4ff5fb2c289183edfc8b`
and package manifest
`3fd4d004e92c4238c69c2d07bc1eedcee92c1968984f17d2b796b7bf01b4e0be`.
Its 3.984750 GMAC graph uses 11,796,480 arena bytes and 576 aligned attention
tokens. Full per-class, size-bin, memory, route-count, pipeline, and camera
rows are referenced by the Stage61 report directory rather than duplicated
here.

No PTQ, Q1 calibration, training, QAT, distillation, topology change, or
model-executor co-design was performed. The missing prerequisites are listed in
`stage62_calibration_prerequisites.tsv`; in particular, no approved auditable
calibration corpus/list/seed currently exists in this lane.
