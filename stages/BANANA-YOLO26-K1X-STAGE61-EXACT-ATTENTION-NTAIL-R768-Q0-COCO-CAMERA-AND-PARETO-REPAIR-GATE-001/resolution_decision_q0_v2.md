# Stage61 Q0 Resolution Decision

## Decision

The exact attention N-tail dispatcher is selected on the Stage61 research
branch. It removes the whole-MatMul scalar cliff at R448, R416, R352, and R320
without changing any output byte. It does not change the accepted 0.9.3
release branch or select a new public/default input resolution.

No Q0 profile clears either fixed deployment gate. The R640 release profile
therefore remains the control, and `selected_resolution` is `none`. This is a
measurement result, not authorization for PTQ, training, or co-design.

## Repaired Pareto Surface

| R | Mean (ms) | p95 (ms) | Pure FPS | mAP50-95 | Loss vs R640 (AP) | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 256 | 24.350 | 24.560 | 41.068 | 0.231262 | 13.948 | diagnostic lower bound |
| 320 | 34.209 | 34.325 | 29.232 | 0.276269 | 9.447 | diagnostic-only |
| 352 | 40.797 | 40.924 | 24.511 | 0.289709 | 8.103 | diagnostic-only |
| 384 | 47.380 | 47.509 | 21.106 | 0.306537 | 6.420 | diagnostic-only |
| 416 | 55.808 | 56.009 | 17.919 | 0.317789 | 5.295 | latency gate only; accuracy fails |
| 448 | 64.266 | 64.451 | 15.560 | 0.332627 | 3.811 | accuracy gate fails |
| 512 | 94.117 | 94.384 | 10.625 | 0.347630 | 2.311 | highest-accuracy smaller control |
| 640 | 131.155 | 131.555 | 7.625 | 0.370741 | 0.000 | accepted release control |
| 768 | 197.530 | 198.616 | 5.063 | 0.373550 | -0.281 | quality tradeoff; no promotion |

All nine rows are non-dominated in the measured Q0 latency/mAP plane. That
does not make every row a deployment candidate. The predeclared strong gate
requires at most 70 ms and at most 1.0 AP loss; the latency/accuracy gate
requires at most 60 ms and at most 1.5 AP loss. No row passes either gate.

Pure-model FPS is not camera FPS. Values above 20 pure inferences/s at R384,
R352, R320, and R256 carry large COCO accuracy losses and remain diagnostic.

## Attention Dispatcher Result

Same-session 1000-sample-per-arm ABBA shows the complete-model effect:

| R | Stage60 control (ms) | N-tail (ms) | Delta | Tail route |
| ---: | ---: | ---: | ---: | --- |
| 448 | 167.071 | 64.290 | -61.519% | N4 |
| 416 | 130.211 | 55.652 | -57.260% | N8 + N4 |
| 352 | 78.769 | 40.703 | -48.326% | N8 + N4 |
| 320 | 60.215 | 34.121 | -43.334% | N4 |

The 95% confidence interval for each target's paired mean difference is below
zero. R640, R512, R384, and R256 remain within 0.052% of their aligned controls,
well inside the 0.5% non-regression limit. All output hashes are exact, all
selected profiles report zero scalar attention MatMul fallbacks, and CPU4-7
execute zero IME instructions.

The N13-N15 alternatives are both exact in the property matrix, but no live
Stage61 graph exercises those remainders. Neither N8+N8 nor padded N16 is
promoted as a performance-selected live route.

## R768 Interpretation

R768 increases mAP50-95 by 0.281 AP and AP-small by 1.603 AP relative to R640.
AP-medium also improves, but AP-large decreases by 1.745 AP. Per-class changes
are mixed. The 50.6% latency increase and mixed accuracy result do not justify
a default promotion.

R768 preserves the 1069-node sequence and all 1024 learned initializer
payloads. It uses inherited R640 Q0 qparams, 3.984750 GMAC, an 11,796,480-byte
arena, and 576 aligned attention tokens. Conv spatial M12 tails are zero; each
second attention MatMul has M=64 and uses its existing exact M4 tail.

## Quantization And Next Decision

Q1/PTQ was not performed. The source lane still lacks an approved immutable
calibration corpus, list, seed, and algorithm. The Q0 evidence is sufficient
for a later human decision about whether calibration work is worth
authorizing, but it does not itself authorize that work.

The required long-soak and matched camera confirmations are separate evidence
surfaces. They may confirm stability and end-to-end behavior, but they cannot
override the fixed COCO deployment thresholds above.
