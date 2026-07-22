# Stage60 Resolution Decision

## Decision

No smaller unchanged-graph profile is selected for deployment. The mandatory
Q0 sweep is exact and complete, but no arm passes either fixed accuracy/latency
gate. Stage60 therefore triggers a recommendation for a separate, explicitly
authorized model-executor co-design project. Stage60 itself did not train,
distill, recalibrate, select a student, or change the integer contract.

R512 and R384 are retained as camera and long-soak evidence finalists only:

- R512 is the highest-accuracy smaller profile. It measures 99.058 ms but loses
  2.311 AP50-95 points from R640, exceeding the 1.0 AP strong-candidate limit.
- R384 is the best useful latency diagnostic. It measures 47.389 ms and has no
  Conv spatial M12 tails, but loses 6.420 AP50-95 points and is therefore
  diagnostic-only under the predefined near-20-FPS rule.
- R256 is faster at 24.317 ms, but its 13.948 AP loss makes it a lower-bound
  diagnostic rather than a candidate.

## Measured Pareto Surface

| R | Mean (ms) | Pure-model FPS | mAP50-95 | Loss (AP) | Decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 640 | 133.012 | 7.518 | 0.370741 | 0.000 | frozen accuracy control |
| 512 | 99.058 | 10.095 | 0.347630 | 2.311 | quality evidence finalist; gate fail |
| 448 | 169.389 | 5.904 | 0.332627 | 3.811 | dominated; scalar attention fallback |
| 416 | 132.060 | 7.572 | 0.317789 | 5.295 | dominated; scalar attention fallback |
| 384 | 47.389 | 21.102 | 0.306537 | 6.420 | latency evidence finalist; diagnostic-only |
| 352 | 79.714 | 12.545 | 0.289709 | 8.103 | dominated; no measured cache step |
| 320 | 60.898 | 16.421 | 0.276269 | 9.447 | misses 60 ms and accuracy gates |
| 256 | 24.317 | 41.123 | 0.231262 | 13.948 | diagnostic lower bound only |

Pure-model FPS is not camera FPS. The R384 value is not a 20-FPS deployment
claim.

## Why Latency Is Non-Monotonic

The material discontinuity is the existing attention route, not a change in
weights or topology. Its token count is `(R / 32)^2`; the selected IME MatMul
route requires that token N be divisible by 16. R640, R512, R384, and R256 meet
that constraint. R448, R416, R352, and R320 use the exact scalar fallback and
therefore exhibit much larger attention time.

R384 removes all Conv spatial M12 tails, but dense time follows the expected
resolution-squared trend closely enough that the complete-model jump cannot be
attributed to tail elimination. The attention lattice alignment is dominant.

## Cache Interpretation

At R352, `4*R^2` is 495,616 bytes, so one common early C4 activation fits within
the nominal 512 KiB cluster0 L2 capacity. This does not put the physical C8
input (991,232 bytes), input plus first output (1,486,848 bytes), complete
prefix, or 2,478,080-byte arena in L2. No timing or available PMU surface shows
a beneficial R352 threshold discontinuity.

At R256, the physical C8 input is exactly 524,288 bytes. Input plus first output
still does not fit.

## Kernel Policy

The bounded M12/M8/M4 sidecar covered representative R384, R352, and R320
classes and remained byte exact. Isolated M8 differences were too small to meet
the required 0.5% complete-model promotion gate, and no dispatcher change was
selected. L2 counters were not exposed by the verified board PMU mapping, so no
L2-miss claim is made.

## Quantization

Q0 preserves all 215 accepted scale/zero-point contracts; only static
shape-derived assets change. Q1 was not run. The accepted source is already a
fixed manual QDQ graph and the frozen evidence does not include an auditable
calibration corpus, seed, or recipe. Creating one would be new PTQ research and
would not be a valid deterministic sidecar under this stage's authorization.

## Finalist Confirmation

Post-repair 10,000-run soaks passed for R512 and R384. R512 measured 101.435 ms
mean with 132.127 ms p99.9; R384 measured 48.015 ms mean with 78.462 ms p99.9.
Both retained exact output, 1.6 GHz, affinity, CPU4-7 IME-zero, and clean O2
rollback. A failed pre-repair soak is retained as invalid lifecycle evidence and
is not mixed into these rows.

On the matched 640x480 MJPG GUI surface, R512 averaged 8.821 processed/displayed
frames/s and 41.180% latest-slot replacement. R384 averaged 15.001 frames/s and
zero replacement, equal to the measured OpenCV decoded rate. Direct V4L2 dequeue
was 30.002748 compressed buffers/s; none of these values is asserted as sensor
FPS. The R384 1,830-second soak processed 27,446 frames at 14.997374 frames/s
with zero replacement and clean profile rollback.

The camera result does not override the COCO decision: R384 remains
diagnostic-only because it loses 6.420 AP50-95 points.

## Next Decision

A future project should directly measure candidate graphs and prioritize:

1. attention token lattices divisible by 16 on the selected K1X IME path;
2. the R512 quality point and R384 aligned-lattice latency point as bounding
   evidence, not as preselected student resolutions;
3. AP-small preservation, which degrades first and materially at every smaller
   arm;
4. the measured 59 unique M/N/K classes and cache/lifetime tables in this stage;
5. the VLEN512, 32-architectural-register hardware hypothesis documented in
   `future_riscv_vector_register_memo.md`.

Training and model-executor co-design remain unauthorized in Stage60.
