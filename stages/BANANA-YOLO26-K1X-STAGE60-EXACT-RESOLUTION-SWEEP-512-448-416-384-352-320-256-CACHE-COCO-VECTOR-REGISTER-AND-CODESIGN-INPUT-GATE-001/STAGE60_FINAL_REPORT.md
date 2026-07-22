# Stage60 Final Report

Classification: `stage60-exact-resolution-sweep-complete-no-deployment-winner`.

Stage60 measures static 640, 512, 448, 416, 384, 352, 320, and 256 profiles
derived from the frozen YOLO26n graph. The source weights, node topology,
`K1X_INT8_V1` Q62/RNE arithmetic, physical layout, compiler contract, and
CPU0-3-only IME policy remain unchanged. Work is isolated on
`yolo26-k1x-resolution-sweep`; the frozen 0.9.2 branch and release are not
modified.

## Baseline And Identity

The regenerated R640 package is byte-identical across two builds and retains
manifest SHA-256
`fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be`.
Its selected O2 500-sample mean is 133.012150 ms, within 1% of the accepted
133.382 ms reference. The fixed output remains `0xd43f5e018b415631`.

Every smaller graph is a fixed `1x3xRxR` derivative with an explicit profile
identity. The operation order, 1024 initializer payloads, and 215 integer
boundaries match R640. Only static geometry and shape-derived package assets
change. Each package was generated twice with a byte-identical tree.

Q0 reuses all 215 accepted scale/zero-point contracts. Q1 was not manufactured:
the frozen manual-QDQ source has no auditable calibration corpus, seed, or
recipe, and creating one would be new PTQ research outside this stage.

## Exactness

All eight mandatory profiles pass independent Python integer semantics,
portable host scalar, board scalar, and board optimized/IME comparison on
F0-F7, bus/canonical, Zidane, known real inputs, RNE ties, saturation, and all
M/N tails. The final cross-surface replay covers 17,280 boundary artifacts per
surface and is byte-identical at every one of the 215 boundaries.

FRM and vector CSR restoration, affinity, deterministic output, no-SIGILL, and
CPU4-7 IME-zero gates pass. Dynamic profile support is research-gated and the
640 release default remains unchanged.

## Measured Pareto Frontier

| R | Mean (ms) | p95 (ms) | FPS | mAP50-95 | Loss vs R640 (AP) | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 640 | 133.012 | 133.541 | 7.518 | 0.370741 | 0.000 | frozen control |
| 512 | 99.058 | 99.421 | 10.095 | 0.347630 | 2.311 | quality evidence finalist; gate fail |
| 448 | 169.389 | 169.622 | 5.904 | 0.332627 | 3.811 | dominated; scalar attention fallback |
| 416 | 132.060 | 132.236 | 7.572 | 0.317789 | 5.295 | dominated; scalar attention fallback |
| 384 | 47.389 | 47.557 | 21.102 | 0.306537 | 6.420 | latency evidence finalist; diagnostic-only |
| 352 | 79.714 | 79.858 | 12.545 | 0.289709 | 8.103 | dominated; no cache step |
| 320 | 60.898 | 61.030 | 16.421 | 0.276269 | 9.447 | misses latency and accuracy gates |
| 256 | 24.317 | 24.403 | 41.123 | 0.231262 | 13.948 | diagnostic lower bound only |

All eight COCO runs completed 5000/5000 with zero failures. AP-small degrades
first and materially as resolution falls. No arm passes either predefined
no-training selection gate. R384 and R256 satisfy the analytical near-20-FPS
latency condition only with accuracy loss greater than 2 AP, so they remain
diagnostic and are not deployment or camera-FPS claims.

## Why The Curve Is Non-Monotonic

Attention token count is `(R/32)^2`. The selected exact IME MatMul path requires
token N divisible by 16, which is true for R640, R512, R384, and R256. R448,
R416, R352, and R320 use the exact scalar fallback. This existing route
boundary, rather than changed arithmetic or topology, dominates the latency
discontinuities.

R384 has zero Conv spatial M12 tails, but its dense category follows the
resolution-squared trend; zero tails do not independently explain its full
model speed. The bounded M12/M8/M4 sidecar remained exact, and no M8 shape
cleared the required 0.5% complete-model promotion gate.

At R352, `4*R^2` is 495,616 bytes and one common early C4 activation fits in the
nominal 512 KiB cluster0 L2. The physical C8 input is 991,232 bytes, input plus
first output is 1,486,848 bytes, and the arena is 2,478,080 bytes. The complete
prefix therefore does not fit, and timing exposes no beneficial R352 cache
threshold. Verified L2 PMU events are unavailable, so no L2-miss claim is made.

## Stability And Camera

The first attempted finalist soak exposed a frame-gated worker lifecycle race:
one required worker could remain parked after the controller began a four-worker
dispatch. That run was terminated and is invalid performance evidence. GDB,
termination, and O2 restoration evidence are retained. The non-arithmetic
repair serializes active-window transitions, waits for park/wake acknowledgement,
and rejects stale job generations.

After the repair, a 2,000-run R512 lifecycle stress and both official 10,000-run
soaks completed. The statistical surfaces remain separate:

| R | Samples | Mean (ms) | p95 (ms) | p99 (ms) | p99.9 (ms) | Max (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 10,000 | 101.435 | 101.328 | 130.923 | 132.127 | 132.744 |
| 384 | 10,000 | 48.015 | 47.932 | 59.930 | 78.462 | 78.733 |

Both held 1.6 GHz, retained exact output hashes, recorded zero affinity failures
and zero CPU4-7 IME operations, and restored O2. Periodic between-block wake
cost is visible in the long-tail columns; these values are not mixed with the
separate 500-sample finalist ABBA rows. Post-repair ABBA measured R512 at
100.446 ms against a 133.524 ms R640 control and R384 at 47.729 ms against a
133.587 ms R640 control.

The first camera comparison was also rejected because its wrapper omitted the
frozen operator profile. The preserved mismatch measured R384 executor time at
about 78.42 ms. After the wrapper was repaired, a 60-second control measured
47.31 ms and processed every OpenCV-decoded frame. The official matched GUI
matrix then produced:

| R | Runs | Processed frames | OpenCV decoded FPS | Processed/displayed FPS | Application replacements | Executor mean (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 3 x 180 s | 4,765 | 14.997 | 8.821 | 41.180% | 97.108 |
| 384 | 3 x 180 s | 8,103 | 15.001 | 15.001 | 0.000% | 47.655 |

An independent V4L2 MMAP trace measured 5,395 compressed buffers with zero
sequence gaps, monotonic SOE timestamps, and 30.002748 dequeued buffers/s. The
mode still reports 60 FPS, but neither that reported value nor the V4L2 dequeue
rate is sensor or application FPS. The current OpenCV producer is approximately
15 decoded frames/s.

The selected R384 camera soak processed and displayed 27,446 frames over
1,830.054 seconds at 14.997374 frames/s with zero application replacements.
Executor mean/p95 were 47.953995/49.554741 ms, the decoded-read-return to
display-call mean was 62.227337 ms, maximum measured thermal-zone average was
68.5 C, and CPU0-4 remained at 1.6 GHz. The reversible camera profile restored.

R384 is a camera latency diagnostic only. Its 6.420 AP loss prevents deployment
selection even when the application keeps pace with decoded input.

Pipeline measurements are separate from pure model latency. For example, R384
measures 73.204 ms on the serial preloaded-image surface and 52.049 ms as a
double-buffer steady-state interval. The latter is throughput, not pure-model
or camera latency.

## Validation And Maintenance

Host CTest, ASan/UBSan, repeated TSan, Python compilation, RISC-V cross-build,
board loader, exact fixtures, full COCO, and source hygiene pass. Repeated TSan
exposed a pre-existing startup lost-wakeup in the older threaded-convolution
workspace. Readiness publication now holds the condition-variable mutex; this
is a non-arithmetic startup safety repair and passes 20 repeated TSan runs.

The Stage60 active-window worker fix prevents stale generation replay and
waits for frame-gated workers to park before returning to an inactive window.
No selected kernel arithmetic or package format is changed.

## Future Hardware And Decision

For a future non-IME target, the evidence-backed default hypothesis is standard
RISC-V V with 32 architectural vector registers, VLEN512, ELEN64, fractional
LMUL, strong chaining/bypass, at least eight in-flight vector operations, and
48-64 implementation-private physical rename entries if renaming exists.
VLEN256 cannot hold the modeled full M12 fused/load-ahead schedules without
splitting or spilling. `Zvldot`, `Zvbdot`, `Zvqdot`, and `Zvdot4a8i` remain
under-development dependencies, not part of this software contract.

Because no arm reaches 60 ms with at most 1.5 AP loss, the fixed co-design
trigger fires. A future separately authorized project should measure candidate
graphs directly, preserve AP-small, and account for the N-divisible-by-16
attention lattice. Stage60 performed no training, student selection, or
model-executor co-design.
