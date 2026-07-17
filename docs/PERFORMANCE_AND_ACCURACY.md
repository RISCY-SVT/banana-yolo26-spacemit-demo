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

### Stage59 Final Release Reconciliation

Stage59 found that the published 0.9.1 top-level cross build had omitted
`-mtune=spacemit-x60 -funroll-loops`. A neutral ABI1 `dlopen` harness rebuilt
Stage57 and the final 0.9.2 library with identical accepted flags, prepared
outside the timed loop, and interleaved 1,000 samples per arm in one O2 window.

| Surface | Samples | Mean (us) | Median (us) | p95 (us) | p99 (us) | Max (us) |
|---|---:|---:|---:|---:|---:|---:|
| rebuilt Stage57 control | 1000 | 133356.369 | 133127.000 | 134949.350 | 135336.040 | 135718.000 |
| final 0.9.2 O2 | 1000 | 134100.921 | 133846.000 | 135724.150 | 136137.060 | 136465.000 |
| final 0.9.2 O2 soak | 13500 | 133381.666593 | 133355.000 | 133912.000 | 134478.050 | 135853.000 |

The final same-session mean delta is +0.558%, inside the required 1%
equivalence gate and below the absolute 135500 us mean / 137000 us p95 limits.
Every arm retained `0xd43f5e018b415631`, and the final COCO prediction remained
byte-identical. The 1,000-sample and 13,500-run rows are separate statistical
surfaces and their tail columns must not be mixed.

### Stage57 Selected Source Reference

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

The 500-sample O2 mean is 6.395000% lower than the accepted Stage56 O2 mean and
71.222934% lower than matched B120 ORT. It corresponds to 7.501581 pure-model
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

Stage57 final executor values remain the arithmetic baseline for the installed
0.9.2 maintenance route. Stage58 and Stage59 report camera throughput
separately. None of these stages demonstrates 20 FPS, an installed camera
service, or production certification.

### Stage59 Matched Camera Presets

| Preset | Requested/reported mode | Runs | Processed/displayed FPS | OpenCV decoded FPS | App slot replacement | Mean call latency |
|---|---|---:|---:|---:|---:|---:|
| quality-wide | 1280x720 MJPG, 60 requested | 3 x 180 s | 5.976975 | 9.999262 | 40.225839% | 219.143 ms |
| performance | 640x480 MJPG, 60 requested | 3 x 180 s | 6.619983 | 14.993076 | 55.846398% | 184.200 ms |
| performance + async record | 640x480 MJPG, 60 requested | 180 s | 6.562994 | 15.002716 | 56.254626% | 185.375 ms |
| quality-wide soak | 1280x720 MJPG, 60 requested | 1830 s | 5.980417 | 9.999235 | 40.191257% | 218.781 ms |
| performance + camera profile soak | 640x480 MJPG, 60 requested | 1830 s | 6.818437 | 14.983294 | 54.493071% | 180.007 ms |
| performance + async record soak | 640x480 MJPG, 60 requested | 1830 s | 6.716931 / 6.712772 recorded | 14.983587 | 55.171408% | 182.481 ms |

Direct V4L2 MMAP telemetry measured 30.0016 dequeued buffers per second and
zero driver-visible sequence gaps for both MJPG modes. The timestamp flag was
monotonic SOE. This is not a sensor-to-display timestamp chain. The public fast
launcher selects the measured 640x480 performance preset and reversible
camera-only CPU5/xHCI IRQ profile without O2; O2 remains an explicitly named
diagnostic. Long-soak rows are separate statistical surfaces from the matched
three-run comparison and are not pooled into its percentages.

### Stage58 Release Revalidation

| Surface | Samples | Mean (us) | Median (us) | p95 (us) | p99 (us) | p99.9 (us) | Max (us) |
|---|---:|---:|---:|---:|---:|---:|---:|
| compatibility | 500 | 154319.032 | 154055.500 | 156507.350 | 157870.420 | 163059.241 | 164926.000 |
| low-latency, original OS | 500 | 141800.914 | 141213.000 | 145614.250 | 151626.520 | 154852.454 | 156077.000 |
| low-latency-dedicated, O2 | 500 | 140875.230 | 140853.000 | 141455.100 | 141744.460 | 142935.286 | 143078.000 |
| low-latency-dedicated O2 soak | 10000 | 140877.187 | 140848.000 | 141485.050 | 142478.040 | 144386.185 | 145307.000 |

The Stage58 source changes are release and demo maintenance; they do not retune
the frozen executor. The final soak retained `0xd43f5e018b415631` for all 10,000
runs, kept IME on CPU0-3, and restored O2 after measurement. Other Stage58
surfaces were 132505.910 us for the 100-image corpus, 138989 us for the already
letterboxed RGB8 input, 186537.301 us for the serial preloaded pipeline, and
139561.631 us per double-buffer interval (7.165293 FPS throughput).

### Stage58 Camera Surface

| Surface | Requested/reported mode | Frames | Processed/displayed FPS | OpenCV decoded FPS | App slot replacement | Mean/p95 call latency |
|---|---|---:|---:|---:|---:|---:|
| selected GUI, no recording | 1280x720@60 MJPG | 3200 | 5.916864 | 9.980414 | 40.869097% | 218.716 / 262.011 ms |
| GUI with MJPG AVI recording | 1280x720@60 MJPG | 146 | 4.833642 / 4.738854 recorded | 9.881496 | 51.771117% | 256.761 / 301.979 ms |

This historical Stage58 row pools three independent 180-second runs. It includes
capture, exact 640x640 preprocessing, executor, output mapping, boxes, overlay,
GUI display, and event handling. It is not pure-model throughput, and the
call latency is not sensor-to-screen latency because sensor timestamps were not
correlated. `40.869097%` is application slot replacement, not a complete camera
drop rate. Stage59's direct V4L2 probe supersedes the older interpretation of
the OpenCV decoded rate but does not construct a sensor-to-display chain.

## HPM Language

Stage56 HPM values are event counts per cycle. They prove a measured surface
dominated by backend/structural/dependency-or-latency events, not by frontend,
I-cache, or branch events. They do not prove that every backend event is a Q62
dependency. The L1D value is not a miss/access ratio, and a DTLB miss ratio is
unknown because the matching access event returned zero.
