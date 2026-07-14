# K1X INT8 Executor Performance

Performance is reported on Banana-Pi BPI-F3 / SpacemiT K1X with the performance
governor, CPU0-3 IME workers, CPU4 controller, and `SCHED_OTHER`. The stable
protocol is 10 warmups, 100 runs, and five repeats. Complete raw samples and
tail distributions are in the corresponding stage evidence.

Timing surfaces remain separate:

```text
preprocess/decode/letterbox
input quantization and layout
pure full executor to 1x300x6
final detection/JSON decode
preloaded-image pipeline
COCO one-pass per-image executor
```

## Stage54 Final Practical Maximization

Stage54 retains exact `K1X_INT8_V1`, the Stage53 package, NCHWc8, and the
selected compiler contract. It adds shape-dispatched direct 1x1/P3 dense
delivery, exact E2c3 C8 LUT/store, depthwise V2, explicit RVV input conversion
with compact C3 stem input, and bounded head V2.

```text
condition-variable compatibility, fixed input, 10/100/5:
  mean:    180238 us
  median:  179922 us
  p95:     182678 us
  p99:     183556 us

epoch-spin low latency, fixed input, 10/100/5:
  mean:    167735 us
  median:  167225 us
  p95:     170693 us
  p99:     178804 us
  max:     183210 us
  rate:    5.961784 inferences/s

100-image in-memory corpus executor mean:
  166140.760690 us

preloaded-image complete pipeline mean:
  190754.405260 us
```

The low-latency mean is 30.08% below the accepted Stage53 239884.016 us
surface and more than 60% below matched B120 ORT. Full COCO and the 10,000-run
soak are frozen in the Stage54 report and release manifest. This remains below
6 FPS, not 20 FPS.

## Stage53 Optimized Research

The selected Stage53 arithmetic, layout, and static graph remain
`K1X_INT8_V1`, `NCHWc8_SPATIAL_INNER_V1`, and exact Q62 E2c2. The optimized
research benchmark additionally sets `Y26_STAGE53_SPIN_POOL=1` before prepare.
It remains `SCHED_OTHER`; the environment setting changes worker wakeup from
condition variables to a persistent epoch poll.

```text
fixed preprocessed input, final source, 10/100/5:
  mean:    239884 us
  median:  239255 us
  p95:     242452 us
  p99:     250782 us
  max:     288609 us
  rate:    4.168682 inferences/s

condition-variable compatibility route, 10/100/5:
  mean:    253069 us
  median:  252149 us
  p95:     260134 us
  p99:     266546 us

epoch-spin selected route, 10,000-run soak:
  mean:    240043.539700 us
  median:  239529.500000 us
  p95:     242403.200000 us
  p99:     250091.230000 us
  p99.9:   260405.010000 us
  max:     263494.000000 us

matched B120 ORT CPU0-3 intra4, 500 per-inference samples:
  mean:    456266.315376 us
  p95:     459138.349850 us
```

The final-source fixed-input mean is 52.369614% lower than the reproduced
Stage52 executor and 47.424562% lower than matched B120 ORT. This is about
2.10x and 1.90x faster, respectively. It is still approximately 4.17
inferences/s, not 20 FPS.

The dedicated preloaded-image benchmark measured:

```text
JPEG decode:                  22410.841346 us
resize/letterbox:              1185.054726 us
color conversion:               505.279534 us
input quantization/layout:     3057.749166 us
pure graph:                  226412.910214 us
executor total:              229470.659380 us
output decode:                    5.403788 us
complete preloaded pipeline: 253694.519464 us
```

The separate 5,000-image COCO pass measured approximately 229406.609224 us
per-image executor mean. It includes real image-dependent head work and is not
substituted for the fixed-input benchmark.

Epoch-spin removes voluntary worker wake switches and improves mean and tail
latency, but it occupies the four worker CPUs continuously while a prepared
executor is active. Condition-variable wake remains the compatibility default.

## Stage52 Functional Reference

The unchanged Stage52 functional-reference bundle measured:

```text
fixed preprocessed input, SCHED_OTHER, 10/100/5:
  mean:    504137.644000 us
  median:  502911.500000 us
  p95:     506182.200000 us
  p99:     527434.100000 us

SCHED_OTHER, 10,000-run soak:
  mean:    503576.174900 us
  p95:     505918.400000 us
  p99:     527271.240000 us
  p99.9:   546540.109000 us
  max:     595021.000000 us
```

Stage51 analytical envelopes are superseded as full-model predictors. Stage53
uses a calibrated 215-operation full-wall profile and a measured cost model;
representative MAC coverage is not treated as optimized wall-time coverage.

No production-readiness or 20 FPS claim is made.
