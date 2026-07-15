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

## Stage56 Final System and Source Ceiling

Stage56 keeps the Stage55 package and exact arithmetic contract. It selects
producer-adjacent head reduction, direct attention packing for the second
MatMul, and the reversible O2 dedicated-board profile. O2 uses an isolated
CPU0-4 cgroup with movable IRQs, unbound workqueues, and nonessential services
placed on CPU5-7. The original boot entry, NVMe runtime, memory policy,
compiler contract, and `SCHED_OTHER` policy remain unchanged.

```text
condition-variable compatibility, fixed input, 500 samples:
  mean:    156620 us
  median:  155860 us
  p95:     160377 us
  p99:     166607 us
  max:     207766 us

O2 frame-gated low latency, fixed input, 500 samples:
  mean:    142413 us
  median:  142416 us
  p95:     142893 us
  p99:     143293 us
  max:     144547 us
  rate:    7.021831 inferences/s

O2 frame-gated low latency, first 10,000 runs of thermal soak:
  mean:    142444.857000 us
  median:  142425.000000 us
  p95:     142984.050000 us
  p99:     143747.080000 us
  p99.9:   144539.027000 us
  max:     145774.000000 us

O2 frame-gated low latency, complete 13,000-run thermal surface:
  mean:    142441.445462 us
  median:  142422.000000 us
  p95:     142978.000000 us
  p99:     143693.100000 us
  p99.9:   144489.050000 us
  max:     145774.000000 us

100-image in-memory corpus executor mean:
  141733.839830 us

serial preloaded-image complete pipeline mean:
  196447.607874 us

double-buffer steady-state frame interval:
  150356.344210 us
  rate:      6.647346 frames/s

complete COCO one-pass executor mean:
  141834.806835 us

matched B120 ORT, 500 per-inference samples:
  mean:    459954.787608 us
  p95:     463463.866350 us
```

The selected fixed-input mean is 4.806% below the official Stage55
149603.240 us surface and approximately 69.04% below matched B120 ORT. The
5,000-image COCO prediction is byte-identical to Stage55 and retains
0.3707408944391919 mAP50-95. Compatibility, 500-sample headline, 10,000-run
soak, 13,000-run thermal, real-corpus, pipeline, COCO, and ORT statistics are
separate surfaces and are not mixed. This is approximately 7.02 pure-model
inferences/s, not 20 FPS, and is not a production-readiness claim.

## Stage55 Residual Ceiling

Stage55 keeps the Stage54 package, arithmetic, layout, compiler, and public
API. It restores V1 stream head selection after a 1,000-sample-per-arm ABBA,
adds legal indexed RVV LUT2 and Q48 attention lookup, exact E2c4 C8,
prepare-time dense Family A dispatch, and frame-gated epoch spin. Workers park
between frames and use epoch spin only inside an active inference window.

```text
condition-variable compatibility, fixed input, 10/100/5:
  mean:    162447.546 us
  median:  161702.500 us
  p95:     166485.750 us
  p99:     175538.580 us

frame-gated low latency, fixed input, 10/100/5:
  mean:    149603.240 us
  median:  149083.000 us
  p95:     153748.300 us
  p99:     159241.830 us
  max:     162339.000 us
  rate:    6.684347 inferences/s

frame-gated low latency, 10,000-run soak:
  mean:    149598.790900 us
  median:  149062.500000 us
  p95:     152636.150000 us
  p99:     158975.130000 us
  p99.9:   203354.276000 us
  max:     211671.000000 us

100-image in-memory corpus executor mean:
  149183.628410 us

preloaded-image complete pipeline mean:
  172265.633994 us

matched B120 ORT, 500 per-inference samples:
  mean:    455028.164774 us
  p95:     457913.059850 us
```

The selected fixed-input mean is 10.637597% below the official Stage54
167411.836 us surface and 67.122202% below the matched B120 ORT mean. The
5,000-image COCO prediction JSON is byte-identical to Stage54 and retains
0.3707408944391919 mAP50-95. This is approximately 6.68 inferences/s, not
20 FPS, and is not a production-readiness claim.

## Stage54 Final Practical Maximization

Stage54 retains exact `K1X_INT8_V1`, the Stage53 package, NCHWc8, and the
selected compiler contract. It adds shape-dispatched direct 1x1/P3 dense
delivery, exact E2c3 C8 LUT/store, depthwise V2, explicit RVV input conversion
with compact C3 stem input, and bounded head V2.

```text
condition-variable compatibility, fixed input, 10/100/5:
  mean:    180237.700 us
  median:  179921.500 us
  p95:     182678.350 us
  p99:     183556.020 us

epoch-spin low latency, fixed input, 10/100/5:
  mean:    167411.836 us
  median:  167151.500 us
  p95:     169621.050 us
  p99:     173464.770 us
  max:     200475.000 us
  rate:    5.973293 inferences/s

epoch-spin low latency, 10,000-run soak:
  mean:    167738.026600 us
  p95:     170529.100000 us
  p99:     177433.070000 us
  p99.9:   182884.190000 us
  max:     186348.000000 us

condition-variable compatibility, 10,000-run soak:
  mean:    180403.108600 us
  p95:     183174.050000 us
  p99:     190387.200000 us
  p99.9:   197072.411000 us
  max:     214786.000000 us

100-image in-memory corpus executor mean:
  166140.760690 us

preloaded-image complete pipeline mean:
  190754.405260 us
```

The post-freeze low-latency mean is 30.211342% below the accepted Stage53
239884.016 us surface and 63.444709% below the reproduced matched B120 ORT
457968.821588 us mean. Full COCO and both 10,000-run scheduler soaks are frozen
in the Stage54 report and release manifest. This remains below 6 FPS, not
20 FPS.

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
