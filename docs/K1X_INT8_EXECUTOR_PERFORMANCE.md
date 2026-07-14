# K1X INT8 Executor Performance

Performance is reported on Banana-Pi BPI-F3 / SpacemiT K1X with the performance
governor, CPU0-3 IME workers, CPU4 controller, and `SCHED_OTHER` as the safe
baseline. The stable protocol is 10 warmups, 100 runs, five repeats. A separate
10,000-run soak reports p99, p99.9, maximum, context switches, migrations,
frequency, and temperature.

Timing surfaces are separate:

```text
preprocess/decode/letterbox
input quantization and layout
pure full executor to 1x300x6
final detection/JSON decode
preloaded-image pipeline
```

The selected exact E2c2 route measured:

```text
SCHED_OTHER, 10/100/5:
  mean:    504137.644000 us
  median:  502911.500000 us
  p95:     506182.200000 us
  p99:     527434.100000 us
  rate:    1.983585 inferences/s

SCHED_RR priority 20 sidecar:
  mean:    503473.764000 us
  p95:     515912.800000 us

SCHED_OTHER, 10,000-run soak:
  mean:    503576.174900 us
  median:  502440.000000 us
  p95:     505918.400000 us
  p99:     527271.240000 us
  p99.9:   546540.109000 us
  max:     595021.000000 us

matched B120 ORT CPU0-3 intra4:
  mean:    460208.112134 us across five 100-inference repeat means
```

E2c2 improves the complete-model mean by 9.707672% versus the exact E2c
control. The selected custom mean is nevertheless 9.545580% slower than the
matched ORT mean. ORT percentiles and custom percentiles are not compared
because the preserved ORT distribution uses repeat means while custom uses
per-inference samples.

All 10,000 soak runs produced `0xd43f5e018b415631`; worker affinity passed
and CPU4-7 executed no IME instructions.

The dedicated preloaded public-image benchmark measured 46464.740510 us for
decode/letterbox/color/input layout, 470259.641654 us for the fixture-specific
pure graph, 5.417302 us for output decode, and 516859.673560 us total. This is
separate from COCO dataset timing, whose preprocessing includes board-NVMe
JPEG file reads.

Complete raw samples, component attribution, the 10,000-run soak, and the
statistical-unit contract are in the Stage52 `full_model_performance_report.md`.
Analytical Stage51 envelopes are not substituted for measured latency. The
optional `rr20` mode is a lab sidecar and is never the default.

No 20 FPS or production claim is made. The measured full executor does not
meet that target.
