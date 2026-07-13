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

Final measured values and the statistically matched B120 ORT diagnostic are in
the Stage52 `full_model_performance_report.md`. Analytical Stage51 envelopes
are not substituted for measured latency. The optional `rr20` mode is reported
only as a lab maximum and is never the default.

No 20 FPS or production claim is made unless a corresponding complete measured
surface passes it.
