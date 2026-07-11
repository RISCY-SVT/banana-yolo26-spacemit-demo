# Model5-8 Isolated Board Profile

Each block used its own graph-derived cut and one in-process board ORT session. Timings are direct isolated measurements, not cumulative-session subtraction.

Protocol: CPU0-3, performance governor at 1.6 GHz, warmup 10, runs 100, repeats 5, sequential ORT, one intra/inter-op thread.

| Block | Mean us | Stddev us | CV | Session init us | First run us |
|---|---:|---:|---:|---:|---:|
| model5 | 17814.502 | 10.090 | 0.0566% | 25101.894 | 23944.740 |
| model6 | 29571.056 | 22.349 | 0.0756% | 91528.991 | 36796.018 |
| model7 | 7545.484 | 20.101 | 0.2664% | 27923.364 | 10088.877 |
| model8 | 17360.697 | 23.523 | 0.1355% | 96366.640 | 21172.900 |

Profile provider fields identify `CPUExecutionProvider` for every captured node event. Board ORT output remains diagnostic under Policy B: against fixed host operational outputs, mismatches were model5 1, model6 21674, model7 2, and model8 11088.
