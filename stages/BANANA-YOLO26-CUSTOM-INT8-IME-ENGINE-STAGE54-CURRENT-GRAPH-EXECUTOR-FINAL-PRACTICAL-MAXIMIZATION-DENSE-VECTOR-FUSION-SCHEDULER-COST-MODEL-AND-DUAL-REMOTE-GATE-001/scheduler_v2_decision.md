# Scheduler V2 decision

Keep condition-variable SCHED_OTHER as compatibility and raw epoch-spin SCHED_OTHER as the dedicated-board low-latency profile. Pause was neutral, adaptive sleep was slower, and S4 static scheduling did not beat dispatch. Thermal/process-CPU costs are explicit.

The 1802-second thermal trace reached 80 C while all CPU0-4 frequency samples remained at 1.6 GHz. With a prepared executor idle for five seconds, condition-variable process CPU/wall was 0.000027 and epoch-spin was 3.993455. A bounded concurrent NVMe-read probe raised the low-latency mean from 167411.836 us to 178697.360 us (+6.741175%). No reliable board power sensor was available.

Raw evidence: `/data/ncnn-logs/ai-team/2026-07-14/2026-07-14_16-36-04__codex__BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE54-CURRENT-GRAPH-EXECUTOR-FINAL-PRACTICAL-MAXIMIZATION-DENSE-VECTOR-FUSION-SCHEDULER-COST-MODEL-AND-DUAL-REMOTE-GATE-001__stage54-final-executor-max`.
