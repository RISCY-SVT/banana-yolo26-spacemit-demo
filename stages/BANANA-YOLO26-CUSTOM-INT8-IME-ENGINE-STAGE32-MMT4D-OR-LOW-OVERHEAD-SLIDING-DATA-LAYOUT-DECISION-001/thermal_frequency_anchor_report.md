# Thermal and Frequency Anchor Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE32-MMT4D-OR-LOW-OVERHEAD-SLIDING-DATA-LAYOUT-DECISION-001

Raw anchor log:

```text
/data/ncnn-logs/ai-team/2026-07-08_12-51-18/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE32-MMT4D-OR-LOW-OVERHEAD-SLIDING-DATA-LAYOUT-DECISION-001/run_logs/board_deploy_anchor.log
```

Board:

```text
hostname: bf3
kernel: Linux bf3 6.6.63 #2.2.7.2 SMP PREEMPT Fri Aug 15 12:32:44 UTC 2025 riscv64
model: Spacemit(R) X60
online_cpus: 0-7
cpu0_scaling_governor: performance
cpu0_scaling_cur_freq: 1600000
```

Stage32 timing decisions use same-session ratios only. No absolute timing is presented as model FPS or full-engine performance.

Binary and fixture hashes were recorded in the raw log. After the Stage32 B4 layout fix, the final deployed Stage32 tool SHA256 was:

```text
3db07df5fa9291e7dc95a6c8ee8124c0c45fa96316c562a7cf870a27d97910ab  bench_stage32_layout_decision
```
