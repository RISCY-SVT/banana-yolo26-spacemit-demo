# Thermal and Frequency Anchor Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE27-CONV-OR-OUTPUT-QUANTIZE-DECISION-AFTER-ACTIVATION-REPAIR-001

Board anchor was recorded before stable timing:

```text
hostname: bf3
kernel: Linux bf3 6.6.63 #2.2.7.2 SMP PREEMPT Fri Aug 15 12:32:44 UTC 2025 riscv64
model: Spacemit(R) X60
cpu_count: 8
allowed_ime_cpus: CPU0-3
forbidden_ime_cpus: CPU4-7
scaling_governor: performance
cpu0_scaling_cur_freq: 1600000
```

No CPU4-7 IME execution was used. All benchmark commands used `taskset -c 0-3`.

Raw board anchor:

```text
run_logs/board_anchor.log
```
