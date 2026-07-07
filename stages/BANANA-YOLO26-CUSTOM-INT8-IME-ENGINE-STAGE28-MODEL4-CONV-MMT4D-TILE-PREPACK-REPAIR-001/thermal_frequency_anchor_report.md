# Thermal And Frequency Anchor Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE28-MODEL4-CONV-MMT4D-TILE-PREPACK-REPAIR-001`

## Board

```text
hostname: bf3
kernel: Linux bf3 6.6.63 #2.2.7.2 SMP PREEMPT Fri Aug 15 12:32:44 UTC 2025 riscv64
architecture: riscv64
cpu_model: Spacemit(R) X60
online_cpus: 0-7
allowed_ime_cpus: CPU0 CPU1 CPU2 CPU3
forbidden_ime_cpus: CPU4 CPU5 CPU6 CPU7
```

## Frequency

```text
cpu0_scaling_governor: performance
cpu0_scaling_cur_freq: 1600000
cpu_max_mhz: 1600.0000
cpu_min_mhz: 614.4000
```

## Thermal

The board command queried `/sys/class/thermal` temp files. No temp file values were returned in stdout for this run.

## Timing Discipline

Stage28 uses same-session ratios for performance decisions. Absolute times are not compared to older stages as if board state were identical.
