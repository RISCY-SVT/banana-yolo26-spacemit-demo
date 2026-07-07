# Thermal And Frequency Anchor Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE25-CONV-THREADING-TILE-DECISION-001

## Board

```text
ssh_target: svt@banana
hostname: bf3
uname: Linux bf3 6.6.63 #2.2.7.2 SMP PREEMPT Fri Aug 15 12:32:44 UTC 2025 riscv64
cpu_model: Spacemit(R) X60
online_cpus: 0-7
allowed_ime_cpus: 0-3
forbidden_ime_cpus: 4-7
```

## Frequency Snapshot

```text
cpu0_scaling_governor: performance
cpu0_scaling_cur_freq: 1600000
cpu_max_mhz: 1600.0000
cpu_min_mhz: 614.4000
```

Thermal `temp` files were queried with:

```text
find /sys/class/thermal -maxdepth 2 -type f -name temp -print -exec cat {} \;
```

No thermal throttle condition was reported by the benchmark logs. Stage25 speedups are interpreted as same-session ratios against the Stage24 selected-path replay.
