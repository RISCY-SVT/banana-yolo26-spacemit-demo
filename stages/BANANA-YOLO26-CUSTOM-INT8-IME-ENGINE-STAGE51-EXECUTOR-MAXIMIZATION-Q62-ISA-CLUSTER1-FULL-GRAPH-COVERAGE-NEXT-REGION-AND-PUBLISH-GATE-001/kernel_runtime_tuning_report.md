# Kernel/runtime tuning

The selected bounded sidecar is SCHED_RR priority 20 with watchdog/cleanup. It reduced the
model4-final to model9 combined mean to 20973.341840 us and p95 to
21161.386150 us. `mlockall` plus prefault and MADV_HUGEPAGE were exact but
smaller/no-win. IRQ retargeting was restored and not selected. Cpuset shielding and boot
isolation were not applied; no persistent system change remains.
