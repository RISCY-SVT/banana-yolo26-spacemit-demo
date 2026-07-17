# System Rollback Report

All Stage59 system changes were reversible measurement profiles. The camera
profile temporarily pinned capture and movable xHCI IRQ 89 to CPU5; O2 was
used only for pure-model measurements. Normal completion, injected failure,
SIGINT, SIGTERM, and SIGHUP paths restored their snapshots.

The final board state records CPUs 0-7 online, the performance governor at
1.6 GHz, global workqueue mask `ff`, O2 inactive, its cgroup absent, camera
profile inactive, and IRQ 89 restored to requested CPUs 0-7 with effective CPU
0. No demo process remains. Original boot, NVMe runtime storage, and all
baseline services remain selected. Stage59 made zero eMMC project writes and
selected no persistent sysctl, boot, kernel, storage, or frequency change.

Status: `pass`.
