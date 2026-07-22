# System Rollback Report

Stage61 selected no persistent system change. The existing reversible O2
profile was used only for pure-model timing and soaks. The existing reversible
camera CPU/IRQ profile was used only for matched camera measurements.

After the final board workload:

- O2 snapshot: absent;
- Stage61 cgroup: absent;
- global workqueue mask: `ff`;
- camera profile state: absent/inactive;
- reviewed xHCI IRQ 89 effective affinity: restored to CPU mask `0`;
- Stage61 executor/demo processes: zero;
- boot ID: unchanged;
- CPU governor: original accepted `performance` state;
- project storage: NVMe `/data` only;
- persistent boot, kernel, sysctl, service, and eMMC changes: none.

The final `restore-stale` calls reported that neither profile had stale state.
Every camera arm also preserved an individual `profile-after.txt` showing
`state=inactive`.
