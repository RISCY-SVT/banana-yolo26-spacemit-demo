# System Rollback Report

Stage60M selected no persistent system change. The existing O2 and camera
profiles were used only through their reversible `run` wrappers.

After the 13,500-run executor soak, 30-minute GUI camera soak, normal and
signal shutdowns, USB capture reset, recorder failure, and clean-extract
camera tests:

- the boot ID still equals `0a0691d1-7502-44c3-903b-444dba83c1d9`;
- the CPU governor is `performance`, maximum and observed frequency are
  1,600,000 kHz;
- the global workqueue mask is restored to `ff`;
- the O2 snapshot and cgroup are absent;
- the camera profile is inactive and xHCI IRQ 89 has requested/effective
  affinity `0-7`;
- no `y26_k1x_demo` process or camera state file remains;
- all stage paths, builds, logs, media, packages, and archives are under the
  NVMe `/data` hierarchy.

The root filesystem remains the original eMMC mount. Stage60M issued no
project-artifact write to an eMMC path; ordinary operating-system metadata is
not classified as a project artifact. No boot, kernel, service, sysctl, THP,
or persistent frequency setting was changed.

Status: `pass`.
