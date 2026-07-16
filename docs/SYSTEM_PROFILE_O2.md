# Reversible O2 System Profile

O2 is the measured dedicated-board placement profile. It creates a cgroup v2
isolated cpuset for CPU0-4 and keeps CPU5-7 for SSH, reviewed device IRQs,
unbound workqueues, and nonessential services.

It does not alter boot parameters, kernel, sysctl, THP, storage, CPU frequency,
devfreq, cpuidle, or real-time scheduling.

## Commands

```bash
scripts/o2-system-profile.sh status
scripts/o2-system-profile.sh dry-run
scripts/o2-system-profile.sh apply
scripts/o2-system-profile.sh restore
scripts/o2-system-profile.sh restore-stale
scripts/o2-system-profile.sh run [--timeout SECONDS] -- COMMAND [ARG...]
```

Use `run` for normal operation. It moves the launched process into the CPU0-4
partition and restores state when the command exits, fails, times out, or receives
`INT`, `TERM`, or `HUP`.

## Safety Model

- Exact online topology `0-7`, cgroup v2 cpuset support, writable `/data`, and
  noninteractive sudo are validated before mutation.
- `flock` prevents nested lifecycle operations.
- A root-owned atomic snapshot is stored under
  `/data/y26-k1x-int8-executor/state/o2`.
- Only reviewed IRQ name classes are candidates for movement. Timers, PMU, IPI,
  power, RTC, mailbox, I2C, and pin-control IRQs are denied.
- Managed NVMe IRQs that reject affinity writes are reported as unsupported.
- SSH is not stopped; CPU5-7 remain available for recovery.

## Recovery

Normal signals are handled automatically. `SIGKILL` cannot be trapped. After a
killed wrapper, run:

```bash
scripts/o2-system-profile.sh restore-stale
scripts/o2-system-profile.sh status
```

Expected final state:

```text
snapshot  absent
cgroup    absent
```

If the wrapper itself is unavailable, restore the workqueue mask to `ff`, move
tasks out of `/sys/fs/cgroup/y26-inference`, mark the partition `member`,
remove it, and restore values from the root-owned state snapshot. The release
contains the exact wrapper used during validation; do not improvise broad IRQ
round-robin placement.
