#!/bin/sh
set -eu
# Idempotent lab rollback for the temporary Stage51 observability/tuning surfaces.
sudo -n sysctl -w kernel.perf_event_paranoid=2
sudo -n sysctl -w kernel.kptr_restrict=1
for irq in 86 99 102 103; do
    test ! -w "/proc/irq/$irq/smp_affinity_list" || printf '%s
' '0-7' | sudo -n tee "/proc/irq/$irq/smp_affinity_list" >/dev/null
done
