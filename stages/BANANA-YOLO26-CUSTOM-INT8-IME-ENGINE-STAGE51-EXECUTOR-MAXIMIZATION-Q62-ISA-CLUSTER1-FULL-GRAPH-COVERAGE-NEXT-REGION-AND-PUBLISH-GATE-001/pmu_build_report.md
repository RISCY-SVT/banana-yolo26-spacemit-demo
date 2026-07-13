# PMU build report

The board had no `perf` CLI and no matching local SpacemiT X60 pmu-events source. No arbitrary
kernel tree was treated as authoritative and no global perf binary was installed. The existing
stage-owned `perf_event_open` helper provided basic per-worker cycles/instructions.
