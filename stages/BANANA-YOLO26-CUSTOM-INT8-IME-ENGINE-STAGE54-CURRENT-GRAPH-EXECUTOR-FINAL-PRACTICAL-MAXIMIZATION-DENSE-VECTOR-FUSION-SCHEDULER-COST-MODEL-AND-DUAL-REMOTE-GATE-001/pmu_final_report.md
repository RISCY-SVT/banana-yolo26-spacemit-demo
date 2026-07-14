# PMU final report

The stage-owned CPU-wide perf_event_open helper measured cycles, instructions, task-clock, context switches, and CPU migrations per CPU around the exact full-model workload, with time_running equal to time_enabled. These counts include any unrelated activity on CPU0-4 and are not presented as worker-owned counters. The board has no installed perf binary and no authoritative X60 cache/stall mapping, so no such events are invented.

CPU-wide prefix subtraction for individual dense rows was repeated but remained noisy and sign-changing; those rows are retained as diagnostic only. Wall time is selection authority.
