# PMU report

Basic cycles, instructions, branches, branch misses, task clock, and context switches were readable CPU-wide on CPU0-3 with `time_running == time_enabled`. Counts cover the complete command envelope, including package prepare and unrelated CPU-wide activity, so they are diagnostic and not normalized to selected-route cycles. Aggregate cycles were 732112483, instructions 509498786, and cross-run diagnostic IPC 0.695930.

Generic cache events returned `time_running=0` and are classified `unmapped-or-unsupported`, not zero activity. X60 frontend/backend/vector event names and per-worker attribution remain unavailable. Wall clock remains the selection authority.
