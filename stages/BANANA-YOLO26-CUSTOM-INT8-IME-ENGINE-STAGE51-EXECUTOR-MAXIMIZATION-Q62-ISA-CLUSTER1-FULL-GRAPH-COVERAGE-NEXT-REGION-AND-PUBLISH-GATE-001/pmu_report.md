# PMU report

Basic per-worker cycles and instructions were measured after worker affinity. Counter-enabled
profiles perturb execution and are not headline timing. Generic cache events returned
`time_running=0` and are labeled unsupported, not zero. Named X60 stall/L1/L2 events remain
unavailable because matching event JSON/source was absent. Wall time remains the selector.
