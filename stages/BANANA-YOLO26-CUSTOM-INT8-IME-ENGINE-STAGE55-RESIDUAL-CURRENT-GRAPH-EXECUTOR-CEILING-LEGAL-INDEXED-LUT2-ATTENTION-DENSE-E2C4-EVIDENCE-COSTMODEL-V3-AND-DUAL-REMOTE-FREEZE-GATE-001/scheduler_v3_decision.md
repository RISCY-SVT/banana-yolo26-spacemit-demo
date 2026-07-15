# Scheduler V3 decision

Select frame-gated epoch-spin as the dedicated-board low-latency profile and retain the condition-variable pool as compatibility. Across 0, 5, 16.7, 33.3, and 100 ms gaps, frame gating preserves inference latency while removing between-frame spin. At 100 ms, process user CPU fell 30.273438% versus raw spin.
