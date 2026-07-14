# Scheduler decision

The SCHED_OTHER epoch-spin arm is selected for the optimized-research benchmark because it improves mean, p95, and p99 and removes voluntary worker switches. Its higher process CPU occupancy is explicit; condition-variable wake remains the compatibility default.
