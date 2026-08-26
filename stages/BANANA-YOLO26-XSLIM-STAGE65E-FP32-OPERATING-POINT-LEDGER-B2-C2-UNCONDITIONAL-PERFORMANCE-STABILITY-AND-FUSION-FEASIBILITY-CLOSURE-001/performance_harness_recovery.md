# Stage65E performance harness recovery

Two partial performance roots were excluded before the accepted clean run.

1. `performance-watchdog-v1`: a stage-local watchdog subshell could leave its
   `sleep 900` child alive and block the parent `wait`. The incomplete root was
   isolated; all stage-created watchdog sleeps were terminated. No model or
   runtime fault occurred.
2. `performance-proc-race-v2`: the resource sampler tested a process path and
   then read `/proc/<pid>/status` after the measured process had exited. The
   resulting normal process-exit race was made fail-closed and the incomplete
   root was isolated.

The accepted `stage65e-passport` run started from a fresh non-existing output
root. It uses no internal watchdog process, treats disappearing `/proc` state
as an expected terminal sampling condition, verifies every samples/resource
hash, and requires the complete 120-slot schedule. Neither partial root is an
accepted timing surface or contributes to any summary statistic.

No host or board reboot occurred during this recovery. A partial command root
is not evidence of an operating-system restart.
