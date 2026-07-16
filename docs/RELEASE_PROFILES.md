# Release Profiles

| Public profile | Wake policy | CPU/system contract | Expected use |
|---|---|---|---|
| `compatibility` | condition variable | CPU0-3 workers, CPU4 controller, original OS placement | shared boards, integration, diagnosis |
| `low-latency` | frame-gated epoch spin | same CPUs, original OS placement | continuously active dedicated executor |
| `low-latency-dedicated` | frame-gated epoch spin | low-latency plus reversible O2 wrapper | measured dedicated-board handoff |

All three profiles use the same frozen exact operator route, package, layout,
weights, and output. Profile selection changes worker wake/system placement only.

The release CLI selects a profile with `--profile`. The C API selects wake policy
through `y26_executor_options.wake_policy`. The dedicated profile additionally
requires `scripts/o2-system-profile.sh run -- ...`; the C API does not mutate the
operating system.

Release builds do not require or honor `Y26_STAGE53_*` through `Y26_STAGE57_*`
variables for operator selection. Those names remain research-only controls.

O2 is not persistent. It keeps the original B0 boot profile, NVMe runtime,
governor, sysctls, THP, cpuidle, and scheduler class. See `SYSTEM_PROFILE_O2.md`.
