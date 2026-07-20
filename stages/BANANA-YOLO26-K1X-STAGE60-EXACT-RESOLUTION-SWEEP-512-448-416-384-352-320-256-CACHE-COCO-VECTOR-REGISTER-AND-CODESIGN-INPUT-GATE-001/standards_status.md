# RISC-V Vector Dot-Product Standards Status

Status captured on 2026-07-20 from RISC-V International sources.

| Item | Current public status | Stage60 treatment |
| --- | --- | --- |
| V vector extension | Ratified, version 1.0 | Standards-compatible baseline with 32 architectural vector registers, `v0` through `v31` |
| `Zvldot` / `Zvbdot` | Freeze stage on the RISC-V specification dashboard; not ratified | Under-development option only |
| `Zvqdot` / `Zvdot4a8i` | Stabilization work in progress; not ratified | Under-development option only |

The [RISC-V specification dashboard](https://riscv.github.io/adm-spec-dashboard/)
is the status authority used here. At capture time it showed the long/batched
dot-product project through Freeze, with ratification work still pending, and
the `Zvqdot`/`Zvdot4a8i` project in stabilization. The latter project's
[ratification plan](https://riscv.atlassian.net/wiki/spaces/PSXX/pages/766672912/Dot-Product%2BZvqdot%2BRatification%2BPlan)
also schedules future review and ratification milestones rather than claiming
a ratified specification.

The [ratified V specification](https://docs.riscv.org/reference/isa/extensions/vector/_attachments/riscv-v-spec.pdf)
defines 32 architectural vector registers. Stage60 therefore does not propose
64 architectural registers as a standards-compatible target. Configurations
with 40, 48, or 64 entries in the budget refer to implementation-private
physical rename capacity behind the standard 32-register interface.

No Stage60 software route uses any of the unratified dot-product proposals or
a new custom opcode. Their possible benefit is a future hardware research
question and cannot be part of the frozen executor contract.
