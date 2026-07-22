# Future Non-IME RISC-V Vector Register Memo

## Scope

This memo asks what a future standards-compatible, non-IME RISC-V CPU would
need to run the rewritten RVV executor without avoidable register spills. It
does not select hardware, authorize a custom opcode, or change the Stage60
software contract. The detailed machine-readable calculation is in
`future_riscv_vector_register_budget.tsv`.

The measured Stage60 graph keeps the same 59 unique Conv/MatMul M/N/K classes
at every resolution. Resolution changes M and tail frequency, but it does not
remove the register pressure created by N16 accumulation and an exact Q62 C8
epilogue. R384 removes all spatial M12 tails; other mandatory resolutions have
102 Conv instances with an M12 tail. This makes R384 useful for separating
tail cost from register-file requirements.

## Configurations

| ID | Architectural state | Physical state | Main result |
| --- | --- | --- | --- |
| A | VLEN256, 32 registers | no rename | M8 fused epilogue and indexed LUT2 fit; full M12 fusion/load-ahead does not |
| B | VLEN512, 32 registers | no rename | All five bounded destructive schedules fit architecturally |
| C | VLEN512, 32 registers | 40 renamed | Architecturally sufficient; only 4 physical destinations of headroom in the largest modeled case |
| D | VLEN512, 32 registers | 48 renamed | Architecturally sufficient with 12 or more physical destinations of modeled headroom |
| E | VLEN512, 32 registers | 64 renamed | Architecturally sufficient with generous overlap headroom |

The spill decision is made against the 32 software-visible names. Rename
registers cannot make an instruction stream that needs more than `v0-v31`
architecturally legal; they can retain old destinations and improve overlap
for an already legal schedule.

## Kernel Budgets

The bounded destructive schedules require the following peak architectural
names at VLEN512:

| Kernel | Peak names | No-spill result |
| --- | ---: | --- |
| M8xN16 plus fused requant/LUT/store | 24 | pass |
| M12xN16 plus fused requant/LUT/store | 28 | pass |
| M12xN16 plus two-iteration load-ahead | 26 | pass |
| attention MatMul plus C8 epilogue | 26 | pass |
| indexed LUT2 | 7 | pass |

At VLEN256, the same M12 accumulator geometry consumes 24 names before a
complete fused epilogue or two live A/B delivery sets are added. The modeled
full schedules peak at 38-40 names and therefore require splitting or spilling.
The M8 destructive schedule peaks at 32 and remains legal but has no spare
architectural name.

These are register-allocation proofs for the stated schedules, not cycle
predictions. A compiler or assembly implementation can use a split-group
schedule with fewer names at the cost of extra handoffs.

## Raw Capacity And Bandwidth

| VLEN / physical entries | Raw vector-file bytes |
| --- | ---: |
| 256 / 32 | 1,024 |
| 512 / 32 | 2,048 |
| 512 / 40 | 2,560 |
| 512 / 48 | 3,072 |
| 512 / 64 | 4,096 |

For one full-width three-source dot issue per cycle, the diagnostic minimum is
96 bytes read plus 32 bytes written at VLEN256 and 192 bytes read plus 64 bytes
written at VLEN512. The budget assumes at least 8 and 16 banks respectively,
subject to the implementation's banking, operand reuse, and accumulator bypass.
Sustaining these rates requires strong chaining/bypass; raw capacity alone is
not enough.

The target should also provide:

- ELEN64 and fractional LMUL;
- one vector load/store pipeline;
- one or two integer vector arithmetic pipelines;
- at least eight in-flight vector operations;
- enough outstanding loads to overlap packed A/B delivery;
- a bypass path that avoids round-tripping every accumulator update through
  the vector file.

## Recommendation

The default hypothesis for a future non-IME target is standard V with 32
architectural registers, VLEN512, ELEN64, fractional LMUL, strong vector
chaining, and 48-64 implementation-private physical registers if out-of-order
renaming is present. Forty physical entries are a bounded minimum but leave
little modeled overlap headroom.

`Zvldot`, `Zvbdot`, `Zvqdot`, and `Zvdot4a8i` remain unratified at the Stage60
capture date. They may be evaluated after their specifications and toolchains
stabilize, but they are not a production-software dependency or a substitute
for the standard-V register and memory-system budget.
