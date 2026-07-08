# SIGILL Trap PC / Instruction Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE35-VMADOT-SIGILL-EMISSION-REPAIR-AND-THROUGHPUT-001`

## Trap Handler

`bench_stage35_vmadot_sigill` installs a local `SIGILL` handler and reports:

```text
case_name
si_code
pc
faulting_insn32_hex
sched_getcpu()
```

The handler is used only for Stage35 probes. It is not a production/default backend mechanism.

## Pre-Repair Fault

The first CPU0 matrix showed all cases exiting through the trap path. The faulting instruction was not the `smt.vmadot` payload:

```text
observed_faulting_words:
  0xc0002773
  0xc0002873
classification: SIGILL-at-rdcycle
```

Interpretation:

```text
Stage34/early-Stage35 traps were caused by direct cycle-counter reads in the benchmark measurement path.
The accepted engine helper still executed; the new diagnostic wrapper trapped before a valid payload conclusion could be made.
```

Raw evidence:

```text
run_logs/008_board_cpu0_sigill_matrix.log
artifacts/board_cpu0_sigill_matrix_after_chrono.tsv
```

## Post-Repair Matrix

After replacing `rdcycle` with `std::chrono::steady_clock`, CPU0 final matrix:

```text
all_cases_status: pass
all_cases_mismatches: 0
all_cases_trap: 0
all_cases_faulting_insn32_hex: 0x0
```

Summary artifact:

```text
artifacts/board_cpu0_sigill_matrix_final.tsv
```

## CPU1/2/3 Smoke

After CPU0 passed, key raw payloads were run on CPU1/2/3:

```text
case2_stage34_exact_single_wrapper_shape_raw_same_as_helper: pass on CPU1/2/3
A5_raw_independent_6_accumulators_if_register_safe: pass on CPU1/2/3
```

Summary artifact:

```text
artifacts/board_cpu1_3_smoke.tsv
```

No CPU4-7 IME execution was performed.

## Classification

```text
root_cause: measurement-path rdcycle SIGILL
vmadot_payload_status_after_repair: board-executable
named_inline_status_after_repair: board-executable
raw_same_as_helper_status_after_repair: board-executable
standalone_S_status_after_repair: board-executable
```
