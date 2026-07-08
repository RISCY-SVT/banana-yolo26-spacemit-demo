# VMADOT Throughput / Latency Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE34-VMADOT-CV2-PIPELINED-MICROKERNEL-001`

## Purpose

Mandatory Step 0 gate before any `/model.4/cv2/conv/Conv` pipelined kernel candidate.

The diagnostic used named `smt.vmadot` only. No raw opcode route was used.

## Build / Disassembly

```text
binary: .deps/custom_int8_engine/build-riscv-stage34/bench_stage34_vmadot_throughput
objdump_log: run_logs/objdump_stage34_exact.out
```

Observed symbolic instructions include:

```text
smt.vmadot v28,v0,v1
smt.vmadot v30,v0,v1
smt.vmadot v24,v0,v1
smt.vmadot v26,v0,v1
smt.vmadot v20,v0,v1
smt.vmadot v22,v0,v1
```

No `.insn` or raw vmadot encoding was used for Stage34.

## Board Results

Board: Banana-Pi BPI-F3 / SpacemiT K1X, CPU0 for microdiagnostic.

The Stage34 diagnostic tool now defaults to `probe_only_no_vmadot`; that case validates probe/affinity/report plumbing but does not execute an IME payload. All payload cases are selected explicitly with `--case`.

The existing accepted wrapper path still passes:

```text
bench_vmadot_microkernel direct wrapper:
  ime_direct_status: 0
  ime_direct_mean_ns_per_call: 49.487
  ime_direct_stddev_ns_per_call: 0.025
```

The Stage34 inline software-pipeline shapes all trapped with `case_rc=132` (`SIGILL` / core dump), including:

```text
dependent_chain_1acc_loadfree
load_included_1acc
independent_2acc_high_loadfree
independent_2acc_loadfree
independent_4acc_loadfree
independent_6acc_loadfree
safe_vset_each_1acc
safe_vset_each_2acc_high
exact_single_wrapper_shape
```

See `vmadot_throughput_matrix.tsv` for the case matrix.

## Interpretation

The accepted one-dot wrapper route remains executable, but the attempted inline loop/register-blocked shapes are not a safe board-executable substrate for a `/model.4/cv2` software-pipelined microkernel in Stage34.

This is a throughput ceiling / toolchain-or-instruction-shape ceiling for this lane, not evidence that the current MMT4D IME path is non-IME. Current MMT4D still uses plain `smt.vmadot`.

## Decision

Do not implement a cv2 pipelined microkernel candidate in Stage34.

classification_basis: `stage34-vmadot-throughput-ceiling-no-pipeline-win`
