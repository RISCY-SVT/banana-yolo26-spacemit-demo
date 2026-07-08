# Stage34 Replay / Caveat Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE35-VMADOT-SIGILL-EMISSION-REPAIR-AND-THROUGHPUT-001`

## Stage34 Accepted Record

```text
stage34_classification: stage34-vmadot-throughput-ceiling-no-pipeline-win
stage34_start_head: 6c64bdd1f9e00359c4c0a084926a75f338252a3d
stage34_end_head: 71a89f40daf39e4a6675ef6dfc2c8485d2671fc7
```

Stage34 recorded that the existing accepted wrapper path executed successfully, while every new inline/register-blocked diagnostic case trapped with `SIGILL`.

## Stage35 Replay Scope

Stage35 replayed the relevant Stage34 diagnostic shape through a new trap-safe tool:

```text
tool: bench_stage35_vmadot_sigill
board: Banana-Pi BPI-F3 / SpacemiT K1X
initial_cpu: CPU0
later_cpu_smoke: CPU1/2/3
scope: vmadot microkernel diagnostic only
not_yolo26_inference: true
not_model_fps: true
```

## Key Finding

The Stage34 failure was reproduced only until the timing path was inspected. The faulting instruction was not `smt.vmadot`; it was `rdcycle`:

```text
faulting_insn32_hex: 0xc0002773 / 0xc0002873
classification: measurement-path SIGILL
```

After replacing `rdcycle` with `std::chrono::steady_clock`, the same `smt.vmadot` payloads became board-executable.

## Corrected Replay Result

CPU0 final matrix:

```text
case0_existing_helper_call: pass
case1_stage34_exact_single_wrapper_shape_named: pass
case2_stage34_exact_single_wrapper_shape_raw_same_as_helper: pass
case3_standalone_S_known_good_bytes: pass
case4_standalone_S_named_v28_v0_v1: pass
case5_standalone_S_raw_word_same_as_case4: pass
case6_v24_v0_v1: pass
case7_v20_v0_v1: pass
case8_two_accumulators_v28_v30: pass
case9_four_accumulators_v20_v22_v24_v26: pass
A5_raw_independent_6_accumulators_if_register_safe: pass
```

All passed cases reported:

```text
status: 0
mismatches: 0
trap: 0
faulting_insn32_hex: 0x0
```

## Raw Evidence

```text
board_cpu0_sigill_matrix_pre_repair: run_logs/008_board_cpu0_sigill_matrix.log
board_cpu0_sigill_matrix_final: artifacts/board_cpu0_sigill_matrix_final.tsv
board_cpu1_3_smoke: artifacts/board_cpu1_3_smoke.tsv
objdump_after_repair: artifacts/objdump_stage35_bench_after_a5.txt
```

## Conclusion

Stage34 did not prove a hardware `smt.vmadot` throughput ceiling. It proved that the previous diagnostic was not board-safe because it used a privileged/unavailable cycle counter path. Stage35 repairs the diagnostic substrate and reopens the cv2 pipelining question with valid board-executable evidence.
