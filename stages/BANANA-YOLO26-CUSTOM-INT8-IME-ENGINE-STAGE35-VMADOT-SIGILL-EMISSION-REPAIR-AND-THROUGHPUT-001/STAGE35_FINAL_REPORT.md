# STAGE35 FINAL REPORT

classification: `stage35-vmadot-emission-repaired-throughput-measured-ready-for-pipelined-cv2`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE35-VMADOT-SIGILL-EMISSION-REPAIR-AND-THROUGHPUT-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `71a89f40daf39e4a6675ef6dfc2c8485d2671fc7`
end_head: `a8b76072f19ff792bc5afc33ab93a022f2c26eb6`
pushed: `false`
full_engine_implemented: `false`
ncnn_source_mutated: `false`
production_claim_made: `false`

## Scope

Stage35 repaired and re-ran the Step-0 `smt.vmadot` diagnostic. It did not implement a full YOLO26 engine, graph-wide scheduler, graph expansion, camera/full-image path, COCO/mAP, model FPS, production/default backend, vmadotn, or vfmadot.

## Root Cause

Stage34 did not prove a hardware throughput ceiling. The Stage35 trap-safe replay found:

```text
faulting_insn32_hex: 0xc0002773 / 0xc0002873
root_cause: rdcycle in benchmark timing path
```

After replacing direct cycle-counter timing with `std::chrono::steady_clock`, the payloads became board-executable.

## Emission / SIGILL Results

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

All rows:

```text
status: 0
mismatches: 0
trap: 0
```

CPU1/2/3 smoke:

```text
case2 raw same as helper: pass
A5 six accumulator raw: pass
CPU4-7 IME execution: none
```

## Throughput

Mandatory protocol:

```text
CPU0
warmup=10
runs/iterations=100
repeats=5
```

Selected rows:

```text
A1_raw_single_acc_dependent_chain: 4.918 ns/vmadot
A2_raw_single_acc_load_included: 6.166 ns/vmadot
A3_raw_independent_2_accumulators: 2.500 ns/vmadot
A4_raw_independent_4_accumulators: 1.250 ns/vmadot
A5_raw_independent_6_accumulators_if_register_safe: 0.861333 ns/vmadot
```

Supplemental high-iteration diagnostic:

```text
A1_raw_single_acc_dependent_chain: 3.77661 ns/vmadot
A4_raw_independent_4_accumulators: 0.937923 ns/vmadot
A5_raw_independent_6_accumulators_if_register_safe: 0.625296 ns/vmadot
```

Interpretation:

```text
vmadot emission repaired: yes
independent accumulator throughput improvement: yes
tested register shape ceiling: not observed
cv2 candidate integrated: no
```

## Validation

```text
git_diff_check: pass
host_build: pass
host_ctest: pass, 42/42
riscv_cross_build: pass, Y26_K1X_ENABLE_IME=ON
board_validation: pass for Stage35 CPU0 matrix and CPU1/2/3 smoke
symlink_scan: pass
secret_path_scan: pass_with_documented_command_log_self_matches
result_packet_export: pending
final_head_copy: result packet will contain `.with-final-head.md` copies after local commit
```

## Next Recommended Step

`BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE36-CV2-PIPELINED-VMADOT-CANDIDATE-001`

Open a bounded `/model.4/cv2/conv/Conv` candidate using the Stage35 board-executable raw/proven `smt.vmadot` emission and 4/6 independent accumulator groups. Keep signed-storage s8xs8 and explicit correction.

## Non-Claims

```text
This is not full YOLO26 inference.
This is not model FPS.
This is not full-image/camera performance.
This is not COCO/mAP.
This is not production/default-backend readiness.
This does not prove vmadotn support.
This does not prove vfmadot support.
```
