# Stage 10 Baseline Replay Report

selected_subset: `candidate_E_branch1_stage9_split_model2_m0_cv1_conv`

## Pre-Code Replay

- host CTest: `27/27` pass
- RISC-V cross build: pass
- board CPU0/1/2/3 correctness: pass
- RNE ambient-frm regression: pass
- mismatches: `0`

CPU0 pre-code Stage10 A2 microbench:

- total_us: `234421`
- activation_share: `15.3506%`
- pack_layout_share: `0.489263%`

## Post-Code Replay

CPU0 post-code Stage10 A2 microbench:

- total_us: `234474`
- activation_share: `15.365%`
- pack_layout_share: `0.513195%`
- mismatches: `0`

The Stage 10 replay gate passed before Stage 11 expansion and remained passing after Stage 11 changes.
