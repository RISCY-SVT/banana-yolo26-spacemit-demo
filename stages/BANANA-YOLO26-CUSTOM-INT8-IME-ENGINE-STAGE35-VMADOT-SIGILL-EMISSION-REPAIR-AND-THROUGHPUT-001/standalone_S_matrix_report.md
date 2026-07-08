# Standalone S / Inline Matrix Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE35-VMADOT-SIGILL-EMISSION-REPAIR-AND-THROUGHPUT-001`

## CPU0 Case Matrix

Source:

```text
artifacts/board_cpu0_sigill_matrix_final.tsv
```

| case | emission | status |
| --- | --- | --- |
| case0_existing_helper_call | public helper | board-executable |
| case1_stage34_exact_single_wrapper_shape_named | inline named | board-executable |
| case2_stage34_exact_single_wrapper_shape_raw_same_as_helper | inline raw word `0xe2103e2b` | board-executable |
| case3_standalone_S_known_good_bytes | standalone raw word `0xe2103e2b` | board-executable |
| case4_standalone_S_named_v28_v0_v1 | standalone named | board-executable |
| case5_standalone_S_raw_word_same_as_case4 | standalone raw word `0xe2103e2b` | board-executable |
| case6_v24_v0_v1 | inline named `v24` | board-executable |
| case7_v20_v0_v1 | inline named `v20` | board-executable |
| case8_two_accumulators_v28_v30 | inline raw 2 accumulators | board-executable |
| case9_four_accumulators_v20_v22_v24_v26 | inline raw 4 accumulators | board-executable |
| A5_raw_independent_6_accumulators_if_register_safe | inline raw 6 accumulators | board-executable |

All rows:

```text
status: 0
mismatches: 0
trap: 0
```

## CPU1/2/3 Smoke

Source:

```text
artifacts/board_cpu1_3_smoke.tsv
```

The exact raw helper-shaped case and 6-accumulator raw case passed on CPU1, CPU2, and CPU3. No CPU4-7 execution was performed.

## Interpretation

The standalone and inline forms are both valid after the benchmark no longer uses `rdcycle`.

```text
standalone_S_pass: yes
raw_encoding_only_pass: yes
named_encoding_pass: yes
register_shape_limited: no evidence for tested shapes
vtype_or_AVL_limited: no evidence for tested shapes
```
