# Candidate Bench Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE28-MODEL4-CONV-MMT4D-TILE-PREPACK-REPAIR-001`
candidate: `T2_fused_correction_writeback`

## Implementation

T2 changes threaded Conv workers so correction writes directly to the final global output rows:

```text
before: raw local output -> corrected local buffer -> memcpy rows to global output
after: raw local output -> corrected global output rows
```

The correction formula is unchanged:

```text
corrected = raw_dot + (128 - activation_zero_point_u8) * weight_sum_oc + bias_oc
```

Only rows actually written by each worker are corrected, so overcomputed halo rows are not corrected or copied.

## Stable Board Benchmark

Protocol:

```text
taskset: CPU0-3
warmup: 10
runs: 100
repeats: 5
```

| metric | Stage27 replay T1 | Stage28 T2 | ratio |
| --- | ---: | ---: | ---: |
| total_us | 41580.9 | 40231.6 | 1.03354x |
| conv_us | 26753.7 | 25255.4 | 1.05933x |
| conv_compute_us | 18058.5 | 18097.1 | 0.99787x |
| conv_correction_us | 2478.61 | 2500.86 | 0.99110x |
| conv_copy_us | 1251.9 | 0 | eliminated |
| thread_overhead_us | 5007.01 | 4705.99 | 1.06396x |
| output_quantize_us | 7034.03 | 7071.88 | 0.99465x |

## Correctness

```text
status: pass
mismatches: 0
max_abs_diff: 0
actual_output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
expected_output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
frm_sweep: pass
affinity_ok: 1
```

## Decision

```text
candidate_status: accepted
accepted_reason: byte-exact, removes targeted copy/writeback bucket, improves total selected-cut time by 3.35% same-session.
```

T2 does not solve the dominant raw MMT4D compute bucket. Further Conv kernel work should be gated by Track B model value evidence.
