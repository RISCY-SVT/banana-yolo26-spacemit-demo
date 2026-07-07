# Conv Threading Matrix Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE27-CONV-OR-OUTPUT-QUANTIZE-DECISION-AFTER-ACTIVATION-REPAIR-001

## Protocol

```text
affinity: taskset -c 0-3
warmup: 10
runs: 100
repeats: 5
output_quantize: rvv
merge_repair: branch1_add_lut
```

Raw table:

```text
conv_threading_matrix.tsv
```

## Summary

| case | threads branch0/branch1/model4_cv2 | total_us | conv_us | thread_overhead_us | correctness |
|---|---:|---:|---:|---:|---|
| ime_single | 0/0/0 | 91636.9 | 77170.4 | 0 | pass |
| all1 threaded wrapper | 1/1/1 | 94921.7 | 80361.6 | 705.023 | pass |
| b0_2_b1_4_cv2_4 | 2/4/4 | 44666.0 | 29848.9 | 2399.09 | pass |
| b0_3_b1_4_cv2_4 | 3/4/4 | 41798.2 | 26951.6 | 3207.42 | pass |
| b0_4_b1_4_cv2_4 | 4/4/4 | 41642.8 | 26826.0 | 4710.67 | pass |
| b0_4_b1_2_cv2_4 | 4/2/4 | 44084.1 | 29304.9 | 2848.98 | pass |
| b0_4_b1_3_cv2_4 | 4/3/4 | 42621.3 | 27766.0 | 4285.2 | pass |
| b0_4_b1_4_cv2_2 | 4/4/2 | 48457.9 | 33611.5 | 2424.9 | pass |
| b0_4_b1_4_cv2_3 | 4/4/3 | 43361.0 | 28509.5 | 3686.74 | pass |

All cases preserved:

```text
mismatches: 0
max_abs_diff: 0
checksum: 106597930
output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
affinity_ok: 1
```

## Decision

The current 4/4/4 policy remains the fastest measured local thread-count policy. The 1-thread threaded wrapper is worse than true single-thread IME, so any future threshold policy must dispatch small nodes to the non-threaded IME path rather than a 1-thread threaded workspace.
