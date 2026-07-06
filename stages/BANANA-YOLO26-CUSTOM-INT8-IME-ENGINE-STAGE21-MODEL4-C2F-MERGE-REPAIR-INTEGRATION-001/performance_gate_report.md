# Performance Gate Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE21-MODEL4-C2F-MERGE-REPAIR-INTEGRATION-001`

## Registered Gate

```text
Stage20 C2 mean_total_us: 116338
accepted +3% threshold: 119828 us
warning +10% threshold: 127972 us
```

## Observed Transfer Timing

```text
candidate: C2_split0_concat_lut_4t
mean_total_us: 116631
stddev_total_us: 364.855
cv_total_pct: 0.312829
mismatches: 0
affinity_ok: 1
```

## Decision

```text
transfer_timing_gate: pass
reason: 116631 us <= 119828 us
```

## Caveat

The representative/full-shape transfer timing is from the Stage20-compatible full-shape sidecar matrix. The real runner integration was separately proven through `bench_stage21_model4_c2f_integrated` on compact oracle scope and through `test_stage21_c2f_merge_repair`.
