# STAGE41 Final Report

classification: stage41-partial-correctness-only

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE41-POST-MODEL4-BLOCK-PROFILING-AND-FIRST-INPROCESS-RUNNER-GATE-001

repo: /data/banana-yolo26-spacemit-demo

branch: yolo26-custom-int8-engine

start_head: 6559e2a4a146e96df9db37bf748808896d08e147

end_head: unchanged-no-commit-hard-board-gate-failed

pushed: false

## Result

Stage41 built the first C++ in-process scaffold:

```text
Python in measured runtime path: false
Per-block file I/O in measured runtime path: false
Custom /model.4 called through C++ API: true
ORT fallback through in-process C API: true
Default backend changed: false
Full engine implemented: false
Production/model FPS claim: false
```

The scaffold is byte-exact on the host accepted ORT 1.27 oracle using the scalar custom `/model.4` path:

```text
full_ort_vs_expected_output0: mismatches=0 max_abs_diff=0
custom_model4_vs_ort_model4: mismatches=0 max_abs_diff=0
custom_model4_through_suffix_vs_full_ort_output0: mismatches=0 max_abs_diff=0
```

The board selected-mode hard gate failed because board ORT CPU output does not match the accepted Stage40 host ORT oracle, and custom `/model.4` does not match board ORT `/model.4` exactly:

```text
board_spacemit_ort_2.0.1_full_ort_vs_host_expected: mismatches=1597 max_abs_diff=635.707
board_custom_model4_vs_board_ort_model4: mismatches=78351 max_abs_diff=2
board_custom_through_suffix_vs_board_full_ort: mismatches=1508 max_abs_diff=635.707155
affinity_ok: 1
```

An upstream board ORT smoke did not close the mismatch.

## Timing

Host exact scaffold timing, not model FPS:

```text
mean_total_us: 301182.315667
mean_prefix_us: 60853.329333
mean_custom_model4_us: 96848.580667
mean_suffix_us: 130142.634167
mean_layout_conversion_us: 1631.365500
```

Board selected-mode timing is recorded only as blocked evidence:

```text
mean_total_us: 858404.224484
mean_prefix_us: 230282.265274
mean_custom_model4_us: 25354.844728
mean_suffix_us: 554279.618664
mean_layout_conversion_us: 11022.462626
correctness: fail
```

## Suffix Ranking

Exact host C++ in-process cumulative suffix profile identifies a provisional next target:

```text
provisional_target: model.16
incremental_delta_us: 16406.986
node_count: 66
conv_count: 9
operator_mix: Add:2, Concat:2, Conv:9, DequantizeLinear:17, Mul:9, QuantizeLinear:17, Sigmoid:9, Split:1
```

`model.23` has a larger delta but is detect/output-head and postprocess-heavy, so it is not selected as the first expansion target.

## Decision

Stage41 does not accept a new custom block implementation gate because the board selected-mode full-output comparison failed.

Next recommended step:

```text
BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE42-INPROCESS-ORT-CONTRACT-REPAIR-AND-MODEL16-ORACLE-GATE-001
```

Stage42 should repair or explicitly scope the board in-process ORT CPU reference contract, then generate the model.16 same-input cut oracle before any optimized model.16 custom block work.

## Non-Claims

This is not full YOLO26 production inference.

This is not model FPS.

This is not camera/full-image performance.

This is not COCO/mAP.

This is not production/default-backend readiness.
