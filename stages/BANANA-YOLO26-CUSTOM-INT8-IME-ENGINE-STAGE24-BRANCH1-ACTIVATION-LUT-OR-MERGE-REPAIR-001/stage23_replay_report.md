# stage23_replay_report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE24-BRANCH1-ACTIVATION-LUT-OR-MERGE-REPAIR-001`
replayed_stage: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE23-MODEL4-CUT-CLOSED-BUCKET-ATTRIBUTION-AND-TARGETED-REPAIR-001`
selected_path: `/model.4` same-input ONNX cut runner API
mode: `ime_threaded`
output_quantize: `rvv`
merge_repair: `baseline`
protocol: `warmup=10 runs=100 repeats=5`
board_affinity: `taskset -c 0-3`

## Correctness

```text
status: 0
mismatches: 0
max_abs_diff: 0
checksum: 106597930
expected_checksum: 106597930
affinity_ok: 1
actual_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
expected_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
```

## Stable Replay Timing

```text
mean_total_us: 147624
stddev_total_us: 113.47
cv_total_pct: 0.0768645
mean_input_adapter_us: 2610.14
mean_conv_us: 62114.7
mean_activation_requant_us: 32615.2
mean_merge_us: 43334.4
mean_output_quantize_us: 6934.36
mean_attributed_us: 147609
mean_attribution_pct: 99.9898
mean_other_us: 15.0449
conv_share_pct: 42.0763
activation_share_pct: 22.0935
merge_share_pct: 29.3546
output_quantize_share_pct: 4.69731
```

## Rounding Replay

The accepted runner path was replayed under ambient `frm` values:

```text
RNE: mismatches=0 after_frm=0
RTZ: mismatches=0 after_frm=1
RDN: mismatches=0 after_frm=2
RUP: mismatches=0 after_frm=3
RMM: mismatches=0 after_frm=4
```

The replay remains selected-subset `/model.4` cut timing only. It is not full YOLO26 inference, model FPS, full-image/camera performance, COCO/mAP, or production readiness.
