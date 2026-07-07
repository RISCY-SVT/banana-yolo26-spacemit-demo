# Stage24 Selected Path Replay Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE25-CONV-THREADING-TILE-DECISION-001
replayed_path: Y26_STAGE16_MERGE_MODE_STAGE24_B3_SPLIT1_LUT
mode: ime_threaded
output_quantize: rvv
affinity: taskset -c 0-3
protocol: warmup=10 runs=100 repeats=5

## Result

```text
status: pass
mismatches: 0
max_abs_diff: 0
checksum: 106597930
expected_checksum: 106597930
output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
affinity_ok: 1
frm_sweep: pass
```

## Timing

```text
mean_total_us: 125176
stddev_total_us: 69.4332
min_total_us: 125090
max_total_us: 125277
cv_total_pct: 0.0554683
mean_conv_us: 62070.3
mean_activation_requant_us: 32592.5
mean_merge_us: 20961.5
mean_output_quantize_us: 6945.9
mean_branch0_conv_us: 6529.83
mean_branch1_conv_us: 18189.1
mean_model4_cv2_conv_us: 37351.4
mean_attribution_pct: 99.9871
conv_share_pct: 49.5863
activation_share_pct: 26.0372
merge_share_pct: 16.7456
output_quantize_share_pct: 5.54889
```

## Board Anchor

```text
hostname: bf3
kernel: Linux bf3 6.6.63 #2.2.7.2 SMP PREEMPT Fri Aug 15 12:32:44 UTC 2025 riscv64
cpu: Spacemit(R) X60, CPU0-7 online
scaling_governor: performance
cpu0_scaling_cur_freq: 1600000
```

This is selected `/model.4` ONNX-cut timing only. It is not full YOLO26 inference, not model FPS, not camera/full-image performance, and not COCO/mAP.
