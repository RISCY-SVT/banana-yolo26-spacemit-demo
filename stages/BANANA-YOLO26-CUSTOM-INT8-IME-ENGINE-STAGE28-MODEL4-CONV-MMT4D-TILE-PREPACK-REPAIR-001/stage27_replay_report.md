# Stage27 Replay Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE28-MODEL4-CONV-MMT4D-TILE-PREPACK-REPAIR-001`
replayed_mode: `Y26_STAGE16_MERGE_MODE_STAGE26_BRANCH1_ADD_LUT`
selected_path: `/model.4` same-input ONNX cut

## Command

```text
taskset -c 0-3 ./bench_stage23_model4_runner_cut --fixture-dir fixtures --mode ime_threaded --output-quantize rvv --merge-repair branch1_add_lut --thread-branch0 4 --thread-branch1 4 --thread-model4-cv2 4 --warmup 10 --runs 100 --repeats 5 --frm-sweep --dump-actual stage28_stage27_replay_actual.bin
```

## Binary And Fixtures

```text
bench_sha256: 77e8957ee945579d3f2b6570a776298bf00f177e3b8033c8b5911fefad20d176
input_sha256: e4ec6700e37e974e5bf9814b90c415169b5e514ed9554592238dd836f84fdc5b
expected_output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
actual_output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
```

## Correctness

```text
status: pass
mismatches: 0
max_abs_diff: 0
checksum: 106597930
expected_checksum: 106597930
affinity_ok: 1
frm_sweep: pass
ambient_frm_values: RNE RTZ RDN RUP RMM
post_call_frm_restored: pass
```

## Stable Timing

```text
warmup: 10
runs: 100
repeats: 5
mean_total_us: 41580.9
stddev_total_us: 659.604
min_total_us: 40808
max_total_us: 42424
cv_total_pct: 1.58632
mean_attribution_pct: 99.9412
```

## Buckets

```text
input_adapter_us: 2620.89
conv_us: 26753.7
activation_requant_us: 2999.07
merge_us: 2148.66
post_concat_qdq_us: 2148.66
output_quantize_us: 7034.03
copy_layout_us: 0
pack_layout_us: 0
thread_overhead_us: 5007.01
correction_us: 2478.61
conv_compute_us: 18058.5
conv_copy_us: 1251.9
conv_worker_other_us: 3.88932
```

## Per-Conv Buckets

| node | conv_us | compute_us | correction_us | copy_us | worker_other_us |
| --- | ---: | ---: | ---: | ---: | ---: |
| `/model.4/m.0/cv1/conv/Conv` | 7718.13 | 5652.96 | 233.884 | 61.1128 | 0.653634 |
| `/model.4/m.0/cv2/conv/Conv` | 6324.68 | 4312.3 | 521.524 | 158.699 | 1.33702 |
| `/model.4/cv2/conv/Conv` | 12710.9 | 8093.26 | 1723.2 | 1032.09 | 1.89867 |

## Non-Claims

This replay is selected `/model.4` ONNX-cut evidence only. It is not full YOLO26 inference, model FPS, full-image/camera performance, COCO/mAP, production readiness, or default-backend readiness.
