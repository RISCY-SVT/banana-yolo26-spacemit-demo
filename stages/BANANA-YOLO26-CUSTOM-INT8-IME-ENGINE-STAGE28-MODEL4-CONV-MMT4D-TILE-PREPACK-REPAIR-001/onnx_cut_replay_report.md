# ONNX Cut Replay Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE28-MODEL4-CONV-MMT4D-TILE-PREPACK-REPAIR-001`
cut_input: `/model.4/cv1/conv/Conv_output_0_QuantizeLinear_Output`
cut_output: `/model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output`

## Replay Gate

```text
status: pass
runner_api_path: real `y26_stage16_model4_c2f_run_cut_u8_output`
mode: `Y26_STAGE16_MERGE_MODE_STAGE26_BRANCH1_ADD_LUT`
mismatches: 0
max_abs_diff: 0
actual_output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
expected_output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
checksum: 106597930
expected_checksum: 106597930
```

## FRM Sweep

```text
RNE: pass
RTZ: pass
RDN: pass
RUP: pass
RMM: pass
post_call_frm_restored: pass
```

## CPU Affinity

```text
taskset: CPU0-3
worker_affinity_ok: 1
cpu4_7_ime_execution: none observed
```

## Non-Claims

This is same-input `/model.4` ONNX-cut correctness evidence. It is not full YOLO26 inference, model FPS, full-image/camera performance, COCO/mAP, production readiness, or default-backend readiness.
