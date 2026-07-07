# Stage25 Final Report

classification: `stage25-conv-threading-expand-selected`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE25-CONV-THREADING-TILE-DECISION-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `e3bbacba79f2c58b10057735c514a280577223c2`
end_head: `b382bd71c4091cc3476d59f77cb35c2a0d246513`
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false
selected_conv_lane: `C1`
selected_candidate: `C1_thread_branch1_and_model4_cv2_4t`
onnx_cut_status: pass
rounding_regression_status: pass
bucket_attribution_status: pass
conv_roofline_status: pass
threading_matrix_status: pass
host_tests: pass
board_tests: pass

## Summary

Stage25 replayed the Stage24 selected `/model.4` ONNX-cut path, confirmed same-input ONNX-cut correctness, and selected C1 cluster0 threading propagation. The selected path keeps the Stage24 merge repair and adds explicit threaded sidecars for:

```text
/model.4/m.0/cv2/conv/Conv
/model.4/cv2/conv/Conv
```

No graph expansion, full engine, default backend switch, `/data/ncnn` mutation, XSlim, vmadot1/2/3, vmadotn, FP/vfmadot, CPU4-7 IME, COCO/mAP, or model FPS claim was made.

## Timing

| path | total_us | stddev_us | conv_us | activation_us | merge_us | output_quantize_us | mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage24 selected replay | 125176 | 69.4332 | 62070.3 | 32592.5 | 20961.5 | 6945.9 | 0 |
| Stage25 C1 selected | 89178.9 | 268.184 | 26164.1 | 32800.7 | 20964.3 | 6579.45 | 0 |

## Speedups

```text
total_speedup_vs_stage24_replay: 1.4037x
branch1_conv_speedup: 2.9547x
model4_cv2_conv_speedup: 3.1191x
```

## Validation

```text
host_ctest: 38/38 passed
riscv_cross_build: pass
board_correctness: pass
board_affinity: CPU0-3 only, affinity_ok=1
frm_sweep: pass for RNE/RTZ/RDN/RUP/RMM
output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
```

## Next

next_recommended_step: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE26-ACTIVATION-REQUANT-REPAIR-AFTER-CONV-THREADING-001`
