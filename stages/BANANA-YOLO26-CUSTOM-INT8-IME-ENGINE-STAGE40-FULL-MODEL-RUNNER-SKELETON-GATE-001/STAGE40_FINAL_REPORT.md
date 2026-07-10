# STAGE40 FINAL REPORT

classification: stage40-full-model-skeleton-correct-ready-for-custom-block-expansion
stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE40-FULL-MODEL-RUNNER-SKELETON-GATE-001
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
start_head: 57ad1bf94d9322108fdca453e03a340e1ce0b1f2
end_head: 6559e2a4a146e96df9db37bf748808896d08e147
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false

## Summary

Stage40 created a correctness-first full-model skeleton/dataflow gate:

```text
full ORT CPU reference
all-ORT prefix/model4/suffix fallback skeleton
custom /model.4 C++ runner insertion + ORT CPU suffix
```

All tested same-input paths matched the full ORT CPU `output0` exactly.

## Correctness

| comparison | status | mismatches | max_abs_diff |
|---|---:|---:|---:|
| prefix_vs_full_model4_input | pass | 0 | 0 |
| all_ort_model4_vs_full_model4_output | pass | 0 | 0 |
| all_ort_final_vs_full_reference | pass | 0 | 0 |
| custom_model4_output_vs_full_model4_output | pass | 0 | 0 |
| custom_model4_skeleton_final_vs_full_reference | pass | 0 | 0 |

Custom `/model.4` board runner:

```text
mode: branch3x3_fastpack + rvv_direct output quantize
mean_total_us: 26428.9
stddev_total_us: 129.331
mismatches: 0
max_abs_diff: 0
affinity_ok: 1
FRM sweep: pass
```

## Skeleton Profiling

| block | implementation | mean_us |
|---|---|---:|
| full_ort_reference | ORT CPU | 198259.272 |
| prefix_images_to_model4_input | ORT CPU | 56796.029 |
| model4_cut_all_ort | ORT CPU | 9466.208 |
| suffix_model4_output_to_output0 | ORT CPU | 129572.132 |

This is skeleton profiling only, not model FPS.

## Validation

- host build + CTest: pass, 42/42
- RISC-V cross build with `Y26_K1X_ENABLE_IME=ON`: pass
- board custom `/model.4` correctness and FRM sweep: pass
- Python tools compileall: pass
- hygiene: pass
- result packet export: pending at repo-local report-generation time

## Non-Claims

This is not full YOLO26 production inference. This is not final model FPS. This is not full-image/camera performance. This is not COCO/mAP. This is not production/default-backend readiness.

## Next Recommended Step

BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE41-POST-MODEL4-BLOCK-PROFILING-AND-FIRST-EXPANSION-GATE-001
