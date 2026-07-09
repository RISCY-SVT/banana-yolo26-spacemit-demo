# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE39-BRANCH3X3-IM2COL-PACK-REPAIR-001

## Mission

Continue in `/data/banana-yolo26-spacemit-demo` on branch `yolo26-custom-int8-engine` after Stage38.

Stage38 selected `Y26_STAGE16_OUTPUT_QUANTIZE_STAGE38_RVV_DIRECT_STORE` for the existing `/model.4` same-input ONNX-cut path and reduced output QuantizeLinear:

```text
output_quantize_us: 7055.2 -> 4551.97
output_quantize_speedup: 1.54994x
selected_cut_total_us: 32890.5 -> 30341.5
selected_cut_total_speedup: 1.08401x
output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
```

The next material local bucket is branch 3x3 im2col/pack:

```text
branch0_im2col_pack_us: 3601.25
branch1_im2col_pack_us: 2000.66
combined_branch3x3_im2col_pack_us: 5601.91
combined_branch3x3_im2col_share_of_branch_conv: 54.67%
```

## Scope

Do not expand graph coverage. Do not implement full YOLO26 inference. Do not run camera/full-image/COCO/mAP. Do not mutate `/data/ncnn`. Keep IME on CPU0-3 only. Do not use `vmadotus`, `vmadot1/2/3` direct/sliding, `vmadotn`, FP/vfmadot, or all-core/OpenMP default dispatch.

## Required First Gate

Replay Stage38 selected mode:

```text
Y26_STAGE16_MERGE_MODE_STAGE37_BRANCH3X3_PIPELINED4
Y26_STAGE16_OUTPUT_QUANTIZE_STAGE38_RVV_DIRECT_STORE
warmup=10 runs=100 repeats=5
taskset -c 0-3
output_sha256=70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
```

Require `mismatches=0`, `max_abs_diff=0`, FRM sweep pass, and attribution >=98%.

## Candidate Lane

Target only branch 3x3 im2col/pack:

- fused im2col+pack for `/model.4/m.0/cv1/conv/Conv`
- fused im2col+pack for `/model.4/m.0/cv2/conv/Conv`
- interior fast path plus separate edge/padding path if exact
- reduce materialized intermediate copies

Acceptance:

```text
combined_branch3x3_im2col_pack_us speedup >= 1.30x
combined_branch3x3_conv_total_us speedup >= 1.15x
selected_cut_total_us speedup >= 1.05x
mismatches=0
FRM sweep pass
```

If im2col/pack cannot be repaired locally, write a decision report recommending either selected-cut persistent thread region work or a full-model runner skeleton gate.
