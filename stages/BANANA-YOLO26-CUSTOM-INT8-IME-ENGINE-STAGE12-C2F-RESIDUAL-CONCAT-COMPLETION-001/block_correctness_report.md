# Block Correctness Report

selected_subset: `candidate_G_model2_c2f_add_concat_cv2_conv`

## Host

Host-native CTest:

```text
29/29 tests passed
```

New test:

```text
test_stage12_c2f_block_runner: pass
```

Host validates exact compact scalar `A0_int8_lut` path. RISC-V-only `A2_rvv_f32_lut`
is tested on board because host has no RVV.

## Board

Board CPU0/1/2/3:

- `test_stage10_rvv_rounding_control`: pass
- `test_stage11_branch_block_runner`: pass
- `test_stage12_c2f_block_runner`: pass

Stage 12 fixture status on all CPUs:

- `concat_mismatches=0`
- `model2_cv2_mismatches=0`

## Scope

This proves the selected C2f subset only. It is not full YOLO26 inference, not
COCO/mAP, not camera, and not production readiness.
