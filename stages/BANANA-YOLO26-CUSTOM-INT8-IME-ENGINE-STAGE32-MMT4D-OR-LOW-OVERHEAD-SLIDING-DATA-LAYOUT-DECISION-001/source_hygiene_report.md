# Source Hygiene Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE32-MMT4D-OR-LOW-OVERHEAD-SLIDING-DATA-LAYOUT-DECISION-001

Raw log:

```text
/data/ncnn-logs/ai-team/2026-07-08_12-51-18/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE32-MMT4D-OR-LOW-OVERHEAD-SLIDING-DATA-LAYOUT-DECISION-001/run_logs/final_hygiene.log
```

## Checks

| check | status |
|---|---|
| git diff --check | pass |
| git diff --cached --check | pass |
| symlink scan under custom_int8_engine and stages | pass, no symlinks listed |
| changed-file secret-like scan | pass, no findings |
| changed-file control-character scan | pass, no findings |
| host build + CTest | pass, 41/41 |
| RISC-V cross build with Y26_K1X_ENABLE_IME=ON | pass |
| board CPU0-3 selected-cut correctness | pass |
| board Stage32 signedness CPU0-3 smoke | pass |

## Cross-Track Note

`/data/ncnn` showed unrelated dirty files:

```text
M src/layer/riscv/convolution_1x1_int8_xsmtvdot.S
M src/layer/riscv/convolution_1x1_int8_xsmtvdot.cpp
M src/layer/riscv/convolution_1x1_int8_xsmtvdot.h
```

Stage32 did not touch `/data/ncnn`, did not depend on those changes, and did not mutate the ncnn source tree.
