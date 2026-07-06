# Cross-Track Notes

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE22-MODEL4-ONNX-CUT-ORACLE-AND-NEXT-BLOCK-GATE-001`

## YOLO11 Production Repo

`/data/banana-yolo11-spacemit-demo` had pre-existing untracked files during Stage22 preflight:

```text
?? .claude/
?? AGENTS.md
```

Stage22 did not modify the YOLO11 production repo.

## ncnn Repo

`/data/ncnn` had pre-existing modified files during Stage22 preflight:

```text
 M src/layer/riscv/convolution_1x1_int8_xsmtvdot.S
 M src/layer/riscv/convolution_1x1_int8_xsmtvdot.cpp
 M src/layer/riscv/convolution_1x1_int8_xsmtvdot.h
```

Stage22 did not modify `/data/ncnn`.
