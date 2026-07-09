# Source Hygiene Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE40-FULL-MODEL-RUNNER-SKELETON-GATE-001

## Checks

| check | status | note |
|---|---|---|
| `git diff --check` | pass | no whitespace errors |
| Python compileall, stage venv | pass | `custom_int8_engine/tools` |
| Python compileall, requested system command | pass | `custom_int8_engine/tools scripts tools`, command allowed `|| true` |
| host build + CTest | pass | 42/42 |
| RISC-V cross build | pass | `Y26_K1X_ENABLE_IME=ON` |
| symlink scan | pass | no symlinks under `custom_int8_engine` or `stages` |
| changed-file secret/path scan | pass | retry command returned `exit_code=0`, no findings |

The broad repository secret scan reported historical `commands.txt` self-matches containing old scan regex command lines. Those are not Stage40 changed-file findings.

## Cross-Track Notes

`/data/ncnn` had unrelated dirty files before Stage40:

```text
src/layer/riscv/convolution_1x1_int8_xsmtvdot.S
src/layer/riscv/convolution_1x1_int8_xsmtvdot.cpp
src/layer/riscv/convolution_1x1_int8_xsmtvdot.h
```

Stage40 did not read deeply, clean, or mutate `/data/ncnn`.

`/data/banana-yolo11-spacemit-demo` was checked for HEAD/tag context only and was not mutated.
