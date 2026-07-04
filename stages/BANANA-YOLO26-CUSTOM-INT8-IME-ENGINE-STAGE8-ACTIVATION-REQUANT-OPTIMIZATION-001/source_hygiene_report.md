# Source Hygiene Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE8-ACTIVATION-REQUANT-OPTIMIZATION-001`

## Checks

| check | status |
|---|---|
| `git diff --check` | pass |
| `git diff --cached --check || true` | pass |
| `find custom_int8_engine stages -type l -print` | pass, no symlinks |
| secret-like scan | pass, no findings |
| large changed files scan | pass, no files over 1 MiB |

## Scope Guard

- `/data/ncnn` was not mutated.
- `/data/banana-yolo11-spacemit-demo` was not mutated.
- `/control` was not mutated.
- No board packages, sysroot, toolchain, kernel, firmware, or device tree changes were made.
- No closed vendor binaries, ONNX model copies, layer dumps, COCO outputs, or board binaries were added to git.

## Known Safe Non-Source Outputs

Raw build logs and board logs remain under:

- `/data/ncnn-logs/ai-team/2026-07-04_10-26-52/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE8-ACTIVATION-REQUANT-OPTIMIZATION-001/run_logs/`

Result-packet candidate artifacts are small text reports only.
