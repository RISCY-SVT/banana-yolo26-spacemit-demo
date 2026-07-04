# Source Hygiene Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE10-BACKBONE-EXPANSION-POST-ACTIVATION-GATE-001`

## Checks

- `git diff --check`: pass
- `git diff --cached --check || true`: pass/no output
- `find custom_int8_engine stages -type l -print`: no symlinks
- secret-like scan over `custom_int8_engine stages scripts docs`, excluding `commands.txt`: no findings

## Artifact Policy

No ONNX model copy, large tensor dump, board binary, COCO output, camera output, closed vendor binary, or credential file was staged for git.

## Scope Guard

- `/data/ncnn` was not mutated.
- `/data/banana-yolo11-spacemit-demo` was not mutated.
- XSlim was not used.
- `vmadot1/2/3`, `vmadotn`, and FP/vfmadot were not implemented.
