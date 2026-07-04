# Source Hygiene Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE12-C2F-RESIDUAL-CONCAT-COMPLETION-001`

## Checks

- `git diff --check`: pass
- `git diff --cached --check || true`: pass
- `find custom_int8_engine stages -type l -print`: no symlinks found
- secret-like scan over `custom_int8_engine stages scripts docs`, excluding `commands.txt`: no findings

## Scope Guard

- `/data/ncnn` was not mutated.
- `/data/banana-yolo11-spacemit-demo` was not mutated.
- `/control` was not mutated.
- XSlim was not used.
- `vmadot1/2/3`, `vmadotn`, and FP/vfmadot were not implemented.
- No full YOLO26 engine, graph-wide scheduler, camera/full-image path, COCO/mAP, or production/model FPS claim was added.

## Artifact Policy

Large generated data stayed under `.deps` or log roots. The repo tracks only
source, small fixtures, scripts, and stage reports.
