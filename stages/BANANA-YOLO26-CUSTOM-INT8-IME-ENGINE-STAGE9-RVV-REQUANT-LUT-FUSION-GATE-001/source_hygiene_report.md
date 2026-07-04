# Source Hygiene Report

classification: pass

Checks:

- `git diff --check`: pass.
- `git diff --cached --check || true`: no staged diff at pre-hygiene check time.
- `find custom_int8_engine stages -type l -print`: no symlinks found.
- Secret-like scan over `custom_int8_engine stages scripts docs`, excluding `commands.txt`: no findings.

Scope controls:

- `/data/ncnn` was not mutated.
- `/data/banana-yolo11-spacemit-demo` was not mutated.
- `/control` was not mutated.
- XSlim was not used.
- No closed vendor binaries, ONNX model copies, tensor dumps, COCO outputs, or board binaries were added to git.
- Stage-local ONNX Runtime usage was limited to host-side 256-code LUT oracle tooling.

Known local artifacts not committed:

- Build dirs under `.deps/custom_int8_engine/`.
- Board binaries deployed under `/home/svt/yolo26-custom-int8-stage9/2026-07-04_13-13-27`.
- Raw logs under `/data/ncnn-logs/ai-team/2026-07-04_13-13-27/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE9-RVV-REQUANT-LUT-FUSION-GATE-001/`.
