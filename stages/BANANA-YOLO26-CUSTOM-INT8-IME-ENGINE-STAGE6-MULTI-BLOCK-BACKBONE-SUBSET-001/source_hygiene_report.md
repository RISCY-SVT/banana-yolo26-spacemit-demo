# Source Hygiene Report

Scope:

- `custom_int8_engine/`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE6-MULTI-BLOCK-BACKBONE-SUBSET-001/`

Checks:

- `git diff --check`: pass
- `git diff --cached --check || true`: pass
- `find custom_int8_engine stages -type l -print`: no symlinks
- Secret-like scan over changed source/docs/reports excluding `commands.txt`: no findings
- Large generated dumps remain under `.deps/custom_int8_engine/stage6_oracle/2026-07-04_06-17-01/` and are not committed
- No credentials, `.env`, SSH keys, Codex auth/config state, closed vendor binaries, ONNX model copies, tensor dumps, COCO outputs, or board binaries are added to git

Policy confirmations:

- `/data/ncnn` was not mutated.
- `/data/banana-yolo11-spacemit-demo` was not mutated.
- `/control` was not mutated.
- XSlim was not used.
- `vmadot1/2/3`, `vmadotn`, and FP/vfmadot were not implemented.
- IME board execution was limited to CPU0-3.

