# Source Hygiene Report

Scope:

- `custom_int8_engine/`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE5-FIRST-BLOCK-INTEGRATION-001/`

Checks:

- `git diff --check`: pass
- symlink scan under `custom_int8_engine stages`: no symlinks found
- filtered secret-like scan excluding `commands.txt`: no findings
- changed file size scan: no large model, tensor dump, board binary, ONNX copy, or closed vendor binary staged

Notes:

- Full Stage 5 oracle dumps are under `.deps/custom_int8_engine/stage5_oracle/2026-07-03_22-07-15/` and are not committed.
- `commands.txt` contains the literal secret-scan pattern as command text, so it is excluded from filtered content scanning and kept as command evidence.
- No `/data/ncnn` mutation was performed.
- No `/data/banana-yolo11-spacemit-demo` mutation was performed.
- No XSlim path was used.
