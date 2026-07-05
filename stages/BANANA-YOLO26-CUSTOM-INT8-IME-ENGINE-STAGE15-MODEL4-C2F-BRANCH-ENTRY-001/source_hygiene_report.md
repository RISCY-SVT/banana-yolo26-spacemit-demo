# Stage 15 Source Hygiene Report

## Checks

| check | status | log |
|---|---|---|
| `git diff --check` | pass | `run_logs/git-diff-check-stage15.log` |
| `git diff --cached --check || true` | pass | `run_logs/git-diff-cached-check-stage15.log` |
| symlink scan | pass | `run_logs/symlink-scan-stage15.log` |
| tracked secret-like scan excluding `commands.txt` | pass | `run_logs/secret-scan-tracked-no-commands.log` |

## Scope

Changed source is limited to:

- `custom_int8_engine/`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE15-MODEL4-C2F-BRANCH-ENTRY-001/`
- Stage 14 final-report traceability field.

No large generated ONNX model copies, tensor dumps, board binaries, COCO outputs, closed vendor binaries, credentials, SSH keys, `.env` files, or agent config files were staged.

`/data/ncnn` was not mutated.

`/data/banana-yolo11-spacemit-demo` was not mutated.

`/control` was not mutated.

XSlim was not used.
