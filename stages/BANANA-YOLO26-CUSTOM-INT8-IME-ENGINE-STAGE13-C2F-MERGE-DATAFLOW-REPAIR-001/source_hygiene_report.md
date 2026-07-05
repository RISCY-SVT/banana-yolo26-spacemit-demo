# Source Hygiene Report

## Checks

| check | status |
|---|---|
| `git diff --check` | pass |
| `git diff --cached --check || true` | pass |
| `find custom_int8_engine stages -type l -print` | pass, no symlinks printed |
| secret-like scan | pass with command-log self-match caveat |

## Secret Scan Caveat

The secret-like scan matched historical `commands.txt` lines that contain the
scan regex itself. No private keys, tokens, `.env`, credentials, or secret
values were found in changed source, scripts, reports, or docs.

## Mutation Scope

- `/data/ncnn`: not mutated.
- `/data/banana-yolo11-spacemit-demo`: not mutated.
- `/control`: not mutated.
- XSlim: not used.
- `vmadot1/2/3`, `vmadotn`, FP/vfmadot: not used or implemented.
- Full YOLO26 engine, scheduler, camera, COCO/mAP, production claim: not added.

## Large Artifacts

No ONNX model copies, tensor dumps, board binaries, closed vendor binaries, COCO
outputs, or large generated artifacts were added to git.
