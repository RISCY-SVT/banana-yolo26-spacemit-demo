# source_hygiene_report

stage_id: BANANA-YOLO26-VENDOR-ORT-RT204-MAP-BASELINE-001
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
start_head: b54c8767e691dc57cbd035a13d2d2d348d2f5366

## Git scope

Tracked/source changes are limited to:

- `CMakeLists.txt`
- `tools/yolo26_coco_predict.cpp`
- `stages/BANANA-YOLO26-VENDOR-ORT-RT204-MAP-BASELINE-001/*`

No `custom_int8_engine` files were modified.
No `/data/ncnn` mutation was performed by this task.
No `/data/banana-yolo11-spacemit-demo` mutation was performed by this task.

## Validation commands

- `git diff --check`: pass
- `find custom_int8_engine stages -type l -print`: pass, no symlinks printed
- secret-like scan over changed source and stage reports excluding `commands.txt`: pass, no findings
- `BANANA_DEMO_BUILD_VARIANTS=rt204 scripts/build_cross.sh`: pass

## Cross-track dirty notes

Read-only status checks found unrelated dirty state outside this task:

- `/data/ncnn`: existing modified riscv convolution source files
- `/data/banana-yolo11-spacemit-demo`: existing untracked `.claude/` and `AGENTS.md`

These trees were not modified, cleaned, staged, or used as source authority for this Track B task.

## Large/generated artifact policy

Large generated COCO prediction JSON, timing TSV, board logs, and COCOeval stdout are kept under:

- `/data/ncnn-logs/ai-team/2026-07-08_06-06-54/BANANA-YOLO26-VENDOR-ORT-RT204-MAP-BASELINE-001/`

They are not staged into git. The repo stage directory contains only small text reports and summary tables.

## Non-claims

This hygiene report does not claim production readiness, full custom-engine inference, camera readiness, or default backend readiness.
