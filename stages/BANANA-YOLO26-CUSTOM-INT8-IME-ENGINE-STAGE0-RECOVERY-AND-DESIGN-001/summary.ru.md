# Сводка Stage 0

classification: stage0-recovery-design-skeleton-complete-ready-for-microkernel-stage-with-caveats
stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE0-RECOVERY-AND-DESIGN-001
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
base_branch: yolo26-rd-bootstrap
start_head: 9c307f8a2d2fed5f39375ebacb0dbc92b59a0510
end_head: 9c307f8a2d2fed5f39375ebacb0dbc92b59a0510
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false
output_contract_decision: traditional
cpu_good_qdq_artifact: /data/banana-yolo26-spacemit-demo/.deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx
quantization_audit_status: pass
cluster0_safety_plan: created
static_model_format_v0: created
skeleton_created: true
stage1_prompt_created: true
big_artifacts_path: /data/banana-yolo26-spacemit-demo/.deps/custom_int8_engine/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE0-RECOVERY-AND-DESIGN-001
log_dir: /data/ncnn-logs/ai-team/2026-07-03_09-07-17/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE0-RECOVERY-AND-DESIGN-001
next_recommended_step: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE1-SMTVMADOT-MICROKERNEL-001 after review/approval
timestamp: 2026-07-03_09-07-17

## Что сделано

Stage 0 завершён как recovery/design/skeleton stage. Восстановлена принятая
основа `smt.vmadot`, найден CPU-good manual ORT Q/DQ artifact, создана
инвентаризация графа, выполнен audit signedness/zero-points, выбран первый
контракт реализации `traditional/trunk-first [1,84,8400]`, описан static model
format v0, создан standalone C++ skeleton и подготовлен prompt для Stage 1.

`xslim` не использовался как источник моделей или решений, потому что для
YOLO26 в нём указан bug.

## Broken / Proven / Unknown

Proven:

- `smt.vmadot` принят только для cluster0 CPU0-3 как `s8 x s8 -> s32` 4x4x8.
- CPU-good Q/DQ artifact найден: `manual_e2e_rep_conv_matmul_qdq.onnx`.
- Native host build через `/usr/bin/g++` проходит, 5/5 CTest pass.

Broken:

- rt204 EP путь для YOLO26 Q/DQ всё ещё заблокирован ошибкой `output_type not implemented for clip minmax`.
- Первый validation run унаследовал `CC/CXX=/opt/riscv/...`; этот bad `build-host` удалён после того, как host shell случайно интерпретировал RISC-V ELF и создал мусорные имена в test dir.
- E2E head содержит `TopK`/`ReduceMax`/`GatherElements`, поэтому не выбран первым engine contract.

Unknown:

- Лучший способ graph-level correction для asymmetric activation `MatMul`.
- Полная скорость, COCO/mAP, production readiness и ncnn integration.

## Валидация

- bad cross-compiled `.deps/custom_int8_engine/build-host` и temporary `build-host-native` удалены.
- `build-host` пересобран native через `/usr/bin/g++` with `LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8`: pass.
- native CTest в `.deps/custom_int8_engine/build-host`: pass, 5/5.
- post-fix suspicious path scan: pass, issue_count 0.
- `commands.txt` sanitized: UTF-8 valid, control byte count 0.
- `git diff --check`: pass.
- `scan-export-candidates.sh` on task-run artifacts: pass.

Рекомендуемый следующий шаг: Stage 1 только после review/approval, без full
engine, без ncnn source mutation и без production FPS claims.
