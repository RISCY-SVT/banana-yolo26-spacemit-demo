# Stage 16 Summary RU

classification: `stage16-model4-c2f-compact-correct-fullshape-gate-proven-conv-dominates`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE16-MODEL4-C2F-COMPLETION-AND-FULLSHAPE-GATE-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `14d0e74affce5abbb0667f9b759972b56ccb5b2b`
end_head: `pending-local-commit-see-final-response`
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false

## Итог

Stage16 закрыл главную проблему Stage15: compact-only доказательство не было принято как performance evidence. Теперь для текущего `/model.4` branch-entry есть representative/full-shape gate с реальными размерами `80x80x64 -> 80x80x32 -> 80x80x16`, `mismatches=0`, и board CPU0 timing.

После этого был добавлен compact proof для завершения `/model.4` C2f-style candidate через float-domain Add, float-domain Concat, post-Concat Q/DQ и `/model.4/cv2/conv/Conv`. Он проходит host и board correctness, но его timing остается compact evidence only.

## Доказано

- Stage15 replay проходит.
- Stage16A representative/full-shape branch-entry проходит с `mismatches=0`.
- Stage16 compact `/model.4` C2f completion проходит с `mismatches=0`.
- Host CTest: `33/33` pass.
- Cross build: pass.
- Board CPU0/1/2/3 correctness: pass.
- `A2_rvv_f32_lut` остается выбранным activation/requant путем.

## Сломано / ограничено

- Full-shape для полного `/model.4` C2f completion еще не доказан.
- Stage16A representative/full-shape path стал Conv-dominated: `conv_share_pct=79.8539`.
- Это не full YOLO26 inference, не model FPS, не camera/full-image result, не COCO/mAP.

## Следующий шаг

`BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE17-CONV-IME-ROOFLINE-AND-CLUSTER0-THREADING-FEASIBILITY-001` после review/approval.
