# STAGE32 Summary RU

classification: stage32-mixed-signedness-proof-ready-for-mmt4d-correction-stage
stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE32-MMT4D-OR-LOW-OVERHEAD-SLIDING-DATA-LAYOUT-DECISION-001
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
start_head: 00aa667b8770cd9e6c7a5cdd24ac2714bb1d52a9
end_head: pending-local-commit-see-final-response

## Proven

Stage31 replay подтвердил прежний вывод: direct/sliding `smt.vmadot1/2/3` kernel корректен, но медленнее текущего MMT4D path.

Текущий selected `/model.4` ONNX-cut path остался byte-exact:

```text
mismatches: 0
max_abs_diff: 0
output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
frm_sweep: pass
attribution_pct: 99.9412
```

Семейство integer dot signedness доказано на CPU0-3:

```text
smt.vmadot:   s8xs8
smt.vmadotu:  u8xu8
smt.vmadotsu: s8xu8
smt.vmadotus: u8xs8
```

## Broken

Low-overhead sliding layout gate не прошёл. Лучший attachable вариант:

```text
B3_interior_fast_path: 18447.2 us
required: <= 7800 us
```

Поэтому `vmadot1/2/3` direct/sliding lane не интегрируется.

## Unknown

Пока не доказано, даст ли `smt.vmadotus` или другая mixed signedness форма выигрыш в реальном MMT4D path. Это следующий узкий proof stage.

## What Changed

Добавлен Stage32 measurement tool:

```text
custom_int8_engine/tools/bench_stage32_layout_decision.cpp
```

Он измеряет layout-only candidates и proof-only signedness family. Default backend и production path не менялись.

## Next

Рекомендуемый следующий этап:

```text
BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE33-MMT4D-MIXED-SIGNEDNESS-CORRECTION-PROOF-001
```

Цель Stage33: проверить, может ли mixed signedness MMT4D уменьшить correction/conversion bucket без изменения ONNX-cut bytes.

## Non-Claims

Это не full YOLO26 inference.
Это не model FPS.
Это не full-image/camera performance.
Это не COCO/mAP.
Это не production/default-backend readiness.
