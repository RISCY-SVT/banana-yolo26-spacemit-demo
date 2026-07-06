# STAGE24_SUMMARY_RU

classification: `stage24-merge-dataflow-repaired-ready-for-conv-thread-tile-decision`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE24-BRANCH1-ACTIVATION-LUT-OR-MERGE-REPAIR-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `fce411e20eb649e7f7f0cfe65573848c0e8a1fd4`
end_head: `pending-local-commit-see-final-response`
pushed: `false`
full_engine_implemented: `false`
ncnn_source_mutated: `false`
production_claim_made: `false`

## Proven

Stage23 same-input ONNX cut replay снова прошел через real runner API:

```text
mismatches: 0
max_abs_diff: 0
SHA256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
frm sweep: pass
```

Stage24 выбрал Lane B и принял `B3_split1_concat_lut_scalar_add`:

```text
merge_us: 43334.4 -> 20953.9
merge_speedup: 2.06808x
total_us: 147624 -> 125229
```

## Broken

Первый вариант B3 не был exact в compact host test, потому что новый merge mode не вызывал `build_split0_concat_lut_activation`. Это исправлено до принятия кандидата; финальные host и board проверки проходят.

## Unknown

Полная YOLO26 inference, model FPS, full-image/camera behavior, COCO/mAP и production readiness остаются неизвестными и не заявлялись.

## Next

После Stage24 главным bucket стал Conv:

```text
conv_share_pct: 49.5835
activation_share_pct: 26.0505
merge_share_pct: 16.7325
output_quantize_share_pct: 5.54777
```

Рекомендуемый следующий этап:

```text
BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE25-CONV-THREADING-TILE-DECISION-001
```
