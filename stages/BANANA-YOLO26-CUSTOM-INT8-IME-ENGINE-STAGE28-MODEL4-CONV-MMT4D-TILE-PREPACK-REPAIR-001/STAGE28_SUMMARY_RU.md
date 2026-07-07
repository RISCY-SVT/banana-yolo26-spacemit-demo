# Stage28 Summary RU

classification: `stage28-conv-component-split-complete-candidate-selected-and-accepted`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE28-MODEL4-CONV-MMT4D-TILE-PREPACK-REPAIR-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `502f7abe06aaba413310731971176ede603f527f`
end_head: `see-final-head-copy-after-local-commit`
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false

## Что сделано

Stage28 сначала добавил обязательную декомпозицию Conv bucket для текущего `/model.4` same-input ONNX-cut пути. Replay Stage27/Stage26 selected path прошел byte-exact:

```text
mismatches: 0
max_abs_diff: 0
output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
frm_sweep: pass
affinity_ok: 1
```

После декомпозиции выбран ровно один локальный кандидат:

```text
selected_candidate: T2_fused_correction_writeback
```

Он убирает промежуточный corrected-buffer copy: worker теперь применяет ту же correction formula сразу в финальные output rows.

## Числа

```text
total_us_before: 41580.9
total_us_after: 40231.6
total_speedup: 1.03354x
conv_us_before: 26753.7
conv_us_after: 25255.4
conv_copy_us_before: 1251.9
conv_copy_us_after: 0
```

После T2:

```text
conv_share_pct: 62.775
activation_share_pct: 7.41481
merge_share_pct: 5.42096
output_quantize_share_pct: 17.5779
```

## Proven

```text
ONNX-cut correctness: pass
FRM sweep RNE/RTZ/RDN/RUP/RMM: pass
host CTest: pass, 39/39
RISC-V cross build: pass
board stable benchmark: pass
CPU0-3 affinity: pass
```

## Broken

Корректность не сломана. Но T2 не решает главный raw MMT4D compute bucket: Conv остается крупнейшим bucket.

## Unknown

```text
full YOLO26 inference performance
model FPS
full-image/camera performance
COCO/mAP
production/default backend readiness
выгодность vmadot1/2/3 без Track B YOLO26 mAP/value gate
```

## Next

Рекомендуемый следующий шаг:

```text
BANANA-YOLO26-VENDOR-ORT-RT204-MAP-BASELINE-001
```

Причина: Stage28 доказал structural low utilization в selected-cut Conv после локального repair, но крупная инвестиция в `vmadot1/2/3` должна быть gated both by Stage28 structural evidence and Track B YOLO26 mAP/value result.

Non-claims: это не full YOLO26 inference, не model FPS, не camera/full-image performance, не COCO/mAP и не production/default-backend readiness.
