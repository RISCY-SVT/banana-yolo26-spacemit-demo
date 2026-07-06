# STAGE22_SUMMARY_RU

classification: `stage22-onnx-cut-pass-ready-for-next-repair`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE22-MODEL4-ONNX-CUT-ORACLE-AND-NEXT-BLOCK-GATE-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `d8025985bff6373aaf7082a47ad532a18bd64134`
end_head: `8350c57bd015f044a51800dcd318cb43976e534a`
pushed: `false`
full_engine_implemented: `false`
ncnn_source_mutated: `false`
production_claim_made: `false`

## Proven

- Построен ONNX Runtime CPU cut для `/model.4` C2f с тем же full-shape boundary input, который подается в C++ runner.
- ONNX cut совпал с full-model ORT output на этой boundary: `mismatches=0`.
- Host C++ scalar path и board `ime_threaded` path совпали с ONNX cut output: `mismatches=0`, `max_abs_diff=0`.
- Rounding regression для ambient `frm` RNE/RTZ/RDN/RUP/RMM прошел; `frm` после вызова сохраняется.
- Host CTest: `36/36`, RISC-V cross build: pass, board correctness: pass.

## Broken

- До исправления Stage22 ambient `RTZ` давал один mismatch. Добавлен scoped RNE guard для Stage22 same-input verifier path.

## Unknown

- Полная YOLO26 inference correctness неизвестна.
- COCO/mAP, camera, full-image и model FPS не проверялись.
- Stage22 timing не является full-model performance.

## Stable Timing

```text
mode: ime_threaded
warmup: 10
runs: 100
repeats: 5
mean_total_us: 225214
stddev_total_us: 44.6982
cv_total_pct: 0.019847
conv_share_pct: 23.1169
activation_share_pct: 14.381
merge_share_pct: 18.8474
mismatches: 0
```

## Next

`BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE23-MODEL4-CUT-CLOSED-BUCKET-ATTRIBUTION-AND-TARGETED-REPAIR-001`
