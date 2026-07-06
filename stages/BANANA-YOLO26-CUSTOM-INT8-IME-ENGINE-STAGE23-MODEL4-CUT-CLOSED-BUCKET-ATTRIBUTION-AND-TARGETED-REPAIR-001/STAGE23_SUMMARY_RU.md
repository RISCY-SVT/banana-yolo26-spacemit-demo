# STAGE23_SUMMARY_RU

classification: `stage23-runner-api-onnx-cut-pass-output-quant-repaired-ready-for-next-stage`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE23-MODEL4-CUT-CLOSED-BUCKET-ATTRIBUTION-AND-TARGETED-REPAIR-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `8350c57bd015f044a51800dcd318cb43976e534a`
end_head: `pending-local-commit-see-final-response`
pushed: `false`
full_engine_implemented: `false`
ncnn_source_mutated: `false`
production_claim_made: `false`

## Proven

Реальный runner API `y26_stage16_model4_c2f_run_cut_u8_output`, а не только bench-local composition, закрывает same-input ONNX cut для `/model.4`:

```text
input: /model.4/cv1/conv/Conv_output_0_QuantizeLinear_Output
output: /model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output
mismatches: 0
max_abs_diff: 0
SHA256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
```

Режимы ambient `frm` RNE/RTZ/RDN/RUP/RMM проходят через real runner API с `mismatches=0`, и `frm` восстанавливается после вызова.

## What changed

- Добавлен локальный cut API для real `/model.4` runner path.
- Добавлен `bench_stage23_model4_runner_cut`.
- Добавлен exact RVV path для финального `/model.4/cv2` `int32 -> uint8 NHWC` QuantizeLinear.
- Добавлен host CTest `test_stage23_runner_api_cut`.
- Stage22 traceability patched to final head `8350c57bd015f044a51800dcd318cb43976e534a`.

## Performance Numbers

Это только selected `/model.4` ONNX-cut timing, не full YOLO26 FPS.

```text
Stage22 mean_total_us: 225214
Stage23 scalar output quant mean_total_us: 205098
Stage23 RVV output quant mean_total_us: 137547
Stage23 RVV stddev_total_us: 81.7884
output_quantize scalar_us: 73983.9
output_quantize rvv_us: 6849.5
output_quantize_speedup: 10.8014x
```

Post-repair buckets:

```text
conv_share_pct: 37.9583
activation_share_pct: 23.7138
merge_share_pct: 31.494
output_quantize_share_pct: 4.97977
```

## Broken

Явного Stage23 correctness blocker не осталось.

## Unknown

```text
full_yolo26_correctness: unknown
full_image_camera_performance: unknown
coco_map: unknown
model_fps: unknown
production_readiness: unknown
default_backend_readiness: unknown
```

## Validation Status

```text
host_ctest: pass (37/37)
riscv_cross_build: pass
board_tests: pass
rounding_regression: pass
stable_benchmark: pass
git_diff_check: pass
symlink_scan: pass
secret_like_scan: pass
```

## Human Decision Needed

Следующий шаг рекомендуется как targeted Stage24, а не graph expansion:

```text
BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE24-BRANCH1-ACTIVATION-LUT-OR-MERGE-REPAIR-001
```

Причина: после ремонта output quantize крупные buckets стали `conv=37.96%`, `merge=31.49%`, `activation=23.71%`; Stage24 должен выбрать один локальный lane по повторному измерению.

## Non-Claims

Это не full YOLO26 inference, не model FPS, не full-image/camera performance, не COCO/mAP, не production readiness и не default backend readiness.
