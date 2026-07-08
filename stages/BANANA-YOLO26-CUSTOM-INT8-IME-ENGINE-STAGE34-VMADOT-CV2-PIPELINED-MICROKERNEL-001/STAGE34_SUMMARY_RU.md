# STAGE34 SUMMARY RU

classification: `stage34-vmadot-throughput-ceiling-no-pipeline-win`

Stage34 проверил главный оставшийся Conv lever для выбранного `/model.4` ONNX-cut пути: software-pipelined/register-blocked `smt.vmadot` для `/model.4/cv2/conv/Conv`.

## Proven

```text
same-input ONNX-cut replay: pass
mismatches: 0
max_abs_diff: 0
output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
FRM sweep: pass
host CTest: pass (42/42)
RISC-V cross build: pass
```

Текущий accepted wrapper path для plain `smt.vmadot` остаётся рабочим. Старый `bench_vmadot_microkernel` показал `ime_direct_mean_ns_per_call=49.487`.

## Broken

Новые прямые inline/register-blocked shapes для Stage34 не стали безопасной основой для kernel-кандидата: все проверенные loop/pipeline cases завершились `SIGILL` (`rc=132`) на board CPU0.

## Unknown

Не доказан безопасный способ сделать software-pipelined cv2 microkernel поверх raw inline `smt.vmadot` без отдельного proof stage для exact loop shape. Full YOLO26 FPS, camera/full-image поведение и COCO/mAP не измерялись.

## What Changed

Добавлен только диагностический tool:

```text
custom_int8_engine/tools/bench_stage34_vmadot_throughput.cpp
```

И CMake target:

```text
bench_stage34_vmadot_throughput
```

Runner/default backend не изменены. `/data/ncnn` не изменялся.

## Timing

```text
selected-cut mean_total_us: 40178.5
stddev_total_us: 283.996
model4_cv2_conv_us: 12096.5
model4_cv2_compute_us: 8071.68
model4_cv2_correction_us: 1753.73
output_quantize_us: 7070.4
thread_overhead_us: 5243.32
```

Это selected `/model.4` cut timing, не model FPS.

## Next

Рекомендуемый Stage35:

```text
BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE35-OUTPUT-QUANTIZE-OR-THREAD-OVERHEAD-LOCAL-REPAIR-001
```

Сначала заново атрибутировать `output_quantize_us` и `thread_overhead_us`, затем выбрать ровно один локальный exact repair lane.
