# Финальный отчет YOLO26 R&D

Корень доказательных логов для closeout gate:

```text
/data/ncnn-logs/ort-logs/2026-06-30_18-30-54/
```

Этот отчет подводит итог изолированной YOLO26 R&D линии. Он не меняет
замороженный production-релиз YOLO11.

## Краткое резюме

YOLO26n технически работает на Banana-Pi BPI-F3 / SpacemiT K1X со SpacemiT
ONNX Runtime `2.0.4` в FP32 и в native
body-FP16/head-FP32 keep-I/O варианте. Лучший локальный путь YOLO26:

```text
YOLO26n 640 e2e, native body-FP16/head-FP32 keep-IO, rt204 SpaceMIT EP
```

Full-I/O FP16 кандидат также принят по корректности и ускорению, но он не
быстрее keep-I/O FP16. YOLO26 INT8 остается заблокированным до исправлений
vendor/runtime/tooling. Замороженный YOLO11 production остается deployable
релизом.

## Замороженный YOLO11 baseline

Production-репозиторий:

```text
/data/banana-yolo11-spacemit-demo
production-2026-07-02 -> 9c0933be58ee122389d1a43f45f81e80655d6904
```

Замороженная политика:

| Путь | Runtime | Статус |
| --- | --- | --- |
| Primary image visual | dynamic640 INT8 на `rt201` | production |
| Normal camera | dynamic640 INT8 на `rt201` | production |
| Fast-live camera | vendor320 INT8 на `rt123`, 320 letterbox | production |
| Vendor320 trusted visual | `rt123` | production |
| Vendor320 low-latency perf | raw `rt201` | только perf |
| Vendor320 rt201 workaround | SHA256-guarded | non-default |
| FP16 | keep_io 640 на `rt201`/`rt202b1` | experimental |
| YOLO26n | P2 only в YOLO11 repo | не production |
| Stable rt202 | протестирован, не принят | не production |

## Почему YOLO26 был изолирован

YOLO11 уже был принят, tagged и mirrored. YOLO26 требовал нового поведения
Ultralytics export, SpacemiT ORT `2.0.4`, FP16 и INT8 runtime forensics,
XSlim PTQ расследования и compatibility gates. Поэтому все эксперименты
выполнялись в:

```text
/data/banana-yolo26-spacemit-demo
```

YOLO11 production-репозиторий использовался только для read-only сравнения.

## Находки по rt204 runtime и API

SpacemiT ORT `2.0.4` запускается на K1X/X60 без SIGILL в default mode. Он
экспонирует новые диагностические/runtime поверхности, включая
`SPACEMIT_EP_DISABLE_PASSES_FILTER`, а также operator strings `YoloDecode`,
`GridSample`, `RotaryEmbedding` и `ArgMax`.

Строка `SPACEMIT_EP_PERFER_CORE_ARCH` существует, но полезное документированное
значение K1/K3/X60 override найдено не было. Для этой R&D линии принят default
mode rt204.

## YOLO26 export/API mismatch и исправление

Старый Ultralytics `8.3.233` экспортировал `yolo26n.pt` как traditional
`[1,84,N]`, отклонял документированный аргумент `end2end` и воспроизводил
крупный ложный `refrigerator` результат.

Текущий Ultralytics исправил oracle:

| Export | Output contract | Статус |
| --- | --- | --- |
| default/e2e | `[1,300,6]` xyxy/conf/class_id | принят |
| `end2end=False` | `[1,84,8400]` traditional output | принят с decoder/NMS |

PyTorch, ONNX Runtime CPU и rt204 SpaceMIT EP семантически согласованы для
исправленного FP32 oracle.

## FP32 baseline

YOLO26 FP32 e2e 640 является замороженным R&D baseline. Он корректен, но медлен
на K1X.

| Metric class | Runtime | Mean ms | FPS |
| --- | --- | ---: | ---: |
| `perf_test forward` | rt204 SpaceMIT EP | 562.934799 | 1.776405 |
| `app forward-only` | rt204 SpaceMIT EP | 573.281930 | 1.744342 |
| `app full image benchmark` | rt204 SpaceMIT EP | 523.119274 | 1.911610 |

Это R&D baseline метрики, не production claims YOLO11.

## FP16 результаты

Текущий closeout gate проверил FP32, body/head keep-I/O FP16, full-I/O FP16,
direct full-model FP16 и XSlim FP16.

| Candidate | Input dtype | Output dtype | rt204 status | Решение |
| --- | --- | --- | --- | --- |
| FP32 e2e baseline | float32 | float32 | pass | baseline |
| Native body-FP16/head-FP32 keep-I/O | float32 | float16 | pass | лучший локальный путь |
| Native body-FP16/head-FP32 full-I/O | float16 | float16 | pass | принят, но не быстрее |
| Native full-model FP16 | float16/float32 variants | mixed invalid head | load fail | rejected |
| XSlim FP16 | float32 | mixed invalid head | load fail | rejected |

Benchmark rows из closeout gate:

| Variant | Metric class | Mean ms | FPS |
| --- | --- | ---: | ---: |
| FP32 e2e | `perf_test forward` | 562.934799 | 1.776405 |
| FP32 e2e | `app forward-only` | 573.281930 | 1.744342 |
| FP32 e2e | `app full image benchmark` | 523.119274 | 1.911610 |
| FP16 keep-I/O | `perf_test forward` | 379.777774 | 2.633119 |
| FP16 keep-I/O | `app forward-only` | 383.229967 | 2.609399 |
| FP16 keep-I/O | `app full image benchmark` | 398.091562 | 2.511985 |
| FP16 full-I/O | `perf_test forward` | 380.472087 | 2.628314 |
| FP16 full-I/O | `app forward-only` | 388.609808 | 2.573275 |
| FP16 full-I/O | `app full image benchmark` | 399.345008 | 2.504100 |

Решение: full-I/O FP16 принят по корректности и rt204 execution, но keep-I/O
FP16 остается лучшим локальным YOLO26 путем, потому что он немного быстрее и не
требует FP16 input в приложении.

## INT8 попытки и блокеры

YOLO26 INT8 acceleration не принят.

| Путь | Результат |
| --- | --- |
| Ultralytics `quantize=8` | Q/DQ ONNX создан, CPU oracle схлопнулся в ноль detections. |
| Manual ORT Q/DQ | CPU-good, но rt204/legacy runtimes блокируют Q/DQ Conv offload. |
| QOperator | Не принят; нет доказательства полезного QLinear offload, слабые parity/timing. |
| XSlim dynamic | CPU-good diagnostic only, не static activation INT8. |
| XSlim static e2e | Нужен upstream fix для two-input `ReduceMax`. |
| XSlim static traditional | В некоторых configs создает ONNX, но CPU scores схлопываются в ноль. |
| Legacy runtimes | Нет принятого accelerated Q/DQ path. |

Actionable rt204 blocker:

```text
output_type not implemented for clip minmax
```

Минимальные repro:

| Repro | Назначение |
| --- | --- |
| `15_conv_qdq_attr_kernel_shape.onnx` | Tiny Q/DQ Conv with explicit `kernel_shape=[3,3]`. |
| `07_yolo26_first_conv_qdq_output_block.onnx` | Real YOLO26 first-Conv extracted repro. |

## Vendor и upstream packets

Vendor/upstream отчеты доступны в R&D docs и raw log roots:

| Packet | Статус |
| --- | --- |
| rt204 Q/DQ Conv `clip minmax` bug report | готов к отправке vendor |
| XSlim ReduceMax static PTQ report | готов к отправке upstream |
| XSlim traditional zero-score report | готов к отправке upstream |

В этой задаче issues автоматически не открывались.

## YOLO11 rt204 и XSlim retrospective

R&D-only копии YOLO11 artifacts были протестированы под rt204. Замороженный
YOLO11 репозиторий не изменялся.

| Variant | Результат |
| --- | --- |
| YOLO11 dynamic640 INT8 on rt204 | pass, но медленнее production rt201 |
| YOLO11 FP16 keep-I/O on rt204 | pass как R&D signal |
| YOLO11 XSlim FP32/FP16 on rt204 | fail или timeout после `YoloDecode` dispatch errors |

Пропущенная YOLO11 production opportunity не доказана.

## Финально рекомендованный YOLO26 путь

Рекомендованный локальный YOLO26 R&D artifact:

```text
YOLO26n 640 e2e native body-FP16/head-FP32 keep-I/O on rt204
```

Он корректен на public sanity suite и private canonical reference, и это самый
быстрый локально принятый YOLO26 precision path. Это все еще R&D path, не
production replacement.

## Почему YOLO26 не заменяет YOLO11 production

YOLO26 FP16 улучшает YOLO26 FP32, но остается медленнее замороженных YOLO11 INT8
production paths на K1X. YOLO26 INT8 acceleration заблокирован. YOLO11
production остается принятым deployable release.

## Open P2 и future work

- Отправить rt204 Q/DQ Conv repro runtime vendor.
- Отправить XSlim ReduceMax и traditional zero-score reports upstream.
- Возвращаться к INT8 только после vendor/runtime/tooling изменений.
- Optional future YOLO26 FP16 polish, если app-level FP16 I/O станет полезным.
- YOLO11 rt204 reevaluation держать отдельным adoption gate.

## Repositories, commits и runtime evidence

| Item | Value |
| --- | --- |
| Frozen YOLO11 production commit/tag | `9c0933be58ee122389d1a43f45f81e80655d6904`, `production-2026-07-02` |
| YOLO26 R&D branch | `yolo26-rd-bootstrap` |
| YOLO26 closeout start HEAD | `02ae1a0c760598e6fd7e396944a4395e9db941b9` |
| Runtime | `spacemit-ort.riscv64.2.0.4` (`rt204`) |
| Closeout log root | `/data/ncnn-logs/ort-logs/2026-06-30_18-30-54/` |

Точный финальный R&D commit, содержащий этот отчет, записан в closeout run
summary и в git history.

