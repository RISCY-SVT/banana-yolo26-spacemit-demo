# STAGE41 Summary RU

Классификация: `stage41-partial-correctness-only`.

Сделано:

```text
создан первый C++ in-process scaffold;
Python отсутствует в измеряемом runtime path;
per-block file I/O отсутствует в измеряемом runtime path;
custom /model.4 вызывается напрямую через C++ API;
ORT fallback подключён через in-process C API;
default backend не менялся.
```

Что доказано:

```text
host C++ scalar scaffold byte-exact против accepted Stage40 ORT oracle:
  output0 mismatches=0
  model4 boundary mismatches=0
```

Что сломано/заблокировано:

```text
board selected-mode full-output gate не прошёл;
board ORT CPU runtime выдаёт другой reference output, чем Stage40 host ORT oracle;
custom /model.4 vs board ORT model4 имеет mismatches=78351, max_abs_diff=2;
через suffix это даёт output0 mismatches=1508.
```

Профиль suffix:

```text
точный host C++ in-process profile показывает provisional next target: model.16;
model.16 delta ~= 16406.986 us;
model.16: 66 nodes, 9 Conv, C2f-like operator mix.
```

Следующий шаг:

```text
BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE42-INPROCESS-ORT-CONTRACT-REPAIR-AND-MODEL16-ORACLE-GATE-001
```

Не заявлялось:

```text
full YOLO26 FPS;
camera/full-image performance;
COCO/mAP;
production/default-backend readiness.
```
