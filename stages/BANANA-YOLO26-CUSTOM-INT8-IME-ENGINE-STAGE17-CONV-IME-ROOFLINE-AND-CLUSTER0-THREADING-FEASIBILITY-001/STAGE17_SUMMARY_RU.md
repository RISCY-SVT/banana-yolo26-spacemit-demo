# Stage17 Summary RU

classification: `stage17-threading-positive-ready-for-threaded-conv-integration`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE17-CONV-IME-ROOFLINE-AND-CLUSTER0-THREADING-FEASIBILITY-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `3ca94995a7b18276dac9ac7c660061cd93377994`
end_head: `92e7d8763295cc514b09d0db32ed2795b092dc44`
pushed: false

## Итог

Stage17 исправил методику измерения для Stage16A representative/full-shape branch-entry и подтвердил корректность с `mismatches=0`.

Принятый протокол:

```text
warmup=10
runs=100
repeats=5
main thread pinned to CPU0
threading workers pinned to CPU0-3
```

Основные числа:

```text
IME selected subset mean_total_us: 25670.974780
IME total stddev_us: 345.670192
IME conv_us: 20458.001284
activation_requant_us: 5035.606050
4-thread spatial split mean_us: 5583.807628
4-thread speedup: 3.680290x
threading_feasibility: strong_positive
```

Это не full YOLO26 FPS, не full-image/camera performance, не COCO/mAP и не production/default-backend claim.

next_recommended_step: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE18-CLUSTER0-THREADED-CONV-INTEGRATION-001`
