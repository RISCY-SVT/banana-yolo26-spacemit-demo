# STAGE19 SUMMARY RU

classification: `stage19-model4-threaded-c2f-correct-but-compact-overhead-dominates`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE19-MODEL4-THREADED-C2F-INTEGRATION-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `6c4c8258fb10cf25476a8380870d624200855f9b`
end_head: `53ac15ad253ac70e594cc7e1ac6c117e92da85ca`
pushed: `false`
full_engine_implemented: `false`
ncnn_source_mutated: `false`
production_claim_made: `false`

## Кратко

Stage19 встроил explicit cluster0 threaded Conv sidecar в узкий `/model.4` C2f runner и добавил bounded activation/requant threading sidecar. Корректность сохранена, но на compact oracle-scope C2f fixture потоковый режим медленнее из-за накладных расходов. Это не отменяет Stage18 representative/full-shape результат для branch-entry Conv, но не позволяет выбирать threaded mode как default для compact C2f.

## Proven

```text
Stage18 representative/full-shape replay:
  A4 4-thread total: 11082.483550 us
  A4 4-thread conv: 5905.210462 us
  total speedup vs A0: 2.308881x
  conv speedup vs A0: 3.455020x
  mismatches: 0

Stage19 compact C2f correctness:
  thread counts: 1/2/3/4
  activation sidecar: tested
  mismatches: 0
  checksum stable: -143848
```

## Broken

```text
Stage19 compact C2f A4 4-thread:
  total speedup vs compact A0: 0.656284x
  branch0 conv speedup vs compact A0: 0.084759x

Stage19 compact A5 activation-threading:
  total speedup vs compact A0: 0.402947x
  CV: 37.251185%
```

## Unknown

```text
representative/full-shape timing for full model4 C2f threaded runner
full YOLO26 FPS
full-image/camera performance
COCO/mAP
production/default-backend readiness
```

## Decision

Следующий этап должен сначала доказать representative/full-shape timing для model4 C2f, затем ремонтировать activation/requant fusion или memory/dataflow, если они остаются bottleneck. `vmadot1/2/3` остается будущей отдельной proof lane, не Stage20 default.
