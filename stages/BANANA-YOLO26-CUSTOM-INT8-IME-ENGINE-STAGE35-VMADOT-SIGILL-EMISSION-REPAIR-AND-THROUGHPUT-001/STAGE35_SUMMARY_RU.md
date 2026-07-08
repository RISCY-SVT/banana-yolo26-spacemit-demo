# STAGE35 SUMMARY RU

classification: `stage35-vmadot-emission-repaired-throughput-measured-ready-for-pipelined-cv2`

Stage35 исправил первопричину Stage34 SIGILL в диагностике `smt.vmadot`: trap был вызван `rdcycle` в benchmark timing path, а не самим `smt.vmadot`.

После замены измерения на `std::chrono::steady_clock`:

```text
case0 existing helper: pass
case1 named inline: pass
case2 raw same as helper: pass
standalone named/raw: pass
2/4/6 accumulator raw groups: pass
mismatches: 0
CPU0 first: pass
CPU1/2/3 smoke: pass
CPU4-7 IME: not used
```

Throughput diagnostic показал, что независимые accumulator groups board-executable и дают microbench throughput improvement:

```text
mandatory runs=100:
  A1 single raw: 4.918 ns/vmadot
  A4 4 accumulators: 1.250 ns/vmadot
  A5 6 accumulators: 0.861 ns/vmadot

supplemental high-iteration:
  A1 single raw: 3.77661 ns/vmadot
  A4 4 accumulators: 0.937923 ns/vmadot
  A5 6 accumulators: 0.625296 ns/vmadot
```

`/model.4/cv2/conv/Conv` pipelined candidate не реализован в Stage35: этот stage был repair/proof gate, а не broad kernel implementation. Следующий шаг должен быть отдельным bounded Stage36 для cv2 candidate.

Non-claims:

```text
no full YOLO26 inference
no model FPS
no full-image/camera performance
no COCO/mAP
no production/default-backend readiness
no vmadotn support
no vfmadot support
```
