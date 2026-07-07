# Track B mAP Gate Note

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE28-MODEL4-CONV-MMT4D-TILE-PREPACK-REPAIR-001`

Stage28 is selected `/model.4` ONNX-cut custom-engine evidence only. It does not answer whether YOLO26 itself has enough model-quality value on the board to justify a deeper direct-conv or `vmadot1/2/3` proof investment.

Recommended separate stage:

```text
BANANA-YOLO26-VENDOR-ORT-RT204-MAP-BASELINE-001
```

Purpose:

```text
Measure YOLO26 vendor-ORT rt204 mAP on a fixed COCO validation subset or full val2017, under a separate protocol.
Compare carefully to YOLO11 production numbers without mixing model/runtime/precision claims.
Do not mutate /data/ncnn.
Do not make production claims.
```

Any major `vmadot1/2/3` investment should be gated by both Stage28 structural custom-engine evidence and this Track B YOLO26 mAP/value result.
