# Track B MAP Baseline Note

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE27-CONV-OR-OUTPUT-QUANTIZE-DECISION-AFTER-ACTIVATION-REPAIR-001

Do not run COCO/mAP in Stage27.

Track B should be launched as a separate first-class task:

```text
BANANA-YOLO26-VENDOR-ORT-RT204-MAP-BASELINE-001
```

Purpose:

```text
Measure YOLO26 vendor-ORT rt204 mAP on a fixed COCO validation protocol and compare carefully to YOLO11 production numbers without mixing model/runtime/precision claims.
```

Reason:

```text
Custom-engine selected-cut optimization does not answer whether YOLO26 itself beats YOLO11 production on board mAP/FPS.
```

This note is intentionally separate from the custom INT8 selected-cut engine lane.
