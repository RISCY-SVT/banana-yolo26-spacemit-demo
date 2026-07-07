# Track B mAP Baseline Note

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE25-CONV-THREADING-TILE-DECISION-001

Stage25 is selected-cut custom engine work only. It does not answer full-model model-quality questions.

Recommended separate task:

```text
BANANA-YOLO26-VENDOR-ORT-RT204-MAP-BASELINE-001
```

Scope:

```text
- measure YOLO26 vendor-ORT rt204 mAP on fixed COCO val subset or full val2017;
- keep separate from custom-engine Stage25;
- no production claim;
- no ncnn mutation;
- no custom-engine graph expansion dependency.
```

Stage25 did not run COCO/mAP.
