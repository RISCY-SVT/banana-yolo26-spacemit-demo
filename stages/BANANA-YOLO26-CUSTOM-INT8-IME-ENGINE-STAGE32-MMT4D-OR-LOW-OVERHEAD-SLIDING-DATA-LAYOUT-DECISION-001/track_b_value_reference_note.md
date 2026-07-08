# Track B Value Reference Note

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE32-MMT4D-OR-LOW-OVERHEAD-SLIDING-DATA-LAYOUT-DECISION-001

Track B is not run in Stage32.

Accepted Track B value context from prior packet:

```text
fp32_e2e_rt204 AP: 0.404730
fp32_e2e_rt204 AP50: 0.571221
fp32_e2e_rt204 AP75: 0.435028
fp16_keepio_rt204 AP: 0.404748
fp16_keepio_rt204 AP50: 0.571417
fp16_keepio_rt204 AP75: 0.435241
fp16_keepio_rt204 full COCO generation mean: 397.128 ms
fp16_keepio_rt204 full COCO generation FPS: 2.518
```

This value evidence is why a narrow custom IME proof lane remains reasonable. Stage32 still makes no full-model, model FPS, camera, COCO/mAP, or production claim for the custom INT8 engine.
