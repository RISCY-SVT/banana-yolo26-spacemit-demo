# Stage 52 prompt: measured K1X model-executor co-design preparation

```yaml
task_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE52-MODEL-EXECUTOR-CODESIGN-PREPARATION-AND-ACCURACY-TARGET-GATE-001
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
expected_start_head: use-the-final-stage51-remote-head
training_authorized: false
student_selection_authorized: false
push_authorized: false
```

Read the complete Stage51 packet. Preserve K1X_INT8_V1, NCHWc8, exact Q62 E2c, and the selected
resident model4-final to model9 evidence. Freeze both student hypotheses: 416 latency-oriented
and 512 accuracy-oriented. Define a measured operator/layout/head contract, explicit full-COCO
accuracy targets, distillation/QAT data requirements, and latency envelopes using Stage51 LUT-v2.
Do not train, select a resolution, claim 20 FPS, or implement a production full graph. End with a
single decision on whether architecture/training preparation is sufficiently specified for a
separately authorized training stage.
