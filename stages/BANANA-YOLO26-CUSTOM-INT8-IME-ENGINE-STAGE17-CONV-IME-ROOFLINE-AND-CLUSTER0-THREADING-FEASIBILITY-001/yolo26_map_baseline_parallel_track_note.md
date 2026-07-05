# YOLO26 mAP Baseline Parallel Track Note

Stage17 did not run mAP.

Reasons:

```text
custom engine still has no full-model output
custom-engine mAP is premature
Stage17 is a Conv/IME roofline and cluster0 threading feasibility gate
```

YOLO26 vendor-ORT rt204 mAP baseline remains important for the overall project question, but it should be a separate parallel task and not mixed into Stage17.
