# YOLO26 mAP Baseline Parallel Track Note

Stage18 did not run mAP.

Reasons:

```text
custom engine still has no full-model output
Stage18 is a selected-mode threaded Conv sidecar gate
custom-engine mAP remains premature
```

YOLO26 vendor-ORT rt204 mAP baseline remains important for the wider project, but it should be a separate parallel task and not mixed into Stage18.
