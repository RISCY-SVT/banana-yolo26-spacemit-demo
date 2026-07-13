# K1X INT8 Executor Accuracy

Accuracy uses official COCO val2017 (5,000 images) and
`instances_val2017.json`. The frozen path uses OpenCV linear resize,
letterbox padding value 114, RGB order, NCHW float32 normalization by 255,
confidence threshold 0.001, the graph's direct 300-row e2e output, and the
standard COCO80 category mapping. No extra NMS is applied.

Accepted comparison surfaces:

```text
FP32 ORT_DISABLE_ALL:          0.401438855549 mAP50-95
legacy semantic INT8:         0.372453424642 mAP50-95
K1X_INT8_V1 full executor:    see Stage52 full_coco_report.md
```

The Stage52 report records mAP50-95, mAP50, AP small/medium/large, per-class AP,
prediction count/hash, and a fixed-seed paired bootstrap with at least 2,000
resamples. Release handoff requires loss no larger than 0.005 absolute versus
the accepted semantic INT8 surface.

Intermediate exactness does not substitute for this task-level evaluation.
