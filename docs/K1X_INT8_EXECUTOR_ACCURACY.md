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
K1X_INT8_V1 full executor:    0.370740894439 mAP50-95
delta versus semantic INT8:  -0.001712530203 absolute
```

The K1X result passes the preferred Stage52 gate of at most 0.002 absolute loss
versus semantic INT8. The final prediction JSON contains 721,755 rows and has
SHA-256
`cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda`.

The fixed-seed paired bootstrap uses 2,000 resamples of a per-image
IoU-averaged F1 proxy. It is not represented as a bootstrap confidence interval
for COCO mAP. Complete metrics and per-class AP are in the Stage52
`full_coco_report.md`.

Intermediate exactness does not substitute for this task-level evaluation.
