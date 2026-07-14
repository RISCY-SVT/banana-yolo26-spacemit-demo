# Full COCO val2017 accuracy

The complete `K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001` executor was evaluated
on all 5,000 official COCO val2017 images. The run completed without an image
failure and produced 721,755 predictions. The final prediction JSON SHA-256 is
`cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda`.

The full COCO run used the exact E2c Q62 implementation. E2c2 was promoted
afterward as a byte-exact implementation of the same `K1X_INT8_V1` arithmetic:
its package-channel, adversarial, F0-F7, and real-image boundary gates match
E2c exactly. Consequently this is a contract-level accuracy result, not an
approximation attributed to a different numerical route. E2c2 changes the
implementation and timing only.

## Frozen surface

- Model SHA-256: `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`
- Package manifest SHA-256: `d3b4cb794f1373aa712d77bab177a5f7da58530361c9af58c0caf5bbcd6dc75f`
- Image-list SHA-256: `09b3bbeda289610ec9fd4e4b5e6da32ec04f98a9a4111e99790de863be0f8f9e`
- Annotation SHA-256: `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`
- OpenCV linear letterbox, pad 114, RGB, NCHW `/255`
- Confidence threshold 0.001, maximum 300 detections, no extra NMS

## Results

| Surface | mAP50-95 | mAP50 | mAP75 | AP small | AP medium | AP large | AR@100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Accepted semantic INT8 | 0.372453424642 | 0.526269698607 | 0.406055938173 | 0.181116238958 | 0.415342169837 | 0.547344785459 | 0.594392621073 |
| K1X_INT8_V1 | 0.370740894439 | 0.525846530087 | 0.403315604236 | 0.183972946262 | 0.414262735261 | 0.544043381180 | 0.593621370936 |

The K1X delta versus semantic INT8 is `-0.001712530203` absolute, or
`-0.171253` AP point. This passes the preferred Stage 52 gate of at most 0.002
absolute loss and the handoff hard gate of at most 0.005.

The evaluator's `mean_precision_grid` and `mean_recall_grid` fields are
`0.370740894439` and `0.593621370936`. They are COCO tensor aggregates, not a
single confidence-threshold precision/recall operating point.

## Paired bootstrap

The fixed-seed paired bootstrap used 2,000 resamples and seed 52001. Its metric
is per-image IoU-averaged F1 delta, not mAP delta. The mean delta was
`-0.000684060335`; the 95% interval was
`[-0.001806710052, 0.000423906450]`. The deterministic COCO mAP delta above is
the task-level acceptance authority.

The full per-class and bootstrap rows are in `full_coco_per_class.tsv` and
`full_coco_bootstrap.tsv`. The accepted FP32 reference remains
`0.401438855549` mAP50-95; this stage does not claim that INT8 removes the
previously measured PTQ loss.
