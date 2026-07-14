# Final COCO val2017 accuracy

The selected Stage53 E2c2 structural route completed all 5,000 COCO val2017 images and emitted 721755 predictions.

FP32 reference mAP50-95 is 0.401438855549. Legacy semantic INT8 mAP50-95 is 0.3724534246416940. Stage53 K1X_INT8_V1 mAP50-95 is 0.3707408944391919, a delta of -0.0017125302025021 versus semantic INT8.

Prediction JSON SHA-256 is `cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda`. It is byte-identical to Stage52, so the Stage53 delta versus the accepted Stage52 K1X result is exactly zero.

The paired image-level diagnostic used 2000 resamples with seed 52001. This bootstrap is an IoU-averaged F1 diagnostic, not a confidence interval for global COCO mAP.

Accuracy classification: preferred. Per-class AP and every bootstrap sample are preserved in adjacent TSV files.
