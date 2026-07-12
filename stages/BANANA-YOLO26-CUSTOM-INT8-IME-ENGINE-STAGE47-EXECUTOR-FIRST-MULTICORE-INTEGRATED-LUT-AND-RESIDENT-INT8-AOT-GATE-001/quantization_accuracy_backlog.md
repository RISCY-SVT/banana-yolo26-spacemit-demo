
# Quantization accuracy backlog

Accepted full COCO evidence remains: semantic PTQ loses 2.899 AP versus FP32;
host optimized INT8 loses another 3.884 AP. `reduce_range` addresses x86 U8S8
pair saturation and is not a semantic-PTQ remedy. Future bounded work should test
per-channel weights, calibration methods, activation-loss localization, selective
exclusion, a symmetric S8S8 contract, then QAT only if PTQ remains outside target.
