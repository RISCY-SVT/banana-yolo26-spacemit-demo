# Stage47 - K1X student 416/512 architecture and training preparation

## Mission

Prepare, but do not start, a reproducible K1X-specific student-model training
lane after the RT205 vendor INT8 regression. Preserve both 416 latency-primary
and 512 accuracy-fallback candidates. Freeze teacher, COCO data, preprocessing,
operator set, quantization arithmetic, export contracts, distillation losses,
latency LUT inputs, and acceptance gates.

## Required gates

1. Reproduce the Stage46 FP32 and semantic-INT8 full-COCO surfaces.
2. Specify two static-shape students using K1X-measured operators and aligned
   channels, without assuming depthwise or attention efficiency.
3. Produce training/QAT manifests and dry-run export/oracle checks only.
4. Define separate accuracy and board latency gates before training approval.
5. Keep RT205/plugin routes rejected unless a new verified vendor package fixes
   the Q/DQ Conv, QOperator, and plugin ABI regressions.

No training, default dispatch, production claim, push, or vendor binary commit
is authorized without a separate direct-user packet.
