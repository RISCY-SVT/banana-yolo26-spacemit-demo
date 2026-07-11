# K1X model/executor co-design specification

## Model envelope

- Primary: static 416x416 student; fallback: 512x512.
- Teacher: YOLO26n or YOLO26s; distillation plus structured pruning and exact INT8 QAT.
- Operators: tile-aligned plain 3x3/1x1 Conv, Add/views, measured pooling, minimal materialized Concat/Split.
- Avoid unmeasured depthwise assumptions and attention/Softmax in the backbone.
- Structurally reparameterize training branches at export.
- Symmetric int8 activation/weight storage where QAT permits, activation zero point 0, per-channel weight scales.
- Simple static NMS-free one-to-one head; evaluate two- and three-scale variants explicitly.

## Executor contract

- Static AOT schedule and one preallocated arena; quantized tensors remain resident.
- One physical tile-compatible layout with conversions only where measured to repay cost.
- Prepare-time immutable packed weights; no hot-loop allocation, graph-name lookup, or file I/O.
- Fuse exact integer correction, bias, requant, and activation where semantics allow.
- CPU0-3 only for IME; CPU4-7 may be evaluated later for non-IME work only.
- Per-block fixed-host integer oracles remain correctness gates.

## Training and acceptance

- Freeze COCO preprocessing/postprocessing and full model hashes.
- Gates: 500-image direction first, then full val2017; target within 1 AP, relaxed within 2 AP.
- Benchmark exported graph with measured operator LUT before full training commitment.
- No accuracy or <=45 ms claim until a trained/exported model passes both gates.
