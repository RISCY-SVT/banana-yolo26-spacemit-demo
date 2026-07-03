# Known Limitations

- No full YOLO26 inference engine is implemented.
- No graph scheduler is implemented.
- No ncnn source was changed.
- No `BACKEND=custom-int8` default was changed.
- No COCO, camera, full-image, model FPS, or production benchmark was run.
- Conv1x1/Conv3x3 kernels are synthetic/local fixtures, not integrated into a real graph block.
- Current Conv wrappers do on-the-fly packing/im2col and are slower than scalar in packing-included board benchmarks.
- Zero-point correction is synthetic-only; raw kernels assume signed symmetric int8 inputs.
- Requantization and activation functions are outside Stage 2.
- `vmadot1/2/3`, `vmadotn`, and FP/`vfmadot` are not implementation lanes in Stage 2.
- Public cached hotpath relies on the caller maintaining cluster0 affinity after the dispatch-boundary check; debug builds may enable extra per-call CPU checks.
