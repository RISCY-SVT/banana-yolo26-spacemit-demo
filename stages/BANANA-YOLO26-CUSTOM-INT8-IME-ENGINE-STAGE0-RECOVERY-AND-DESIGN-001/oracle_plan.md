# Oracle Plan

## Per-Layer ONNX CPU Dump

Use Python `onnx`/`onnxruntime` only in converter/oracle tools. Dump selected
activation tensors from the accepted CPU-good Q/DQ model and compare against
custom-engine layer outputs.

## Quantization Metadata Oracle

Generate a stable metadata table with:

```text
tensor name, dtype, scale, zero-point, axis, per-tensor/per-channel
```

`signedness_zero_point_audit.tsv` is the Stage 0 seed.

## Scalar INT8 Oracle

Stage 0 creates tiny scalar fixtures for:

- 4x4x8 signed `s8 x s8 -> s32`
- A row-major packing
- B transposed K-contiguous packing
- requant rounding and clamp

## Full-Image Smoke

Later stages should use the existing small oracle suite:

- canonical production photo
- Ultralytics `bus.jpg`
- Ultralytics `zidane.jpg`
- Day 4 real camera still
- blank white sanity image

Full COCO/mAP is a later stage only.
