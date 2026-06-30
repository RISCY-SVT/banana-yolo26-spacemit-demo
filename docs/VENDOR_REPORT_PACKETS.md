# YOLO26 Vendor Report Packets

This repository keeps only lightweight pointers to vendor-ready report packets.
The packet directories and tarballs are raw evidence artifacts, not tracked git
content.

Latest packaging task:

```text
BANANA-YOLO26-VENDOR-REPORT-PACKETS-AND-RND-PUSH-001
```

Log root:

```text
/data/ncnn-logs/ort-logs/2026-06-30_20-54-29/
```

## Packet A: SpaceMIT EP rt20x Q/DQ Conv `clip minmax`

Audience:

```text
SpacemiT ONNX Runtime / SpaceMITExecutionProvider team
```

Packet directory:

```text
/data/ncnn-logs/ort-logs/2026-06-30_20-54-29/vendor_packets/spacemit_ep_rt20x_qdq_conv_clip_minmax/
```

Archive:

```text
/data/ncnn-logs/ort-logs/2026-06-30_20-54-29/vendor_packets/spacemit_ep_rt20x_qdq_conv_clip_minmax_vendor_packet.tar.gz
```

Archive SHA256:

```text
5eced1e706178d258adb0fe6c8b7225491008e3e1c6d74a7155aa9f85de62e90
```

Purpose:

- Documents the runtime/compiler blocker for CPU-good YOLO26 manual ONNX
  Runtime Q/DQ INT8 candidates.
- Includes tiny synthetic repro `15_conv_qdq_attr_kernel_shape.onnx`.
- Includes real YOLO26-derived repro `07_yolo26_first_conv_qdq_output_block.onnx`.
- Captures the SpaceMIT EP error:

```text
output_type not implemented for clip minmax
```

The known correctness-only workaround
`SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=QuantizeLinear;DequantizeLinear;Conv`
is CPU-heavy and is not accepted as accelerated INT8.

## Packet B: XSlim YOLO26 static PTQ blockers

Audience:

```text
SpacemiT XSlim / PTQ / PPQ team
```

Packet directory:

```text
/data/ncnn-logs/ort-logs/2026-06-30_20-54-29/vendor_packets/spacemit_xslim_yolo26_static_ptq_blockers/
```

Archive:

```text
/data/ncnn-logs/ort-logs/2026-06-30_20-54-29/vendor_packets/spacemit_xslim_yolo26_static_ptq_blockers_vendor_packet.tar.gz
```

Archive SHA256:

```text
ceaf3e29b5d26486a20e2fb08407b3fcbbaf79c6fa8d53c34c52182e38c5b24c
```

Purpose:

- Documents XSlim static PTQ e2e failure around two-input `ReduceMax`:

```text
ValueError: too many values to unpack (expected 1)
```

- Documents traditional static PTQ `minmax`/`percentile` output models whose
  CPU oracle is bad because class/object scores collapse to zero.
- Records why XSlim `--dynq` is diagnostic only and not proof of static
  activation INT8 acceleration.

## Current Decision

YOLO26 INT8 board acceleration remains blocked pending vendor/runtime/tooling
fixes. The current best local YOLO26 path remains native
body-FP16/head-FP32 keep-IO on rt204. Full-I/O FP16 is accepted but slightly
slower than keep-IO. Frozen YOLO11 production remains unchanged.
