# Bbox/confidence separation audit

The canonical FP32 graph contains all six vendor-named truncation tensors.
They map to three bbox heads (`cv2`, four channels) and three confidence heads
(`cv3`, 80 channels) at 80x80, 40x40, and 20x20.

XSlim's split output was reconstructed into exactly two executable graphs:

```text
signed-INT8 QDQ inference graph
  images -> six separate float32 boundary tensors

floating-point post-processing graph
  six boundary tensors -> output0 (1x300x6)
```

No concatenation combines bbox and confidence before their quantization
boundaries. The generated post-processing graph contains zero
`QuantizeLinear`, `DequantizeLinear`, or QLinear nodes. All 12 retained tail
initializers match the canonical source byte-for-byte.

The FP32 source split reproduces the unsplit FP32 graph exactly for all 100
holdout images. Recombining either accepted official 2.1.1 quantized
inference output through the retained source tail has zero difference from
running the same boundary values through the candidate tail.
