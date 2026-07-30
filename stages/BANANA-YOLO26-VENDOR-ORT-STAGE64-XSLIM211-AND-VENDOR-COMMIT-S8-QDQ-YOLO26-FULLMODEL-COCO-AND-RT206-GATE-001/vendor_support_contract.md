# Vendor support contract

## Source

This document records the support policy from the captured SpacemiT ORT issue
1 comment, the captured XSlim issue 45 reply, and the direct vendor statement
preserved in the Stage64 authorization. Raw public issue/comment JSON is
preserved in the Stage64 evidence root. This is a tested vendor contract, not
an independent claim that every conforming graph is supported.

## Quantization representation

| Surface | Vendor-stated contract |
|---|---|
| activations | asymmetric signed INT8, per tensor, represented by Q/DQ |
| weights | symmetric signed INT8, per tensor or per channel, represented by Q/DQ |
| zero points | UINT8 zero points unsupported |
| operators | Q/DQ is the unified 2.x representation |
| legacy quantized ops | `QLinearConv` and `QLinearMatMul` unsupported |
| Conv | explicit valid `kernel_shape` required |

The exported model is the authority for actual dtype, zero point, and scale
granularity. Stage64 does not infer those properties solely from source code
or vendor prose.

## YOLO26 graph boundary

The vendor-prescribed path truncates the inference graph at six named
one-to-one detection-head tensors. The three bbox and three confidence/class
branches remain separate through quantization. The decoder, TopK, and final
`1x300x6` post-processing tail remain floating point and execute on CPU by
design.

An intentionally CPU-only tail is not counted as unexpected provider fallback.
The quantized inference graph must still show meaningful Conv/MatMul placement
on the SpacemiT EP.

## Stage64 controls

The test matrix independently covers:

- signed activation zero point `0` and nonzero signed zero points;
- per-channel and per-tensor signed symmetric Conv weights;
- signed Q/DQ MatMul;
- omitted Conv `kernel_shape`;
- UINT8 Q/DQ negative controls;
- historical QLinear negative controls.

No result from the historical U8/QLinear model is reused as evidence that a
new S8-QDQ graph works.
