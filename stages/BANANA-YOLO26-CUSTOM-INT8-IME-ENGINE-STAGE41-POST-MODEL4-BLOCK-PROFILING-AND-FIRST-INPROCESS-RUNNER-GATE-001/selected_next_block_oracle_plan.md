# Selected Next Block Oracle Plan

provisional_block: model.16

Before implementing `model.16`, Stage42 must create a same-input ONNX cut oracle for:

```text
start: /model.15/Concat_output_0_DequantizeLinear_Output
end: /model.16/cv2/act/Mul_output_0_DequantizeLinear_Output
```

Stage42 should also inspect whether a quantized internal output boundary is safer:

```text
/model.16/cv2/conv/Conv_output_0_QuantizeLinear_Output
/model.16/cv2/act/Mul_output_0_QuantizeLinear_Output if present
```

The board in-process ORT CPU contract mismatch must be repaired or explicitly scoped before accepting a custom `model.16` expansion.
