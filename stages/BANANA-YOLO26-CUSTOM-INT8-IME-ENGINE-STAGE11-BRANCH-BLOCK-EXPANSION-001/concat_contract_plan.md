# Concat Contract Plan

status: `deferred`

## Future Target

`/model.2/Concat` inputs:

- `/model.2/Split_output_0`
- `/model.2/Split_output_1_DequantizeLinear_Output`
- `/model.2/m.0/Add_output_0`

Output:

- `/model.2/Concat_output_0`
- Q/DQ after Concat:
  - scale: `/model.2/Concat_output_0_scale`
  - zero-point: `/model.2/Concat_output_0_zero_point`

## Stage 12 Requirements

- Determine whether Concat can be represented as views/slices plus one final quantized layout.
- Keep Split/Concat copy time separate.
- Generate ONNX CPU oracle for Add output and Concat output.
- Do not implement generic graph scheduler.
- Keep `/model.2/cv2/conv/Conv` out until Concat output contract is verified.
