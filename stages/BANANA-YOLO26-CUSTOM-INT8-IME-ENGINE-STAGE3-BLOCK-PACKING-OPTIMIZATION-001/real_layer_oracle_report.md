# Real Layer Oracle Report

## Method

1. Loaded the accepted manual Q/DQ ONNX model with stage-local `onnx`.
2. Ran ONNX shape inference.
3. Added selected intermediate quantized activation tensors and Conv float outputs as graph outputs.
4. Ran ONNX Runtime CPU with deterministic input seed `20260703`.
5. Computed integer Conv oracle:
   `sum((q_u8 - za) * (w_s8 - zw)) + bias_i32`.
6. Compared dequantized integer oracle with ORT Conv float outputs.
7. Generated small tracked C++ fixtures for board and host tests.

## Results

| case | node | max abs dequant diff vs ORT | status |
|---|---|---:|---|
| Conv1x1 | `/model.2/cv1/conv/Conv` | `9.5367431640625e-06` | pass |
| Conv3x3 | `/model.2/m.0/cv1/conv/Conv` | `2.384185791015625e-07` | pass |

## Artifacts

- `.deps/custom_int8_engine/stage3_oracle/stage3_selected_conv_outputs.onnx`
- `.deps/custom_int8_engine/stage3_oracle/*_input_s8.bin`
- `.deps/custom_int8_engine/stage3_oracle/*_weights_ohwi_s8.bin`
- `.deps/custom_int8_engine/stage3_oracle/*_bias_i32.bin`
- `.deps/custom_int8_engine/stage3_oracle/*_expected_i32.bin`
- `custom_int8_engine/tests/stage3_real_conv_fixture.h`
- `real_layer_oracle_data.json`

The `.deps` dumps are raw evidence only and are not committed.
