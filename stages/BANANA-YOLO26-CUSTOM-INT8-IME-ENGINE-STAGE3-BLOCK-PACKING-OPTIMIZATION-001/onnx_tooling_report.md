# ONNX Tooling Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE3-BLOCK-PACKING-OPTIMIZATION-001`

## Route

- `xslim` was not used.
- System Python did not provide usable `onnx` / `onnxruntime`.
- A stage-local venv was created under `.deps/custom_int8_engine/venv-onnx-stage3`.
- Installed into that venv only: `onnx`, `numpy`, `onnxruntime` and their Python package dependencies.
- Runtime C++ library does not depend on Python, ONNX Runtime, protobuf, or ONNX.

## Versions

- Python: `3.12.3`
- `onnx`: `1.22.0`
- `numpy`: `2.5.0`
- `onnxruntime`: `1.27.0`

## Model

- Path: `.deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx`
- SHA256: `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`
- Opset: `ai.onnx` 18
- Input: `images` `[1,3,640,640]`, `float`
- Output: `output0` `[1,300,6]`, `float`

## Generated Oracle Data

- Intermediate-output model: `.deps/custom_int8_engine/stage3_oracle/stage3_selected_conv_outputs.onnx`
- Binary dumps: `.deps/custom_int8_engine/stage3_oracle/`
- Small tracked C++ fixture: `custom_int8_engine/tests/stage3_real_conv_fixture.h`
- Structured oracle metadata: `real_layer_oracle_data.json`

The `.deps` oracle files are intentionally not committed as model/runtime artifacts.
