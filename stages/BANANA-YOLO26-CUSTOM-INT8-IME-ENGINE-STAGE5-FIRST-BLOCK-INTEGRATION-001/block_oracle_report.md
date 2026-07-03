# Block Oracle Report

Tool:

- `custom_int8_engine/tools/extract_stage5_block_oracle.py`
- Python venv: `.deps/custom_int8_engine/venv-stage5-onnx`
- packages: `onnx 1.22.0`, `onnxruntime 1.27.0`, `numpy 2.5.0`
- `xslim_used`: false

Model:

- path: `.deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx`
- SHA256: `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`

Oracle output root:

- `.deps/custom_int8_engine/stage5_oracle/2026-07-03_22-07-15/`

Generated small C++ fixture:

- `custom_int8_engine/tests/stage5_block0_fixture.h`

## Inputs

The extractor supports two deterministic inputs:

- `synthetic_seeded`: seed `20260703`, random float32 in `[0,1)`
- `synthetic_gradient`: deterministic H/W gradient float32 in `[0,1]`

No existing real preprocessed `.npy` input tensor was found under the searched project paths. Existing `.jpg` files were not decoded because Stage 5 only authorized `onnx`, `onnxruntime`, and `numpy` in the isolated venv.

## Oracle Checks

| case | ROI | input SHA256 | expected int32 SHA256 | max abs diff vs ORT Conv float ROI |
|---|---:|---|---|---:|
| `synthetic_seeded` | `8x8 input -> 4x4 output` | `def132eca896bf791e7b3bd7e04b243e701d75e89b11316ad8f22788213ab2b6` | `7982b5f4a3068fe26923130413074e211a708ac5e5b382c56117eb0a561447b4` | `7.62939453125e-06` |
| `synthetic_gradient` | `8x8 input -> 4x4 output` | `8166435867441d53724104b2aa87b257a80cd5e4590f873f46dbe7633c3abeaf` | `37adfdeb0782276e678ebbf8545a8b8cf2683b53d72e68037c74a08d41edcad8` | `1.9073486328125e-06` |

The full tensors are retained under `.deps` and are not committed.
