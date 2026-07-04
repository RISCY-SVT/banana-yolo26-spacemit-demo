# Block Oracle Report

Selected subset: `candidate_C_block0_silu_model1_conv`

Oracle source:

- model: `.deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx`
- SHA256: `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`
- offline tool: `custom_int8_engine/tools/extract_stage6_multiblock_oracle.py`
- dump root: `.deps/custom_int8_engine/stage6_oracle/2026-07-04_06-17-01/`
- metadata: `stage6_oracle_metadata.json`
- `xslim_used`: false

ONNX tooling:

- Python environment: reused `.deps/custom_int8_engine/venv-stage5-onnx`
- `onnx`: `1.22.0`
- `onnxruntime`: `1.27.0`
- `numpy`: `2.5.0`

The tool adds selected intermediate tensors as graph outputs and runs ONNX Runtime CPU with graph optimizations disabled. Large dumps are kept under `.deps`; the repo only tracks the small deterministic fixture header.

## Cases

| case | Conv0 dequant max abs diff vs ORT ROI | Conv1 dequant max abs diff vs ORT ROI | status |
| --- | ---: | ---: | --- |
| `synthetic_seeded` | `3.814697265625e-06` | `1.1444091796875e-05` | pass |
| `synthetic_gradient` | `1.9073486328125e-06` | `3.814697265625e-06` | pass |

The exact C++ fixture compares corrected int32 outputs and int8 activation handoff tensors, not float model-level outputs.

