# ONNX Cut Construction Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE22-MODEL4-ONNX-CUT-ORACLE-AND-NEXT-BLOCK-GATE-001`

## Tool

```text
tool: custom_int8_engine/tools/extract_model4_c2f_onnx_cut.py
python_env: .deps/custom_int8_engine/venv-stage20-onnx
onnx: 1.22.0
onnxruntime: 1.27.0
numpy: 2.5.1
provider: CPUExecutionProvider
```

## Command Summary

The tool loaded the accepted CPU-good QDQ ONNX model, ran the full model once with deterministic `synthetic_seeded` input, extracted the `/model.4` C2f cut with `onnx.utils.extract_model`, then ran the cut model with the exact saved cut input.

Cut construction:

```text
cut_input: /model.4/cv1/conv/Conv_output_0_QuantizeLinear_Output
cut_output: /model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output
cut_model: .deps/custom_int8_engine/stage22_onnx_cut/model4_c2f_cut.onnx
cut_model_sha256: bde82b0130615717ffcbdbaca8fa274e5de00c111cf0b0a518023b6a674d841a
```

## Result

```text
construction_status: pass
full_model_provider: CPUExecutionProvider
cut_provider: CPUExecutionProvider
cut_vs_full_model_mismatches: 0
cut_vs_full_model_max_abs_diff: 0
```

Large `.onnx`, `.npy`, and `.bin` artifacts are kept under `.deps/custom_int8_engine/stage22_onnx_cut/` and are not staged for git.
