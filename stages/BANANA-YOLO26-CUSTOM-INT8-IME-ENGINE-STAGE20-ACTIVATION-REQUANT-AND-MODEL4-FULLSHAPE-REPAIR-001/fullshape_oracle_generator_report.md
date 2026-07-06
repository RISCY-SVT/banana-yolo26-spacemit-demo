# Fullshape Oracle Generator Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE20-ACTIVATION-REQUANT-AND-MODEL4-FULLSHAPE-REPAIR-001`
tool: `custom_int8_engine/tools/extract_fullshape_oracle.py`
status: `pass`

## Model

model: `.deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx`
model_sha256: `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`
provider: `CPUExecutionProvider`
input: `images`, shape `1x3x640x640`, dtype `float32`
input_mode: `synthetic_seeded`

## Output

large_tensor_dir: `.deps/custom_int8_engine/stage20_fullshape_oracles/model4_c2f_synthetic_seeded`
metadata_json: `fullshape_oracle_metadata.json`
boundary_manifest: `fullshape_boundary_manifest.tsv`
checksums: `fullshape_tensor_checksums.tsv`

## Boundaries

The generator extracted 25 full-shape `/model.4` C2f boundary tensors, including Conv float outputs, QuantizeLinear uint8 outputs, Split outputs, Add output, Concat output, post-Concat Q/DQ output, and `/model.4/cv2/conv/Conv` output.

Large `.npy` dumps are intentionally under `.deps/` and are not tracked.
