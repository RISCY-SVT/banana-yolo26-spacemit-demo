# Model Artifact Manifest

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE40-FULL-MODEL-RUNNER-SKELETON-GATE-001

## Accepted ONNX/QDQ Authority

```text
path: .deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx
sha256: 30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c
provider: ONNX Runtime CPUExecutionProvider
input: images, float32, 1x3x640x640
output: output0, float32, 1x300x6
```

This stage did not regenerate, rewrite, simplify, quantize, or export a new model.

## Checkpoint Provenance

The Stage40 bounded repo scan used the accepted ONNX/QDQ artifact as authority. A repository-local `yolo26n.pt` checkpoint was not required for this stage and was not used as source authority.

## Track B Context Only

Track B confirmed YOLO26 model value using vendor ORT rt204:

| variant | AP | AP50 | AP75 | full COCO generation |
|---|---:|---:|---:|---:|
| fp32_e2e_rt204 | 0.404730 | 0.571221 | 0.435028 | 526.130 ms |
| fp16_keepio_rt204 | 0.404748 | 0.571417 | 0.435241 | 397.128 ms |

Those are vendor-runtime model-value facts, not custom INT8 engine performance claims.
