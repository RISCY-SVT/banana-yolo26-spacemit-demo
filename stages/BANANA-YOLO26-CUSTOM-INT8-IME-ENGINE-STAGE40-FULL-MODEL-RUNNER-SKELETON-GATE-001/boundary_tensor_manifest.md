# Boundary Tensor Manifest

Primary machine-readable manifest:

```text
stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE40-FULL-MODEL-RUNNER-SKELETON-GATE-001/boundary_tensor_manifest.tsv
```

Key tensors:

| tensor | shape | dtype | SHA/source |
|---|---:|---|---|
| `images` | 1x3x640x640 | float32 | deterministic `synthetic_seeded` |
| `output0` | 1x300x6 | float32 | full ONNX Runtime CPU reference |
| `/model.4/cv1/conv/Conv_output_0_QuantizeLinear_Output` | 1x64x80x80 | uint8 | prefix/full reference match |
| `/model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output` | 1x128x80x80 | uint8 | all-ORT/custom model4 match |

NHWC C++ runner fixtures:

```text
model4 input bin SHA256:  94dcc2e954f7b9b112bc3f11cfe4a147dfe91c52b37f94058e2dd09b4e08b1b8
model4 output bin SHA256: 517db620fca8465888ec387673f888d5e7c43c86d613c88cbf4bb5ffcbe4cd91
```
