# Model4 Branch0 Oracle Report

model_path: `.deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx`
model_sha256: `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`
provider: `CPUExecutionProvider` for activation micro-oracles
selected_subset: `candidate_I_model4_split_first_branch`

## Compact Fixtures

| fixture | split1_count | branch0_i32_count | branch0_act_count | split1_checksum | branch0_i32_checksum | branch0_act_checksum |
|---|---:|---:|---:|---:|---:|---:|
| `synthetic_seeded` | `32` | `16` | `16` | `-3077` | `333333` | `-1059` |
| `synthetic_gradient` | `32` | `16` | `16` | `-2877` | `309823` | `-1163` |

## Conv Oracle

`/model.4/m.0/cv1/conv/Conv` compact oracle uses accepted ONNX quantized weights, per-channel weight scales, quantized bias, and the Stage 0-14 signed-storage correction formula.

The compact branch Conv shape is `1x1x32 -> 1x1x16`, kernel `3x3`, stride `1`, padding `1`.
