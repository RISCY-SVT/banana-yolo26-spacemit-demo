# Stage 6 Multi-Block Subset Selection

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE6-MULTI-BLOCK-BACKBONE-SUBSET-001`

## Authority

- Repo start HEAD: `fcbcd0fa72e3649d85ec2281bf8dce8dc92e78da`
- Previous stage: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE5-FIRST-BLOCK-INTEGRATION-001`
- CPU-good Q/DQ artifact: `.deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx`
- Model SHA256: `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`
- `xslim_used`: false

## Candidate Review

| candidate | boundary | decision | reason |
| --- | --- | --- | --- |
| `candidate_A_block0_conv_only` | `/model.0/conv/Conv` only | baseline replay | Already proved in Stage 5. |
| `candidate_B_block0_conv_silu_requant` | `/model.0/conv/Conv` + Q/DQ + `Sigmoid` + `Mul` + Q/DQ | included | Immediate downstream activation/requant metadata is clear and linear. |
| `candidate_C_block0_silu_model1_conv` | candidate B + `/model.1/conv/Conv` | selected | Adds one additional real backbone Conv with clear input/output quant metadata and no branch before Conv1. |

## Selected Subset

Selected subset id: `candidate_C_block0_silu_model1_conv`

Node sequence:

1. `images_QuantizeLinear`
2. `images_DequantizeLinear`
3. `/model.0/conv/Conv`
4. `/model.0/conv/Conv_output_0_QuantizeLinear`
5. `/model.0/conv/Conv_output_0_DequantizeLinear`
6. `/model.0/act/Sigmoid`
7. `/model.0/act/Mul`
8. `/model.0/act/Mul_output_0_QuantizeLinear`
9. `/model.0/act/Mul_output_0_DequantizeLinear`
10. `/model.1/conv/Conv`

Output boundary: corrected int32 accumulator output of `/model.1/conv/Conv`.

## Deferred

- `/model.1/conv/Conv_output_0` Q/DQ and `/model.1/act/Sigmoid` + `/model.1/act/Mul` are deferred to the next stage.
- `/model.2/cv1/conv/Conv` is not included because adding Conv1 activation plus the next Conv would exceed the Stage 6 bounded scope.
- Branching at `/model.2/Split` is explicitly deferred.
- No graph-wide scheduler, full YOLO26 inference, COCO/mAP, camera path, XSlim path, `vmadot1/2/3`, `vmadotn`, or FP/vfmadot implementation is included.

