# Add Contract Report

node_name: `/model.2/m.0/Add`
op_type: `Add`

## ONNX Inputs

| input | producer | shape | domain | scale | zero_point |
|---|---|---:|---|---:|---:|
| `/model.2/Split_output_1_DequantizeLinear_Output` | `/model.2/Split_output_1_DequantizeLinear` | `[1,16,160,160]` | float after Q/DQ | `0.18348428606987` | `2` |
| `/model.2/m.0/cv2/act/Mul_output_0` | `/model.2/m.0/cv2/act/Mul` | `[1,16,160,160]` | float after Conv Q/DQ + SiLU, no activation-output Q/DQ | n/a | n/a |

Output tensor: `/model.2/m.0/Add_output_0`

## Decision

`/model.2/m.0/Add` is a float-domain merge in the accepted Q/DQ ONNX graph.
There is no accepted exact integer-domain Add candidate at this boundary because
one input is dequantized Split output and the other is a float SiLU result with
no activation-output Q/DQ before Add.

accepted_candidate: `B/C measured float fallback Add as part of Add+Concat merge`
risk: hidden float path if not timed and reported
mitigation: Stage 12 exposes `add_us`, `concat_us`, and `post_concat_qdq_us`
as first-class timing buckets.
