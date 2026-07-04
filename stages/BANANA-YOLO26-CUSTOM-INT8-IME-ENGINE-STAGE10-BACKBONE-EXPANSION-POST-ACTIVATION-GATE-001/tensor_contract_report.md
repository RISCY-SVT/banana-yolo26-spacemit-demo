# Tensor Contract Report

selected_subset: `candidate_E_branch1_stage9_split_model2_m0_cv1_conv`

| tensor | shape | dtype/domain | producer | consumer | scale | zero_point | signedness/storage | layout | lifetime |
|---|---|---|---|---|---:|---:|---|---|---|
| `images` | `[1,3,640,640]` source / NHWC in runner | float input -> signed test storage | input Q/DQ | `/model.0/conv/Conv` | 0.00392156885937 | 0 | int8 storage `q-128` | NHWC in runner | input |
| Act0 handoff | `[320,320,16]` | int8 storage | Conv0 Q/DQ+SiLU LUT | `/model.1/conv/Conv` | 0.311162889004 | 1 | int8 storage `q-128` | NHWC compact | workspace |
| Act1 handoff | `[160,160,32]` | int8 storage | Conv1 Q/DQ+SiLU LUT | `/model.2/cv1/conv/Conv` | 0.582500994205 | 0 | int8 storage `q-128` | NHWC compact | workspace |
| Conv2 corrected output | `[160,160,32]` | int32 accumulator | `/model.2/cv1/conv/Conv` | Conv2 activation LUT | accumulator scale per output channel | n/a | int32 | NHWC | workspace |
| Conv2 activation full | `[160,160,32]` | int8 storage | Conv2 Q/DQ+SiLU LUT | `/model.2/Split` | split target: 0.18348428607 | 2 | int8 storage `q-128` | NHWC compact full | workspace |
| `/model.2/Split_output_1` | `[160,160,16]` | int8 storage | Split channel slice 16..31 | `/model.2/m.0/cv1/conv/Conv` | 0.18348428607 | 2 | int8 storage `q-128` | NHWC compact copied | workspace |
| Branch0 corrected output | `[160,160,8]` | int32 accumulator | `/model.2/m.0/cv1/conv/Conv` | Stage 10 output boundary | branch output scale 0.0381805039942 | output zp 176 | int32 corrected | NHWC | output |

## Workspace

Stage 10 reuses Stage 7 workspaces and adds persistent prepacked weights/workspace for `/model.2/m.0/cv1/conv/Conv`. No heap allocation is performed inside the measured hot loop after `prepare`.
