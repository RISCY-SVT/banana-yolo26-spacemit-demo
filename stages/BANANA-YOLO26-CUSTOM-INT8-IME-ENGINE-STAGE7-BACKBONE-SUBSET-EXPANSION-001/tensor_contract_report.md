# Tensor Contract Report

Selected subset: `candidate_D_block0_silu_model1_silu_model2_cv1_conv`

| Tensor | Shape | Dtype/domain | Producer | Consumer | Scale/ZP | Layout | Lifetime |
|---|---|---|---|---|---|---|---|
| `images` | `[1,3,640,640]` model, cropped fixture `8x8x3` | float/Q/DQ | model input | `images_QuantizeLinear` | `0.00392156885937/0` | NCHW model, NHWC signed fixture | input |
| Conv0 input signed | fixture `8x8x3` | `int8` storage | Stage 7 fixture | Conv0 runner | storage ZP `-128` | NHWC | hot |
| `/model.0/conv/Conv` corrected | fixture `4x4x16` | `int32` | Conv0 | Act0 fallback | output Q scale/ZP `0.620968401432/128` | NHWC | workspace |
| Act0 signed handoff | fixture `4x4x16` | `int8` storage | Act0 fallback | Conv1 runner | act scale/ZP `0.311162889004/1`, storage ZP `-127` | NHWC | workspace |
| `/model.1/conv/Conv` corrected | fixture `2x2x32` | `int32` | Conv1 | Act1 fallback | output Q scale/ZP `1.1142437458/122` | NHWC | workspace |
| Act1 signed handoff | fixture `2x2x32` | `int8` storage | Act1 fallback | Conv2 runner | act scale/ZP `0.582500994205/0`, storage ZP `-128` | NHWC | workspace |
| `/model.2/cv1/conv/Conv` corrected | fixture `2x2x32` | `int32` | Conv2 | Stage 7 output | output Q scale/ZP `0.836495816708/155` | NHWC | output |

Workspace needed: persistent prepacked weights for Conv0/Conv1/Conv2, three reusable Conv workspaces, corrected int32 buffers for Conv0/Conv1, signed activation handoff buffers for Conv1/Conv2, and raw int32 buffers for all Conv nodes.
