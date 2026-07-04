# Backbone Expansion Selection

selected_subset: `candidate_E_branch1_stage9_split_model2_m0_cv1_conv`

## Start Boundary

Stage 9 selected subset:

- `/model.0/conv/Conv`
- `/model.0/act/Sigmoid` + `/model.0/act/Mul`
- `/model.1/conv/Conv`
- `/model.1/act/Sigmoid` + `/model.1/act/Mul`
- `/model.2/cv1/conv/Conv`

## Stage 10 Expansion

Added:

- `/model.2/cv1/conv/Conv_output_0` Q/DQ + SiLU activation
- `/model.2/Split`
- `/model.2/Split_output_1` Q/DQ handoff
- `/model.2/m.0/cv1/conv/Conv`

Output boundary:

- corrected int32 output of `/model.2/m.0/cv1/conv/Conv`

## Rejected/Deferred

- `/model.2/m.0/cv2/conv/Conv`: deferred because it requires another activation boundary and residual Add.
- `/model.2/Concat`: deferred because it introduces multi-input branch join and larger layout policy.
- `/model.2/cv2/conv/Conv`: deferred because it depends on Concat output contract.

## Decision

The selected subset crosses the first branch point safely while avoiding generic graph scheduling. Split output 1 is represented as an explicit channel-slice copy into compact NHWC storage for the first branch Conv.
