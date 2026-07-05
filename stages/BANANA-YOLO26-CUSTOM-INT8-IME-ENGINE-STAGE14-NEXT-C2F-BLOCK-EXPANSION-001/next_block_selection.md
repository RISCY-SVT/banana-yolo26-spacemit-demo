# Stage 14 Next Block Selection

selected_subset: `candidate_H3_model2_act_model3_act_model4_cv1_conv`

## Candidate Progression

| candidate | boundary | decision |
|---|---|---|
| `H1` | Stage13 `/model.2` output to `/model.3/conv/Conv` | tractable |
| `H2` | `H1` plus `/model.3/act` Q/DQ | tractable |
| `H3` | `H2` plus `/model.4/cv1/conv/Conv` | selected |

Stage 14 stops before `/model.4/Split`. The next split/branch point belongs to a later stage.

## Included Boundaries

1. Stage 13 selected subset through corrected int32 output of `/model.2/cv2/conv/Conv`.
2. `/model.2/cv2/act/Sigmoid` and `/model.2/cv2/act/Mul`.
3. `/model.2/cv2/act/Mul_output_0` Q/DQ handoff into `/model.3/conv/Conv`.
4. `/model.3/conv/Conv`.
5. `/model.3/act/Sigmoid` and `/model.3/act/Mul`.
6. `/model.3/act/Mul_output_0` Q/DQ handoff into `/model.4/cv1/conv/Conv`.
7. `/model.4/cv1/conv/Conv` corrected int32 output.

## Why This Boundary Is Safe

- No new Add/Concat branch merge is crossed after `/model.2`.
- Each new activation boundary has a boundary-specific 256-code ONNX Runtime LUT oracle with mismatches `0`.
- Both new Conv nodes have symmetric signed weights with zero-point `0`.
- The runner remains a narrow selected-subset runner, not a graph scheduler.
