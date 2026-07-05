# Stage 15 Scope Decision

selected_subset: `candidate_I_model4_split_first_branch`

## Included

Stage 15 starts from the Stage 14 output boundary:

`/model.4/cv1/conv/Conv` corrected int32 output.

The Stage 15 selected subset includes:

1. `/model.4/cv1/conv/Conv_output_0` Q/DQ semantics.
2. `/model.4/cv1/act/Sigmoid`.
3. `/model.4/cv1/act/Mul`.
4. `/model.4/Split`.
5. `/model.4/Split_output_1` Q/DQ to signed int8 storage.
6. `/model.4/m.0/cv1/conv/Conv`.
7. `/model.4/m.0/cv1/conv/Conv_output_0` Q/DQ semantics.
8. `/model.4/m.0/cv1/act/Sigmoid`.
9. `/model.4/m.0/cv1/act/Mul`.
10. `/model.4/m.0/cv1/act/Mul_output_0` Q/DQ to signed int8 storage.

## Deferred

The following are explicitly deferred:

- `/model.4/m.0/cv2/conv/Conv`
- `/model.4/m.0/Add`
- `/model.4/Concat`
- `/model.4/cv2/conv/Conv`
- `/model.5/conv/Conv`
- any graph-wide scheduler
- full YOLO26 inference

## Reason

The ONNX graph shows `/model.4/cv1/act/Mul_output_0` is split in float domain. Only `Split_output_1` receives Q/DQ before `/model.4/m.0/cv1/conv/Conv`.

Stage 15 therefore models the exact branch-entry handoff and does not invent an integer-domain Split shortcut for future branch merge tensors.

full_shape_stage15_timing: `not_proven` until a representative/full-shape benchmark is implemented and validated.
