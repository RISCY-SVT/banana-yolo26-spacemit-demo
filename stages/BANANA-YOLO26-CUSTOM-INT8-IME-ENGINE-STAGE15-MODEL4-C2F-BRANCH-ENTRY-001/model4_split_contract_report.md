# Model4 Split Contract Report

selected_subset: `candidate_I_model4_split_first_branch`

`/model.4/Split` consumes `/model.4/cv1/act/Mul_output_0` in float domain.

Split attributes:

- axis: `1`
- output0: `/model.4/Split_output_0`, shape `[1,32,80,80]`
- output1: `/model.4/Split_output_1`, shape `[1,32,80,80]`

Stage 15 uses only `/model.4/Split_output_1` for the first branch Conv.

`/model.4/Split_output_1` Q/DQ:

- scale: `0.0226610563695`
- zero_point_u8: `12`
- signed_storage_zero_point_s8: `-116`

`/model.4/Split_output_0` is deferred for future `/model.4/Concat`.
