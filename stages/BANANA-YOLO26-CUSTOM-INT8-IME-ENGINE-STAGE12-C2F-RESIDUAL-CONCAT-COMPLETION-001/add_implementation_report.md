# Add Implementation Report

accepted_candidate: `B/C measured float fallback`

## Implementation

Stage 12 materializes:

1. `split1_f32` from `/model.2/Split_output_1_DequantizeLinear_Output`
2. `branch1_act_f32` from `/model.2/m.0/cv2/conv/Conv` corrected int32 through Conv-output Q/DQ and SiLU
3. `add_f32 = split1_f32 + branch1_act_f32`

No integer-domain shortcut is used.

## Correctness

Compact fixtures:

- `concat_mismatches=0`
- `model2_cv2_mismatches=0`

Board CPU0/1/2/3 Stage 12 fixture tests passed.

## Timing

CPU0 full-shape Stage 12 IME A2:

- `add_us=2504.42`
- `concat_us=4335.56`
- `post_concat_qdq_us=83007.7`
- `add_concat_share_pct=15.4979`
