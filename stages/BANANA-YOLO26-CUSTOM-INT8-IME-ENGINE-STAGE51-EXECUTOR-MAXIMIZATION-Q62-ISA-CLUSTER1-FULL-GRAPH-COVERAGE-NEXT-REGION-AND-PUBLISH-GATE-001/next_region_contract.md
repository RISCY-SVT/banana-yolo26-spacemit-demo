# Next resident region contract

Selected region F begins at `/model.8/cv2/act/Mul_output_0_QuantizeLinear_Output` and ends at
`/model.9/Add_output_0_QuantizeLinear_Output`. It executes model9 cv1 Conv+SiLU, three exact 5x5
stride-1 MaxPools, four-way producer-direct Concat, cv2 Conv preactivation, and exact residual
Add/activation. It uses K1X_INT8_V1 and resident NCHWc8 throughout, with no float fallback.
