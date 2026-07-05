# Model4 Add Contract Report

node: `/model.4/m.0/Add`
inputs:
- `/model.4/Split_output_1_DequantizeLinear_Output`: float-domain Q/DQ output from Split output1
- `/model.4/m.0/cv2/act/Mul_output_0`: float-domain SiLU output without intermediate Q/DQ
output: `/model.4/m.0/Add_output_0`
classification: `float-domain Add`
accepted Stage 16 path: measured float fallback, not an integer-domain shortcut
