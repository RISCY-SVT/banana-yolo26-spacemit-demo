# Multiblock Subset Selection

selected_subset: `candidate_D_block0_silu_model1_silu_model2_cv1_conv`

## Boundary

1. `images_QuantizeLinear`
2. `images_DequantizeLinear`
3. `/model.0/conv/Conv`
4. `/model.0/conv/Conv_output_0_QuantizeLinear`
5. `/model.0/conv/Conv_output_0_DequantizeLinear`
6. `/model.0/act/Sigmoid`
7. `/model.0/act/Mul`
8. `/model.0/act/Mul_output_0_QuantizeLinear`
9. `/model.0/act/Mul_output_0_DequantizeLinear`
10. `/model.1/conv/Conv`
11. `/model.1/conv/Conv_output_0_QuantizeLinear`
12. `/model.1/conv/Conv_output_0_DequantizeLinear`
13. `/model.1/act/Sigmoid`
14. `/model.1/act/Mul`
15. `/model.1/act/Mul_output_0_QuantizeLinear`
16. `/model.1/act/Mul_output_0_DequantizeLinear`
17. `/model.2/cv1/conv/Conv`

Preferred output boundary: corrected int32 output of `/model.2/cv1/conv/Conv`.

## Why This Boundary Is Safe

Graph inspection in `run_logs/012_inspect_model2_cv1_graph.txt` shows a linear, quantized handoff from `/model.1/conv/Conv` through Q/DQ and SiLU (`Sigmoid` + `Mul`) to `/model.2/cv1/conv/Conv`. `/model.2/Split` appears after `/model.2/cv1/act/Mul`, so Stage 7 stops before branch/Split expansion.

## Deferred

- `/model.2/cv1/conv` output Q/DQ and activation.
- `/model.2/Split` and branch/Concat handling.
- Graph-wide scheduler and full engine integration.
- Dedicated activation/requant optimization.
