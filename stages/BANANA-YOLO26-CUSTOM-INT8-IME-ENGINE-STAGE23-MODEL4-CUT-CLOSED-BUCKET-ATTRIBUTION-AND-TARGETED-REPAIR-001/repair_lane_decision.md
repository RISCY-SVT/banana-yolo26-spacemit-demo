# Repair Lane Decision

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE23-MODEL4-CUT-CLOSED-BUCKET-ATTRIBUTION-AND-TARGETED-REPAIR-001`

## Gates Before Repair

```text
Gate A runner API vs ONNX cut: pass
Gate B runner frm robustness: pass
Gate C bucket attribution >=90%: pass
```

## Bucket Evidence

Scalar output quantization path:

```text
mean_total_us: 205098
mean_output_quantize_us: 73983.9
output_quantize_share_pct: 36.0726
mismatches: 0
```

The final `/model.4/cv2` int32-to-uint8 output QuantizeLinear boundary was the largest single bucket before repair.

## Selected Lane

```text
selected_repair_lane: D1_output_quantize_rvv
```

Reason:

```text
The output_quantize bucket was 36.07% of scalar-output selected-cut runtime.
The operation is local to the selected ONNX cut output boundary.
It has exact ONNX QuantizeLinear semantics and can use explicit-RNE RVV conversion.
It does not expand the graph and does not change model math.
```

## Rejected Lanes

```text
D2_branch1_activation_lut:
  Rejected for Stage23 because activation_requant_share was 15.91% before D1 and 23.71% after D1. It is material but not the measured primary blocker before repair.

D3_merge_rvv_or_lut:
  Rejected for Stage23 because merge_share was 21.13% before D1. After D1 it becomes the next likely target at 31.49%, so it is a Stage24 candidate.

D4_conv_or_threading:
  Rejected for Stage23 because Conv was not the largest pre-repair blocker. After D1, Conv is the largest bucket at 37.96%, but still below a clear >45-50% Conv-only gate.
```

## Acceptance Criteria

```text
runner_api_vs_onnx_cut: mismatches=0
rounding_regression: pass
output_quantize_bucket_speedup: >=3x
selected_cut_total_improves_vs_scalar_output_quantize_baseline: yes
selected_cut_total_improves_vs_stage22_total: yes
```
