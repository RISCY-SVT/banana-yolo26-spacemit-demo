# Activation Boundary Report

## Boundary

```text
/model.2/m.0/cv1/conv/Conv_output_0
-> QuantizeLinear/DequantizeLinear
-> /model.2/m.0/cv1/act/Sigmoid
-> /model.2/m.0/cv1/act/Mul
-> QuantizeLinear/DequantizeLinear
-> /model.2/m.0/cv2/conv/Conv
```

## Implementation

- selected mode: `A2_rvv_f32_lut`
- fallback mode retained: `A0_int8_lut`
- fixed-point mode: sidecar only, not selected
- rounding: explicit RNE inherited from Stage 10
- LUT: per-boundary 256-code table
- correctness: compact fixture and board CPU0-3 mismatches `0`

## Timing

On CPU0 Stage 11 selected-subset microbench:

- branch cv1 activation: `4022.68 us`
- total activation: `40070.5 us`
- activation share: `14.8755%`

Activation remains below the 40% gate.
