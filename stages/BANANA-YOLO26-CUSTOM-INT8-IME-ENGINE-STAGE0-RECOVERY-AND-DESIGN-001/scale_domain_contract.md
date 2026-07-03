# Scale Domain Contract

## Domains

```text
input_scale: activation quantization scale, fp32 in v0
weight_scale: weight quantization scale, fp32 in v0
accumulator_scale = input_scale * weight_scale
output_scale: target output activation scale, fp32 in v0
```

For per-output-channel weights:

```text
accumulator_scale[oc] = input_scale * weight_scale[oc]
```

## Requantization

Stage 0 skeleton uses a clear scalar policy:

```text
effective_scale = accumulator_scale / output_scale
rounded = round_to_nearest_ties_away_from_zero(acc * effective_scale)
q = clamp(rounded + output_zero_point, qmin, qmax)
```

Future optimized kernels may use fixed-point multipliers, but must match the
chosen scalar oracle exactly or document a bounded tolerance before adoption.

## Clamp Range

- signed int8: `[-128, 127]`
- unsigned int8 activation fallback: `[0, 255]`

The accepted `smt.vmadot` primitive remains signed int8 only.

## Scale Blob Layout

Model format v0 stores scale descriptors separately from scale bytes:

```text
scale_descriptor {
  dtype: fp32 or fixed-point
  granularity: per-tensor / per-output-channel / per-group
  axis
  count
  alignment
  blob_offset
}
```

Stage 0 default storage dtype is `fp32`. Fixed-point is a future performance
choice, not a Stage 0 commitment.
