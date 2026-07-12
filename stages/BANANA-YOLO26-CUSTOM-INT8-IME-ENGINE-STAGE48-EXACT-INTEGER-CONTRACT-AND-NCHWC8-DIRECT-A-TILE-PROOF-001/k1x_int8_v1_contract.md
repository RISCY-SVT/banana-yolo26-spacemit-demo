# K1X_INT8_V1 contract

## Identity

- Contract ID: `K1X_INT8_V1`
- General profile: `K1X_INT8_V1_GENERAL`
- Symmetric profile: `K1X_INT8_V1_SYMMETRIC`
- Asset byte order: little-endian
- Current direct-layout ID: `NCHWc8_SPATIAL_INNER_V1`

This contract is independent of ONNX Runtime execution order. The integer assets
in a versioned package are authoritative. A deployment must reject unknown IDs,
malformed sizes, unsupported shifts, unsafe accumulator bounds, or hash mismatch.

## General profile

- Logical activations are uint8 semantic codes with an explicit per-tensor scale
  identity and zero point in `[0,255]`.
- Physical activation storage is signed int8 code storage: `physical = code - 128`.
- Padding uses the physical encoding of the semantic input zero point.
- Weights are signed int8, zero point 0, with one scale identity per output channel.
- Bias is signed int32 in `input_scale * weight_scale[channel]` units.
- Conv accumulation is the exact integer sum of `(input_code - input_zero_point) * weight`
  plus bias. Raw vmadot signed-storage accumulation is corrected with
  `(128 - input_zero_point) * sum(weight)`.

## Symmetric profile

- Logical and physical activations are signed int8 with zero point 0.
- Weights are signed int8, zero point 0, with per-output-channel symmetric scales.
- The same accumulator, multiplier, rounding, saturation, and asset rules apply.
- This is the preferred future-student profile. The accepted current model5 QDQ
  metadata is not symmetric and is represented by the general profile.

## Accumulator and overflow

For every Conv package, preparation records and validates:

`abs(bias) + K * max(abs(0-zp), abs(255-zp)) * max(abs(weight))`

An int32 accumulation route is accepted only when the recorded bound is no larger
than `INT32_MAX`. Correction and epilogue arithmetic use int64. Unsupported bounds
are rejected; narrowing is never implicit.

## Requantization

Each output channel carries:

- non-negative signed-int64 `multiplier`;
- non-negative `right_shift` in `[0,126]`;
- output zero point;
- saturation interval, currently `[0,255]`.

The deployment computes `product = accumulator * multiplier` in signed 128-bit
precision, rounds `abs(product) / 2^right_shift` to nearest with ties to even,
then restores the sign. Positive and negative ties therefore use the same
unsigned-magnitude rule. It adds the output zero point and saturates to the
declared interval. Negative signed values are never left-shifted. Unsupported
shift or overflow surfaces are rejected or saturate only where the schema
explicitly allows output saturation.

The exporter derives the multiplier from exact IEEE-754 float32 bit-pattern
fractions and performs integer ties-to-even encoding. Board prepare does not call
`frexp`, `llround`, `exp`, or floating threshold search.

## Activation and graph operations

- Model5 SiLU is an authoritative 256-entry signed-storage LUT in the package.
  Its bytes, contract ID, source scale bits, and hash define the integer mapping.
- Split is view-only and preserves dtype, scale, zero point, layout, and bytes.
- Concat is byte concatenation only when every input has identical quantization
  metadata and a compatible physical axis/layout. Otherwise prepare rejects it
  unless an explicit requant operation is present.
- Add/residual is exact integer addition only under a declared shared output
  contract or explicit per-input integer multipliers/shifts. Saturation and RNE
  are identical to Conv requant. Undeclared mixed scales are rejected.

## Layout and assets

`NCHWc8_SPATIAL_INNER_V1` uses:

`offset(n, cb, y, x, ci) = ((((n * C_blocks + cb) * H + y) * W + x) * 8 + ci)`

where `cb = channel / 8` and `ci = channel % 8`. Channels must be divisible by 8.
Every package records the model, source asset, generated asset, fixture, and
generator hashes. Generated integer multipliers, shifts, packed weights, weight
sums, bias, LUT, metadata, and fixture outputs are immutable after hashing.

## Authority

1. Python arbitrary-precision integer oracle.
2. Portable C++ scalar contract implementation.
3. K1X scalar execution.
4. K1X IME execution.

Legacy host float-QDQ ORT is a model-replay diagnostic. Board ORT is an
integration/timing diagnostic. Neither overrides `K1X_INT8_V1` integer bytes.
