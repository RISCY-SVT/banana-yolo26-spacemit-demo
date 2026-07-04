# Stage 10 Pre-registered Hypotheses

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE10-BACKBONE-EXPANSION-POST-ACTIVATION-GATE-001`

## H1

Stage 9 `A2_rvv_f32_lut` should remain below 20% activation share on the existing Stage 9 subset.

## H2

On newly introduced boundaries, `A2` must not be assumed valid until the boundary-specific LUT, scale contract, and oracle are generated and verified.

## H3

The next likely bottleneck after Stage 9 is not activation/requant on the old subset, but one of:

- conv/IME
- Split/branch handling
- pack/layout
- memory/workspace
- new boundary scales

## H4

`/model.2/Split` and first `/model.2` branch integration are likely to expose tensor-contract complexity; correctness gates win over speed.

## H5

`A5` packA sidecar must remain sidecar unless a measured pack/layout bottleneck appears in the expanded subset.
