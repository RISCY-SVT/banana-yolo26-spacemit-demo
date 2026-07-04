# Stage 8 Final Report

classification: `stage8-activation-improved-but-still-dominates`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE8-ACTIVATION-REQUANT-OPTIMIZATION-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `889b6717f1df74459c715342167588ca8b9d9100`
end_head: `71e143271b2d09eb35511725e360c3c95bddfc09`
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false
xslim_used: false
vmadot_sliding_used: false

## Selected Subset

`candidate_D_block0_silu_model1_silu_model2_cv1_conv`

The graph was not expanded in Stage 8. The stage only optimized Act0 and Act1 activation/requant handoff for the Stage 7 selected subset.

## Implemented

- Added `custom_int8_engine/include/y26_k1x_activation.h`.
- Added `custom_int8_engine/kernels/activation_requant.cpp`.
- Added `activation_mode` to `Y26Stage7BackboneSubsetConfig`.
- Added `scalar_float_reference`, `fixed_requant_only`, `int8_lut`, and `fused_lut_pack` mode IDs.
- Added per-boundary 256-entry SiLU LUT generation for Act0 and Act1.
- Added diagnostic fixed-point requant helper and tests.
- Integrated `int8_lut` mode into the Stage 7 selected-subset runner without changing Conv IME kernels.
- Added `bench_stage8_activation_requant`.

## Proven

- Stage 7 baseline was reproduced on board.
- Host CTest passed: `21/21`.
- RISC-V cross build passed with `Y26_K1X_ENABLE_IME=ON`.
- Board CPU0/1/2/3 correctness passed for Stage 8 LUT mode.
- LUT exhaustive oracle passed for Act0 and Act1.
- Stage 8 selected mode `int8_lut` matched the Stage 7 scalar reference output with `mismatches=0`.
- No `/data/ncnn` mutation, no XSlim use, no full engine, no graph scheduler, no model FPS claim.

## Board CPU0 Timing

| mode | total us | activation us | activation share | mismatches |
|---|---:|---:|---:|---:|
| scalar_float_reference | 620735 | 465901 | 75.06% | 0 |
| fixed_requant_only | 516970 | 361666 | 69.96% | 0 |
| int8_lut | 350092 | 192568 | 55.0052% | 0 |
| fused_lut_pack alias | 347546 | 192589 | 55.41% | 0 |

The minimum Stage 8 timing gate was met:

- `activation_total_us <= 220000`
- `selected_subset_ime_total_us <= 400000`

## Broken / Residual

- Activation/requant remains the largest selected-subset bucket after optimization: `192568 us`, `55.0052%`.
- The remaining cost is mainly per-element Conv-output-code requantization plus LUT lookup/write.
- `fused_lut_pack` is currently an alias of LUT write-to-current-layout and does not pack directly for the next Conv.
- RVV LUT gather mechanisms were not implemented or measured.

## Unknown

- Whether fused `requant -> LUT -> packA` can reduce activation share below 40%.
- Full YOLO26 inference behavior.
- COCO/mAP impact.
- Full-image or camera performance.

## Next

Recommended next stage:

`BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE9-ACTIVATION-FUSION-AND-PACK-HANDOFF-001`

Do not expand the graph further until activation/requant share is below 40% or clearly no longer the dominant bottleneck.
