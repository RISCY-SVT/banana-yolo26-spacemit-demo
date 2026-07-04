# Stage 8 LUT Integration Report

classification: pass
selected_mode: `int8_lut`

## API Changes

Added:

- `Y26ActivationMode`
- `Y26ActivationRequantParams`
- `Y26FixedRequantParams`
- `Y26ActivationSubbucketTimingUs`
- `activation_mode` in `Y26Stage7BackboneSubsetConfig`
- Act0/Act1 LUT storage and per-channel fixed-requant metadata in `Y26Stage7BackboneSubsetWorkspace`

Existing default behavior remains `scalar_float_reference` for older call sites.

## Correctness

| gate | status |
|---|---|
| host CTest | pass, `21/21` |
| RISC-V cross build | pass |
| board CPU0 Stage 8 correctness | pass |
| board CPU1 Stage 8 correctness | pass |
| board CPU2 Stage 8 correctness | pass |
| board CPU3 Stage 8 correctness | pass |

The Stage 8 test exercises the IME hotpath with `activation_mode=int8_lut` when built with IME enabled.

## Performance

| mode | total us | activation us | mismatches |
|---|---:|---:|---:|
| scalar_float_reference | 620735 | 465901 | 0 |
| fixed_requant_only | 516970 | 361666 | 0 |
| int8_lut | 350092 | 192568 | 0 |
| fused_lut_pack alias | 347546 | 192589 | 0 |

The `fused_lut_pack` mode currently aliases the LUT write-to-current-layout path. No next-Conv pack fusion was integrated in this stage.
