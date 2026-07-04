# Block Microbench Report

classification: `pass-with-activation-dominant-caveat`

Benchmark is selected-subset only, not model FPS or production evidence.

## CPU0 Command

`taskset -c 0 ./bench_stage7_backbone_subset 1`

## Result

- subset: `candidate_D_block0_silu_model1_silu_model2_cv1_conv`
- shape: `640x640x3->320x320x16_silu->160x160x32_silu->160x160x32`
- scalar total: `1.22366e+06 us`
- IME total: `593347 us`
- speedup IME vs scalar: `2.0623x`
- checksum scalar/IME: `707794080` / `707794080`
- prepacked bytes: `6464`
- workspace bytes: `25396032`
- activation/requant total: `436780 us`, `73.6129%` of IME total

The expanded subset is correct and faster than scalar, but activation/requant fallback dominates the IME path.
