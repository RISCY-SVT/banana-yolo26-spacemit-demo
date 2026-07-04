# Stage 6 Baseline Replay Report

Recovered baseline from repo-local Stage 6 reports before Stage 7 implementation:

- Stage 6 commit: `246d6011865d5cd246e8a701c501c14f1193a060`
- selected subset: `candidate_C_block0_silu_model1_conv`
- board scalar total: `1009980 us`
- board IME total: `419769 us`
- speedup: `2.41x`
- Conv0 IME component: `67775.3 us`
- activation/requant fallback: `286942 us`
- Conv1 IME component: `63886.5 us`
- Stage 5 Conv0 replay IME: `70203.2 us`

Stage 7 will also run `bench_stage6_multiblock` on the board as a replay smoke and record raw stdout under the Stage 7 log root when board access is available.
