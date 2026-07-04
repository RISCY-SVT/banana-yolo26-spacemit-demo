# Requant Optimization Report

classification: pass

## Candidates

| candidate | path | correctness | CPU0 total_us | activation_us | activation_share | decision |
|---|---|---|---:|---:|---:|---|
| A0 | Stage 8 `int8_lut` | pass | 350531 | 192885 | 55.0266% | baseline |
| A1 | scalar-unrolled float requant + LUT | pass | 306914 | 149004 | 48.5492% | improvement, not enough |
| A2 | RVV f32 requant + LUT | pass | 182420 | 24471.3 | 13.4148% | accepted |
| A3 | fixed-requant + LUT | pass | 215074 | 57203.8 | 26.5972% | secondary exact path |
| A4 | fused current-layout scalar-unrolled | pass | 304990 | 146948 | 48.1811% | improvement, not enough |
| A5 | packA handoff sidecar | pass pack/unpack | n/a | 4059.95 packA only | n/a | not integrated |

## Findings

- H1 confirmed for Stage 8 baseline: scalar per-element conv-output-code quantization dominated after LUT removed SiLU math.
- H2 confirmed: removing `% channels` and function-call overhead in A1 reduced activation from `192885 us` to `149004 us`.
- H3 confirmed: RVV f32 requant was much faster on X60 than scalar-unrolled and fixed-requant paths for this selected subset.
- H4 partially confirmed: A5 packA handoff is correct as a sidecar, but A2 already cleared the activation gate, so packA fusion should move to a later integration stage only if needed.
- H5 satisfied for accepted A2 path on selected fixtures: mismatches remained `0` against Stage 8 scalar/LUT reference and selected-subset oracle.

Accepted path: `A2_rvv_f32_lut`.
