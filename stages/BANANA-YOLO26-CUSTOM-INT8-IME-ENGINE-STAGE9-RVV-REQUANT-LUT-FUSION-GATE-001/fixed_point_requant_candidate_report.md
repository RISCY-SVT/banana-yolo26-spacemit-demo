# Fixed-Point Requant Candidate Report

classification: pass-secondary

Candidate: `A3_fixed_requant_lut`

The Stage 8 diagnostic fixed-point requant path was combined with the 256-entry SiLU LUT.

Correctness:

- Direct Act0/Act1 fixture comparison vs Stage 8 LUT baseline: mismatches `0`.
- Full selected-subset runner on board CPU0/1/2/3: Conv2 mismatches `0`.

CPU0 benchmark:

| metric | value |
|---|---:|
| selected_subset_total_us | 215074 |
| activation_total_us | 57203.8 |
| activation_share | 26.5972% |

Decision:

`A3_fixed_requant_lut` is kept as an exact secondary path for the selected subset. `A2_rvv_f32_lut` is faster and selected for Stage 9.
