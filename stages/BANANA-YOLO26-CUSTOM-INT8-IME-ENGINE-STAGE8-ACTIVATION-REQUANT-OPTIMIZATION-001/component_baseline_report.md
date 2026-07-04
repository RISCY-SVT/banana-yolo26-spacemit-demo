# Component Baseline Report

subset: `candidate_D_block0_silu_model1_silu_model2_cv1_conv`
measurement_scope: selected subset microbench only
board_cpu: CPU0

## Stage 7 Replay Components

| component | Stage 7 replay us |
|---|---:|
| Conv0 IME | 71872.9 |
| Act0/requant float fallback | 317064 |
| Conv1 IME | 68206.8 |
| Act1/requant float fallback | 157572 |
| Conv2 IME | 24966 |
| IME total | 640886 |
| activation total | 474636 |
| activation share | 74.0593% |

## Stage 8 Same-Binary Baseline

| component | `scalar_float_reference` us |
|---|---:|
| Conv0 IME | 68199.6 |
| Act0/requant float fallback | 311321 |
| Conv1 IME | 63955.9 |
| Act1/requant float fallback | 154580 |
| Conv2 IME | 21541.5 |
| IME total | 620735 |
| activation total | 465901 |

The same-binary baseline is used for Stage 8 speedup calculations because it removes binary and layout drift from the comparison.
