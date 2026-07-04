# Activation Requant Sidecar Report

classification: `characterized`

Stage 7 keeps activation/requant as scalar float fallback and times it as first-class buckets.

## CPU0 Board Timing

| bucket | mean us | notes |
|---|---:|---|
| `act0_requant_us` | `293585` | Conv0 Q/DQ + SiLU + Act0 requant, feeds `/model.1/conv/Conv` |
| `act1_requant_us` | `143195` | Conv1 Q/DQ + SiLU + Act1 requant, feeds `/model.2/cv1/conv/Conv` |
| activation/requant total | `436780` | `73.6129%` of Stage 7 IME total |

No low-risk activation optimization was made default in Stage 7. The fallback remains dominant after graph expansion, so Stage 8 should focus on activation/requant optimization.
