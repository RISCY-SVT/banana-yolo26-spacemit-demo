# Activation Requant Subbucket Report

## Baseline Subbuckets

The Stage25 selected C1 replay with Stage26 instrumentation produced non-overlapping activation subbuckets:

```text
activation_requant_us: 32790.8
branch0_activation_us: 1253.91
branch1_activation_silu_or_lut_us: 31536.9
add_slot_quant_or_merge_us: 21101.5
other_activation_requant_us: approximately 0 within available instrumentation
frm_guard_overhead_us: not separately measurable; covered by total/other
```

Branch1 activation is 96.18% of the activation bucket:

```text
31536.9 / 32790.8 = 96.18%
```

## Candidate Subbuckets

The accepted A3 candidate produced:

```text
activation_requant_us: 3004.46
branch0_activation_us: 1260.4
branch1_activation_us: 1744.06
merge_us: 2156.81
```

A3 replaces branch1 per-element SiLU float materialization with branch1 conv-code quantization plus a precomputed split1-code x branch1-code Add/post-QDQ LUT.
