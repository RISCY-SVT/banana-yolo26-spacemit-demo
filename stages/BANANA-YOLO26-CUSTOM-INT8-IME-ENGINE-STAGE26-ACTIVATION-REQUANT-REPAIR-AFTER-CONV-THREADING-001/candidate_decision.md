# Candidate Decision

selected_lane: A3
selected_candidate: A3_branch1_add_lut
classification: stage26-activation-requant-repaired-ready-for-next-stage

## Why A3

Stage26 replay proved branch1 activation dominated the activation/requant bucket:

```text
branch1_activation_us: 31536.9
activation_requant_us: 32790.8
branch1_share_of_activation: 96.18%
```

A3 is local to the existing real model4 cut runner API and does not expand the graph. It builds a prepare-time 256x256 LUT:

```text
split1_u8_code x branch1_conv_u8_code -> concat_s8 add-slot code
```

This preserves the ONNX float-domain Add/post-QDQ behavior by using the same per-code dequant, SiLU, add, and RNE quantization semantics as the baseline path.

## Rejected Lanes

A2 threaded activation was not selected because the dominant branch1 activation was scalar std::exp/math, not a pure memory copy. The LUT candidate is simpler and removes the math directly.

Lane B merge residual repair was not selected as the primary lane because branch1 activation was the dominant measured subbucket. A3 also reduces merge as a side effect because the add-slot post-QDQ is fused into the LUT.

Lane C no-repair was rejected because A3 is exact and materially faster.
