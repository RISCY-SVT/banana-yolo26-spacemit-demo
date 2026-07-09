# Selected Lane Decision

## Decision

- selected_lane: `A`
- selected_candidate: `A2_fuse_clamp_store_remove_intermediate_buffer`
- selected_mode: `Y26_STAGE16_MERGE_MODE_STAGE37_BRANCH3X3_PIPELINED4 + Y26_STAGE16_OUTPUT_QUANTIZE_STAGE38_RVV_DIRECT_STORE`

## Evidence

Stage37 replay:

- output_quantize_us: `7055.2`
- output_quantize_share_pct: `21.4506`
- threshold: select Lane A if `output_quantize_us >= 6000` or share `>= 18%`

Lane C was also plausible because branch 3x3 im2col/pack was `55.10%` of combined branch 3x3 conv. Stage38 selected only one lane as required. Lane A was first in the decision tree and had a smaller local exact implementation.

## Candidate Description

The selected candidate adds an explicit local output quantize mode:

```text
Y26_STAGE16_OUTPUT_QUANTIZE_STAGE38_RVV_DIRECT_STORE
```

It preserves the same ONNX QuantizeLinear math and RNE conversion policy as the accepted RVV path, but removes the intermediate temporary code buffer and writes `u8` output directly with RVV narrowing and `vse8`.

## Acceptance

| gate | required | observed | status |
|---|---:|---:|---|
| output_quantize speedup | >= 1.30x | 1.54994x | pass |
| selected-cut total speedup | >= 1.05x | 1.08401x | pass |
| mismatches | 0 | 0 | pass |
| max_abs_diff | 0 | 0 | pass |
| FRM sweep | pass | pass | pass |
| host CTest | pass | 42/42 | pass |
| RISC-V cross build | pass | pass | pass |
| board CPU0-3 | pass | pass | pass |

## Rejected Lanes For Stage38

- Lane B cluster1 non-IME offload: not selected; Lane A was clearer and local.
- Lane C im2col/pack repair: deferred to Stage39 because it remains the next material local bucket after Lane A.
- Lane D thread overhead repair: deferred; overhead is material but less directly isolated than Lane A.
- Lane E no safe local repair: rejected because Lane A passed.
