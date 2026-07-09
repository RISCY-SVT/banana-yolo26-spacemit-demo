# Selected Lane Correctness Report

## Candidate

- selected_lane: `A`
- output quantize mode: `rvv_direct`
- explicit mode: `Y26_STAGE16_OUTPUT_QUANTIZE_STAGE38_RVV_DIRECT_STORE`
- merge mode: `Y26_STAGE16_MERGE_MODE_STAGE37_BRANCH3X3_PIPELINED4`

## Same-Input ONNX Cut Gate

| metric | value |
|---|---|
| status | pass |
| mismatches | 0 |
| max_abs_diff | 0 |
| checksum | 106597930 |
| output_sha256 | `70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433` |
| expected_sha256 | `70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433` |
| affinity_ok | 1 |

## FRM Sweep

The accepted candidate passed ambient FRM values `RNE RTZ RDN RUP RMM` with `mismatches=0`, `max_abs_diff=0`, checksum `106597930`, and post-call FRM restored.

## Scope

This correctness gate is only for the selected `/model.4` same-input ONNX cut. It is not full YOLO26 inference.
