# Lane Selection Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE37-MODEL4-POST-CV2-PIPELINED-VMADOT-BOTTLENECK-GATE-001

## Input Evidence

Stage36 replay showed:

```text
conv_share_pct: 58.9327
combined branch 3x3 conv_us: 13747.47
combined branch 3x3 compute_us: 10257.22
model4_cv2_conv_us: 7335.33
output_quantize_us: 7110.83
```

## Candidate Lanes

| lane | condition | decision | reason |
| --- | --- | --- | --- |
| Lane A branch 3x3 pipelined MMT4D/GEMM repair | combined branch 3x3 Conv buckets are the clearest local target | selected | branch0+branch1 Conv was 13.747 ms and Stage36 pipelined kernel shape was directly reusable in signed s8xs8 MMT4D |
| Lane B output QuantizeLinear repair | output quantize around 6-7 ms or >=18-20% | deferred | material at 7.111 ms, but smaller than combined branch 3x3 Conv at selection time |
| Lane C thread overhead / persistent-region repair | material thread overhead and low-risk local repair | deferred | thread overhead remained material, but branch 3x3 compute had a clearer bounded kernel transfer path |
| Lane D no safe local repair | no credible >=5% total improvement | rejected | Lane A had a credible local candidate and passed speed/correctness gates |

## Selected Lane

```text
selected_lane: Lane A
selected_candidate: Y26_STAGE16_MERGE_MODE_STAGE37_BRANCH3X3_PIPELINED4
kernel_policy: reuse Stage36 4-accumulator smt.vmadot software-pipelined MMT4D core
storage: signed s8 activation, signed s8 weights
correction: existing explicit correction semantics
forbidden paths avoided: smt.vmadotus, smt.vmadot1/2/3 direct/sliding, vmadotn
```

## Acceptance Result

```text
combined_branch3x3_compute_speedup: 1.433051x
selected_cut_total_speedup: 1.107313x
mismatches: 0
max_abs_diff: 0
output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
frm_sweep: pass
```
