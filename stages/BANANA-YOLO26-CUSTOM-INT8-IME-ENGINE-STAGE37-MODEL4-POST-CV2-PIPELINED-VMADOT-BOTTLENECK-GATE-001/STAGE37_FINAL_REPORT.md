# STAGE37 Final Report

classification: stage37-branch3x3-pipelined-mmt4d-selected-ready-for-next-bottleneck-gate

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE37-MODEL4-POST-CV2-PIPELINED-VMADOT-BOTTLENECK-GATE-001

repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
start_head: a945d60a5fedf3d5b74483a02e5b95214c5cd973
end_head: pending-local-commit-see-final-response
pushed: false

## Non-Claims

```text
full_engine_implemented: false
graph_expansion_done: false
model_fps_claim_made: false
full_image_camera_claim_made: false
coco_map_claim_made: false
production_claim_made: false
ncnn_source_mutated: false
default_backend_changed: false
```

## Selected Lane

```text
selected_lane: Lane A branch 3x3 pipelined MMT4D/GEMM repair
selected_candidate: Y26_STAGE16_MERGE_MODE_STAGE37_BRANCH3X3_PIPELINED4
instruction: smt.vmadot
storage: signed s8 activation x signed s8 weight
correction: existing explicit correction preserved
CPU policy: cluster0 CPU0-3 only
```

## Correctness

```text
same_input_onnx_cut_status: pass
mismatches: 0
max_abs_diff: 0
output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
frm_sweep: pass
affinity_ok: 1
CPU4_7_IME_execution: none
```

## Stable Timing

Protocol:

```text
taskset -c 0-3
warmup: 10
runs: 100
repeats: 5
```

| metric | Stage36 baseline | Stage37 candidate |
| --- | ---: | ---: |
| mean_total_us | 35774.4 | 32307.4 |
| stddev_total_us | 272.052 | 112.917 |
| cv_total_pct | 0.760466 | 0.349509 |
| conv_us | 21082.8 | 17742.7 |
| activation_requant_us | 2986.17 | 2993.66 |
| merge_us | 2131.95 | 2099.23 |
| output_quantize_us | 7110.83 | 7081.0 |
| combined_branch3x3_compute_us | 10257.22 | 7157.61 |
| combined_branch3x3_conv_us | 13747.47 | 10241.72 |

```text
selected_cut_total_speedup: 1.107313x
combined_branch3x3_compute_speedup: 1.433051x
combined_branch3x3_conv_speedup: 1.342301x
```

## Bucket State After Candidate

```text
conv_share_pct: 54.9182
output_quantize_share_pct: 21.9176
activation_requant_share_pct: 9.26617
merge_share_pct: 6.49766
attribution_pct: 99.9188
```

## Validation

```text
host_build: pass
host_ctest: pass, 42/42
riscv_cross_build: pass
board_correctness: pass
board_stable_benchmark: pass
frm_sweep: pass
source_hygiene: see source_hygiene_report.md
```

## Caveats

Current per-Conv timing does not expose `im2col_pack_us` as a separate non-overlapping counter. It is reported as `included_in_compute`. No im2col-specific optimization claim is made in Stage37.

## Next Recommended Step

Run Stage38 as a fresh bottleneck gate after Stage37 selected mode. Based on current same-session evidence, likely candidates are:

```text
1. output QuantizeLinear repair if it remains >=18-20% of selected-cut total;
2. remaining Conv/thread-overhead work if Conv stays dominant and a local >=5% selected-cut total lane exists;
3. stop/next-boundary planning if no local lane clears the threshold.
```
