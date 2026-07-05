# Stage17 Final Report

classification: `stage17-threading-positive-ready-for-threaded-conv-integration`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE17-CONV-IME-ROOFLINE-AND-CLUSTER0-THREADING-FEASIBILITY-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `3ca94995a7b18276dac9ac7c660061cd93377994`
end_head: `pending-local-commit-see-final-response`
pushed: false
full_engine_implemented: false
graph_scheduler_implemented: false
ncnn_source_mutated: false
production_claim_made: false
xslim_used: false
sliding_vmadot_used: false
vmadotn_used: false

## Stable Replay

Protocol: `warmup=10 runs=100 repeats=5`

| candidate | mean_total_us | stddev_total_us | cv_pct | mean_conv_us | mean_activation_requant_us | mismatches |
|---|---:|---:|---:|---:|---:|---:|
| `scalar_reference_int8_lut` | 90950.820736 | 37.846523 | 0.041612 | 50738.654554 | 40039.630260 | 0 |
| `stage17_IME_A2_rvv_f32_lut` | 25670.974780 | 345.670192 | 1.346541 | 20458.001284 | 5035.606050 | 0 |

## Roofline

Measured node: `/model.4/m.0/cv1/conv/Conv`

```text
MAC_count: 29491200
IME single-thread conv_us: 20458.001284
effective_GMAC_s: 1.441548
bottleneck_class: structural_low_K_or_packing
```

## Threading Feasibility

| threads | mean_us | stddev_us | speedup_vs_1thread | mismatches |
|---:|---:|---:|---:|---:|
| 1 | 20550.030600 | 4.898223 | 1.000000 | 0 |
| 2 | 10750.271882 | 9.551604 | 1.911582 | 0 |
| 3 | 7258.029130 | 9.618641 | 2.831351 | 0 |
| 4 | 5583.807628 | 11.380425 | 3.680290 | 0 |

threading_feasibility: `strong_positive`

## Decision

Stage17 repaired benchmark methodology and showed that the representative branch-entry Conv work scales strongly across cluster0 CPU0-3 using spatial row split. The next stage should be bounded threaded Conv integration, not full graph expansion and not a vmadot1/2/3 implementation lane yet.

This is not full YOLO26 FPS, not full-image/camera performance, not COCO/mAP, and not production/default-backend evidence.

next_recommended_step: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE18-CLUSTER0-THREADED-CONV-INTEGRATION-001`
