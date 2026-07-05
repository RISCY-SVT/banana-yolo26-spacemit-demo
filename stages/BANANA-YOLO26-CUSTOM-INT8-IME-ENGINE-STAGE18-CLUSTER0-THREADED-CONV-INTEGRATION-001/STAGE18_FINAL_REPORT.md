# Stage18 Final Report

classification: `stage18-threaded-conv-integrated-ready-for-model4-threaded-c2f-stage`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE18-CLUSTER0-THREADED-CONV-INTEGRATION-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `92e7d8763295cc514b09d0db32ed2795b092dc44`
end_head: `pending-local-commit-see-final-response`
pushed: false
full_engine_implemented: false
graph_wide_scheduler_implemented: false
ncnn_source_mutated: false
production_claim_made: false
xslim_used: false
sliding_vmadot123_used: false
vmadotn_used: false
cluster1_ime_used: false

## Selected Mode

```text
selected_threaded_mode: A4_integrated_threaded_conv_4t
target Conv: /model.4/m.0/cv1/conv/Conv
threading: explicit cluster0 CPU0-3
default path changed: false
```

## Results

| candidate | threads | mean_total_us | stddev_total_us | mean_conv_us | stddev_conv_us | total_speedup_vs_A0 | conv_speedup_vs_A0 | mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0_stage17_single_thread_replay | 1 | 25588.395318 | 3.740899 | 20414.513636 | 10.350443 | 1.000000 | 1.000000 | 0 |
| A4_integrated_threaded_conv_4t | 4 | 11211.333822 | 184.481542 | 6025.979842 | 187.863483 | 2.282369 | 3.387750 | 0 |

Acceptance:

```text
target Conv 4-thread speedup >= 3.0x: pass
4-thread Conv mean_us <= 6500: pass
total branch-entry mean_us <= 12000: pass
mismatches=0: pass
CPU4-7 IME: no
```

## Validation

```text
host CTest: pass, 34/34
RISC-V cross build: pass
board CPU0-3 smoke: pass
board threaded correctness 1/2/3/4: pass
board stable microbench: pass
```

## Caveat

This is selected-subset evidence for the `/model.4` branch-entry sidecar. It is not full YOLO26 FPS, not full-image/camera performance, not COCO/mAP, and not production readiness.

next_recommended_step: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE19-MODEL4-THREADED-C2F-INTEGRATION-001`
