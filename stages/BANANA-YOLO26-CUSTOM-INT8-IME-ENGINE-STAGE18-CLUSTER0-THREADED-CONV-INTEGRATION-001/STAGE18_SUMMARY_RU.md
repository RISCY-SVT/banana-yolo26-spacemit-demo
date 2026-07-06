# Stage18 Summary RU

classification: `stage18-threaded-conv-integrated-ready-for-model4-threaded-c2f-stage`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE18-CLUSTER0-THREADED-CONV-INTEGRATION-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `92e7d8763295cc514b09d0db32ed2795b092dc44`
end_head: `6c4c8258fb10cf25476a8380870d624200855f9b`
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false

## Proven

Stage18 добавил явный cluster0 threaded Conv sidecar для `/model.4/m.0/cv1/conv/Conv`.

```text
selected_mode: A4_integrated_threaded_conv_4t
CPU policy: CPU0-3 only
OpenMP/default all-core dispatch: not used
mismatches: 0
checksum: 1324192976
```

Stable board microbench:

```text
A0 single-thread total: 25588.395318 us ± 3.740899
A0 single-thread Conv: 20414.513636 us ± 10.350443
A4 4-thread total: 11211.333822 us ± 184.481542
A4 4-thread Conv: 6025.979842 us ± 187.863483
total speedup: 2.282369x
Conv speedup: 3.387750x
```

## Broken

No correctness break found. No CPU4-7 IME execution found. No default backend change.

## Unknown

Full YOLO26 FPS is unknown. Full-image/camera performance is unknown. COCO/mAP is unknown. Production readiness is not claimed.

## Next

Recommended next stage:

```text
BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE19-MODEL4-THREADED-C2F-INTEGRATION-001
```
