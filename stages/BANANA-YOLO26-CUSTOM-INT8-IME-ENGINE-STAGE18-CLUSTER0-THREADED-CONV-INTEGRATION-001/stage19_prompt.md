# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE19-MODEL4-THREADED-C2F-INTEGRATION-001

User-facing summaries must be in Russian. Code, commands, identifiers, report filenames, and artifact names stay in English.

## Mission

Integrate the Stage18 selected threaded Conv sidecar into the bounded `/model.4` C2f completion path, without implementing a full graph scheduler or full YOLO26 engine.

## Required Starting Evidence

```text
Stage18 selected mode: A4_integrated_threaded_conv_4t
target Conv: /model.4/m.0/cv1/conv/Conv
4-thread mean_total_us: 11211.333822
4-thread mean_conv_us: 6025.979842
total speedup vs A0: 2.282369
conv speedup vs A0: 3.387750
mismatches: 0
CPU4-7 IME: not used
```

## Scope

Allowed:

```text
reuse y26_threaded_conv_create_spatial_rows
apply explicit threaded mode to /model.4 C2f selected Conv nodes where oracle is already clear
measure activation share after threaded Conv integration
preserve CPU0-3 only
preserve single-thread default
```

Forbidden:

```text
full YOLO26 inference
graph-wide scheduler
camera/full-image/COCO/mAP
model FPS or production claims
ncnn mutation
XSlim
vmadot1/2/3 implementation
vmadotn
CPU4-7 IME
OpenMP/all-core default dispatch
```

## Gate

If activation share remains above 40% after threaded Conv integration, open a focused activation/layout follow-up before further graph expansion.
