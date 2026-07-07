# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE26-ACTIVATION-REQUANT-REPAIR-AFTER-CONV-THREADING-001

## Mission

Continue the isolated YOLO26 custom INT8 IME engine track in:

```text
/data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
```

Stage25 selected C1 Conv threading propagation for the existing `/model.4` ONNX-cut path:

```text
selected_candidate: C1_thread_branch1_and_model4_cv2_4t
selected_total_us: 89178.9
conv_share_pct: 29.3389
activation_share_pct: 36.7808
merge_share_pct: 23.5081
output_quantize_share_pct: 7.37782
mismatches: 0
frm_sweep: pass
```

Stage26 must not expand the graph. It must target exactly one local activation/requant repair lane on the same `/model.4` ONNX-cut runner API path.

## Hard Boundaries

Do not implement full YOLO26 inference, graph-wide scheduler, camera/full-image path, COCO/mAP, production/model FPS, `/data/ncnn` mutation, XSlim, vmadot1/2/3, vmadotn, FP/vfmadot, CPU4-7 IME, OpenMP/all-core dispatch, or default backend switch.

## Required Gates

```text
- replay Stage25 selected C1 path with warmup=10 runs=100 repeats=5;
- same-input ONNX-cut mismatches=0 and max_abs_diff=0;
- ambient frm sweep RNE/RTZ/RDN/RUP/RMM pass;
- non-overlapping bucket attribution >=99%;
- select exactly one activation/requant candidate;
- host CTest pass;
- RISC-V cross build pass;
- board CPU0-3 correctness and stable timing pass.
```

## Candidate Lanes

```text
A1_branch1_activation_lut_or_table:
  Replace branch1 scalar float activation/requant with exact LUT/RVV/table path if the boundary permits byte-exact ONNX cut output.

A2_threaded_activation_requant:
  Use the existing cluster0 worker infrastructure for large activation/requant tensors only, with an element-count threshold so compact tensors remain single-thread.

A3_fused_activation_merge_handoff:
  Fuse branch1 activation output generation with existing merge/post-QDQ path only if byte-exact and local.
```

## Next Decision

If activation/requant is repaired and merge becomes dominant, Stage27 should revisit merge/dataflow. If Conv becomes dominant again, Stage27 should consider MMT4D tile/prepack or future vmadot123 proof only with a separate proof prompt.
