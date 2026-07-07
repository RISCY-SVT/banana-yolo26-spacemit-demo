# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE27-CONV-OR-OUTPUT-QUANTIZE-DECISION-AFTER-ACTIVATION-REPAIR-001

Continue in `/data/banana-yolo26-spacemit-demo` on branch `yolo26-custom-int8-engine` after Stage26.

Stage26 accepted `A3_branch1_add_lut` for the selected `/model.4` ONNX-cut runner path:

```text
Stage25 replay total_us: 90086.8
Stage25 replay activation_requant_us: 32790.8
Stage26 A3 total_us: 41573.9
Stage26 A3 activation_requant_us: 3004.46
mismatches: 0
max_abs_diff: 0
frm_sweep: pass
```

Stage27 must not expand the graph until it replays Stage26 and selects exactly one next lane from measured buckets. The current Stage26 post-repair dominant bucket is Conv:

```text
conv_share_pct: 64.3721
output_quantize_share_pct: 16.8706
activation_share_pct: 7.22679
merge_share_pct: 5.18788
```

Recommended lane: Conv/tile/vmadot123 decision note first, not immediate vmadot1/2/3 implementation. Continue to forbid full engine, graph scheduler, graph expansion, `/data/ncnn` mutation, XSlim, vmadotn, FP/vfmadot, CPU4-7 IME, COCO/mAP, camera/full-image tests, model FPS, and production/default-backend claims.
