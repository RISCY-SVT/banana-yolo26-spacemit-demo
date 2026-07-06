# BANANA-YOLO26 Custom INT8 IME Engine Stage 20 Prompt

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE20-ACTIVATION-REQUANT-AND-MODEL4-FULLSHAPE-REPAIR-001`

## Mission

Use Stage19 evidence to repair the next real bottleneck without expanding the graph blindly.

Mandatory first step:

```text
Build representative/full-shape timing and correctness for the current model4 C2f runner.
Do not rely on compact oracle-scope timing for performance decisions.
```

Then choose one repair lane based on measured full-shape model4 C2f buckets:

```text
activation/requant fusion or threading repair
memory planner / concat-view / direct-QDQ-to-packed-layout repair
Conv tile/kernel tuning
```

## Required starting facts

```text
Stage18 representative branch-entry A4:
  total_us: 11082.483550
  conv_us: 5905.210462
  activation_requant_us: 4983.945734
  total_speedup_vs_A0: 2.308881x
  conv_speedup_vs_A0: 3.455020x
  mismatches: 0

Stage19 compact C2f A4:
  total_us: 283.534780
  total_speedup_vs_A0: 0.656284x
  mismatches: 0

Stage19 compact C2f A5:
  total_us: 461.796518
  total_speedup_vs_A0: 0.402947x
  mismatches: 0
```

## Restrictions

Do not implement full YOLO26 inference, graph-wide scheduler, camera/full-image path, COCO/mAP, production/model FPS claim, XSlim, vmadot1/2/3, vmadotn, FP/vfmadot, CPU4-7 IME, OpenMP/all-core default dispatch, or `/data/ncnn` mutation.
