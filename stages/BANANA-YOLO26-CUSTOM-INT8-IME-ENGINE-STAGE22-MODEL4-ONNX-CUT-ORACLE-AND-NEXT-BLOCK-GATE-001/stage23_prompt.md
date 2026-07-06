# BANANA-YOLO26 Custom INT8 IME Engine Stage 23 Prompt

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE23-MODEL4-CUT-CLOSED-BUCKET-ATTRIBUTION-AND-TARGETED-REPAIR-001`

## Mission

Use the Stage22 same-input ONNX cut closure as the mandatory correctness oracle for any next `/model.4` C2f work.

Primary goal:

```text
Tighten Stage22 full-shape selected-subset bucket attribution and select one targeted local repair lane.
```

Allowed lanes after bucket attribution:

```text
1. branch1 activation/requant repair if activation remains material;
2. model4 cv2 Conv threading/tuning if cv2 Conv dominates;
3. merge/dataflow repair if split/add/concat/post-QDQ remains material;
4. next graph expansion only if buckets are balanced and same-input ONNX-cut gate remains pass.
```

Hard requirements:

```text
- no full YOLO26 engine;
- no graph-wide scheduler;
- no camera/full-image/COCO/mAP;
- no model FPS or production claim;
- no /data/ncnn mutation;
- no XSlim;
- no vmadot1/2/3, vmadotn, FP/vfmadot;
- no CPU4-7 IME;
- all accepted changes must compare against the Stage22 ONNX cut boundary with mismatches=0.
```
