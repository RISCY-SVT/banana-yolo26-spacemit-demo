# BANANA-YOLO26 Vendor ORT rt204 mAP Baseline Gate

## Stage ID

`BANANA-YOLO26-VENDOR-ORT-RT204-MAP-BASELINE-001`

## Mission

Run a separate Track B model-value gate for YOLO26 on the Banana-Pi BPI-F3 / SpacemiT K1X board before authorizing deeper custom-engine direct-conv or `vmadot1/2/3` proof investment.

Stage28 proved selected `/model.4` custom-engine Conv structural low utilization after a local T2 copy/writeback repair:

```text
stage28_t2_total_us: 40231.6
stage28_t2_conv_us: 25255.4
stage28_t2_conv_share: 62.775%
stage28_t2_conv_compute_us: 18097.1
stage28_t2_onnx_cut_mismatches: 0
```

Do not treat this as full-model performance. This Track B stage must answer whether YOLO26 model quality/value justifies a future custom-engine `vmadot1/2/3` semantics/direct-conv applicability proof lane.

## Scope

Allowed:

```text
- Use vendor ORT rt204 only for a model-quality/performance baseline track.
- Use a fixed COCO validation subset or full val2017 with documented preprocessing/postprocessing.
- Compare carefully against YOLO11 production numbers without mixing model/runtime/precision claims.
- Produce mAP and board runtime evidence as a separate report.
```

Forbidden:

```text
- Do not mutate /data/ncnn.
- Do not mutate YOLO11 production repo.
- Do not change the custom INT8 engine.
- Do not claim production readiness.
- Do not mix Track B numbers into selected-cut custom-engine speed claims.
```

## Output

Create a result packet with:

```text
MAP_BASELINE_FINAL_REPORT.md
MAP_BASELINE_SUMMARY_RU.md
dataset_manifest.md
runtime_manifest.md
commands.txt
source_hygiene_report.md
```

If Track B is positive, the next custom-engine stage may be:

```text
BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-VMADOT123-SEMANTICS-AND-CONV-APPLICABILITY-001
```
