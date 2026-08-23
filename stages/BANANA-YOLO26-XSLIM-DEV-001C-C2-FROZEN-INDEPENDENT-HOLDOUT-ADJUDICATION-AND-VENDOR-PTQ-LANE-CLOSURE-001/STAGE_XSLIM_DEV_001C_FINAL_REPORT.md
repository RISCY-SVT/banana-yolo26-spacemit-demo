# XSLIM-DEV-001C Final Report

## Classification

```text
xslim-dev-001c-frozen-c2-independent-holdout-pass-
full-val-pareto-fail-vendor-ptq-lane-closed
```

Publication classification: `not-authorized-not-attempted`.

Stage ID: `BANANA-YOLO26-XSLIM-DEV-001C-C2-FROZEN-INDEPENDENT-HOLDOUT-ADJUDICATION-AND-VENDOR-PTQ-LANE-CLOSURE-001`.

## Input Closure

- Banana started at `6e7a2baf0d4b8dc2922c72ddfdcb8c83b85356f1`, tree `c2774c6beb88f558f71202c2a08f86a9fbd5929c`.
- XSlim remained read-only at `46d5d36bcb6979bab6567fb4fe62839689f1881c`, tree `1788779cd0887a1c8e6924cd63ad7d16d42f41ca`, version `2.1.2+riscy.2.dev2`.
- DEV-001B packet identity passed: tree `139092b15cede35760edf5d6fdfef98503a9bab788974cd4f7409d5bdfb997f9`, 59 files, 181,055 bytes.
- Frozen B2, A1, C2 deployable/inference models and the common tail matched every expected SHA-256. No model, qparam, reconstruction or XSlim source generation occurred.

## Fresh H5000 Surface

`H5000_C2_ADJUDICATION` was selected by stable SHA-256 rank under policy `xslim-dev-001c-fresh-h5000-v1`. The 5,000-image list SHA-256 is `55021fbc1a58109f22239b02d433bb92caa25589613a5883288d7dd71ff4dfb9`.

The surface has zero overlap with all prior calibration, reconstruction and candidate-selection inputs, H500 and val2017 by image ID, exact JPEG SHA-256 and canonical decoded-pixel SHA-256. Internal duplicate counts are also zero. It covers all 80 detection categories and contains 14,609 small, 12,241 medium and 8,524 large annotated objects.

Classification: quantization-selection-independent `yes`; model-training-independent `no` because the source model used COCO train2017; final-generalization authority `no`.

## H5000 Adjudication

| Surface | mAP50-95 | AP-S | AP-M | AP-L | AR-S | AR-M | AR-L | Predictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B2 | 0.423400181 | 0.223132289 | 0.481681625 | 0.618003194 | 0.402980128 | 0.694155043 | 0.833943112 | 641204 |
| A1 | 0.429818184 | 0.223556396 | 0.487131015 | 0.632701484 | 0.403813647 | 0.691133483 | 0.828684778 | 615204 |
| C2 | 0.434699252 | 0.224535484 | 0.487446124 | 0.650722718 | 0.403088784 | 0.694460464 | 0.832741867 | 628372 |

The shared 10,000-draw bootstrap used seed `65007`; draw-matrix content SHA-256 is `8405b60ab498ad7af36afb3900292022569f9f9b300d7553faa46c638fd947a8` and replicate NPZ SHA-256 is `4e9b6eae5b067e424ac5ce56636a123685befb71fc80f8b03889eaf8aa5041c3`.

C2 improved mAP over B2 by `+0.011299070376`, with `P(delta>0)=1.0`. Every AP/AR size-bin point and probability gate passed. AR-small delta was `+0.000108655427`; AR-large delta was `-0.001201244191`. All six descriptive non-inferiority intervals were `interval-pass`. C2 also exceeded A1 mAP by `+0.004881067260` and AR-large by `+0.004057089301`, so conditional full val2017 opened.

## Full Val2017

Accepted B2/A1 host predictions were reused only after exact model, tail, runner, preprocessing, threshold, image-list, input-tensor and evaluator identities matched. C2 alone was run fresh through the same host runner for 5,000/5,000 images with zero failures, non-finite outputs or collapse.

| Surface | mAP50-95 | AP-S | AP-M | AP-L | AR-S | AR-M | AR-L | Predictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B2 | 0.365859229 | 0.180146851 | 0.419955250 | 0.520106705 | 0.373046142 | 0.646686907 | 0.789256171 | 651637 |
| A1 | 0.372935154 | 0.179846689 | 0.424335192 | 0.541288514 | 0.373047868 | 0.644739018 | 0.785733487 | 625640 |
| C2 | 0.378508778 | 0.180461061 | 0.425631820 | 0.558723160 | 0.373183454 | 0.646379022 | 0.786853666 | 638745 |

The full-val shared bootstrap used the same predeclared seed and draw contract. Replicate NPZ SHA-256 is `6cce04f783a3c1a2d171b00a78a016384e99b9615fe124c22583f18595a30446`.

C2 passed every universal gate versus B2. Its mAP delta was `+0.012649549127`, 95% CI `[+0.011938434367, +0.015801015699]`. AR-large delta was `-0.002402504907`, with CI lower `-0.004377891003`, inside the frozen non-inferiority contract.

C2 did not pass Pareto repair versus A1: mAP was higher by `+0.005573624403`, but AR-large recovery was only `+0.001120178248` against a required `+0.002`, and `P(C2-A1 AR-large>0)=0.876` against a required `0.95`.

## Disposition

- C2 remains an exact frozen host research artifact. It is not ready for a K1X gate.
- B2 remains the vendor-lane universal control.
- A1 remains a frozen research artifact and is not promoted.
- The vendor all-S8 PTQ lane is closed for YOLO26 under the predeclared rules.
- Deferred, separately authorized routes are head-only S8 QAT, model/executor co-design, custom-executor rank-aware terminal calibration and stable XSlim source-hardening closure.

No board, custom-executor, publication, tag, release, PyPI or runtime-promotion action occurred.

## Protected State

Banana protected main, the custom-executor tree, XSlim, upstream main and `/data/ncnn` head/tree/diff plus its three accepted dirty paths are unchanged.

Evidence completion timestamp: `2026-08-23T10:34:11Z`.
