# Full COCO comparison

## Identity

All rows use COCO val2017 5,000 images, annotation SHA-256
`e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`,
the same 640-letterbox preprocessing contract, and confidence threshold
0.001. There are zero image failures and zero non-finite outputs.

| Surface | mAP50-95 | mAP50 | AP small | AP medium | AP large | Predictions | Prediction SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---|
| same-source FP32 CPU | 0.404730651123 | 0.571261907189 | 0.197788572586 | 0.441452359314 | 0.586958794540 | 548128 | `e8c97ebf44727670cdc44c3fa5ce50df6748e849dd4211b9367db92b5da96c1a` |
| XSlim 2.1.1 project-exact S8 CPU split | 0.358768508793 | 0.514176307052 | 0.179113696686 | 0.417639616374 | 0.516550936249 | 782544 | `6162fc26a654f19e21a7ba65f064ab1c3f651a318453944e25026f2e75ae3a00` |
| XSlim 2.1.1 project-exact S8 SpacemiT split | 0.359072683728 | 0.514173727408 | 0.179166788668 | 0.420416372236 | 0.516591419978 | 791650 | `8d16e1cdc0436c0ff8f5dae0d411778bca70c870206db55b6ef25d4c7af494a8` |

## Interpretation

The SpacemiT route loses 0.045657967395 mAP50-95, or 4.5658 AP points,
against the same-source FP32 control. This is a quantization/tool output
comparison, not a regression against the custom executor's separate integer
contract.

The CPU and EP rows use the same generated S8 model and explicit floating
tail. Their aggregate and per-class differences characterize provider
execution differences; byte identity is not required. EP mAP50-95 is
0.000304174935 higher than CPU, while mAP50 differs by only
-0.000002579644. Prediction counts and hashes differ, and the largest
per-class EP-minus-CPU AP deltas are +0.026287965942 and -0.023083261493.
The close aggregate result therefore supports task-level correctness but does
not imply per-image or per-class identity.

The selected 50-image calibration list is disjoint from the selected
100-image holdout. Its source corpus nevertheless overlaps COCO val2017, so
these metrics identify the measured artifact but are not independent
calibration-generalization evidence.
