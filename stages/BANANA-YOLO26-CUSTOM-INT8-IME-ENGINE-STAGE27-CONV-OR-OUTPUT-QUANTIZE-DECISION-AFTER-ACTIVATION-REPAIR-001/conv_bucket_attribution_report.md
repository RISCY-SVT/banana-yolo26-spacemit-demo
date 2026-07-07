# Conv Bucket Attribution Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE27-CONV-OR-OUTPUT-QUANTIZE-DECISION-AFTER-ACTIVATION-REPAIR-001
selected_mode: Y26_STAGE16_MERGE_MODE_STAGE26_BRANCH1_ADD_LUT

## Non-Overlapping Bucket Map

The Stage26 accepted path was replayed with the existing runner timing buckets. Attribution was:

```text
mean_total_us: 41669.2
mean_attributed_us: 41647.2
mean_attribution_pct: 99.9472
mean_other_us: 21.9888
```

Buckets:

| bucket | mean_us | share_pct | notes |
|---|---:|---:|---|
| input_adapter | 2615.76 | 6.277 | cut input split/LUT adapter |
| conv | 26869.6 | 64.4832 | branch0 + branch1 + model4_cv2 Conv including correction/copy in threaded kernels |
| activation_requant | 2998.92 | 7.19698 | branch0 activation + branch1 code LUT path |
| merge | 2088.94 | 5.01316 | Stage26 branch1 add LUT concat merge |
| output_quantize | 7073.94 | 16.9765 | final model4_cv2 int32 -> uint8 output boundary |
| other | 21.9888 | 0.0528 | timer residual |

Thread overhead is reported separately by the threaded Conv runner:

```text
thread_overhead_us: 4980.87
```

It is already included inside the Conv bucket and must not be added again when computing shares.

## Decision Impact

Conv remains the dominant selected-cut bucket after Stage26. Output quantize is secondary at `16.9765%`, below the Stage27 `>20%` threshold for selecting output quantize repair.
