# CV2 Baseline Component Report

Target node:

`/model.4/cv2/conv/Conv`

Shape:

`80x80x96 -> 80x80x128`, 1x1 Conv, signed activation storage, signed weights, explicit correction.

Baseline mode: `branch1_add_lut`

| bucket | mean_us |
| --- | ---: |
| model4_cv2_conv_us | 10420.4 |
| model4_cv2_compute_us | 7541.75 |
| model4_cv2_correction_us | 1187.69 |
| model4_cv2_copy_us | 0 |
| model4_cv2_worker_other_us | 0.225858 |

Context buckets:

| bucket | mean_us |
| --- | ---: |
| branch0_conv_us | 7199.42 |
| branch0_compute_us | 5355.08 |
| branch1_conv_us | 5587.03 |
| branch1_compute_us | 4075.49 |
| thread_overhead_us | 4797.69 |

Conclusion: `/model.4/cv2/conv/Conv` was the largest single Conv target in the selected cut before Stage36. The raw compute bucket was large enough to justify one bounded pipelined `smt.vmadot` candidate.
