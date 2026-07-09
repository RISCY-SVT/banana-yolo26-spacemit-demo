# Candidate Benchmark Report

Protocol:

- board: Banana-Pi BPI-F3 / SpacemiT K1X / X60
- affinity: `taskset -c 0-3`
- warmup: 10
- runs: 100
- repeats: 5
- no `rdcycle`

| candidate | mean_total_us | stddev_total_us | cv_pct | model4_cv2_compute_us | model4_cv2_conv_us | thread_overhead_us | attribution_pct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0_branch1_add_lut | 37341.1 | 405.161 | 1.08503 | 7541.75 | 10420.4 | 4797.69 | 99.9258 |
| A1_cv2_pipelined4 | 33192.7 | 364.104 | 1.09694 | 3616.14 | 6307.54 | 4362.11 | 99.9195 |
| A2_cv2_pipelined6 | 33217.5 | 418.787 | 1.26074 | 3822.91 | 6521.77 | 4347.32 | 99.9179 |

Speedup vs A0:

| candidate | total_speedup | model4_cv2_compute_speedup | model4_cv2_conv_speedup | decision |
| --- | ---: | ---: | ---: | --- |
| A1_cv2_pipelined4 | 1.124979 | 2.085580 | 1.652055 | selected |
| A2_cv2_pipelined6 | 1.124139 | 1.972777 | 1.597787 | correct but not selected |

Acceptance gates:

- minimum `model4_cv2_compute_us` speedup >= 1.25x: pass
- minimum selected-cut total speedup >= 1.05x: pass
- good selected-cut total speedup >= 1.10x: pass
- excellent `model4_cv2_compute_us` speedup >= 2.00x: pass for A1

Conclusion: A1 transfers Stage35 throughput headroom into the real `/model.4/cv2/conv/Conv` selected ONNX-cut path.
