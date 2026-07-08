# Candidate Benchmark Report

## Baseline

```text
candidate: A0_current_baseline
mean_total_us: 40178.5
stddev_total_us: 283.996
mean_model4_cv2_conv_us: 12096.5
mean_model4_cv2_compute_us: 8071.68
mean_model4_cv2_correction_us: 1753.73
mean_output_quantize_us: 7070.4
```

## Pipelined Candidates

No pipelined `/model.4/cv2` candidate reached selected-path timing. Step 0 rejected the lane before runner integration because the direct inline/register-blocked `smt.vmadot` diagnostic cases trapped on CPU0.

Performance gates were therefore not evaluated for A1-A5:

```text
minimum cv2_compute_us speedup >= 1.25x: not reached
minimum selected-cut total speedup >= 1.05x: not reached
```

Classification basis: `stage34-vmadot-throughput-ceiling-no-pipeline-win`.
