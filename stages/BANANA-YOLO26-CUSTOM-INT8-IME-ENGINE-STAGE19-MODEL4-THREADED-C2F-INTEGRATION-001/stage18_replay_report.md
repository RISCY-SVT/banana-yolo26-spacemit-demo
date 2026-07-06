# Stage18 Replay Report

Replay target:

```text
/model.4/m.0/cv1/conv/Conv representative/full-shape branch-entry
stable protocol: warmup=10, runs=100, repeats=5
pinning: taskset CPU0-3
board target: svt@banana
```

Raw log:

```text
/data/ncnn-logs/ai-team/2026-07-06_06-41-42/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE19-MODEL4-THREADED-C2F-INTEGRATION-001/run_logs/board_stage18_replay_stable_after_fix.log
```

| candidate | threads | CPUs | mean_total_us | stddev_total_us | CV % | mean_conv_us | mean_activation_requant_us | total speedup | conv speedup | mismatches |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0_stage17_single_thread_replay | 1 | 0 | 25588.139540 | 24.851779 | 0.097122 | 20402.617866 | 5000.416242 | 1.000000 | 1.000000 | 0 |
| A1_integrated_threaded_conv_1t | 1 | 0 | 26195.022234 | 534.310124 | 2.039739 | 20956.839720 | 5040.938930 | 0.976832 | 0.973554 | 0 |
| A2_integrated_threaded_conv_2t | 2 | 0-1 | 16471.759794 | 613.091522 | 3.722077 | 11271.260658 | 5004.468012 | 1.553455 | 1.810145 | 0 |
| A3_integrated_threaded_conv_3t | 3 | 0-2 | 12622.782918 | 38.511657 | 0.305096 | 7420.070616 | 5008.923600 | 2.027139 | 2.749653 | 0 |
| A4_integrated_threaded_conv_4t | 4 | 0-3 | 11082.483550 | 96.463441 | 0.870414 | 5905.210462 | 4983.945734 | 2.308881 | 3.455020 | 0 |

Checksum remained `1324192976` for all replayed candidates. Worker affinity was reported as ok for all candidates.

Interpretation:

```text
Stage18 representative/full-shape branch-entry threading still passes correctness and stable timing.
The 4-thread Conv path remains useful for the representative branch-entry Conv.
Activation/requant is the largest non-Conv bucket after Conv threading: 44.971379% of A4 total.
```
