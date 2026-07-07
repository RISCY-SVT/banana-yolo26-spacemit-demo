# Stage25 Replay Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE26-ACTIVATION-REQUANT-REPAIR-AFTER-CONV-THREADING-001

## Command

```text
taskset -c 0-3 ./bench_stage23_model4_runner_cut --fixture-dir fixtures --mode ime_threaded --output-quantize rvv --merge-repair split1_lut --thread-branch0 4 --thread-branch1 4 --thread-model4-cv2 4 --warmup 10 --runs 100 --repeats 5 --frm-sweep
```

## Result

```text
mismatches: 0
max_abs_diff: 0
checksum: 106597930
affinity_ok: 1
frm_sweep: pass for RNE/RTZ/RDN/RUP/RMM
output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
mean_total_us: 89074.9
stddev_total_us: 257.943
cv_total_pct: 0.28958
mean_conv_us: 26075.5
mean_activation_requant_us: 32781.3
mean_merge_us: 20971.2
mean_output_quantize_us: 6574.1
```

A second replay with Stage26 subbucket instrumentation preserved correctness and showed:

```text
mean_total_us: 90086.8
mean_activation_requant_us: 32790.8
mean_branch0_activation_us: 1253.91
mean_branch1_activation_us: 31536.9
```

The subbucket replay proves branch1 activation dominates the activation/requant bucket.
