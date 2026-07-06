# Model4 Fullshape Timing Gate Report

selected_subset: `candidate_K_model4_c2f_representative_fullshape_synthetic`
shape_class: `representative_full_shape_model4_c2f_synthetic`
protocol: `warmup=10 runs=100 repeats=5`
board_affinity: `taskset -c 0-3`
status: `pass`

## Correctness

All measured candidates reported:

```text
status=0
mismatches=0
checksum=-3094964234
affinity_ok=1
```

## Baseline Gate

Best pre-repair threaded candidate:

```text
candidate: B1_threaded_branch0_4t
mean_total_us: 149539
stddev_total_us: 76.3189
cv_total_pct: 0.0510361
mean_conv_us: 52979.2
mean_activation_requant_us: 29780.3
mean_merge_us: 66564.3
conv_share_pct: 35.4283
activation_share_pct: 19.9147
merge_share_pct: 44.513
```

This made merge/post-Concat-QDQ the dominant local bucket before repair.

## Selected Repair Result

```text
candidate: C2_split0_concat_lut_4t
mean_total_us: 116338
stddev_total_us: 121.933
cv_total_pct: 0.104809
mean_conv_us: 52867.5
mean_activation_requant_us: 29765.2
mean_merge_us: 29791.6
conv_share_pct: 45.443
activation_share_pct: 25.5851
merge_share_pct: 25.6078
```

This is selected-subset evidence only, not YOLO26 full-model FPS or full-image/camera performance.
