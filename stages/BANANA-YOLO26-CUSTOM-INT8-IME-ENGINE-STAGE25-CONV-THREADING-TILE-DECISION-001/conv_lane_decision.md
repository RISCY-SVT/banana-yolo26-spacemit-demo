# Conv Lane Decision

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE25-CONV-THREADING-TILE-DECISION-001

## Selected Lane

selected_conv_lane: C1
selected_candidate: C1_thread_branch1_and_model4_cv2_4t
classification: stage25-conv-threading-expand-selected

## Evidence

Stage24 selected-path replay showed Conv was dominant:

```text
total_us: 125176
conv_share_pct: 49.5863
activation_share_pct: 26.0372
merge_share_pct: 16.7456
```

Stage25 threaded the two remaining large Conv nodes in the selected cut path:

```text
/model.4/m.0/cv2/conv/Conv:
  0-thread-sidecar baseline: 17902.8 us
  4-thread: 6059.11 us
  speedup: 2.9547x

/model.4/cv2/conv/Conv:
  0-thread-sidecar baseline: 37545.2 us
  4-thread: 12037.1 us
  speedup: 3.1191x
```

The final selected C1 path:

```text
total_us: 89178.9
stddev_total_us: 268.184
mismatches: 0
max_abs_diff: 0
output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
frm_sweep: pass
affinity_ok: 1
```

## Rejected Lanes

```text
C2_tile_prepack: rejected for this stage because C1 already moved the dominant Conv bucket and Conv is no longer largest after C1.
C3_vmadot123: not authorized and not needed as immediate next lane after C1.
C4_no_conv_repair: rejected because Conv was dominant in same-session replay.
```

## Next Bucket

After C1:

```text
activation_share_pct: 36.7808
merge_share_pct: 23.5081
conv_share_pct: 29.3389
```

Stage26 should target activation/requant on the same `/model.4` cut path before graph expansion.
