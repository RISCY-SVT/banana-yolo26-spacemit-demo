# Stage32 Replay Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE33-MMT4D-MIXED-SIGNEDNESS-CORRECTION-PROOF-001

baseline_mode: `branch1_add_lut`
runtime: board `bf3`, `taskset -c 0-3`
protocol: `warmup=10 runs=100 repeats=5`

## Correctness

```text
status: 0
mismatches: 0
max_abs_diff: 0
checksum: 106597930
expected_checksum: 106597930
output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
affinity_ok: 1
frm_sweep: pass for RNE/RTZ/RDN/RUP/RMM
```

## Timing

| bucket | mean_us |
|---|---:|
| total | 40380.4 |
| conv | 25447.5 |
| activation_requant | 2980.39 |
| merge | 2133.42 |
| output_quantize | 7071.04 |
| thread_overhead | 5113.7 |
| correction | 2449.85 |
| conv_compute | 17929.6 |
| model4_cv2_conv | 11852.7 |
| model4_cv2_correction | 1742.83 |
| model4_cv2_compute | 8129.4 |

Raw log:

```text
/data/ncnn-logs/ai-team/2026-07-08_14-41-34/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE33-MMT4D-MIXED-SIGNEDNESS-CORRECTION-PROOF-001/run_logs/stage32_baseline_replay_board.log
```
