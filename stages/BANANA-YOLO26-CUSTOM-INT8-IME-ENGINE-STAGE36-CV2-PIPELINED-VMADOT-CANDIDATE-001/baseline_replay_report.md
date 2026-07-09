# Baseline Replay Report

Baseline mode: `branch1_add_lut`
Selected-cut path: current `/model.4` same-input ONNX cut path before Stage36 cv2 pipelined candidate.

Protocol:

- board affinity: `taskset -c 0-3`
- warmup: 10
- runs: 100
- repeats: 5
- output quantize: `rvv`
- thread counts: branch0=4, branch1=4, model4_cv2=4

Result:

| metric | value |
| --- | ---: |
| status | 0 |
| mismatches | 0 |
| max_abs_diff | 0 |
| output_sha256 | 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433 |
| mean_total_us | 37341.1 |
| stddev_total_us | 405.161 |
| cv_total_pct | 1.08503 |
| mean_conv_us | 23206.8 |
| mean_activation_requant_us | 2932.35 |
| mean_merge_us | 2104.46 |
| mean_output_quantize_us | 6603.46 |
| mean_thread_overhead_us | 4797.69 |
| mean_attribution_pct | 99.9258 |

Raw evidence:

- `/data/ncnn-logs/ai-team/2026-07-09_05-29-08/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE36-CV2-PIPELINED-VMADOT-CANDIDATE-001/run_logs/stage36_stable_candidates.log`
