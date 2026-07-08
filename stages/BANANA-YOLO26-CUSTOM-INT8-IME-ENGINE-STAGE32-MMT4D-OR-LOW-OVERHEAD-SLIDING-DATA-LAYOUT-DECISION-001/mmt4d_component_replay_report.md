# MMT4D Component Replay Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE32-MMT4D-OR-LOW-OVERHEAD-SLIDING-DATA-LAYOUT-DECISION-001
selected_path: Stage26 A3 branch1/add LUT + Stage25 C1 threaded Conv policy
mode: ime_threaded
merge_repair: branch1_add_lut
output_quantize: rvv
thread_branch0: 4
thread_branch1: 4
thread_model4_cv2: 4
protocol: warmup=10 runs=100 repeats=5
affinity: taskset -c 0-3

## Raw Log

```text
/data/ncnn-logs/ai-team/2026-07-08_12-51-18/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE32-MMT4D-OR-LOW-OVERHEAD-SLIDING-DATA-LAYOUT-DECISION-001/run_logs/stage32_selected_cut_mmt4d_replay_board.log
```

## Correctness

```text
status: 0
mismatches: 0
max_abs_diff: 0
output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
expected_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
affinity_ok: 1
frm_sweep: pass for RNE/RTZ/RDN/RUP/RMM
```

## Selected-Cut Buckets

| bucket | mean_us | share_pct | note |
|---|---:|---:|---|
| input_adapter_us | 2399.96 | 5.904 | non-Conv input load/adapter |
| conv_us | 25988.2 | 63.9366 | includes correction and thread overhead |
| activation_requant_us | 2988.19 | 7.3516 | post Stage26 A3 |
| merge_us | 2127.29 | 5.2336 | same as post_concat_qdq_us in runner output |
| output_quantize_us | 7119.27 | 17.515 | final selected-cut output QuantizeLinear |
| other_us | 23.9033 | 0.0588 | residual attribution |
| total_us | 40646.8 | 100.0 | selected `/model.4` cut only |

Attribution:

```text
mean_attributed_us: 40622.9
mean_attribution_pct: 99.9412
```

## Per-Conv Buckets

| node | total_conv_us | compute_us | correction_us | copy_us | worker_other_us |
|---|---:|---:|---:|---:|---:|
| /model.4/m.0/cv1/conv/Conv | 7489.04 | 5766.48 | 217.943 | 0 | 0.236508 |
| /model.4/m.0/cv2/conv/Conv | 6430.69 | 4334.1 | 520.71 | 0 | 0.50217 |
| /model.4/cv2/conv/Conv | 12068.4 | 8164.37 | 1758.66 | 0 | 0.576072 |
| aggregate | 25988.2 | 18265.0 | 2497.31 | 0 | 1.31475 |

Thread overhead:

```text
mean_thread_overhead_us: 5277.9
```

## Answer

The current MMT4D path is not limited by copy/writeback after Stage28 cleanup. The largest measured Conv sub-bucket is raw plain `smt.vmadot` compute, followed by thread overhead and correction. Correction is small compared with total Conv, but it is still a plausible bounded next proof lane because Stage32 proved the mixed signedness matrix dot family.
