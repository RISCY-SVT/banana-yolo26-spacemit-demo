# VMADOT Throughput / Latency Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE35-VMADOT-SIGILL-EMISSION-REPAIR-AND-THROUGHPUT-001`

## Mandatory Protocol

```text
board: Banana-Pi BPI-F3 / SpacemiT K1X
cpu: CPU0
warmup: 10
runs/iterations: 100
repeats: 5
timing_source: std::chrono::steady_clock
scope: vmadot microkernel diagnostic only
```

Source:

```text
artifacts/vmadot_throughput_gate_cpu0.tsv
```

## Mandatory Results

| case | status | mismatches | mean_ns_per_iteration | ns_per_vmadot |
| --- | ---: | ---: | ---: | ---: |
| A0 existing helper | 0 | 0 | 27.418 | 27.418 |
| A1 raw single accumulator dependent chain | 0 | 0 | 4.918 | 4.918 |
| A2 raw single accumulator load included | 0 | 0 | 6.166 | 6.166 |
| A3 raw independent 2 accumulators | 0 | 0 | 5.000 | 2.500 |
| A4 raw independent 4 accumulators | 0 | 0 | 5.000 | 1.250 |
| A5 raw independent 6 accumulators | 0 | 0 | 5.168 | 0.861 |

All rows:

```text
trap: 0
CPU: 0
```

## Supplemental High-Iteration Diagnostic

The mandatory `runs=100` protocol is intentionally recorded, but it is coarse for sub-10 ns loops. A supplemental diagnostic used `iterations=100000`, `warmup=10`, `repeats=5` to reduce timer quantization.

Source:

```text
artifacts/vmadot_throughput_high_iteration_cpu0.tsv
```

Key rows:

| case | mean_ns_per_iteration | ns_per_vmadot |
| --- | ---: | ---: |
| A1 raw single accumulator dependent chain | 3.77661 | 3.77661 |
| A2 raw single accumulator load included | 5.05265 | 5.05265 |
| A3 raw independent 2 accumulators | 3.75153 | 1.87576 |
| A4 raw independent 4 accumulators | 3.75169 | 0.937923 |
| A5 raw independent 6 accumulators | 3.75178 | 0.625296 |

## Interpretation

```text
vmadot_pipelined: yes/likely for tested raw independent accumulator loops
independent_accumulators_improve_throughput: yes in microbench
board_executable_shapes:
  v28
  v24
  v20
  v28/v30
  v20/v22/v24/v26
  v16/v18/v20/v22/v24/v26
register_shape_ceiling: not observed for tested shapes
named_encoding_status: pass
raw_encoding_status: pass
```

This is microkernel evidence only. It is not selected-cut timing, not full YOLO26 FPS, and not proof that a `/model.4/cv2` kernel candidate will beat the existing MMT4D path after real loads, correction, writeback, and threading overhead.
