# Benchmark Protocol Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE17-CONV-IME-ROOFLINE-AND-CLUSTER0-THREADING-FEASIBILITY-001`

## Accepted Protocol

```text
pin: main thread CPU0; worker threads CPU0-3 for threading matrix
warmup: 10
runs: 100
repeats: 5
report: mean_us, stddev_us, min_us, max_us, coefficient_of_variation
```

The final accepted run used:

```text
taskset -c 0-3 ./bench_stage17_roofline_threading 10 100 5
```

The benchmark tool pins the main thread to CPU0 internally. Threading workers are explicitly pinned to CPU0, CPU1, CPU2, and CPU3 according to the thread-count matrix.

## Stage16 One-Shot Comparison

Stage16A used one board run:

```text
iterations=1
warmup=0
repeats=0
stddev=not available
```

Replayed Stage16A one-shot on CPU0:

```text
candidate=stage16A_IME_A2_rvv_f32_lut
total_us=25467.3
conv_us=20323.4
activation_requant_us=4960.69
mismatches=0
```

Stage17 stable replay is the only performance basis for Stage17 decisions.
