# Stage17 Replay Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE18-CLUSTER0-THREADED-CONV-INTEGRATION-001`
previous_stage: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE17-CONV-IME-ROOFLINE-AND-CLUSTER0-THREADING-FEASIBILITY-001`

## Recovered Stage17 Evidence

```text
classification: stage17-threading-positive-ready-for-threaded-conv-integration
start_head: 3ca94995a7b18276dac9ac7c660061cd93377994
end_head: 92e7d8763295cc514b09d0db32ed2795b092dc44
protocol: warmup=10 runs=100 repeats=5
stage16a single-thread IME mean_total_us: 25670.974780
stage16a single-thread IME stddev_total_us: 345.670192
single-thread Conv node: /model.4/m.0/cv1/conv/Conv
single-thread Conv mean_us: 20458.001284
MAC_count: 29491200
effective_GMAC_s: 1.441548
bottleneck_class: structural_low_K_or_packing
threading matrix 1/2/3/4 threads: 20550.03 / 10750.27 / 7258.03 / 5583.81 us
4-thread speedup_vs_1thread: 3.680290
mismatches: 0
checksum: 1324192976
no CPU4-7 IME execution
```

## Stage18 Replay

Stage18 replayed the single-thread selected mode with the same stable protocol.

```text
candidate: A0_stage17_single_thread_replay
correctness_status: pass
mismatches: 0
checksum: 1324192976
mean_total_us: 25588.395318
stddev_total_us: 3.740899
mean_conv_us: 20414.513636
stddev_conv_us: 10.350443
mean_activation_requant_us: 4985.479680
```

The Stage18 replay is within the Stage17 evidence range. Stage18 performance decisions use the Stage18 stable replay as baseline.
