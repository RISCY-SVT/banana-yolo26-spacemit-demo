# Microbench Report

This is a microkernel-only benchmark, not a YOLO26 inference benchmark.

## Board command

```bash
taskset -c 0 /home/svt/contcodex/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE1-SMTVMADOT-MICROKERNEL-001/bench_vmadot_microkernel 10000 5
```

## Protocol

- CPU affinity: CPU0 only
- warmup: 1000 calls
- iterations: 10000 calls per repeat
- repeats: 5
- packing included: no
- scalar compiler flags: cross Release build with `-march=rv64gcv_zvfh -mabi=lp64d`
- IME route: named `smt.vmadot`

## Result

```text
benchmark_scope=microkernel_only_not_yolo26_inference
iterations=10000
warmup=1000
repeats=5
packing_included=no
public_guard_overhead_included_for_public_api=yes
direct_ime_guard_overhead_included=no
scalar_mean_ns_per_call=254.827
scalar_stddev_ns_per_call=5.193
ime_status=0
ime_mean_ns_per_call=2324.896
ime_stddev_ns_per_call=6.967
public_guarded_speedup_vs_scalar=0.110
ime_direct_status=0
ime_direct_mean_ns_per_call=31.624
ime_direct_stddev_ns_per_call=0.805
direct_speedup_vs_scalar=8.058
```

Interpretation:

- The public guarded API includes `sched_getcpu` and SIGILL handler setup on each call, so it is intentionally not the hot-loop timing target.
- The benchmark-only direct IME body excludes packing and per-call safety guard overhead. It is the relevant microkernel body timing.
- No model-level FPS, full-image pipeline speed, COCO mAP, or production readiness claim is made.
