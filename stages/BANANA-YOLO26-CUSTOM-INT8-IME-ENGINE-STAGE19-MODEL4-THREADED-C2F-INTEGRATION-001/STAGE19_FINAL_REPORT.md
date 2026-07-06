# STAGE19 FINAL REPORT

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE19-MODEL4-THREADED-C2F-INTEGRATION-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `6c4c8258fb10cf25476a8380870d624200855f9b`
end_head: `pending-local-commit-see-final-response`
pushed: `false`
full_engine_implemented: `false`
ncnn_source_mutated: `false`
production_claim_made: `false`

## Classification

classification: `stage19-model4-threaded-c2f-correct-but-compact-overhead-dominates`

## Scope

Stage19 integrated the Stage18 explicit cluster0 threaded Conv sidecar into the narrow `/model.4` C2f runner path and added a bounded activation/requant threading sidecar using the same worker pool.

No full YOLO26 engine, graph-wide scheduler, default backend switch, camera path, COCO/mAP, model FPS claim, XSlim path, vmadot1/2/3 implementation, vmadotn path, FP/vfmadot path, or CPU4-7 IME path was added.

## Selected Modes

Representative/full-shape Stage18 replay:

```text
A0_stage17_single_thread_replay
A4_integrated_threaded_conv_4t
```

Compact Stage19 C2f oracle-scope modes:

```text
A0_single_thread_c2f
A1_threaded_conv_1t
A2_threaded_conv_2t
A3_threaded_conv_3t
A4_threaded_conv_4t
A5_threaded_conv_threaded_activation_4t
```

## Representative Stage18 Replay

Stable protocol:

```text
warmup: 10
runs: 100
repeats: 5
pinning: taskset CPU0-3
```

Key replay results:

```text
A0 single-thread:
  mean_total_us: 25588.139540
  stddev_total_us: 24.851779
  mean_conv_us: 20402.617866
  mean_activation_requant_us: 5000.416242
  mismatches: 0
  checksum: 1324192976

A4 4-thread:
  mean_total_us: 11082.483550
  stddev_total_us: 96.463441
  mean_conv_us: 5905.210462
  mean_activation_requant_us: 4983.945734
  total_speedup_vs_A0: 2.308881x
  conv_speedup_vs_A0: 3.455020x
  activation_share_pct: 44.971379
  mismatches: 0
  checksum: 1324192976
```

This confirms the Stage18 representative/full-shape branch-entry threading result remains valid.

## Stage19 Compact C2f Integration

The Stage19 `/model.4` C2f selected runner currently uses compact oracle-scope fixtures, not representative/full-shape tensors. These timings are correctness and local overhead evidence only.

Stable compact protocol:

```text
warmup: 10
runs: 100
repeats: 5
pinning: taskset CPU0-3
shape_class: compact_oracle_scope
```

Key compact results:

```text
A0_single_thread_c2f:
  mean_total_us: 186.079392
  mean_conv_us: 122.624718
  branch0_conv_us: 7.192042
  activation_share_pct: 14.670393
  mismatches: 0
  checksum: -143848

A4_threaded_conv_4t:
  mean_total_us: 283.534780
  mean_conv_us: 212.156010
  branch0_conv_us: 84.853328
  thread_overhead_us: 76.068746
  total_speedup_vs_A0: 0.656284x
  branch0_conv_speedup_vs_A0: 0.084759x
  mismatches: 0
  checksum: -143848

A5_threaded_conv_threaded_activation_4t:
  mean_total_us: 461.796518
  stddev_total_us: 172.024673
  mean_activation_requant_us: 216.748804
  thread_overhead_us: 258.251034
  total_speedup_vs_A0: 0.402947x
  mismatches: 0
  checksum: -143848
```

The compact C2f thread sidecar is correct but not worth selecting for compact oracle-scope execution. The useful Stage18 full-shape Conv threading result does not automatically transfer to tiny compact fixtures.

## Validation

```text
host-native build: pass
host CTest: pass, 35/35
RISC-V cross build: pass
board Stage19 correctness CPU0-3: pass
board stable Stage18 replay: pass
board stable Stage19 compact bench: pass
git diff --check: pass
source hygiene: pass
result packet: exported after repo-local report generation
```

## Broken

```text
Stage19 compact C2f A4/A5 threading did not meet speedup gates.
A5 activation threading sidecar is correct but slower and high variance on compact tensors.
The compact Stage19 C2f fixture is not representative/full-shape timing evidence.
```

## Proven

```text
Stage18 representative/full-shape branch-entry A4 replay remains correct and fast.
The threaded Conv sidecar can be called through the model4 C2f runner without correctness regressions.
The bounded activation sidecar preserves exact outputs, but is not performance-selected.
CPU0-3 affinity checks passed; no CPU4-7 IME path was used.
```

## Unknown

```text
Representative/full-shape timing for the completed model4 C2f threaded runner remains unproven.
Whether activation/requant fusion or memory/dataflow repair wins next is not proven for full model4 C2f shapes.
Model FPS, full-image behavior, COCO/mAP, and production readiness remain unknown and were not tested.
```

## Next Recommended Step

`BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE20-ACTIVATION-REQUANT-AND-MODEL4-FULLSHAPE-REPAIR-001`

Stage20 should first produce representative/full-shape model4 C2f timing, then decide between activation/requant fusion and memory/dataflow repair. Do not open vmadot1/2/3 implementation from this Stage19 result.
