# STAGE34 FINAL REPORT

classification: `stage34-vmadot-throughput-ceiling-no-pipeline-win`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE34-VMADOT-CV2-PIPELINED-MICROKERNEL-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `6c64bdd1f9e00359c4c0a084926a75f338252a3d`
end_head: `final-head-copy-in-result-packet-after-local-commit`
pushed: `false`
full_engine_implemented: `false`
ncnn_source_mutated: `false`
production_claim_made: `false`

## Scope

Selected `/model.4` same-input ONNX-cut path only.

Target considered:

```text
/model.4/cv2/conv/Conv
1x1, 80x80, C_in=96, C_out=128
current accepted mainline: signed-storage s8xs8 smt.vmadot MMT4D + explicit correction
```

No graph expansion, no full YOLO26 engine, no full-image/camera path, no COCO/mAP, no model FPS, and no production/default-backend claim.

## Step 0 Result

Stage34 added `bench_stage34_vmadot_throughput` to test named `smt.vmadot` issue/loop/register-blocking shapes before any `/model.4/cv2` candidate.

Existing accepted wrapper path:

```text
bench_vmadot_microkernel ime_direct_status: 0
ime_direct_mean_ns_per_call: 49.487
ime_direct_stddev_ns_per_call: 0.025
```

Stage34 direct inline/register-blocked diagnostic cases:

```text
dependent_chain_1acc_loadfree: SIGILL / rc=132
load_included_1acc: SIGILL / rc=132
independent_2acc_high_loadfree: SIGILL / rc=132
independent_2acc_loadfree: SIGILL / rc=132
independent_4acc_loadfree: SIGILL / rc=132
independent_6acc_loadfree: SIGILL / rc=132
safe_vset_each_1acc: SIGILL / rc=132
safe_vset_each_2acc_high: SIGILL / rc=132
exact_single_wrapper_shape: SIGILL / rc=132
```

The diagnostic assembled and disassembled symbolically, but the attachable direct inline shapes were not board-executable. Stage34 therefore did not implement or select a `/model.4/cv2` pipelined candidate.

## Baseline Replay

Stable board replay:

```text
taskset -c 0-3
warmup=10 runs=100 repeats=5
merge_repair=branch1_add_lut
output_quantize=rvv
thread_branch0=4 thread_branch1=4 thread_model4_cv2=4
```

Correctness:

```text
status: 0
mismatches: 0
max_abs_diff: 0
output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
frm_sweep: pass RNE/RTZ/RDN/RUP/RMM
affinity_ok: 1
```

Timing:

```text
mean_total_us: 40178.5
stddev_total_us: 283.996
cv_total_pct: 0.706837
mean_conv_us: 25585.8
mean_model4_cv2_conv_us: 12096.5
mean_model4_cv2_compute_us: 8071.68
mean_model4_cv2_correction_us: 1753.73
mean_output_quantize_us: 7070.4
mean_thread_overhead_us: 5243.32
mean_attribution_pct: 99.9385
```

## Decision

selected_candidate: `none`

Reason:

```text
The accepted wrapper path remains executable, but the required direct inline/register-blocked smt.vmadot software-pipeline shapes are not a safe board-executable substrate in Stage34.
```

Do not continue trying to force `smt.vmadotus` for `/model.4/cv2` in this stage. Stage33 already proved it correct but slower.

## Validation

```text
host_native_build: pass
host_ctest: pass (42/42)
riscv_cross_build_Y26_K1X_ENABLE_IME_ON: pass
board_CPU0_3_correctness: pass
board_stable_benchmark: pass
frm_rounding_regression: pass
same_input_onnx_cut: pass
```

## Next Recommended Step

`BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE35-OUTPUT-QUANTIZE-OR-THREAD-OVERHEAD-LOCAL-REPAIR-001`

Recommended decision gate:

```text
Re-attribute output_quantize_us (~7070 us) and thread_overhead_us (~5243 us total), then choose exactly one exact local repair lane.
```

## Non-claims

```text
This is not full YOLO26 inference.
This is not model FPS.
This is not full-image/camera performance.
This is not COCO/mAP.
This is not production/default-backend readiness.
```
