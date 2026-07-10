# Stage42 Final Report

classification: stage42-host-oracle-policy-selected-model16-oracle-ready
run_attempt: 2
start_head: 6559e2a4a146e96df9db37bf748808896d08e147
end_head: pending-local-commit-see-final-response
workspace_inheritance_mode: task-local-continuation
workspace_inheritance_status: stage41-uncommitted-input
push: false

## Proven

1. Stage41 host exactness reproduces under host ORT 1.27.0 with the accepted operational session contract: full output0, custom scalar model4, and custom-through-suffix all have zero mismatches.
2. The graph-verified model4 contract is uint8 NCHW `1x64x80x80` in and uint8 NCHW `1x128x80x80` out; custom adapters use NHWC.
3. With one identical saved model4 input, host scalar equals board scalar exactly, board scalar equals board IME exactly, and both board custom arms equal the fixed host operational oracle exactly.
4. Host and board ORT do not match. Under `ORT_DISABLE_ALL`, the float first-Conv output is byte-identical, then the first QuantizeLinear boundary differs in 5/1,638,400 uint8 codes by one.
5. Verbose board logs observe all six model.0 diagnostic nodes on CPUExecutionProvider. Provider registration and assignment are reported separately.
6. Reference Policy B is selected: fixed host ORT is correctness authority; board ORT is fallback/integration/debug/timing only.
7. The corrected benchmark excludes full/model4 ORT reference execution, comparisons, and file I/O from the custom hot loop. Its attribution is 99.998379% with 0.132606% CV.
8. The model16 semantic and quantized fixed-host oracle package is complete and replayed diagnostically on board.
9. Existing IME route is present in disassembly, executes on CPU0-3 only, passes the same-input oracle, and restores ambient RNE/RTZ/RDN/RUP/RMM.

## Broken Or Scoped

- Board ORT 1.20.2+spacemit is not byte-compatible with host ORT 1.27.0 for this model. Same-input model4 ALL differs in 77,196/819,200 bytes, max diff 2.
- Board operational full ORT differs from fixed host output0 in 1597/1800 floats; the board custom pipeline differs in 1604/1800. These are cross-runtime diagnostics, not custom integer-boundary failures.
- The cross runner embeds an absolute build-tree RPATH. Controlled loader proof passes, but deployable packaging remains future work.
- The inherited Stage41 runner still includes the model4 fixture implementation source from `tests/`. Moving the fixture factory into a normal internal source is deferred technical debt because it is unrelated to the numerical contract repair.

## Unknown

- The vendor QuantizeLinear source-level cause is not proven; the first divergent boundary is localized, not reverse-engineered.
- Full-graph provider assignment was not profiled; CPU assignment is directly observed for the decisive model.0 cut.
- Individual model5-8 board costs remain unknown because Stage41 cumulative-session subtraction is not an isolated profile.
- Production allocator, final ORT-free scheduler, and deployment layout are outside Stage42.

## Correctness Policy

```text
authority: host ORT 1.27.0
optimization: ORT_ENABLE_ALL
execution: sequential
intra/inter threads: 1/1
memory pattern/CPU arena/spinning: enabled
integer custom gate: exact bytes, no tolerance
board ORT role: integration/debug/timing scaffold
```

Primary same-input model4 results:

```text
host ORT vs board ORT:       mismatches=77196 max_abs_diff=2
host scalar vs board scalar: mismatches=0 max_abs_diff=0
board scalar vs board IME:   mismatches=0 max_abs_diff=0
host ORT vs board IME:       mismatches=0 max_abs_diff=0
```

## Timing Status

Board protocol: `taskset -c 0-3`, warmup 10, runs 100, repeats 5, performance governor at 1.6 GHz.

```text
custom_pipeline_mean_us: 826008.582826
custom_pipeline_stddev_us: 1095.337938
prefix_us: 229662.287042
layout_in_us: 5293.200918
custom_model4_us: 25149.098496
layout_out_us: 11838.504756
suffix_us: 554052.102166
full_board_ort_mean_us: 796866.390970
same_input_custom_ime_mean_us: 26815.626798
same_input_custom_scalar_mean_us: 378322.274994
same_input_board_ort_model4_mean_us: 37072.764532
```

All timing is scaffold/component evidence. It is not model FPS, production latency, camera/full-frame performance, or accuracy.

## Model16 Oracle

Semantic and quantized contracts, cuts, input/output tensors, quantization metadata, hashes, runtime contract, and replay results are in the model16 reports. Board ORT replay differs, as expected under Policy B. No optimized model16 code was written or authorized.

## Implementation Authorization

Stage42 authorizes no production/default dispatch, new ISA lane, optimized model16, full custom engine, CPU4-7 IME, camera, COCO/mAP, source publication, or push.

## Next Readiness

Selected next objective: a Stage43 fixed-host oracle and isolated board profile for the contiguous model5-8 island, followed by at most one byte-exact first-block implementation. The model16 oracle remains ready as a separate reusable-C2f proof asset.

Raw log: `/data/ncnn-logs/ai-team/2026-07-10/2026-07-10_06-32-46__codex__BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE42-INPROCESS-ORT-CONTRACT-REPAIR-AND-MODEL16-ORACLE-GATE-001__stage42-technical-rerun`

Result packet: `/exchange/results/outbox/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE42-INPROCESS-ORT-CONTRACT-REPAIR-AND-MODEL16-ORACLE-GATE-001`
