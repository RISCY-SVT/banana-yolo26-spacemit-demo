# Stage 6 Final Report

classification: `stage6-multiblock-ready-for-backbone-subset-stage`

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE6-MULTI-BLOCK-BACKBONE-SUBSET-001`

repo: `/data/banana-yolo26-spacemit-demo`

branch: `yolo26-custom-int8-engine`

start_head: `fcbcd0fa72e3649d85ec2281bf8dce8dc92e78da`

pushed: false

full_engine_implemented: false

ncnn_source_mutated: false

production_claim_made: false

XSlim used: false

## Selected Subset

Selected subset id: `candidate_C_block0_silu_model1_conv`

Boundary:

```text
images Q/DQ
/model.0/conv/Conv
Conv0 Q/DQ
/model.0/act/Sigmoid
/model.0/act/Mul
Act0 Q/DQ
/model.1/conv/Conv
```

Output boundary: corrected int32 output of `/model.1/conv/Conv`.

Deferred:

- `/model.1` output Q/DQ and activation
- `/model.2/cv1/conv/Conv`
- `/model.2/Split`
- graph-wide scheduling

## Implementation

Added a narrow Stage 6 runner:

- `custom_int8_engine/include/y26_k1x_multiblock_runner.h`
- `custom_int8_engine/src/multiblock_runner.cpp`

The runner reuses Stage 4 persistent weight prepack and reusable Conv workspace, keeps scalar and IME paths, and implements Conv0 activation/requant as an explicit scalar float fallback. The measured hot loop does not allocate after `prepare`.

Added tooling and fixtures:

- `custom_int8_engine/tools/extract_stage6_multiblock_oracle.py`
- `custom_int8_engine/tests/stage6_multiblock_fixture.h`
- `custom_int8_engine/tests/test_stage6_multiblock_runner.cpp`
- `custom_int8_engine/tools/bench_stage6_multiblock.cpp`

## Oracle

ONNX CPU oracle: pass.

| case | Conv0 max abs diff vs ORT ROI | Conv1 max abs diff vs ORT ROI |
| --- | ---: | ---: |
| `synthetic_seeded` | `3.814697265625e-06` | `1.1444091796875e-05` |
| `synthetic_gradient` | `1.9073486328125e-06` | `3.814697265625e-06` |

Exact C++ fixtures compare corrected int32 and int8 activation handoff.

## Validation

- Host-native CMake build with `/usr/bin/g++`: pass
- Host CTest: `18/18` pass
- RISC-V cross build with `Y26_K1X_ENABLE_IME=ON`: pass
- Board CPU0/1/2/3 Stage 6 correctness: pass, mismatches `0`
- Board CPU0 Stage 6 microbench: pass
- `git diff --check`: pass
- symlink scan: no symlinks under `custom_int8_engine` or `stages`
- secret-like scan: no findings

## Board Microbench

Command:

```text
taskset -c 0 ./bench_stage6_multiblock 3
```

Results:

| metric | value |
| --- | ---: |
| scalar total mean | `1009980 us` |
| IME total mean | `419769 us` |
| speedup IME vs scalar | `2.41x` |
| Conv0 IME component | `67775.3 us` |
| activation/requant fallback | `286942 us` |
| Conv1 IME component | `63886.5 us` |
| Stage 5 Conv0 replay IME | `70203.2 us` |

This is selected-subset evidence only, not full YOLO26 inference or model FPS.

## Known Limitations

- Activation/requant fallback is scalar float and dominates the selected-subset IME time.
- Conv1 output activation/requant and `/model.2` branch handling are not integrated.
- No full engine, camera, COCO/mAP, ncnn integration, XSlim, production claim, or model FPS claim was made.

## Next Recommended Step

`BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE7-BACKBONE-SUBSET-EXPANSION-001` after review/approval.

