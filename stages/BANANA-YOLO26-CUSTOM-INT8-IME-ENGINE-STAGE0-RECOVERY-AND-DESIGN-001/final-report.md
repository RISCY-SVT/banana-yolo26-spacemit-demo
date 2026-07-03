# Stage 0 Final Report

classification: stage0-recovery-design-skeleton-complete-ready-for-microkernel-stage-with-caveats
stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE0-RECOVERY-AND-DESIGN-001
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
base_branch: yolo26-rd-bootstrap
start_head: 9c307f8a2d2fed5f39375ebacb0dbc92b59a0510
end_head: 9c307f8a2d2fed5f39375ebacb0dbc92b59a0510
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false
output_contract_decision: traditional
cpu_good_qdq_artifact: /data/banana-yolo26-spacemit-demo/.deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx
quantization_audit_status: pass
cluster0_safety_plan: created
static_model_format_v0: created
skeleton_created: true
stage1_prompt_created: true
big_artifacts_path: /data/banana-yolo26-spacemit-demo/.deps/custom_int8_engine/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE0-RECOVERY-AND-DESIGN-001
log_dir: /data/ncnn-logs/ai-team/2026-07-03_09-07-17/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE0-RECOVERY-AND-DESIGN-001
next_recommended_step: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE1-SMTVMADOT-MICROKERNEL-001 after review/approval
timestamp: 2026-07-03_09-07-17

## Summary

Stage 0 recovered the accepted `smt.vmadot` evidence, selected the CPU-good
manual ORT Q/DQ artifact, audited Q/DQ signedness/zero-points, chose a
traditional/trunk-first implementation contract, defined static model format
v0, created a standalone C++ skeleton, and prepared the Stage 1 microkernel
prompt.

`xslim` was not used as authority or model source because the user reported a
YOLO26 bug in that path.

## Broken / Proven / Unknown

Proven:

- `smt.vmadot` is accepted only as cluster0 CPU0-3 `s8 x s8 -> s32` 4x4x8 with exact scalar oracle evidence.
- `manual_e2e_rep_conv_matmul_qdq.onnx` is present and matches the repo docs CPU-good/blank-clean acceptance claim.
- Stage 0 native host skeleton builds and 5/5 unit tests pass with `/usr/bin/g++`.

Broken:

- rt204 SpaceMIT EP remains blocked for CPU-good YOLO26 Q/DQ Conv path by `output_type not implemented for clip minmax`.
- The first validation attempt inherited `CC/CXX=/opt/riscv/...`; its generated `build-host` directory was removed after it produced invalid test-side filenames when RISC-V ELF binaries were accidentally interpreted by the host shell.
- e2e head is control-heavy (`TopK`, `ReduceMax`, `GatherElements`) and is not selected as the first engine contract.

Unknown:

- Best graph-level correction path for asymmetric activation `MatMul` into signed `smt.vmadot`.
- Final full-image speed, mAP, production readiness, and ncnn integration.

## Files Created Or Modified

- `custom_int8_engine/`: standalone runtime skeleton, scalar oracle fixtures, and CMake tests.
- `app/backend_custom_int8/README.md`: backend placeholder only.
- `scripts/bench_custom_int8.sh`: benchmark protocol placeholder.
- `models/custom_int8/schema.md`: schema pointer.
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE0-RECOVERY-AND-DESIGN-001/`: Stage 0 reports, TSV audits, and command log.

## Validation

- Bad cross-compiled `.deps/custom_int8_engine/build-host` and temporary `build-host-native` were removed.
- `env -u CC -u CXX LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 cmake -S custom_int8_engine -B .deps/custom_int8_engine/build-host -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=/usr/bin/g++`: pass.
- `env -u CC -u CXX LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 cmake --build .deps/custom_int8_engine/build-host -j$(nproc)`: pass.
- `env -u CC -u CXX LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 ctest --test-dir .deps/custom_int8_engine/build-host --output-on-failure`: pass, 5/5.
- Post-fix suspicious path scan under touched areas and `.deps/custom_int8_engine`: pass, issue_count 0.
- `commands.txt` files were sanitized after the failed cross-run; UTF-8 valid and control byte count is 0.
- `git diff --check`: pass.
- `find . -type l -print`: ran; printed pre-existing symlinks under ignored `.deps`, `.cache`, and `third_party/vendor`.
- `scan-export-candidates.sh` on task-run artifacts: pass.

## Human Decision Needed

Approve Stage 1 before any IME asm implementation. Stage 1 should implement
only the `smt.vmadot` microkernel and scalar oracle. Full engine integration,
asymmetric graph correction, ncnn integration, and production benchmarking
remain out of scope.
