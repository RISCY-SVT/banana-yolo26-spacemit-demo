# Stage 7 Final Report

classification: `stage7-backbone-subset-correct-but-activation-dominates`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE7-BACKBONE-SUBSET-EXPANSION-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `246d6011865d5cd246e8a701c501c14f1193a060`
end_head: `889b6717f1df74459c715342167588ca8b9d9100`
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false
xslim_used: false
vmadot_sliding_used: false

## Selected Subset

`candidate_D_block0_silu_model1_silu_model2_cv1_conv`

Boundary:

- images Q/DQ
- `/model.0/conv/Conv`
- Conv0 Q/DQ
- `/model.0/act/Sigmoid`
- `/model.0/act/Mul`
- Act0 Q/DQ
- `/model.1/conv/Conv`
- Conv1 Q/DQ
- `/model.1/act/Sigmoid`
- `/model.1/act/Mul`
- Act1 Q/DQ
- `/model.2/cv1/conv/Conv`

Output boundary: corrected int32 output of `/model.2/cv1/conv/Conv`.

## Proven

- Stage 5/6 evidence recovered from repo-local reports.
- Accepted CPU-good manual Q/DQ ONNX artifact recovered and inspected.
- ONNX CPU oracle extraction passed for two deterministic fixtures.
- Host-native CTest: `19/19` pass.
- RISC-V cross build: pass.
- Board CPU0/1/2/3 Stage 7 correctness: pass, mismatches `0`.
- Stage 7 selected-subset CPU0 microbench: scalar `1.22366e+06 us`, IME `593347 us`, speedup `2.0623x`.
- Stage 6 replay on same board run: IME `418971 us`, matching prior evidence scale.

## Broken / Residual

- Activation/requant fallback dominates Stage 7 IME total: `436780 us`, `73.6129%`.
- `/model.2/cv1/conv` output Q/DQ and activation remain deferred.
- `/model.2/Split` and branch handling remain deferred.
- No full engine, no graph scheduler, no full-image pipeline.

## Unknown

- Full YOLO26 inference speed.
- COCO/mAP and accuracy beyond selected oracle fixtures.
- Best activation/requant implementation strategy.

## Next

Recommended next stage: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE8-ACTIVATION-REQUANT-OPTIMIZATION-001` after review/approval.
