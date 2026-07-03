# Stage 5 Final Report

classification: `stage5-first-block-ready-for-multiblock-stage`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE5-FIRST-BLOCK-INTEGRATION-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `55f133ca0900cb91d891a77149479b6fd392c420`
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false
xslim_used: false

## Result

Stage 5 integrated the Stage 4 real Conv kernel path into the first coherent graph block boundary:

- selected block: `block0_conv_only`
- selected node: `/model.0/conv/Conv`
- shape: `640x640x3 -> 320x320x16`
- kernel: `3x3`, stride `2`, padding `1`

The implementation is still a narrow block runner, not a full YOLO26 engine and not a graph scheduler.

## Proven

- Stage 4 state recovered from local reports.
- ONNX CPU oracle created with isolated `.deps/custom_int8_engine/venv-stage5-onnx`.
- Host-native CTest after Stage 5 changes: `17/17` pass.
- RISC-V cross build with `Y26_K1X_ENABLE_IME=ON`: pass.
- Board runtime probe: pass.
- Stage 1 `smt.vmadot` board fixture: pass.
- Stage 4 packing repair board smoke: pass.
- Stage 5 first-block correctness on CPU0/1/2/3: pass.
- Board block microbench: scalar `463480 us`, IME total packing included `71932.7 us`.

## Broken / Residual

- Downstream SiLU activation (`Sigmoid` + `Mul`) is not integrated.
- Output requant and full graph activation policy are deferred.
- PackA remains visible at `38912.7 us` inside the selected block timing.
- Real preprocessed `.npy` input was not found; two deterministic synthetic inputs were used for oracle fixtures.

## Unknown

- Full YOLO26 inference speed.
- Full-image pipeline behavior.
- COCO/mAP.
- Accuracy after integrating downstream activation/requant and more graph blocks.

## Next

Recommended next stage:

`BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE6-MULTI-BLOCK-BACKBONE-SUBSET-001`
