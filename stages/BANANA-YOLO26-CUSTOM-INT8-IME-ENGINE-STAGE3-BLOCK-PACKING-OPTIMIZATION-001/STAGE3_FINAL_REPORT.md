# Stage 3 Final Report

classification: `stage3-conv-real-node-correct-but-packing-dominates`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE3-BLOCK-PACKING-OPTIMIZATION-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `d242181cfb7219a26c07925fe7e927e7aa71a603`
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false

## What Changed

- Added stage-local ONNX/ORT oracle extraction tooling; `xslim` was not used.
- Selected real Conv nodes from accepted Q/DQ artifact.
- Added prepacked-B MMT4D Conv1x1/Conv3x3 IME APIs.
- Added caller-owned A-panel workspace path.
- Added real selected-node `uint8` activation to signed-storage correction.
- Added real Conv fixture generated from ONNX CPU oracle.
- Added Stage 3 board-capable real Conv fixture test.
- Added Stage 3 packing microbench.

## Proven

- Real Conv1x1 selected node correctness passes on host and board.
- Real Conv3x3 selected node correctness passes on host and board.
- Conv1x1 prepacked IME improves over scalar and old wrapper for measured shape.
- Board IME execution stayed on CPU0-3 via `taskset -c 0-3`.

## Broken

- Conv3x3/im2col packing still dominates.
- Stage 3 is not ready for first graph-block integration.

## Unknown

- Full YOLO26 inference speed.
- Full-image pipeline speed.
- COCO/mAP.
- Accuracy after downstream activations and full graph requant.

## Validation

- Native host build/CTest: pass, `15/15`.
- RISC-V cross build: pass.
- Board baseline smoke: pass.
- Board real Conv fixture: pass.
- Board packing microbench: completed.

## Next

Recommended Stage 4:

`BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE4-PACKING-REPAIR-001`
