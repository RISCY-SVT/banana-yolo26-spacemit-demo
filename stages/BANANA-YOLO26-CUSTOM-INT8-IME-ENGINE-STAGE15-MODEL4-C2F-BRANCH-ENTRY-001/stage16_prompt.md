# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE16-MODEL4-C2F-COMPLETION-AND-FULLSHAPE-GATE-001

User-facing summaries must be in Russian. Code, comments, identifiers, commands, paths, report filenames, and artifact names must stay in English.

## Mission

Complete the bounded `/model.4` C2f block after Stage 15 branch-entry correctness, while adding a representative/full-shape selected-subset timing gate.

Start from:

`candidate_I_model4_split_first_branch`

Attempt in gates:

1. `/model.4/m.0/cv1/act` replay.
2. `/model.4/m.0/cv2/conv/Conv`.
3. `/model.4/m.0/cv2/act`.
4. `/model.4/m.0/Add` only after float-domain contract/oracle.
5. `/model.4/Concat` only after contract/oracle.
6. post-Concat Q/DQ.
7. `/model.4/cv2/conv/Conv` only after merge correctness.

## Required Carry-Forward Facts

- Stage 15 compact correctness passed on host and board CPU0-3.
- Stage 15 full-shape timing is not proven.
- Stage 14 compact `139.04 us` and Stage 15 compact `160.038 us` are not full-shape or model-FPS evidence.
- Use A2 RVV activation path only with boundary-specific LUT oracle and explicit RNE.
- Preserve Stage13/14 merge dataflow lessons, but select per boundary by correctness, timing, and maintainability.

## Non-Actions

Do not:

- implement full YOLO26 inference;
- create a graph-wide scheduler;
- run full-image/camera demo;
- run COCO/mAP;
- claim model FPS or production readiness;
- mutate `/data/ncnn`;
- use XSlim;
- implement `vmadot1/2/3`, `vmadotn`, FP/vfmadot;
- enable default multithreading or CPU4-7 IME.

## Gates

Gate 16A:

- replay Stage 15 CPU0-3 correctness;
- prove representative/full-shape timing strategy or explicitly scope it.

Gate 16B:

- extract ONNX CPU oracle for `/model.4/m.0/cv2/conv/Conv` and activation.

Gate 16C:

- implement and validate branch cv2 Conv/activation.

Gate 16D:

- discover Add/Concat contracts before any merge implementation.

Gate 16E:

- board CPU0-3 correctness and CPU0 selected-subset microbench.

Recommended classification if full-shape timing remains unavailable:

`stage16-model4-c2f-correct-but-fullshape-unproven`
