# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE11-BRANCH-BLOCK-EXPANSION-001

You are Codex working in the Banana-Pi BPI-F3 / SpacemiT K1X / X60 / YOLO26 custom INT8 IME `smt.vmadot` engine project.

User-facing summaries must be in Russian. Code, comments, identifiers, commands, paths, report filenames, artifact names, and commit messages stay in English.

## Mission

Expand the Stage 10 selected subset from:

`candidate_E_branch1_stage9_split_model2_m0_cv1_conv`

towards the next bounded branch block:

- `/model.2/m.0/cv1/conv/Conv` activation/requant
- `/model.2/m.0/cv2/conv/Conv`
- residual `/model.2/m.0/Add` only if oracle and scale contracts are clear

Do not implement full YOLO26 inference, graph-wide scheduling, camera/full-image demo, COCO/mAP, model FPS claims, ncnn source mutation, XSlim, or vmadot1/2/3/vmadotn/FP lanes.

## Required Start State

Expected branch: `yolo26-custom-int8-engine`
Expected previous stage: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE10-BACKBONE-EXPANSION-POST-ACTIVATION-GATE-001`
Expected Stage 10 classification: `stage10-backbone-expanded-ready-for-branch-stage`

## Required Gates

- Recover Stage 10 reports and CPU0-3 board correctness.
- Preserve explicit RVV RNE A2 activation path.
- Generate boundary-specific 256-code ONNX Runtime LUT oracle for every new activation/requant boundary.
- Keep Split/branch/Concat/Add timing as separate buckets.
- Use only plain `smt.vmadot` MMT4D Conv kernels.
- Keep no heap allocation in measured hot loop after prepare.
- Run host CTest, RISC-V cross build, and board CPU0-3 correctness.

## Acceptance

Ready only if selected branch expansion passes oracle/correctness on host and board with `mismatches=0`, activation share stays below 40%, and the new bottleneck is explicitly classified.
