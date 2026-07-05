# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE15-MODEL4-C2F-BRANCH-ENTRY-001

You are Codex working in the Banana-Pi BPI-F3 / SpacemiT K1X / X60 / YOLO26 custom INT8 IME `smt.vmadot` engine project.

User-facing summaries must be in Russian. Code, comments, identifiers, commands, paths, report filenames, and artifact names must be in English.

## Mission

Expand from Stage 14 `candidate_H3_model2_act_model3_act_model4_cv1_conv` to the first safe `/model.4` C2f branch boundary.

Primary target:

```text
/model.4/cv1/conv/Conv output
/model.4/cv1/act/Sigmoid
/model.4/cv1/act/Mul
/model.4/Split
first tractable /model.4 branch Conv if oracle and tensor contracts are clear
```

## Required Boundaries

1. Recover Stage 14 reports and verify `stage14-next-c2f-expanded-ready-for-next-stage`.
2. Replay Stage 14 host tests, cross build, CPU0-3 board correctness, and CPU0 microbench.
3. Inspect `/model.4` graph from the accepted manual Q/DQ ONNX artifact.
4. Generate boundary-specific LUT oracles for any new activation Q/DQ.
5. Document Split contract before implementation.
6. Do not cross Add/Concat unless contracts are explicit and oracle-safe.

## Non-actions

Do not implement full YOLO26 inference, graph-wide scheduler, camera/full-image demo, COCO/mAP, production/model FPS, ncnn mutation, XSlim, `vmadot1/2/3`, `vmadotn`, FP/vfmadot, or push.

## Suggested Classification Set

```text
stage15-model4-branch-entry-ready-for-c2f-completion
stage15-model4-split-correct-branch-deferred
stage15-correct-but-conv-dominates
stage15-blocked-model4-split-contract
stage15-blocked-board-correctness
stage15-partial-needs-human-decision
```
