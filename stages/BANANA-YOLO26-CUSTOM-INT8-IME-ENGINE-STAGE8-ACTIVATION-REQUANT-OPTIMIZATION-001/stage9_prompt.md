# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE9-ACTIVATION-FUSION-AND-PACK-HANDOFF-001

You are Codex working in the Banana-Pi BPI-F3 / SpacemiT K1X / X60 / YOLO26 custom INT8 IME `smt.vmadot` engine project.

User-facing summaries must be in Russian. Code, comments, identifiers, commands, paths, report filenames, and artifact names must be in English.

## Mission

Continue after Stage 8, which reduced selected-subset activation/requant from about `465901 us` to `192568 us`, but activation still consumed about `55%` of selected-subset IME total.

Stage 9 must focus on activation fusion and handoff to the next Conv input layout before graph expansion.

## Scope

Allowed:

- keep the same Stage 7 selected subset;
- optimize Act0 and Act1 only;
- test fused `requant -> LUT -> packA/next Conv handoff` if it can be isolated;
- measure activation-only, Conv-only, and selected-subset timings separately;
- run host CTest, RISC-V cross build, and board CPU0-3 correctness;
- use only plain `smt.vmadot` MMT4D for Conv kernels.

Forbidden:

- full YOLO26 inference;
- graph-wide scheduler;
- camera/full-image demo;
- COCO/mAP;
- model FPS or production claims;
- `/data/ncnn` mutation;
- XSlim;
- `vmadot1/2/3`, `vmadotn`, FP/vfmadot;
- toolchain/sysroot/board OS mutation;
- push.

## Acceptance

Ready for graph expansion only if:

- correctness passes on host and board CPU0-3;
- selected-subset IME total remains faster than scalar;
- activation share is below `40%`, or the remaining bottleneck is no longer activation/requant;
- no full-engine or production claim is made.
