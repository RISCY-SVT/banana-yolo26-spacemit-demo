# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE4-PACKING-REPAIR-001

You are Codex working in the Banana-Pi BPI-F3 / SpacemiT K1X / X60 / YOLO26 / custom INT8 IME `smt.vmadot` engine project.

User-facing summaries must be in Russian. Code, comments, identifiers, commands, paths, report filenames, and artifact names must be in English.

## Mission

Repair Conv3x3 packing/im2col dataflow before first graph-block integration.

Stage 3 proved selected real Conv1x1 and Conv3x3 correctness against ONNX CPU oracle, and showed Conv1x1 prepacked IME is faster than scalar. Conv3x3 correctness passed, but im2col/A packing cost dominated and prepacked IME did not beat scalar for the selected real-node shape.

## Scope

Implement only:

- tiled Conv3x3 A-panel reuse across adjacent output columns/rows;
- workspace layout that avoids re-reading the same input pixels for each `4xK` panel;
- optional row-ring im2col cache for stride1/pad1;
- separate microbench for A-pack only, compute only, and combined path;
- selected real-node correctness against existing Stage 3 fixture and ONNX oracle.

Do not:

- implement full YOLO26 inference;
- run COCO;
- run camera;
- mutate `/data/ncnn`;
- use `vmadot1/2/3`, `vmadotn`, FP, or `vfmadot`;
- make model FPS or production claims;
- push source branches.

## Acceptance

- Host CTest passes.
- Cross build passes.
- Board cluster0 real Conv3x3 fixture passes with mismatches `0`.
- Conv3x3 packing-included path improves materially over Stage 3 prepacked result for `160x160x16->8`.
- If improvement is not achieved, produce a blocker report with exact cost breakdown.
