# Stage 44 Prompt

Stage ID:

`BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE44-MODEL5-STRIDE2-PACK-AND-EXACT-REQUANT-REPAIR-001`

Start from the unchanged Stage43 start HEAD plus the Stage43 result packet. Do not implement model6.

Mission: run a bounded model5-only repair that preserves the Stage43 semantic fixed-host oracle and persistent NHWC handoff. Decompose the exact model5 path into stride-2 im2col/pack, vmadot compute, correction, fixed requant, and worker overhead. Test at most one stride-2 pack candidate and one exact requant candidate. Do not emulate x86 MLAS pair saturation.

Hard gates:

- F0-F7 scalar and IME exact against host ORT 1.27 `ORT_DISABLE_ALL` semantic cuts;
- operational `ORT_ENABLE_ALL` outputs remain a separately labeled integration artifact;
- CPU0-3 IME only;
- no new ISA, model6, graph expansion, full engine, FPS, camera, COCO, or production claim;
- model5 Conv plus postactivation must beat same-session isolated board ORT model5 by a stable margin before any full scaffold integration.

If no plausible repair can remove at least the measured `8716.941 us` gap, stop custom model5 work and return to runtime/island strategy selection.
