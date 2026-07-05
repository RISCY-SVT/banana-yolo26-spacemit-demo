# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE17-CONV-IME-ROOFLINE-AND-CLUSTER0-THREADING-FEASIBILITY-001

User-facing summaries must be in Russian. Code, commands, paths, identifiers, report filenames, and artifact names stay in English.

Mission: investigate the Stage16 representative/full-shape Conv-dominant bottleneck without expanding to a full engine.

Scope:

- Replay Stage16A representative/full-shape branch-entry gate.
- Preserve Stage16 compact model4 C2f correctness.
- Build a Conv/IME roofline for `/model.4/m.0/cv1/conv/Conv`, `/model.4/m.0/cv2/conv/Conv`, and `/model.4/cv2/conv/Conv` where representative/full-shape fixtures exist.
- Test controlled cluster0 threading feasibility: CPU0, CPU0-1, CPU0-2, CPU0-3.
- Keep IME on CPU0-3 only.
- Do not use CPU4-7 for IME.
- Do not implement full YOLO26 inference, graph-wide scheduler, camera, COCO/mAP, model FPS claim, XSlim, `/data/ncnn` mutation, `vmadot1/2/3`, `vmadotn`, or FP/vfmadot.

Acceptance:

- Host tests pass.
- RISC-V cross build passes.
- Board CPU0-3 correctness passes.
- Representative/full-shape timing is reported with non-overlapping buckets.
- Any threading result is selected-subset evidence only, not model FPS.
