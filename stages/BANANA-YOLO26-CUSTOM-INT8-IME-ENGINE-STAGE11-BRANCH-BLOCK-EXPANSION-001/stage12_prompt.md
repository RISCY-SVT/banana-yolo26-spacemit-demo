# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE12-C2F-RESIDUAL-CONCAT-COMPLETION-001

User-facing summaries must be in Russian. Code, commands, paths, report filenames, and artifact names stay in English.

## Mission

Complete the bounded `/model.2/m.0` C2f branch contract after Stage 11:

- add `/model.2/m.0/cv2` activation/QDQ boundary;
- define and prove `/model.2/m.0/Add`;
- define and prove `/model.2/Concat`;
- stop before `/model.2/cv2/conv/Conv` unless Concat Q/DQ is fully verified and scoped.

## Required Starting Evidence

- Stage 11 classification: `stage11-branch-cv2-correct-add-deferred`
- selected subset: `candidate_F_model2_m0_cv1_act_cv2_conv`
- Stage 11 board CPU0-3 correctness: pass, mismatches `0`
- Stage 11 CPU0 microbench: IME A2 total `269372 us`, activation share `14.8755%`
- Add deferred because current ONNX Add is float-domain before Concat Q/DQ.

## Hard Limits

Do not implement full YOLO26 inference, graph-wide scheduler, camera/full-image demo, COCO/mAP, production/model FPS, XSlim, `/data/ncnn` mutation, `vmadot1/2/3`, `vmadotn`, or FP/vfmadot.

## Stage 12 Gates

- ONNX CPU oracle for cv2 activation, Add, and Concat.
- Boundary-specific 256-code LUT oracle for every new activation/QDQ boundary.
- Host CTest pass.
- RISC-V cross build pass.
- Board CPU0-3 correctness pass with mismatches `0`.
- Component timings for cv2 activation, Add, Concat, copy/layout, and Conv buckets.
