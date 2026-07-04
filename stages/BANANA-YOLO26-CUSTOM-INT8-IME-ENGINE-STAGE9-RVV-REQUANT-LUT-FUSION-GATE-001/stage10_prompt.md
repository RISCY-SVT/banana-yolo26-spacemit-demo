# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE10-BACKBONE-EXPANSION-POST-ACTIVATION-GATE-001

Use this only after review/approval of Stage 9.

## Mission

Expand the custom YOLO26 INT8 IME prototype beyond the Stage 7/8/9 selected subset now that activation/requant no longer dominates the selected subset.

## Starting Evidence

- Stage 9 selected mode: `A2_rvv_f32_lut`.
- Selected subset: `candidate_D_block0_silu_model1_silu_model2_cv1_conv`.
- CPU0 selected-subset total: `182420 us`.
- Activation total: `24471.3 us`.
- Activation share: `13.4148%`.
- Board CPU0/1/2/3 correctness: pass.

## Hard Boundaries

- Do not implement full YOLO26 inference.
- Do not implement a graph-wide scheduler.
- Do not run COCO/mAP.
- Do not run camera/full-image demo.
- Do not make model FPS or production claims.
- Do not mutate `/data/ncnn`.
- Do not use XSlim.
- Do not implement `vmadot1/2/3`, `vmadotn`, or FP/vfmadot.
- IME remains CPU0-3 cluster0 only.

## Suggested Scope

Select the next bounded backbone subset after `/model.2/cv1/conv/Conv`, including only the next branch/Split boundary if tensor contracts and ONNX CPU oracle are clear.

Required:

- Recover Stage 9 reports.
- Re-run Stage 9 selected-subset baseline.
- Extract ONNX CPU oracle for the proposed next subset.
- Keep A2 RVV activation/requant path explicit and fallback-safe.
- Preserve scalar reference path.
- Run host CTest, RISC-V cross build, and board CPU0-3 correctness.
- Benchmark selected subset only; no full-model claims.

Candidate next classification:

`stage10-backbone-expansion-ready-for-next-subset`
