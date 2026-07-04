# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE7-BACKBONE-SUBSET-EXPANSION-001

You are Codex working in the Banana-Pi BPI-F3 / SpacemiT K1X / X60 / YOLO26 custom INT8 IME `smt.vmadot` engine project.

User-facing summaries must be in Russian. Code, commands, paths, identifiers, report filenames, and artifact names stay in English.

## Mission

Extend Stage 6 from `candidate_C_block0_silu_model1_conv` to the next bounded backbone subset only if tensor contracts remain clear.

Stage 6 proved:

```text
selected_subset: candidate_C_block0_silu_model1_conv
boundary: /model.0/conv/Conv -> Conv0 Q/DQ -> SiLU -> Act0 Q/DQ -> /model.1/conv/Conv
host CTest: pass
board CPU0/1/2/3 correctness: pass
board CPU0 microbench: scalar 1009980 us, IME 419769 us
activation fallback: 286942 us and dominant
```

## Hard Boundaries

Do not implement full YOLO26 inference, graph-wide scheduler, COCO/mAP, camera path, production claims, ncnn source mutation, XSlim, `vmadot1/2/3`, `vmadotn`, or FP/vfmadot.

## Recommended Stage 7 Scope

1. Integrate `/model.1` output Q/DQ + SiLU/requant.
2. Include `/model.2/cv1/conv/Conv` only if the intervening tensor contract is clear and branch handling can still be avoided.
3. Keep activation/requant timing separate.
4. Add a targeted optimization decision for activation/requant:
   - scalar float fallback remains correctness reference;
   - optional LUT or fixed-point approximation can be evaluated only against ONNX CPU oracle.
5. Maintain persistent prepack/workspace and cluster0-only IME execution.
6. Stop before `/model.2/Split` unless a stage-specific branch contract is explicitly approved.

## Acceptance

- ONNX CPU oracle pass for the selected Stage 7 subset.
- Host CTest pass.
- RISC-V cross build pass.
- Board CPU0-3 correctness pass with mismatches `0`.
- Component timing shows Conv and activation costs separately.
- No full-model or production claims.

