# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE42-INPROCESS-ORT-CONTRACT-REPAIR-AND-MODEL16-ORACLE-GATE-001

## Mission

Repair or explicitly scope the Stage41 board in-process ORT CPU reference contract mismatch, then build the same-input ONNX cut oracle for the provisional next custom block:

```text
model.16
start: /model.15/Concat_output_0_DequantizeLinear_Output
end: /model.16/cv2/act/Mul_output_0_DequantizeLinear_Output
```

Do not implement optimized `model.16` kernels until the in-process reference contract is closed or the stage explicitly chooses a host-oracle-only debug scaffold policy.

## Required Gates

```text
1. Reproduce Stage41 host C++ in-process exactness.
2. Diagnose board ORT CPU mismatch:
   - board full ORT vs host accepted ORT
   - board model4 cut vs custom model4
   - runtime/library version and graph optimization policy
3. Choose one reference policy:
   - accepted host ORT oracle for custom correctness; board ORT fallback timing only
   - or board ORT runtime that matches accepted oracle
4. Generate model.16 ONNX cut and tensor manifests.
5. Compare model.16 cut outputs on deterministic input.
6. No full YOLO26 engine, no production FPS, no camera/COCO/mAP, no new ISA lane.
```

## Expected Outcome

```text
classification options:
  stage42-inprocess-ort-contract-repaired-model16-oracle-ready
  stage42-host-oracle-policy-selected-model16-oracle-ready
  stage42-blocked-inprocess-ort-contract
```
