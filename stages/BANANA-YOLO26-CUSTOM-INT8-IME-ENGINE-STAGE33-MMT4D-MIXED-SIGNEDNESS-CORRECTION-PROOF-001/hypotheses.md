# Stage33 Hypotheses

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE33-MMT4D-MIXED-SIGNEDNESS-CORRECTION-PROOF-001

Note: these hypotheses were restored as a tracked Stage33 artifact during implementation after preflight. The implementation and reports still evaluate them explicitly.

H1: `smt.vmadotus` implements `u8 x s8 -> s32` with the same accumulator semantics proven in Stage32.

H2: For `/model.4/cv2/conv/Conv`, activation can be packed as uint8 operand A and weights remain int8 operand B, reducing or eliminating part of the current conversion/correction path.

H3: The candidate can preserve the exact same selected ONNX-cut output SHA:

```text
70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
```

H4: The candidate reduces correction/conversion time measurably. A selected-cut total speedup is useful but not required if the correction bucket is reduced and correctness is exact.

H5: If mixed signedness does not reduce correction/conversion or total time, do not select it as the accepted runner mode; record a negative result and recommend MMT4D tile/thread-overhead or output-quantize follow-up.
