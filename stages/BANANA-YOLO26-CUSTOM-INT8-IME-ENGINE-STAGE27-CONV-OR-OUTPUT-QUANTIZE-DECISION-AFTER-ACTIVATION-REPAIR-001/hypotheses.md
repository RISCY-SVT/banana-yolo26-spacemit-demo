# Stage27 Pre-Registered Hypotheses

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE27-CONV-OR-OUTPUT-QUANTIZE-DECISION-AFTER-ACTIVATION-REPAIR-001

H1: Stage26 A3 replay remains ONNX-cut byte-exact under RNE/RTZ/RDN/RUP/RMM and stable under `taskset -c 0-3`.

H2: After Stage26 A3, Conv remains the dominant selected-cut bucket; per-Conv attribution can identify whether branch0, branch1, or model4_cv2 is the best next target.

H3: The immediate low-risk in-track win is a persistent cluster0 worker pool / thread-region reuse, not a `vmadot1/2/3` implementation, if thread overhead is still material.

H4: If per-Conv roofline shows severe structural low-K/MMT4D underutilization, the correct output is a decision packet for a future Conv/`vmadot123` proof lane, not implementation in Stage27.

H5: Output quantize is secondary unless replay shows it exceeds 20% and Conv repair has no low-risk path.
