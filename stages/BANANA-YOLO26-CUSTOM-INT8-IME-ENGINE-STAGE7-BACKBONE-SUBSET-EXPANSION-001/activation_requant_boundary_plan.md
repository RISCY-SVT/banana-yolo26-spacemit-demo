# Activation Requant Boundary Plan

Stage 7 includes two activation/requant fallback buckets:

- `act0_requant_us`: `/model.0/conv` corrected int32 -> Conv0 Q/DQ -> SiLU -> Act0 Q -> signed storage for `/model.1/conv`.
- `act1_requant_us`: `/model.1/conv` corrected int32 -> Conv1 Q/DQ -> SiLU -> Act1 Q -> signed storage for `/model.2/cv1/conv`.

Implementation policy:

- scalar float fallback for SiLU and dequant/requant in Stage 7;
- exact `nearbyint`/nearest-even quantization to match Stage 6 policy and ONNX CPU oracle;
- clamp to `[0,255]`;
- store Conv input as signed int8 using `q_u8 - 128`;
- measure each activation/requant bucket separately;
- do not make this a full activation optimization stage.

If activation/requant remains dominant after expansion, recommend `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE8-ACTIVATION-REQUANT-OPTIMIZATION-001`.
