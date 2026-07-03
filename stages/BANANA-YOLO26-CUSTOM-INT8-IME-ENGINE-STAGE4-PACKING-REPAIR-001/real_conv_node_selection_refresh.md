# Real Conv Node Selection Refresh

Stage 4 kept the Stage 3 selected real Conv nodes from the accepted CPU-good Q/DQ artifact:

`/data/banana-yolo26-spacemit-demo/.deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx`

No new Q/DQ artifact was generated. XSlim was not used.

## Selected Nodes

| role | node | shape | reason |
|---|---|---|---|
| first real Conv | `/model.0/conv/Conv` | `[1,3,640,640] -> [1,16,320,320]`, 3x3 stride2 pad1 | Documented first graph Conv; not the Stage 4 repair benchmark because `Cin=3` is tail-heavy. |
| real Conv1x1 target | `/model.2/cv1/conv/Conv` | `[1,32,160,160] -> [1,32,160,160]`, 1x1 stride1 pad0 | Selected for prepacked MMT4D Conv1x1 dataflow repair. |
| real Conv3x3 target | `/model.2/m.0/cv1/conv/Conv` | `[1,16,160,160] -> [1,8,160,160]`, 3x3 stride1 pad1 | Selected for fused im2col-to-packA repair. |

## Quantization Boundary

- Real Conv1x1 activation zero-point: `0`.
- Real Conv3x3 activation zero-point: `2`.
- Selected weights are signed int8 with zero-point `0`.
- Stage 4 raw kernels compute `sum(s8 * s8) -> s32`.
- Real selected-node correction remains outside the microkernel.
