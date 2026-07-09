# Stage37 Hypotheses

H1: Stage36 A1 remains byte-exact and reproduces the selected-cut total and bucket shares within same-session noise.

H2: After Stage36, `/model.4/cv2` is no longer the dominant single Conv target; the remaining branch 3x3 convs and/or output QuantizeLinear/thread overhead are the next local bottlenecks.

H3: If the two `/model.4/m.0` branch 3x3 Conv GEMM compute parts dominate, the Stage36 4-accumulator software-pipelined `smt.vmadot` MMT4D kernel can improve their GEMM compute while preserving ONNX-cut output bytes.

H4: For 3x3 Conv nodes, im2col/pack cost must be reported separately. If im2col/pack dominates, Stage37 must not overstate the benefit of GEMM pipelining.

H5: Exactly one repair lane will be selected after replay and attribution. If no lane has a credible >=5% selected-cut total improvement, Stage37 will stop with a decision report rather than forcing a small or risky change.
