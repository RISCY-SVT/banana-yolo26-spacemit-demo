# Hypotheses

H1: The current `/model.4/cv2` MMT4D compute path is partly `smt.vmadot` latency/load-use bound; a software-pipelined kernel with multiple independent accumulator groups can reduce `model4_cv2_compute_us`.

H2: A 6-accumulator kernel can approach Stage35 throughput headroom only if real loads, address arithmetic, correction, and writeback are controlled.

H3: If the candidate fails `>=1.25x` `cv2` compute speedup, the result is still useful as a real-kernel ceiling/diagnostic and must not be forced into the selected path.

H4: ONNX-cut byte-exactness and FRM robustness are hard gates; speed without exact output is rejected.
