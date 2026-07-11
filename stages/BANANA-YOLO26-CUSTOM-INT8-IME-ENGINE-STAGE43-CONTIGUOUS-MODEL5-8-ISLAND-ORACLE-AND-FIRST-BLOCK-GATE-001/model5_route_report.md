# Model5 Route Report

Route:

1. NHWC uint8 model4 preactivation input is consumed directly.
2. A prepared 256-entry LUT produces model4 postactivation signed-code NHWC.
3. A persistent four-thread spatial-row pool runs model5 3x3 stride-2 MMT4D Conv on CPU0-3.
4. The existing Stage37 four-accumulator named `smt.vmadot` kernel computes the GEMM tiles.
5. Existing exact zero-point correction and bias semantics produce corrected int32 NHWC.
6. Prepared fixed-point requant plus the 256-entry SiLU LUT emits model5 signed-code NHWC.

Weights are packed at prepare time. The hot path has no custom allocation, file I/O, Python, graph-name lookup, or NCHW/NHWC conversion. Disassembly records the existing named `smt.vmadot` route; no new ISA lane or raw opcode was introduced.
