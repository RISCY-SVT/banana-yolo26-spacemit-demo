# vmadot123 Future Lane Note

Stage20 did not implement `vmadot1`, `vmadot2`, `vmadot3`, `vmadotn`, FP, or `vfmadot`.

Plain `smt.vmadot` MMT4D remains the only implementation primitive in this lane.

`vmadot1/2/3` may remain future direct-convolution/sliding-window candidates, but Stage20 evidence shows the selected model4 C2f issue was merge/post-Concat-QDQ dataflow, not a need for a new IME instruction. Any sliding-op stage must separately prove parser/assembler/disassembly, board CPU0-3 execution, scalar oracle, and per-node speedup versus the current threaded/MMT4D path.

`vmadotn` remains not authorized.
