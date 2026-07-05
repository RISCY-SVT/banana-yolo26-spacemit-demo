# Sliding Vmadot Future Lane Note

`vmadotn` remains not authorized.

`vmadot1`, `vmadot2`, and `vmadot3` are future direct-conv/sliding candidates only. Stage16 does not implement them and does not use them in any Conv path.

A separate proof stage is required before any sliding-op implementation:

- parser/assembler acceptance
- disassembly evidence
- board CPU0-3 execution
- CPU4/5 negative check only if safe and authorized
- scalar oracle
- direct-conv vs current MMT4D comparison

Open a separate proof stage only if representative/full-shape Conv/IME remains dominant and MMT4D/im2col/packing is a proven bottleneck.
